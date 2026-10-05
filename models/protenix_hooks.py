"""Runtime hooks installed into the backend's sampling path.

Two hooks, both applied before the model is built:

  * **diffusion-side search** (:func:`install_diffusion`), the coupled search inside the sampling
    loop: fk / tds / cast / noise_search / af3_red / boltz_perturb, driven by a
    :class:`DiffusionSearchSpec` read from ``$PROTENIX_DIFFUSION_SPEC``.
  * **trunk-side latent modulation** (:func:`install`), a transform on the pre-Pairformer pair
    representation ``z_pre``, driven by a :class:`LatentModulationSpec` read from
    ``$PROTENIX_LATENT_MOD_SPEC``.

The latent-modulation methods act on z_pre, before the Pairformer, as their references specify:

  * **rescale**, Pair-Representation Scaling (Suzuki et al., bioRxiv 2026.01.23.701250):
    ``z <- (1+beta)*z``, one isotropic scalar beta, applied at EVERY recycle.
  * **conformets**, ConforNets (arXiv:2604.18559): channel-wise affine ``phi(z) = z W^T + b``
    applied at the FINAL Pairformer pass only, with gradients enabled only in that pass. (W, b) come
    from an unsupervised diversity optimisation whose decoder is the real trunk path,
    ``distogram_head(Pairformer(phi(z_pre)))``.

Design, which has to work across a subprocess boundary: ``ProtenixBackend.run`` writes the spec to
JSON and launches the runner through ``models/_protenix_launch.py``, which calls
:func:`enable_from_env` + :func:`install` before the model is built. :func:`install` wraps
``Protenix.get_pairformer_output`` for model-level access to ``distogram_head`` and, during that
call, temporarily intercepts ``PairformerStack.forward`` to modulate its input z_pre.

Import-safe without the backend or torch. Only the install functions touch the backend, and they
raise an actionable error if an attach point cannot be located.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import importlib
import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)

_COFOLD_PKG = os.environ.get("TTS_COFOLD_PKG") or "protenix"
_COFOLD_CLASS_BY_PKG = {"protenix": "Protenix", "opendde": "OpenDDE"}
_COFOLD_MODEL_CLASS = (os.environ.get("TTS_COFOLD_MODEL_CLASS")
                       or _COFOLD_CLASS_BY_PKG.get(_COFOLD_PKG))
if not _COFOLD_MODEL_CLASS:
    raise RuntimeError(
        f"TTS_COFOLD_PKG={_COFOLD_PKG!r} is not a known co-folding package "
        f"({sorted(_COFOLD_CLASS_BY_PKG)}). Set TTS_COFOLD_MODEL_CLASS to its model class name.")


def cofold_pkg() -> str:
    """The co-folding package the hooks are bound to, for callers that must VERIFY it."""
    return _COFOLD_PKG


def _cofold_module(dotted: str):
    """Import ``dotted`` from the ACTIVE co-folding package."""
    return importlib.import_module(dotted.replace("protenix", _COFOLD_PKG))


def _cofold_model_class():
    """The model class the hooks wrap (``Protenix`` or ``OpenDDE``)."""
    return getattr(_cofold_module("protenix.model.protenix"), _COFOLD_MODEL_CLASS)



_SYNC_TIMERS = os.environ.get("PROTENIX_TTS_SYNC_TIMERS", "1") != "0"


def _cuda_sync() -> None:
    """Drain the CUDA stream before reading a clock. ONE copy, called by EVERY GPU-work timer."""
    if not _SYNC_TIMERS:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:                                       # noqa: BLE001 -- timing never breaks a run
        pass

# Import the spec type without importing torch (torch only needed inside the optimisation).
from method.trunk_side.latent_modulation import LatentModulationSpec  # noqa: E402

_ACTIVE: Optional[LatentModulationSpec] = None
ENV_SPEC = "PROTENIX_LATENT_MOD_SPEC"     # path to the per-variant spec JSON
ENV_REAL_RUNNER = "PROTENIX_REAL_RUNNER"  # path to the real runner/inference.py
ENV_CONFORMETS_CACHE = "PROTENIX_CONFORMETS_CACHE"  # shared dir: ConforNets snapshots cached per target
ENV_RECYCLE_TAP = "PROTENIX_USE_RECYCLE"      # 1-based recycle to CONDITION diffusion on
ENV_RECYCLE_PROBE = "PROTENIX_RECYCLE_PROBE"  # dir: per-recycle antibody-antigen contact profiles
ENV_SITE_ARCHIVE = "PROTENIX_SITE_ARCHIVE"      # shared dir: visited binding-site placements, per target
                                          # so all K variants apply the SAME optimised grid (determinism)

from method.diffusion_side import diffusion_search as _ds  # noqa: E402
from method.diffusion_side import boltz_perturb as bpmod  # noqa: E402
from method.diffusion_side.diffusion_search import (  # noqa: E402
    DiffusionSearchSpec, spec_from_dict as diffusion_spec_from_dict)
_ACTIVE_DIFF: Optional[DiffusionSearchSpec] = None
_DIFF_SPEC_PATH: Optional[str] = None      # spec JSON path, diffusion_timing.jsonl is written beside it
ENV_DIFF_SPEC = "PROTENIX_DIFFUSION_SPEC"  # path to the diffusion-search spec JSON
ENV_EPITOPE_MAP = "PROTENIX_EPITOPE_MAP"
ENV_CDR_MAP = "PROTENIX_CDR_MAP"           # JSON: sample_name -> paratope token indices (cdr_rmsd reward)
_PHYS_PENALTY = 50.0        # log-weight penalty for an unphysical particle (finite, not -inf,
                            # so an all-unphysical population still normalises instead of NaN)
_CURRENT_SAMPLE_NAME: Optional[str] = None  # set by the predict patch, the cdr_rmsd reward's map key
                                            # (inference is sequential, one target/process, so a global is safe)


# ---------------------------------------------------------------- PAE-as-.npz saving
def patch_dumper_pae_npz() -> None:
    """Make Protenix save the per-sample PAE as a compact ``*_pae_sample_N.npz`` instead of
        the ~15 MB ``*_full_data_sample_*.json`` (an a1e-scale run is ~78 GB of those JSONs).
    """
    import numpy as np
    import runner.dumper as dm

    if getattr(dm, "_pae_npz_patched", False):
        return
    _orig_save_json = dm.save_json

    def save_json(obj, path, indent=None, *args, **kwargs):
        sp = str(path)
        if "_full_data_sample_" in sp and isinstance(obj, dict) and obj.get("token_pair_pae") is not None:
            npz = (sp[:-5] if sp.endswith(".json") else sp).replace("_full_data_sample_", "_pae_sample_") + ".npz"
            np.savez_compressed(npz, pae=np.asarray(obj["token_pair_pae"], dtype=np.float32))
            return
        return _orig_save_json(obj, path, indent=indent, *args, **kwargs)

    dm.save_json = save_json
    dm._pae_npz_patched = True
    logger.info("dumper: PAE → *_pae_sample_N.npz (full_data JSON suppressed)")


# ---------------------------------------------------------------- confidence PAE offload (OOM fix)
def patch_confidence_offload() -> None:
    """Lower the confidence head's per-sample PAE/PDE CPU-offload threshold."""
    thr = os.environ.get("PROTENIX_CONF_OFFLOAD_TOKENS")
    if not thr:
        return
    try:
        threshold = int(thr)
    except ValueError:
        raise RuntimeError(
            f"PROTENIX_CONF_OFFLOAD_TOKENS={thr!r} is not a number. The backend must resolve 'auto' "
            "to a concrete token threshold before launching inference "
            "(see models.protenix2.ProtenixBackend._resolve_offload_tokens). Refusing to continue "
            "with confidence CPU-offload silently disabled: large targets OOM without it.") from None
    try:
        import torch
        ConfidenceHead = getattr(_cofold_module("protenix.model.modules.confidence"),
                                 "ConfidenceHead")
    except (ImportError, AttributeError) as e:
        logger.warning("conf-offload: attach point not found (%s); skipping", e)
        return
    if getattr(ConfidenceHead, "_tts_offload_patched", False):
        return
    orig_mef = ConfidenceHead.memory_efficient_forward

    def _mef(self, *a, **k):
        out = orig_mef(self, *a, **k)
        try:
            z = k.get("z_pair")
            if z is None and len(a) >= 2:
                z = a[1]
            if z is not None and z.shape[-2] > threshold and not self.training and isinstance(out, tuple):
                plddt, pae, pde, resolved = out
                out = (plddt, pae.cpu(), pde.cpu(), resolved)     # offload the big PAE/PDE to host RAM
                torch.cuda.empty_cache()
        except Exception as e:                                    # noqa: BLE001, never break inference
            logger.warning("conf-offload wrap failed: %s", e)
        return out

    ConfidenceHead.memory_efficient_forward = _mef
    ConfidenceHead._tts_offload_patched = True
    logger.info("confidence PAE/PDE CPU-offload enabled for N_token > %d", threshold)


# ---------------------------------------------------------------- chunk the confidence stage
class _DeferredConf:
    """Placeholder for one of the four confidence-head outputs, until the summary consumes it."""
    __slots__ = ("holder", "slot")

    def __init__(self, holder, slot):
        self.holder, self.slot = holder, slot

    def _ref(self):
        """A tensor sharing the pending output's device/dtype."""
        h = self.holder or {}
        for v in (h.get("kwargs") or {}).values():
            if hasattr(v, "device") and hasattr(v, "dtype"):
                return v
        model = h.get("model")
        if model is not None and hasattr(model, "parameters"):
            for p in model.parameters():
                return p
        return None

    @property
    def device(self):
        t = self._ref()
        if t is None:
            raise AttributeError("device: no reference tensor on the deferred confidence call")
        return t.device

    @property
    def dtype(self):
        t = self._ref()
        if t is None:
            raise AttributeError("dtype: no reference tensor on the deferred confidence call")
        return t.dtype

    @property
    def shape(self):
        """Shape of the pending output, derived from the captured call, materialising nothing."""
        import torch                        # module-level `torch` does not exist in this file
        h = self.holder or {}
        kw = h.get("kwargs") or {}
        x = kw.get("x_pred_coords")
        s = kw.get("s_trunk")
        if self.slot not in (1, 2) or x is None or s is None:
            raise AttributeError(
                f"shape: not derivable for slot {self.slot} of the deferred confidence call")
        n_sample = int(x.shape[-3])          # [..., N_sample, N_atom, 3]
        n_token = int(s.shape[-2])           # [..., N_token, c_s]
        head = getattr(h.get("model"), "confidence_head", None)
        bins = getattr(head, "b_pae" if self.slot == 1 else "b_pde", 64)
        return torch.Size((n_sample, n_token, n_token, int(bins)))

    def to(self, *a, **kw):
        """No-op when the request names the device the outputs already land on."""
        import torch                        # module-level `torch` does not exist in this file
        want = kw.get("device")
        if want is None:
            for x in a:
                if isinstance(x, (torch.device, str)):
                    want = x
                    break
        if want is None:                      # .to(dtype) or .to() -> nothing to move
            return self
        if torch.device(want).type == torch.device(self.device).type:
            return self
        raise AttributeError(
            f"to({want}): a deferred confidence output cannot be moved off "
            f"{self.device} without materialising the full sample stack")


def patch_chunked_confidence() -> None:
    """Produce and consume the confidence logits in SAMPLE SUB-BATCHES instead of all at once."""
    if os.environ.get("TTS_NO_CONF_CHUNK"):          # escape hatch if the proxy bites again
        logger.info("chunked-confidence: disabled via TTS_NO_CONF_CHUNK")
        return
    _odde = _COFOLD_PKG == "opendde"
    if _odde and os.environ.get("TTS_ODDE_CONF_CHUNK", "1") in ("0", "false", "False"):
        logger.info("chunked-confidence: opendde chunking disabled via TTS_ODDE_CONF_CHUNK=0")
        return
    if _COFOLD_PKG not in ("protenix", "opendde"):
        logger.info("chunked-confidence: NOT patching %s (its confidence path reads the tensors "
                    "directly, so the deferred proxy would break every seed)", _COFOLD_PKG)
        return
    try:
        k = int(os.environ.get("PROTENIX_CONF_SAMPLE_CHUNK", "5") or 5)
    except ValueError:
        k = 5
    if k <= 0:
        return
    try:
        import torch
        Protenix = _cofold_model_class()
        sc = _cofold_module(("opendde" if _odde else "protenix") + ".model.sample_confidence")
    except (ImportError, AttributeError) as e:
        logger.warning("chunked-confidence: attach point not found (%s); skipping", e)
        return
    if getattr(Protenix, "_tts_confchunk_patched", False):
        return
    orig_head, orig_sum = Protenix.run_confidence_head, sc.compute_full_data_and_summary

    def run_head(self, *a, **kw):
        # defer only what we can slice safely: inference, keyword call, sample axis present
        if getattr(self, "training", False) or "x_pred_coords" not in kw or a:
            return orig_head(self, *a, **kw)
        holder = {"model": self, "kwargs": kw,
                  "amp": (torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype())}
        return tuple(_DeferredConf(holder, i) for i in range(4))

    def summary(*a, **kw):
        pae = kw.get("pae_logits")
        if not isinstance(pae, _DeferredConf):
            return orig_sum(*a, **kw)
        holder = pae.holder
        model, hk = holder["model"], holder["kwargs"]
        x = hk["x_pred_coords"]
        n = int(x.shape[-3])                       # [..., N_sample, N_atom, 3]
        amp_on, amp_dtype = holder.get("amp", (False, None))

        def _head(kwargs):
            """Run the real head under the autocast state its caller had, not the summary's."""
            if amp_dtype is None:
                return orig_head(model, **kwargs)
            with torch.autocast(device_type="cuda", enabled=amp_on, dtype=amp_dtype):
                return orig_head(model, **kwargs)

        summaries, fulls = [], []
        try:
            for s0 in range(0, n, k):
                s1 = min(s0 + k, n)
                hk_c = dict(hk)
                hk_c["x_pred_coords"] = x[..., s0:s1, :, :]
                plddt_c, pae_c, pde_c, _res = _head(hk_c)
                kw_c = dict(kw)
                kw_c["pae_logits"], kw_c["plddt_logits"], kw_c["pde_logits"] = pae_c, plddt_c, pde_c
                ac = kw.get("atom_coordinate")
                if ac is not None and getattr(ac, "ndim", 0) >= 3 and int(ac.shape[-3]) == n:
                    kw_c["atom_coordinate"] = ac[..., s0:s1, :, :]
                cp = kw.get("contact_probs")
                if cp is not None and getattr(cp, "dim", lambda: 0)() == 3 and int(cp.shape[0]) == n:
                    kw_c["contact_probs"] = cp[s0:s1]
                su, fu = orig_sum(*a, **kw_c)
                summaries += list(su)
                fulls += list(fu)
                del plddt_c, pae_c, pde_c, _res, kw_c
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:                                    # noqa: BLE001
            # never silently half-produce: redo the whole thing the original way
            logger.warning("chunked-confidence failed (%s); falling back to the unchunked path", e)
            plddt_f, pae_f, pde_f, _r = _head(hk)
            kw_f = dict(kw)
            kw_f["pae_logits"], kw_f["plddt_logits"], kw_f["pde_logits"] = pae_f, plddt_f, pde_f
            return orig_sum(*a, **kw_f)
        return summaries, fulls

    Protenix.run_confidence_head = run_head
    sc.compute_full_data_and_summary = summary
    Protenix._tts_confchunk_patched = True
    logger.info("confidence stage chunked at %d sample(s) per sub-batch", k)


# ---------------------------------------------------------------- per-target auto diffusion chunk
def patch_diffusion_chunk_auto() -> None:
    """When ``$PROTENIX_DIFFUSION_CHUNK_AUTO`` is set, size the diffusion chunk PER TARGET from its
        own ``N_token`` (like the offload decision), instead of one global value for the whole runner
        invocation. Wraps ``Protenix._main_inference_loop`` (where ``N_token`` is known) and sets
        ``configs.infer_setting.sample_diffusion_chunk_size`` for that target before ``sample_diffusion``
        reads it, using ``models.protenix2.auto_diffusion_chunk``, so a small target gets a big chunk
        (fewer passes, faster) even in a batch that also has a large target, without per-target reload.
    """
    if not os.environ.get("PROTENIX_DIFFUSION_CHUNK_AUTO"):
        return
    try:
        from models.protenix2 import auto_diffusion_chunk, _detect_gpu_gib
        Protenix = _cofold_model_class()
    except (ImportError, AttributeError) as e:
        logger.warning("diffusion-chunk-auto: attach point not found (%s); skipping", e)
        return
    if getattr(Protenix, "_tts_diffchunk_patched", False):
        return
    gib = _detect_gpu_gib()
    off_thr = os.environ.get("PROTENIX_CONF_OFFLOAD_TOKENS")
    try:
        off_thr = float(off_thr) if off_thr is not None else None
    except ValueError:
        off_thr = None
    orig_loop = Protenix._main_inference_loop

    def _loop(self, input_feature_dict, *a, **k):
        try:
            n_tok = int(input_feature_dict["residue_index"].shape[-1])
            n_samp = int(self.configs.sample_diffusion["N_sample"])
            offload = off_thr is not None and n_tok > off_thr
            if cofold_pkg() == "opendde":
                from models.opendde import opendde_auto_diffusion_chunk
                chunk = opendde_auto_diffusion_chunk(n_tok, n_samp, gpu_gib=gib, offload=offload)
            else:
                chunk = auto_diffusion_chunk(n_tok, n_samp, gpu_gib=gib, offload=offload)
            self.configs.infer_setting.sample_diffusion_chunk_size = int(chunk)
            logger.info("auto diffusion chunk per-target: N_token=%d n_samples=%d offload=%s -> chunk=%d",
                        n_tok, n_samp, offload, chunk)
        except Exception as e:                       # noqa: BLE001, never break inference
            logger.warning("diffusion-chunk-auto: could not size (%s); using config value", e)
        return orig_loop(self, input_feature_dict, *a, **k)

    Protenix._main_inference_loop = _loop
    Protenix._tts_diffchunk_patched = True
    logger.info("diffusion-chunk-auto: per-target sample_diffusion_chunk_size for %s "
                "(gpu=%.0f GiB, offload_thr=%s)", cofold_pkg(), gib, off_thr)


def patch_flop_profile() -> None:
    """When ``$PROTENIX_FLOP_PROFILE`` is set, wrap the FIRST ``Protenix.forward`` in PyTorch's
        ``torch.utils.flop_counter.FlopCounterMode`` and dump the measured per-module FLOP split
        (trunk / diffusion / confidence) to ``$PROTENIX_FLOP_PROFILE_OUT`` (default ./flop_profile.json).
    """
    if not os.environ.get("PROTENIX_FLOP_PROFILE"):
        return
    try:
        from torch.utils.flop_counter import FlopCounterMode
        Protenix = _cofold_model_class()
    except (ImportError, AttributeError) as e:
        logger.warning("flop-profile: unavailable (%s); skipping", e)
        return
    if getattr(Protenix, "_tts_flop_patched", False):
        return
    orig_forward = Protenix.forward
    state = {"done": False}

    def _fwd(self, *a, **k):
        if state["done"]:
            return orig_forward(self, *a, **k)
        state["done"] = True
        fcm = FlopCounterMode(display=False)
        try:
            with fcm:
                out = orig_forward(self, *a, **k)
        except Exception as e:                       # profiling must never break a run
            import traceback as _tb
            logger.warning("flop-profile: FlopCounterMode failed (%s: %s); running unwrapped.\n%s",
                           type(e).__name__, e, _tb.format_exc())
            return orig_forward(self, *a, **k)
        try:
            counts = fcm.get_flop_counts()           # {module_fqn: {op: flops}}
            trunk = diff = conf = 0
            for mod, ops in counts.items():
                if mod == "Global":
                    continue
                s = int(sum(ops.values())); low = mod.lower()
                if "diffusion" in low:
                    diff += s
                elif "confidence" in low:
                    conf += s
                elif any(t in low for t in ("pairformer", "msa", "template", "embedder", "input_feature")):
                    trunk += s
            total = int(fcm.get_total_flops())
            rec = {"convention": "torch FlopCounterMode (multiply-add = 2 FLOPs)",
                   "flops_total": total, "flops_trunk": trunk,
                   "flops_diffusion": diff, "flops_confidence": conf,
                   "n_cycle": int(getattr(self.configs.model, "N_cycle", 0) or 0)
                            if hasattr(self, "configs") else None}
            out_path = os.environ.get("PROTENIX_FLOP_PROFILE_OUT", "flop_profile.json")
            with open(out_path, "w") as f:
                json.dump(rec, f, indent=2)
            logger.info("FLOP PROFILE (true FLOPs): total=%.4e trunk=%.4e diffusion=%.4e conf=%.4e -> %s",
                        total, trunk, diff, conf, out_path)
        except Exception as e:                       # noqa: BLE001
            logger.warning("flop-profile: could not summarise counts (%s)", e)
        return out

    Protenix.forward = _fwd
    Protenix._tts_flop_patched = True
    logger.info("flop-profile: FlopCounterMode will wrap the first Protenix.forward")


# ---------------------------------------------------------------- host-RAM accumulation trace
def patch_mem_trace() -> None:
    """When ``$PROTENIX_MEM_TRACE`` is set, start a daemon thread that logs, every ~15 s, to
        ``$PROTENIX_MEM_TRACE_OUT`` (default ./mem_trace.jsonl): process RSS, the total size + count of
        live CPU torch tensors (the offloaded PAE/PDE land here), and the top Python allocators from
        tracemalloc. Diagnoses WHAT accumulates in host RAM across diffusion samples (PAE tensors vs
        numpy result buffers vs dict retention) so a flush can target the real cause. Opt-in, read-only,
        non-fatal.
    """
    if not os.environ.get("PROTENIX_MEM_TRACE"):
        return
    try:
        import gc
        import threading
        import time
        import tracemalloc
        import psutil
        import torch
    except ImportError as e:
        logger.warning("mem-trace: unavailable (%s); skipping", e)
        return
    out_path = os.environ.get("PROTENIX_MEM_TRACE_OUT", "mem_trace.jsonl")
    proc = psutil.Process()
    tracemalloc.start(1)

    def _loop() -> None:
        while True:
            try:
                rss = proc.memory_info().rss
                cpu_bytes = cpu_n = 0
                shapes: dict = {}
                for o in gc.get_objects():
                    if isinstance(o, torch.Tensor) and o.device.type == "cpu":
                        b = o.numel() * o.element_size()
                        cpu_bytes += b; cpu_n += 1
                        if b > 5 * 1024 ** 2:               # only track tensors > 5 MB
                            k = f"{tuple(o.shape)}:{o.dtype}"
                            shapes[k] = shapes.get(k, [0, 0]); shapes[k][0] += 1; shapes[k][1] += b
                snap = tracemalloc.take_snapshot()
                top = [f"{st.traceback[0]}={st.size/1e6:.0f}MB"
                       for st in snap.statistics("lineno")[:6]]
                big = sorted(({"shape": k, "n": v[0], "gb": round(v[1] / 1e9, 2)}
                              for k, v in shapes.items()), key=lambda d: -d["gb"])[:6]
                rec = {"t": round(time.time(), 1), "rss_gb": round(rss / 1e9, 2),
                       "cpu_tensor_gb": round(cpu_bytes / 1e9, 2), "cpu_tensor_n": cpu_n,
                       "big_cpu_tensors": big, "top_py_alloc": top}
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:                               # noqa: BLE001, a trace must never break a run
                pass
            time.sleep(15)

    threading.Thread(target=_loop, daemon=True).start()
    logger.info("mem-trace: host-RAM / CPU-tensor / tracemalloc logger -> %s", out_path)


# ---------------------------------------------------------------- stage-wall timing (all methods)
def patch_stage_timing() -> None:
    """Surface Protenix's internal stage timer (pairformer / diffusion / confidence wall) so the
        MEASURED trunk-vs-diffusion wall split is available for subset wall(k, m) analysis.
    """
    try:
        Protenix = _cofold_model_class()
    except (ImportError, AttributeError) as e:
        logger.warning("stage-timing: attach point not found (%s); skipping", e)
        return
    if getattr(Protenix, "_tts_stage_patched", False) or not hasattr(Protenix, "main_inference_loop"):
        return
    orig_loop = Protenix.main_inference_loop

    def _num(v):
        return float(sum(v)) if isinstance(v, (list, tuple)) else float(v)

    def _loop(self, *a, **k):
        out = orig_loop(self, *a, **k)
        try:
            tt = out[2] if (isinstance(out, tuple) and len(out) >= 3 and isinstance(out[2], dict)) else None
            name = _extract_sample_name(k.get("input_feature_dict") or (a[0] if a else None))
            if tt and name:
                rec = {"sample": str(name),
                       **{str(kk): _num(vv) for kk, vv in tt.items() if isinstance(vv, (int, float, list, tuple))}}
                rec["confidence_measured"] = float(getattr(self, "_tts_conf_wall", 0.0))
                rec["confidence_calls"] = int(getattr(self, "_tts_conf_calls", 0))
                _tot = float(getattr(self, "_tts_trunk_wall_total", 0.0))
                rec["trunk_measured"] = _tot - float(getattr(self, "_tts_trunk_wall_seen", 0.0))
                self._tts_trunk_wall_seen = _tot
                # per-(target,seed) file INSIDE that sample's own prediction dir (rerun-safe, matches
                # where the cifs go, so it is cleaned/moved with the target). _read_stage_wall sums.
                sdir = os.path.join(_dump_dir(self), str(name))
                os.makedirs(sdir, exist_ok=True)
                with open(os.path.join(sdir, "stage_wall.json"), "w") as f:
                    json.dump(rec, f)
        except Exception as e:                             # noqa: BLE001, instrumentation never breaks a run
            logger.warning("stage-timing write failed: %s", e)
        return out

    Protenix.main_inference_loop = _loop
    Protenix._tts_stage_patched = True
    logger.info("stage-timing hook installed (%s.main_inference_loop → per-sample stage_wall.json)",
                Protenix.__name__)


def patch_trunk_timing() -> None:
    """Synchronised trunk wall clock for EVERY run, steered or not."""
    try:
        Protenix = _cofold_model_class()
    except (ImportError, AttributeError) as e:
        logger.warning("trunk-timing: attach point not found (%s); skipping", e)
        return
    if getattr(Protenix, "_tts_trunk_timed_patched", False) \
            or not hasattr(Protenix, "get_pairformer_output"):
        return
    import time as _time
    orig_gpo = Protenix.get_pairformer_output

    def _timed_gpo(self, *a, **k):
        _t = _time.perf_counter()
        out = orig_gpo(self, *a, **k)
        _cuda_sync()                    # device completion, not enqueue -- see _SYNC_TIMERS
        try:
            dt = _time.perf_counter() - _t
            self._tts_trunk_wall = getattr(self, "_tts_trunk_wall", 0.0) + dt
            self._tts_trunk_wall_total = getattr(self, "_tts_trunk_wall_total", 0.0) + dt
        except Exception:               # noqa: BLE001, instrumentation never breaks a run
            pass
        return out

    Protenix.get_pairformer_output = _timed_gpo
    Protenix._tts_trunk_timed_patched = True
    logger.info("trunk-timing hook installed (%s.get_pairformer_output, synchronised)",
                Protenix.__name__)


_STAGE_MEM: dict = {}

_PARATOPE_IDX = [None]

_RESERVE_FN = [None]
_EMBEDOPT_GRAD_FN = [None]
_EMBEDOPT_DELTAS = [None]
_step_i = {}

_EFFECTIVE_N_SAMPLE = [None]   # effective particle count of the CURRENT sampler call.
                               # set unconditionally in _sample_diffusion, read by
                               # _write_stage_mem. See the note at the assignment.
_JOINT = {"cache": [], "names": []}

_dump_off = []         # latch: one failed write disables the dump, so a full volume costs one
                       # warning rather than 200 failed writes per trajectory

_epi_idx_for_realloc = [None]
# Observer probe for the reallocation signal: path to a JSONL sink, and the current target
# name. Set from TTS_REALLOC_PROBE. None disables every probe write.
_REALLOC_PROBE = [os.environ.get("TTS_REALLOC_PROBE") or None]
_probe_name = [""]


def patch_stage_memory() -> None:
    """Per-stage GPU peak -> ``stage_mem.json``, on every run, every backend."""
    try:
        import torch
        Model = _cofold_model_class()
    except Exception as e:                                  # noqa: BLE001
        logger.warning("stage-memory: attach point not found (%s); skipping", e)
        return
    if getattr(Model, "_tts_stage_mem_patched", False):
        return

    def _gib(x):
        return round(float(x) / (1024 ** 3), 3)

    def _shapes(a, k):
        """The largest tensor shapes this stage was handed."""
        out = []
        for v in list(a) + list(k.values()):
            try:
                if hasattr(v, "shape") and hasattr(v, "dtype") and v.ndim >= 2:
                    out.append([int(x) for x in v.shape])
            except Exception:                               # noqa: BLE001
                pass
        out.sort(key=lambda s: -_prod(s))
        return out[:3]

    def _prod(s):
        n = 1
        for x in s:
            n *= max(int(x), 1)
        return n

    def _wrap(attr, stage):
        fn = getattr(Model, attr, None)
        if fn is None:
            return False

        def _inner(self, *a, **k):
            if not torch.cuda.is_available():
                return fn(self, *a, **k)
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            rec = _STAGE_MEM.setdefault(stage, {"calls": 0, "peak_gib": 0.0})
            rec["calls"] += 1
            rec["entry_gib"] = _gib(before)
            rec.setdefault("input_shapes", []).append(_shapes(a, k))
            try:
                return fn(self, *a, **k)
            except Exception as exc:                        # noqa: BLE001
                # A FAILING stage is the most valuable row there is -- it is the only one that tells
                # you where the ceiling actually is. Record it, then re-raise unchanged.
                rec["failed"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                raise
            finally:
                rec["peak_gib"] = max(rec["peak_gib"], _gib(torch.cuda.max_memory_allocated()))
                rec["exit_gib"] = _gib(torch.cuda.memory_allocated())
        setattr(Model, attr, _inner)
        return True

    attached = [s for a, s in (("get_pairformer_output", "trunk"),
                               ("sample_diffusion", "diffusion"),
                               ("run_confidence_head", "confidence")) if _wrap(a, s)]

    loop = getattr(Model, "_main_inference_loop", None)
    if loop is None:
        logger.warning("stage-memory: _main_inference_loop absent; per-stage stats will not be written")
        Model._tts_stage_mem_patched = True
        return

    def _loop(self, *a, **k):
        _STAGE_MEM.clear()
        feat = k.get("input_feature_dict") or (a[0] if a else None)
        failure = None
        try:
            out = loop(self, *a, **k)
        except BaseException as exc:                        # noqa: BLE001, re-raised below untouched
            failure = f"{type(exc).__name__}: {str(exc)[:400]}"
            _write_stage_mem(self, feat, failure)
            raise
        _write_stage_mem(self, feat, None)
        return out

    def _write_stage_mem(self, feat, failure):
        """Emit one self-describing row. Never raises: instrumentation must not break a run."""
        try:
            name = _extract_sample_name(feat)
            n_token = None
            try:
                n_token = int(feat["residue_index"].shape[-1])
            except Exception:                               # noqa: BLE001
                pass
            cfg = getattr(self, "configs", None)
            infer = getattr(cfg, "infer_setting", None) if cfg is not None else None
            rec = {
                "sample": str(name) if name else None,
                # NOMINAL token count of the input. A stage may run at a DIFFERENT size -- see
                # stages[*].input_shapes -- and when it does, that is the number the model needs.
                "n_token": n_token,
                "n_sample": (_EFFECTIVE_N_SAMPLE[0] if _EFFECTIVE_N_SAMPLE[0] is not None
                             else getattr(getattr(cfg, "sample_diffusion", None), "N_sample", None)),
                "n_sample_requested": getattr(getattr(cfg, "sample_diffusion", None), "N_sample", None),
                "backend": _COFOLD_PKG,
                "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                "gpu_total_gib": (_gib(torch.cuda.get_device_properties(0).total_memory)
                                  if torch.cuda.is_available() else None),
                "dtype": str(getattr(cfg, "dtype", None)),
                # The knobs a memory model is fitted AGAINST -- recorded so a row is self-describing
                # and a later re-fit needs nothing but the file.
                "trunk_chunk": getattr(self, "_tts_trunk_chunk_used", None),
                "diffusion_chunk": getattr(infer, "sample_diffusion_chunk_size", None),
                "conf_sample_chunk": os.environ.get("PROTENIX_CONF_SAMPLE_CHUNK"),
                "conf_offload_tokens": os.environ.get("PROTENIX_CONF_OFFLOAD_TOKENS"),
                "steered": bool(os.environ.get(ENV_DIFF_SPEC)),
                "stages": dict(_STAGE_MEM),
                "peak_gib": max((v.get("peak_gib", 0.0) for v in _STAGE_MEM.values()), default=0.0),
                # Present ONLY on a failed run, and the reason the row is written at all in that case.
                "failure": failure,
            }
            if name:
                sdir = os.path.join(_dump_dir(self), str(name))
                os.makedirs(sdir, exist_ok=True)
                # A failed run must not overwrite the good row from a previous attempt.
                fn_out = "stage_mem_failed.json" if failure else "stage_mem.json"
                with open(os.path.join(sdir, fn_out), "w") as f:
                    json.dump(rec, f, indent=1)
            logger.info("stage-memory %s: n_token=%s peak=%.2f GiB [%s]%s", name, n_token,
                        rec["peak_gib"],
                        " ".join(f"{s}={v.get('peak_gib', 0):.2f}" for s, v in _STAGE_MEM.items()),
                        " FAILED" if failure else "")
        except Exception as e:                              # noqa: BLE001, never break a run
            logger.warning("stage-memory write failed: %s", e)

    Model._main_inference_loop = _loop
    Model._tts_stage_mem_patched = True
    logger.info("stage-memory hook installed on %s (stages: %s) -> per-sample stage_mem.json",
                Model.__name__, ", ".join(attached) or "none")



def _safe_name(name) -> str:
    """Filesystem-safe form of a sample name. One definition: it was duplicated verbatim in the
        archive path and the recycle probe, which is two places for the escaping rule to drift.
    """
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))


def _site_archive_path(sample_name):
    """Where a target's visited-placement archive lives, or None if persistence is off."""
    d = os.environ.get(ENV_SITE_ARCHIVE)
    if not d:
        return None
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{_safe_name(sample_name)}__site_archive.npy")


def _load_site_archive(sample_name):
    """Placements visited by EARLIER runs of this target, or None."""
    import numpy as np                     # numpy is not a module-level import in this file
    p = _site_archive_path(sample_name)
    if not p or not os.path.exists(p):
        return None
    try:
        return np.load(p)
    except Exception as e:                  # noqa: BLE001
        raise RuntimeError(
            f"site archive {p} exists but could not be read ({e}); refusing to silently run "
            f"intra-batch-only repulsion under a cross-seed archive label")


def _save_site_archive(sample_name, arr):
    import numpy as np                     # numpy is not a module-level import in this file
    p = _site_archive_path(sample_name)
    if p is None:                          # persistence off: in-process only, a deliberate mode
        return
    if arr is None or not len(arr):
        logger.warning("site archive %s NOT written: the run produced no placements to archive", p)
        return
    np.save(p, np.asarray(arr, dtype=np.float32))
    logger.info("site archive %s written: %d placements", p, len(arr))

_DECODE_MAX_CHUNK = 64


def conformets_decode_chunk(n_token: int):
    """Attention chunk (tile size) for the ConforNets gradient decode. None = no chunking."""
    if n_token <= 740:
        return None                                  # fits unchunked → full speed
    return max(64, min(256, int(110000 / n_token)))  # ~constant tile, 857→128, 1155→95→96, cap [64,256]


# ---------------------------------------------------------------- enable
def enable(spec: LatentModulationSpec) -> None:
    global _ACTIVE
    _ACTIVE = spec
    logger.info("latent-modulation hook ENABLED: %s", spec.label())


def spec_from_dict(d: dict) -> LatentModulationSpec:
    """Rebuild a spec from a JSON dict (ignores unknown keys like 'label')."""
    fields = LatentModulationSpec.__dataclass_fields__
    return LatentModulationSpec(**{k: v for k, v in d.items() if k in fields})


def enable_from_env() -> bool:
    """Enable from ``$PROTENIX_LATENT_MOD_SPEC`` (a JSON file). Returns True if enabled."""
    p = os.environ.get(ENV_SPEC)
    if not p:
        return False
    with open(p) as f:
        enable(spec_from_dict(json.load(f)))
    return True


# ---------------------------------------------------------------- ConforNets optimisation
def _snapshot_cache_path(spec, sample_name):
    """Per-target cache file for a ConforNets optimisation, or None if caching is off."""
    cache_dir = os.environ.get(ENV_CONFORMETS_CACHE)
    if not cache_dir or not sample_name:
        return None
    import re as _re
    safe = _re.sub(r"[^0-9A-Za-z_.@=-]", "_", str(sample_name))
    snaps = "-".join(str(s) for s in spec.snapshot_steps)
    key = (f"{safe}__s{spec.seed}_k{spec.k_nets}_n{spec.n_steps}_snap{snaps}"
           f"_lr{spec.lr:g}_clip{spec.grad_clip_norm:g}_noise{spec.init_noise_std:g}")
    return os.path.join(cache_dir, key + ".npz")


def patch_sample_name_capture() -> None:
    """Publish each sample's target name into ``_CURRENT_SAMPLE_NAME``, the ONE reliable source of the
        target id inside the model forward (``_extract_sample_name`` falls back to it).
    """
    try:
        dl = _cofold_module("protenix.data.inference.infer_dataloader")
    except ImportError as e:
        logger.warning("sample-name capture: dataloader module not found (%s); cdr_rmsd may fall back", e)
        return
    if getattr(dl, "_tts_name_capture_patched", False):
        return
    orig = dl.get_inference_dataloader

    def _wrapped(*a, **k):
        loader = orig(*a, **k)

        class _NameCapturingLoader:
            def __iter__(self):
                global _CURRENT_SAMPLE_NAME
                for batch in loader:
                    try:                                    # batch[0] = (data, atom_array, err) per infer_predict
                        first = batch[0]
                        data = first[0] if isinstance(first, (list, tuple)) else first
                        sn = data.get("sample_name") if hasattr(data, "get") else None
                        if sn is not None:
                            _CURRENT_SAMPLE_NAME = sn[0] if isinstance(sn, (list, tuple)) and sn else sn
                    except Exception:                       # noqa: BLE001, never break iteration
                        pass
                    yield batch

            def __len__(self):
                return len(loader)

            def __getattr__(self, n):
                return getattr(loader, n)

        return _NameCapturingLoader()

    dl.get_inference_dataloader = _wrapped
    dl._tts_name_capture_patched = True
    logger.info("sample-name capture: inference dataloader wrapped (cdr_rmsd reward key)")


def _extract_sample_name(feat):
    """Target id for this (target, seed). Prefers an in-band key, else the dataloader-captured global."""
    if isinstance(feat, dict):
        sn = feat.get("_conformets_sample_name") or feat.get("sample_name")
        sn = sn[0] if isinstance(sn, (list, tuple)) and sn else sn      # de-batch if listy
        if sn:
            return sn
    return _CURRENT_SAMPLE_NAME


def _diffusion_seed() -> str:
    """The diffusion RNG seed from the launcher argv (``--seeds 101``). ``sample_name`` names only the
        MSA subsample, so two runs differing solely in diffusion seed share it -- anything persisted per
        sample must include this or the later run silently overwrites the earlier one.
    """
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--seeds" and i + 1 < len(argv):
            return str(argv[i + 1]).replace(",", "-")
        if a.startswith("--seeds="):
            return a.split("=", 1)[1].replace(",", "-")
    return "na"


_STAGE_WARNED = set()
_TERM_STAMPED = set()


def _stamp_term(name, weight, **extra):
    """Announce a steering term the FIRST time it is actually active in a run."""
    if name in _TERM_STAMPED:
        return
    _TERM_STAMPED.add(name)
    tail = "".join(f" {k}={v}" for k, v in extra.items())
    logger.info("steering term ACTIVE: %s weight=%.4g%s", name, float(weight), tail)


def _facing_weight_now(spec, state):
    """facing_weight for THIS step, after the STAGED-TILTING gate."""
    w = _ds.facing_weight_of(spec)
    if w <= 0.0:
        return 0.0
    smax = float(getattr(spec, "facing_sigma_max", 0.0))
    if smax <= 0.0:
        _stamp_term("facing", w, staged="no")
        return w                                        # no gate requested
    sig = state.get("sigma")
    if sig is None:
        key = (id(spec), "nosigma")
        if key not in _STAGE_WARNED:
            _STAGE_WARNED.add(key)
            logger.warning("facing_sigma_max=%.3g requested but this sampler exposes no sigma "
                           "(SVGD tilt): the FACING TERM IS DISABLED rather than run un-staged", smax)
        return 0.0
    if sig <= smax:
        _stamp_term("facing", w, sigma_max=smax)
        return w
    return 0.0


def _dump_dir(model=None) -> str:
    """Prediction dump dir. Read off the launcher argv (``--dump_dir X``, always present on this
        launch path), falling back to the model's configs. Used by the per-sample instrumentation writes.
    """
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--dump_dir" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--dump_dir="):
            return a.split("=", 1)[1]
    try:
        return str(getattr(model.configs, "dump_dir", "") or ".")
    except Exception:                                                   # noqa: BLE001
        return "."


def _conformets_grid(model, pf_self, s, z, pa, pk, orig_pf, spec, sample_name):
    """k_nets × |snapshots| optimised (W,b) affines for THIS target, decoded through the real
        trunk path ``distogram_head(Pairformer(φ(z_pre)))``, paper-faithful (arXiv:2604.18559).
    """
    import numpy as np
    import torch
    path = _snapshot_cache_path(spec, sample_name)
    if path and os.path.exists(path):
        d = np.load(path)
        Ws, bs = d["W"], d["b"]
        logger.info("ConforNets: loaded %d cached snapshots for %s", len(Ws), sample_name)
        return [(Ws[i], bs[i]) for i in range(len(Ws))]

    from method.trunk_side.latent_modulation import optimize_conformets
    head = model.distogram_head
    _n_token = int(z.shape[-2])
    pk_decode = dict(pk)
    pk_decode["inplace_safe"] = False
    if "triangle_multiplicative" in pk_decode:
        pk_decode["triangle_multiplicative"] = "torch"
    if "triangle_attention" in pk_decode:
        pk_decode["triangle_attention"] = "torch"
    if "chunk_size" in pk_decode:
        _cfg = getattr(getattr(model, "configs", None), "infer_setting", None)
        _forced = (int(_cfg.chunk_size) if _cfg is not None
                   and getattr(_cfg, "dynamic_chunk_size", True) is False
                   and getattr(_cfg, "chunk_size", None) else None)
        pk_decode["chunk_size"] = (min(_forced, _DECODE_MAX_CHUNK) if _forced
                                   else conformets_decode_chunk(_n_token))

    def decode_fn(mz):
        # Final Pairformer pass on the modulated z_pre → z_post → distogram logits.
        out = orig_pf(pf_self, s, mz, *pa, **pk_decode)
        z_post = out[1] if isinstance(out, (tuple, list)) else out
        return head(z_post)                                 # distogram logits (through the trained head)

    bpc = int(os.environ.get("PROTENIX_CONFORMETS_BPC", "1"))
    old_bpc = getattr(pf_self, "blocks_per_ckpt", None)
    if bpc > 0:
        pf_self.blocks_per_ckpt = bpc
    import torch.utils.checkpoint as _tc
    import time as _time
    _orig_ckpt = _tc.checkpoint
    def _reentrant_ckpt(fn, *a, **k):
        k["use_reentrant"] = True
        return _orig_ckpt(fn, *a, **k)
    _t0 = _time.perf_counter()
    try:
        _tc.checkpoint = _reentrant_ckpt
        try:
            import torch as _t2
            _t2.cuda.empty_cache()
        except Exception:                                   # noqa: BLE001
            pass
        with torch.enable_grad():                           # inference is no_grad, the optimiser needs grad
            snaps = optimize_conformets(z, decode_fn, spec)  # z is z_pre = the final Pairformer INPUT
    finally:
        _tc.checkpoint = _orig_ckpt
        pf_self.blocks_per_ckpt = old_bpc                   # restore (the real forward stays unchecked)
    _opt_wall = _time.perf_counter() - _t0
    _passes = spec.k_nets * spec.n_steps           # decode fwd+bwd passes done this optimisation
    logger.info("ConforNets opt-timing: sample=%s n_token=%d k_nets=%d n_steps=%d decode_passes=%d wall=%.1fs",
                sample_name, _n_token, spec.k_nets, spec.n_steps, _passes, _opt_wall)
    cache_dir = os.environ.get(ENV_CONFORMETS_CACHE)
    if cache_dir:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, "opt_timing.jsonl"), "a") as f:
                f.write(json.dumps({"sample": str(sample_name), "n_token": _n_token,
                                    "wall_s": _opt_wall, "k_nets": spec.k_nets,
                                    "n_steps": spec.n_steps, "decode_passes": _passes}) + "\n")
        except Exception:                                   # noqa: BLE001, timing must never break the run
            pass
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"                   # atomic write, no torn reads by later variants
        np.savez(tmp, W=np.stack([w for w, _ in snaps]), b=np.stack([b for _, b in snaps]))
        os.replace(tmp, path)
        logger.info("ConforNets: computed + cached %d snapshots for %s", len(snaps), sample_name)
    return snaps


def _apply_conformets(model, pf_self, s, z, pa, pk, orig_pf, spec, sample_name):
    """Apply the ``variant_index``-th optimised ConforNet affine to z_pre (forward)."""
    import torch
    if not hasattr(model, "distogram_head"):
        logger.warning("ConforNets: no model.distogram_head; skipping optimisation")
        return z
    snaps = _conformets_grid(model, pf_self, s, z, pa, pk, orig_pf, spec, sample_name)
    idx = spec.variant_index % len(snaps)
    W_np, b_np = snaps[idx]
    c = z.shape[-1]
    W = torch.as_tensor(W_np, dtype=z.dtype, device=z.device)
    b = torch.as_tensor(b_np, dtype=z.dtype, device=z.device)
    logger.info("ConforNets: applied snapshot %d/%d (c_z=%d) for %s", idx, len(snaps), c, sample_name)
    return (z.reshape(-1, c) @ W.T + b).reshape(z.shape)


# ---------------------------------------------------------------- monkey-patch
def install() -> None:
    """Patch Protenix so the pre-Pairformer pair latent z_pre is modulated at the right passes."""
    patch_sample_name_capture()                             # so the ConforNets cache can key on the target
    try:
        Protenix = _cofold_model_class()
        PairformerStack = getattr(_cofold_module("protenix.model.modules.pairformer"),
                                  "PairformerStack")
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            "Could not locate protenix.model.protenix.Protenix / "
            f"protenix.model.modules.pairformer.PairformerStack for the A2 hook: {e!r}")
    orig_gpo = Protenix.get_pairformer_output
    orig_pf = PairformerStack.forward

    def _wrapped_gpo(self, *args, **kwargs):
        if _ACTIVE is None:
            return orig_gpo(self, *args, **kwargs)
        spec = _ACTIVE
        n_cycle = args[1] if len(args) > 1 else kwargs.get("N_cycle")
        try:
            n_cycle = int(n_cycle)
        except (TypeError, ValueError):
            n_cycle = None
        sample_name = _extract_sample_name(args[0] if args else kwargs.get("input_feature_dict"))
        model = self
        state = {"call": 0}

        main_pf = getattr(model, "pairformer_stack", None)
        def _pf_forward(pf_self, s, z, *pa, **pk):
            if pf_self is not main_pf:
                return orig_pf(pf_self, s, z, *pa, **pk)
            state["call"] += 1
            if spec.kind == "rescale":
                z = (1.0 + spec.beta) * z                              # EVERY recycle (Boltz-sample)
            elif spec.kind == "conformets":
                if n_cycle is None:
                    logger.error("ConforNets: N_cycle unknown; skipping (cannot identify final pass)")
                elif state["call"] >= n_cycle:                         # final Pairformer pass only
                    z = _apply_conformets(model, pf_self, s, z, pa, pk, orig_pf, spec, sample_name)
            return orig_pf(pf_self, s, z, *pa, **pk)

        PairformerStack.forward = _pf_forward
        try:
            return orig_gpo(self, *args, **kwargs)
        finally:
            PairformerStack.forward = orig_pf                          # always restore

    Protenix.get_pairformer_output = _wrapped_gpo
    logger.info("latent-modulation hook installed on Protenix.get_pairformer_output "
                "(z_pre @ Pairformer input; rescale=every recycle, conformets=final pass, "
                "decode=distogram∘Pairformer)")



def install_recycle_probe() -> None:
    """Record the antibody-antigen CONTACT PROFILE at EVERY trunk recycle, not just the last."""
    out_dir = os.environ.get(ENV_RECYCLE_PROBE)
    if not out_dir:
        return
    if os.environ.get(ENV_RECYCLE_TAP):
        logger.info("recycle probe SKIPPED: $%s is set and the two hooks cannot coexist "
                    "(they both patch PairformerStack.forward)", ENV_RECYCLE_TAP)
        return
    import numpy as np
    import torch
    Protenix = _cofold_model_class()
    if Protenix is None or not hasattr(Protenix, "get_pairformer_output"):
        logger.warning("recycle probe: Protenix.get_pairformer_output unavailable")
        return
    os.makedirs(out_dir, exist_ok=True)
    PairformerStack = getattr(_cofold_module("protenix.model.modules.pairformer"),
                              "PairformerStack", None)
    if PairformerStack is None:
        logger.warning("recycle probe: PairformerStack unavailable")
        return
    orig_gpo = Protenix.get_pairformer_output
    orig_pf = PairformerStack.forward

    def _wrapped(self, *args, **kwargs):
        ifd = _map_space_ifd(args[0] if args else kwargs.get("input_feature_dict"))
        name = _extract_sample_name(ifd) or "unknown"
        head = getattr(self, "distogram_head", None)
        main_pf = getattr(self, "pairformer_stack", None)
        if head is None or main_pf is None or ifd is None:
            return orig_gpo(self, *args, **kwargs)
        cid = ifd["asym_id"].reshape(-1).long().cpu().numpy()
        par = _paratope_tokens(ifd, None)
        if par is None:
            logger.warning("recycle probe: no paratope map, cannot split antibody from antigen")
            return orig_gpo(self, *args, **kwargs)
        ab_ch = set(cid[np.asarray(par[0], dtype=int)].tolist())
        ab = np.flatnonzero(np.isin(cid, list(ab_ch)))
        ag = np.flatnonzero(~np.isin(cid, list(ab_ch)))
        prof, state = [], {"call": 0}

        def _pf(pf_self, *pa, **pk):
            if pf_self is not main_pf:
                return orig_pf(pf_self, *pa, **pk)
            state["call"] += 1
            out = orig_pf(pf_self, *pa, **pk)
            z_post = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else out
            try:
                with torch.no_grad():
                    lg = head(z_post.float())
                    p = torch.softmax(lg, dim=-1)
                    bins = torch.arange(p.shape[-1], device=p.device, dtype=p.dtype)
                    ebin = (p * bins).sum(-1).reshape(-1, p.shape[-3], p.shape[-2])[0]
                    prof.append((-ebin[np.ix_(ab, ag)].min(dim=0).values).cpu().numpy())
            except Exception as e:      # noqa: BLE001 -- a probe must never kill the run
                logger.warning("recycle probe: readout failed at recycle %d (%s)", state["call"], e)
            return out

        PairformerStack.forward = _pf
        try:
            return orig_gpo(self, *args, **kwargs)
        finally:
            PairformerStack.forward = orig_pf
            if prof:
                np.savez(os.path.join(out_dir, f"{_safe_name(name)}__recycle_profiles.npz"),
                         profiles=np.stack(prof), antigen_tokens=ag, antibody_tokens=ab)
                logger.info("recycle probe: saved %d recycle profiles over %d antigen tokens for %s",
                            len(prof), len(ag), name)

    Protenix.get_pairformer_output = _wrapped
    logger.info("recycle probe INSTALLED -> %s", out_dir)



def install_recycle_tap() -> None:
    """CONDITION the diffusion module on an EARLIER recycle instead of the converged one."""
    k_env = os.environ.get(ENV_RECYCLE_TAP)
    if not k_env:
        return
    try:
        K = int(k_env)
        assert K >= 1
    except (ValueError, AssertionError):
        logger.warning("recycle tap: %r is not a 1-based recycle index, ignoring", k_env)
        return
    Protenix = _cofold_model_class()
    PairformerStack = getattr(_cofold_module("protenix.model.modules.pairformer"),
                              "PairformerStack", None)
    if Protenix is None or PairformerStack is None:
        logger.warning("recycle tap: model or PairformerStack unavailable")
        return
    orig_gpo = Protenix.get_pairformer_output
    orig_pf = PairformerStack.forward

    def _wrapped(self, *args, **kwargs):
        main_pf = getattr(self, "pairformer_stack", None)
        if main_pf is None:
            return orig_gpo(self, *args, **kwargs)
        grab, state = {}, {"call": 0}

        def _pf(pf_self, *pa, **pk):
            # generic signature: Protenix nests other PairformerStacks called with keywords, and
            # binding s/z positionally would fail on those BEFORE the trunk check below.
            if pf_self is not main_pf:
                return orig_pf(pf_self, *pa, **pk)
            state["call"] += 1
            out = orig_pf(pf_self, *pa, **pk)
            if state["call"] == K:
                grab["out"] = out
            return out

        PairformerStack.forward = _pf
        try:
            res = orig_gpo(self, *args, **kwargs)
        finally:
            PairformerStack.forward = orig_pf
        got = grab.get("out")
        if got is None:
            logger.warning("recycle tap: recycle %d never reached (only %d ran), using the final",
                           K, state["call"])
            return res
        gs = got if isinstance(got, (tuple, list)) else (got,)
        if isinstance(res, (tuple, list)):
            out = list(res)
            n_sub = 0
            for i, r in enumerate(out):
                for g in gs:
                    if hasattr(r, "shape") and hasattr(g, "shape") and tuple(r.shape) == tuple(g.shape):
                        out[i] = g
                        n_sub += 1
                        break
            if n_sub == 0:
                logger.warning("recycle tap: no shape match, leaving the final recycle in place")
                return res
            logger.info("recycle tap: conditioning on recycle %d of %d (%d tensor(s) substituted)",
                        K, state["call"], n_sub)
            return type(res)(out) if isinstance(res, tuple) else out
        logger.warning("recycle tap: unexpected trunk return type %s, leaving unchanged", type(res))
        return res

    Protenix.get_pairformer_output = _wrapped
    logger.info("recycle tap INSTALLED: diffusion will condition on recycle %d", K)



def enable_diffusion(spec: DiffusionSearchSpec) -> None:
    global _ACTIVE_DIFF
    _ACTIVE_DIFF = spec
    logger.info("diffusion-search hook ENABLED: %s", spec.label())


def enable_diffusion_from_env() -> bool:
    """Enable from ``$PROTENIX_DIFFUSION_SPEC`` (a JSON file). Returns True if enabled."""
    global _DIFF_SPEC_PATH
    p = os.environ.get(ENV_DIFF_SPEC)
    if not p:
        return False
    _DIFF_SPEC_PATH = p                    # timing jsonl is written next to the spec
    with open(p) as f:
        enable_diffusion(diffusion_spec_from_dict(json.load(f)))
    install_recycle_probe()
    install_recycle_tap()
    return True


def patch_confidence_timing() -> None:
    """Time the OUTPUT confidence head ourselves, because Protenix's own tracker does not."""
    import time as _time
    try:
        Protenix = _cofold_model_class()
    except Exception as e:                                  # noqa: BLE001
        logger.warning("confidence-timing hook not installed: %s", e)
        return
    if getattr(Protenix, "_tts_conf_patched", False) or not hasattr(Protenix, "run_confidence_head"):
        return
    _orig = Protenix.run_confidence_head

    def _timed(self, *a, **k):
        _t = _time.perf_counter()
        out = _orig(self, *a, **k)
        _cuda_sync()
        self._tts_conf_wall = getattr(self, "_tts_conf_wall", 0.0) + (_time.perf_counter() - _t)
        self._tts_conf_calls = getattr(self, "_tts_conf_calls", 0) + 1
        return out

    Protenix.run_confidence_head = _timed
    Protenix._tts_conf_patched = True
    logger.info("confidence-timing hook installed (run_confidence_head -> _tts_conf_wall)")


def _achieved_contact_d_list(spec, kw, coords, sweep, n_sites, n_particles):
    """`achieved_contact_distance` per particle as a plain list, or [] if it cannot be computed."""
    if coords is None:
        return []
    try:
        import numpy as _np
        import method.diffusion_side.base_search as bs
        from method.verifier.diversity import achieved_contact_distance
        if hasattr(coords, "detach"):
            coords = coords.detach().to("cpu").numpy()
        pool = _np.asarray(coords, dtype=_np.float64)
        if pool.ndim > 3:
            pool = pool.reshape((-1,) + pool.shape[-2:])
        pi = _PARATOPE_IDX[0]
        ei = _epi_idx_for_realloc[0]
        if pi is None or ei is None:
            logger.warning("achieved_contact_d not recorded: paratope/epitope idx unavailable")
            return []
        # Build the cell assignment HERE, with the search's own helpers, and hand it to the metric --
        # the metric never recomputes it, so it cannot drift from what the reward acted under.
        sites = bs.epitope_sites(ei)
        tg = bs.sweep_targets(n_particles, sweep) if sweep else None
        tgt_all = _np.broadcast_to(_np.asarray(0.0 if tg is None else tg, dtype=_np.float64),
                                   (len(pool),))
        gid = (bs.epitope_site_ids(len(pool), len(sites), tgt_all) if len(sites) > 1
               else _np.zeros(len(pool), dtype=int))
        d = achieved_contact_distance(pool, pi, sites, gid)
        return [None if v != v else float(v) for v in d]
    except Exception as e:                                 # noqa: BLE001
        logger.warning("achieved_contact_d not recorded: %s", e)
        return []


def _record_cell_map(spec, kw, sweep, n_sites: int, n_particles: int, sub_ids=None,
                     sub_ids0=None, coords=None) -> None:
    """Write the particle -> (umbrella, site) assignment beside the run's other artefacts."""
    try:
        if _DIFF_SPEC_PATH is None:
            return
        # imported here, not at module scope: this file imports the search modules inside functions
        # so that a plain Protenix run never pays for them.
        import method.diffusion_side.base_search as bs
        tg = bs.sweep_targets(n_particles, sweep) if sweep else None
        cells = bs.base_cells(n_particles, tg, n_sites)
        rec = {
            "sample": str(_extract_sample_name(kw.get("input_feature_dict"))),
            "n_particles": int(n_particles),
            "n_sites": int(n_sites),
            "umbrella_sweep": list(sweep) if sweep else [],
            "reserve_frac": float(_ds.reserve_frac_of(spec)),
            "cell_id": [int(c) for c in cells],
            "umbrella_idx": [int(c) // max(int(n_sites), 1) for c in cells],
            "site_idx": [int(c) % max(int(n_sites), 1) for c in cells],
            "target_dist": [float(x) for x in tg] if tg is not None else [],
            "reserved": [bool(x) for x in bs.reserve_mask_for(
                n_particles, _ds.reserve_frac_of(spec), sweep, n_sites)],
            "subsample_idx": [int(x) for x in sub_ids] if sub_ids is not None else [],
            "subsample_idx0": [int(x) for x in sub_ids0] if sub_ids0 is not None else [],
            # K, so the analysis never has to infer it from the ids that survived -- a subsample
            # eliminated everywhere is exactly the case where inferring it gives the wrong K.
            "n_subsamples": int(getattr(spec, "subsample_pool", 1) or 1),
            "achieved_contact_d": _achieved_contact_d_list(spec, kw, coords, sweep, n_sites,
                                                           n_particles),
        }
        path = os.path.join(os.path.dirname(_DIFF_SPEC_PATH), "cell_map.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:                                 # noqa: BLE001, provenance never breaks a run
        logger.warning("cell-map write failed: %s", e)



def _ess_cell_str(ess_cells, n_total):
    """Render the PER-CELL ESS summary for run.log, or "" when there are no cells."""
    import numpy as _np
    if ess_cells is None or len(ess_cells) == 0:
        return ""
    per = n_total / float(len(ess_cells))
    return (f" [per-cell ESS min={_np.min(ess_cells):.2f} med={_np.median(ess_cells):.2f} "
            f"max={_np.max(ess_cells):.2f} of {per:.0f}]")

def _record_diffusion_timing(spec, kw, stats: dict, n_token: int, wall_s: float,
                             trunk_wall_s: float = 0.0, n_emitted: int = 0) -> None:
    """Append one measured compute-budget record to ``diffusion_timing.jsonl`` (next to the spec)."""
    try:
        sample_name = _extract_sample_name(kw.get("input_feature_dict"))
        _ifd_net = kw.get("input_feature_dict") or {}
        try:
            _n_tok_net = int(_ifd_net["asym_id"].reshape(-1).shape[0])
        except Exception:                                   # noqa: BLE001, instrumentation only
            _n_tok_net = int(n_token)
        rec = {"sample": str(sample_name), "algorithm": spec.algorithm, "n_token": int(n_token),
               # Equal to n_token on Protenix and on OpenDDE without structural expansion.
               "n_token_denoiser": _n_tok_net,
               "n_sample": int(kw.get("N_sample", 0)),          # requested; kept for old readers
               "n_structures_emitted": int(n_emitted),          # what the call actually returned
               "denoiser_particle_steps": int(stats["denoiser_particle_steps"]),
               "confidence_passes": int(stats["confidence_passes"]),
               "wall_reward_s": round(float(stats["t_reward"]), 3),
               "wall_guidance_s": round(float(stats.get("t_guidance", 0.0)), 3),
               "wall_resample_s": round(float(stats.get("t_resample", 0.0)), 3),
               "n_score_events": int(stats.get("n_score_events", 0)),
               "n_resample_events": int(stats.get("n_resample_events", 0)),
               "wall_s": float(wall_s), "wall_denoise_s": float(stats["t_denoise"]),
               "wall_confidence_s": float(stats["t_confidence"]),
               "wall_trunk_s": float(trunk_wall_s),
               "wall_confidence_output_s": round(float(stats.get("t_confidence_output", 0.0)), 3),
               "n_confidence_output_calls": int(stats.get("n_confidence_output_calls", 0)),
               "wall_unaccounted_s": round(float(
                   wall_s - stats["t_denoise"] - stats["t_confidence"] - stats["t_reward"]
                   - stats.get("t_guidance", 0.0) - stats.get("t_resample", 0.0)), 3)}
        if _DIFF_SPEC_PATH:
            path = os.path.join(os.path.dirname(_DIFF_SPEC_PATH), "diffusion_timing.jsonl")
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        logger.info("diffusion-search compute: %s", rec)
    except Exception as e:                 # noqa: BLE001, instrumentation must never break a run
        logger.warning("could not record diffusion timing: %s", e)


def _make_raw_reward_grad(model, kw, spec):
    """Bare d(reward)/d(atom coords) for the SVGD tilt, no eps_t schedule, no clip."""
    import numpy as np
    from method.diffusion_side import base_search as bs
    from method.diffusion_side import guidance as gd

    par = _paratope_tokens(kw["input_feature_dict"], spec)
    if par is None:
        return None
    idx, atom_to_token, n_token = par
    sweep = spec.sweep_list()
    epi = None
    if getattr(spec, "reward", "") in _ds.SITE_REWARDS:
        ep = _epitope_tokens(kw["input_feature_dict"], n_token)
        if ep is None:
            logger.warning("svgd tilt: reward=epitope but no epitope map, tilt disabled")
            return None
        epi = ep
    # SAME grid validation as the TDS reward builder, for the same reason: SVGD reads the sweep from
    # the spec, so an over-fine grid empties cells here too.
    if sweep:
        _ns = len(bs.epitope_sites(epi)) if epi is not None else 1
        _np = int(kw.get("N_sample", 0) or 0) or 50
        _lv, _bm = bs.check_cell_budget(_np, len(sweep), _ns,
                                        reserve_frac=_ds.reserve_frac_of(spec))
        if _lv == "error":
            raise ValueError("svgd umbrella grid: %s" % _bm)
        if _lv == "warn":
            logger.warning("svgd umbrella grid: %s", _bm)
    state = {"ref": None}

    def raw(coords):
        c = coords.detach().float().cpu().numpy() if hasattr(coords, "detach") else np.asarray(coords)
        c = c.reshape(-1, c.shape[-2], 3)
        reps = bs.residue_reps_from_atoms(c, atom_to_token, n_token)
        tgt = bs.sweep_targets(len(reps), sweep) if sweep else float(spec.cdr_target_rmsd)
        if epi is not None:
            g_tok = bs.site_contact_grad(reps, idx, epi, target_dist=tgt,
                             contact_weight=_ds.contact_weight_of(spec),
                             contact_saturate_tau=float(getattr(spec, "contact_saturate_tau", 0.0)),
                             r_contact=float(getattr(spec, "contact_r", 8.0)),
                             contact_sigma=float(getattr(spec, "contact_sigma", 2.0)),
                             hinge=bool(getattr(spec, "site_hinge", False)),
                             huber_delta=float(getattr(spec, "site_huber_delta", 0.0)),
                             contact_gate=float(getattr(spec, "site_contact_gate", 0.0)),
                             site_floor=float(getattr(spec, "site_floor", 0.0)),
                             softmin_tau=float(getattr(spec, "site_softmin_tau", 0.0)),
                             dev_cap=float(getattr(spec, "site_dev_cap", 0.0)))
            _cw = _ds.clash_weight_of(spec)
            if _cw > 0.0:
                _stamp_term("clash", _cw)
                _, _cg = bs.clash_penalty(reps, idx, bs.epitope_sites(epi)[0],
                                          d_min=float(getattr(spec, "clash_d_min", 4.0)))
                g_tok = g_tok + _cw * _cg
            _fw = _facing_weight_now(spec, state)
            if _fw > 0.0:
                _, _fg = bs.facing_shell(reps, idx, bs.epitope_sites(epi)[0])
                g_tok = g_tok + _fw * _fg
            _ow = _ds.orient_weight_of(spec)
            if _ow > 0.0:
                _stamp_term("orient", _ow, tol_deg=float(getattr(spec, "orient_tol_deg", 0.0)))
                _abi = bs.antibody_tokens(
                    idx, _map_space_ifd(kw["input_feature_dict"])["asym_id"]
                    .reshape(-1).long().cpu().numpy())
                _, _og = bs.orientation_reward(reps, idx, bs.epitope_sites(epi)[0], _abi,
                                               tol_deg=float(getattr(spec, "orient_tol_deg", 0.0)))
                g_tok = g_tok + _ow * bs.relative_scale(g_tok, _og) * _og
            _rf = _ds.reserve_frac_of(spec)
            if _rf > 0.0:
                _ns_r = len(bs.epitope_sites(epi)) if epi is not None else 1
                g_tok = np.where(bs.reserve_mask_for(len(g_tok), _rf, sweep, _ns_r)[:, None, None],
                                 0.0, g_tok)
        else:
            if state["ref"] is None:
                state["ref"] = reps[0].copy()        # same anchor convention as the cdr reward
            g_tok = bs.cdr_rmsd_grad(reps, state["ref"], idx, target_rmsd=tgt)
        return gd.scatter_token_grad_to_atoms(g_tok, atom_to_token, c.shape[-2])

    return raw


def _trunk_z(model, kw):
    """The REAL trunk pair rep ``z`` for the confidence head (which does ``z_trunk.detach()``)."""
    z = kw.get("z_trunk")
    return getattr(model, "_tts_diff_z", None) if z is None else z


_RESIDUE_LEVEL_KEYS = ("token_index", "asym_id", "residue_index", "entity_id", "sym_id",
                       "atom_to_token_idx", "atom_to_tokatom_idx", "has_frame",
                       "frame_atom_index", "pae_rep_atom_mask", "distogram_rep_atom_mask")


def _map_space_ifd(ifd):
    """``ifd`` viewed in the space the TOKEN MAPS and metrics live in (residue space)."""
    if not isinstance(ifd, dict) or "residue_level_asym_id" not in ifd:
        return ifd
    out = dict(ifd)
    for k in _RESIDUE_LEVEL_KEYS:
        v = ifd.get(f"residue_level_{k}")
        if v is not None:
            out[k] = v
    return out


def _token_plddt(plddt_out, ifd, n_token, model=None):
    """Per-token C-alpha pLDDT (0-100), mirroring bias.py::create_plddt_mask."""
    import numpy as np
    import method.diffusion_side.base_search as bs

    if model is not None:
        _sc = _cofold_module("protenix.model.sample_confidence")
        get_bin_params, logits_to_score = _sc.get_bin_params, _sc.logits_to_score
        _bins = get_bin_params(model.configs.loss.plddt)
        if plddt_out.shape[-1] == _bins["no_bins"]:          # per-bin logits -> atom score, x100
            plddt_out = logits_to_score(plddt_out.detach().float(), **_bins) * 100.0
    pl = plddt_out.detach().float().cpu().numpy().reshape(-1)
    tok = ifd["atom_to_token_idx"].reshape(-1).long().cpu().numpy()
    if pl.shape[0] != tok.shape[0]:                  # already per-token
        if pl.shape[0] != int(n_token):
            logger.warning("token pLDDT: got %d values for %d atoms and %d tokens; cannot place them "
                           "in this token space, falling back to a flat score",
                           pl.shape[0], tok.shape[0], int(n_token))
            return np.full(int(n_token), float(np.mean(pl)) * (100.0 if pl.max() <= 1.0 else 1.0))
        return pl * (100.0 if pl.max() <= 1.0 else 1.0)
    per_tok = bs.residue_reps_from_atoms(pl[None, :, None] * np.ones((1, 1, 3)), tok, n_token)[0, :, 0]
    names = ifd.get("ref_atom_name_chars")
    if names is not None:
        nm = names.reshape(len(tok), -1).long().cpu().numpy()
        ca = np.where((nm[:, 0] == 35) & (nm[:, 1] == 33) & (nm[:, 2] == 0) & (nm[:, 3] == 0))[0]
        if ca.size:
            per_tok = per_tok.copy()
            per_tok[tok[ca]] = pl[ca]
    return per_tok * (100.0 if per_tok.max() <= 1.0 else 1.0)


def _ca_token_coords(atom_coords, ifd, n_token):
    """Per-token C-alpha coordinate, (..., n_token, 3), the AF3-ReD reference's atom set."""
    import numpy as np
    import method.diffusion_side.base_search as bs

    a = np.asarray(atom_coords, dtype=np.float64)
    tok = ifd["atom_to_token_idx"].reshape(-1).long().cpu().numpy()
    out = bs.residue_reps_from_atoms(a, tok, n_token)          # centroid fallback
    names = ifd.get("ref_atom_name_chars")
    if names is None:
        return out
    nm = names.reshape(len(tok), -1).long().cpu().numpy()
    is_ca = (nm[:, 0] == 35) & (nm[:, 1] == 33) & (nm[:, 2] == 0) & (nm[:, 3] == 0)
    if not is_ca.any():
        return out
    flat = a.reshape(-1, a.shape[-2], 3)
    ca_idx = np.where(is_ca)[0]
    out2 = out.reshape(-1, n_token, 3).copy()
    out2[:, tok[ca_idx], :] = flat[:, ca_idx, :]
    return out2.reshape(out.shape)


def _load_token_map(env_var, ifd, n_token):
    """{sample_name: [token idx]} json from ``env_var`` -> a validated index array, or None."""
    import numpy as np

    path = os.environ.get(env_var)
    if not path:
        return None                                  # deliberate: no map requested
    if not os.path.exists(path):
        _stamp_term(f"{env_var}:MISSING", 0.0, path=path)
        logger.warning("%s=%s does NOT EXIST -- this arm will run with NO map for that term", 
                       env_var, path)
        return None
    sn = _extract_sample_name(ifd) or _CURRENT_SAMPLE_NAME
    name = str(sn) if sn else ""
    try:
        with open(path) as f:
            tmap = json.load(f)
    except (OSError, ValueError) as e:
        _stamp_term(f"{env_var}:UNREADABLE", 0.0, path=path)
        logger.warning("%s=%s could not be read (%r) -- this arm will run with NO map for that term",
                       env_var, path, e)
        return None
    tmap = {k: v for k, v in tmap.items() if not str(k).startswith("_")}
    ent = tmap.get(name)
    if ent is None and name:
        cand = [k for k in tmap if name.startswith(k) or k in name]
        if cand:
            ent = tmap[max(cand, key=len)]
    if isinstance(ent, dict):                       # {"paratope_tokens": [...]} / {"epitope_tokens": [...]}
        ent = ent.get("paratope_tokens") or ent.get("epitope_tokens")
    if ent:
        _stamp_term(f"{env_var}:loaded", float(len(ent)),
                    path=os.path.basename(path), matched_key=name[:40] or "<none>")
    if not ent:
        _stamp_term(f"{env_var}:NOMATCH", 0.0, path=os.path.basename(path),
                    sample=name[:40] or "<none>", map_targets=len(tmap))
        logger.warning("%s=%s has no entry for %r (map holds %d target(s), e.g. %s) -- NO map for that "
                       "term on this target; a site arm degrades to its fallback reward under a site "
                       "label. Wrong dataset's map?", env_var, os.path.basename(path),
                       name or "<none>", len(tmap), ", ".join(list(tmap)[:2]) or "<empty>")
        return None
    if isinstance(ent[0], (list, tuple)):
        sites = []
        for s in ent:
            a = np.asarray(s, dtype=int)
            a = a[(a >= 0) & (a < int(n_token))]
            if a.size:
                sites.append(a)
        if not sites:
            return None
        return sites[0] if len(sites) == 1 else sites
    idx = np.asarray(ent, dtype=int)
    idx = idx[(idx >= 0) & (idx < int(n_token))]
    return idx if idx.size else None


def _epitope_tokens(ifd, n_token):
    """Epitope token indices from $PROTENIX_EPITOPE_MAP, or None."""
    return _load_token_map(ENV_EPITOPE_MAP, ifd, n_token)


def _paratope_tokens(ifd, spec=None):
    """(subset_idx, atom_to_token, n_token) for the configured RMSD subset, or None."""
    import method.diffusion_side.base_search as bs

    ifd = _map_space_ifd(ifd)                        # maps are residue-space; see _map_space_ifd
    atom_to_token = ifd["atom_to_token_idx"].reshape(-1).long().cpu().numpy()
    n_token = int(ifd["asym_id"].reshape(-1).shape[0])
    idx = _load_token_map(ENV_CDR_MAP, ifd, n_token)
    if idx is None:
        return None
    if spec is not None:                             # pluggable RMSD subset (paratope | all | chain)
        cid = ifd["asym_id"].reshape(-1).long().cpu().numpy()
        idx = bs.rmsd_subset(getattr(spec, "rmsd_subset", "paratope"), n_token,
                              {"paratope_idx": idx, "chain_id": cid,
                               "rmsd_chain": (None if getattr(spec, "rmsd_chain", -1) < 0
                                              else spec.rmsd_chain)})
    return (idx, atom_to_token, n_token) if idx.size >= 3 else None


def _make_cdr_reward_fn(model, kw: dict, spec: DiffusionSearchSpec):
    """Backend-agnostic CDR/paratope-conformation steering reward (the algorithm lives in
        ``method.diffusion_side.base_search``; this is only the adapter that feeds it coordinates).
    """
    import numpy as np
    from method.diffusion_side import base_search as bs
    from method.diffusion_side import guidance as gd

    path = os.environ.get(ENV_CDR_MAP)
    if not path or not os.path.exists(path):
        return None
    ifd = _map_space_ifd(kw["input_feature_dict"])    # maps are residue-space; see _map_space_ifd
    atom_to_token = ifd["atom_to_token_idx"].reshape(-1).long().cpu().numpy()
    n_token = int(ifd["asym_id"].reshape(-1).shape[0])
    sn = _extract_sample_name(ifd) or _CURRENT_SAMPLE_NAME   # ifd copy drops the key, the global carries it
    sample_name = str(sn) if sn else ""
    # _load_token_map opens and parses this same file behind its own OSError/ValueError guard, so a
    # pre-read here would only duplicate the parse on every reward event.
    paratope_idx = _load_token_map(ENV_CDR_MAP, ifd, n_token)   # one shared loader (see that helper)
    if paratope_idx is None:
        logger.warning("cdr_rmsd: sample %r not in %s, falling back to ipSAE", sample_name, ENV_CDR_MAP)
        return None
    paratope_idx = bs.rmsd_subset(                      # pluggable: every algorithm shares this subset
        getattr(spec, "rmsd_subset", "paratope"), n_token,
        {"paratope_idx": paratope_idx,
         "chain_id": ifd["asym_id"].reshape(-1).long().cpu().numpy(),
         "rmsd_chain": None if getattr(spec, "rmsd_chain", -1) < 0 else spec.rmsd_chain})
    if paratope_idx.size < 3:                       # need >=3 residues for a meaningful superposition
        logger.warning("cdr_rmsd: <3 valid paratope tokens for %r, falling back to ipSAE", sample_name)
        return None
    # PUBLISH for the within-cell repulsion in _search_sample_diffusion (see _PARATOPE_IDX). Done
    # only AFTER the size guard, so the weight update can never repel on an unusable subset.
    _PARATOPE_IDX[0] = paratope_idx
    epitope_idx = None
    if getattr(spec, "reward", "") in _ds.SITE_REWARDS:
        epitope_idx = _load_token_map(ENV_EPITOPE_MAP, ifd, n_token)
        if epitope_idx is None:
            logger.warning("reward=epitope but no usable epitope map for %r (%s) -- falling back to "
                           "cdr_rmsd", sample_name, ENV_EPITOPE_MAP)
    sweep = spec.sweep_list()                       # d* umbrellas (ConforMix), empty = single target
    target = np.asarray(sweep, dtype=float) if sweep else float(spec.cdr_target_rmsd)
    _chain_of = ifd["asym_id"].reshape(-1).long().cpu().numpy()
    state = {"ref": None,                           # reference conformation, captured on the first event
             "sigma": None}
    # Candidate binding sites, auto-read from the MAP (1 for a ground-truth map, several for a
    # predicted one). Only the site reward is multi-modal, cdr_rmsd has no site axis.
    n_sites = len(bs.epitope_sites(epitope_idx)) if epitope_idx is not None else 1
    _P = int(kw.get("N_sample", 0) or 0) or 50
    _probe_name[0] = str(sample_name)     # so each probe line names its target+seed, not just a step
    _rf0 = _ds.reserve_frac_of(spec)
    _lvl, _msg = bs.check_cell_budget(_P, max(len(sweep), 1), n_sites, reserve_frac=_rf0)
    if _lvl == "error":
        raise ValueError("umbrella grid for %r: %s" % (sample_name, _msg))
    if _lvl == "warn":
        logger.warning("umbrella grid for %r: %s", sample_name, _msg)
    _occ, _nres = bs.cell_occupancy(_P, max(len(sweep), 1), n_sites, _rf0)
    logger.info("umbrella grid for %r: %d sites x %d umbrellas = %d cells over %d particles "
                "(%d reserved, %d tilted), %d-%d tilted per cell", sample_name, n_sites,
                max(len(sweep), 1), n_sites * max(len(sweep), 1), _P, _nres, _P - _nres,
                int(_occ.min()), int(_occ.max()))

    def _reserve_of(n_particles):
        """The reserve mask, from the one constructor in base_search (see reserve_mask_for)."""
        return bs.reserve_mask_for(n_particles, _ds.reserve_frac_of(spec), sweep, n_sites)
    _RESERVE_FN[0] = _reserve_of        # PUBLISH for the repulsion (see _RESERVE_FN)
    _epi_idx_for_realloc[0] = epitope_idx   # PUBLISH for site realloc (see the holder)

    def steering_groups(n_particles):
        """Per-particle (umbrella, site) CELL id -- the stratum the SMC resample must stay inside."""
        tg = bs.sweep_targets(n_particles, sweep) if sweep else None
        return bs.steering_group_ids(n_particles, tg, n_sites,
                                     reserve_frac=_ds.reserve_frac_of(spec))

    if n_sites > 1:
        logger.info("multi-site epitope steering: %d candidate sites x %d umbrellas", n_sites,
                    max(len(sweep), 1))

    def reward(coords):
        c = coords.detach().float().cpu().numpy() if hasattr(coords, "detach") else np.asarray(coords)
        c = c.reshape(-1, c.shape[-2], 3)                                   # [P, N_atom, 3]
        reps = bs.residue_reps_from_atoms(c, atom_to_token, n_token)        # [P, n_token, 3]
        _sd = "" if _dump_off else os.environ.get("PROTENIX_STEP_DUMP_DIR", "")
        if _sd:
            safe = _safe_name(sample_name)          # carries the _s{k} MSA-subsample tag
            _step_i[safe] = _step_i.get(safe, 0) + 1
            try:
                os.makedirs(_sd, exist_ok=True)
                np.save(os.path.join(_sd, "reps_%s_step%04d.npy" % (safe, _step_i[safe])),
                        np.asarray(reps, dtype=np.float32))
            except Exception as e:                   # noqa: BLE001 -- never break sampling
                _dump_off.append(1)
                logger.warning("per-step dump disabled after failure (%s)", e)
        if epitope_idx is not None:            # binding-SITE reward: absolute geometry, no anchor
            tg = bs.sweep_targets(len(reps), sweep) if sweep else float(spec.cdr_target_rmsd)
            r = bs.site_contact_reward(reps, paratope_idx, epitope_idx, target_dist=tg,
                             contact_weight=_ds.contact_weight_of(spec),
                             contact_saturate_tau=float(getattr(spec, "contact_saturate_tau", 0.0)),
                             r_contact=float(getattr(spec, "contact_r", 8.0)),
                             contact_sigma=float(getattr(spec, "contact_sigma", 2.0)),
                             hinge=bool(getattr(spec, "site_hinge", False)),
                             huber_delta=float(getattr(spec, "site_huber_delta", 0.0)),
                             contact_gate=float(getattr(spec, "site_contact_gate", 0.0)),
                             site_floor=float(getattr(spec, "site_floor", 0.0)),
                             softmin_tau=float(getattr(spec, "site_softmin_tau", 0.0)),
                             dev_cap=float(getattr(spec, "site_dev_cap", 0.0)))
            _cw = _ds.clash_weight_of(spec)
            if _cw > 0.0:
                _stamp_term("clash", _cw)
                _cl, _ = bs.clash_penalty(reps, paratope_idx, bs.epitope_sites(epitope_idx)[0],
                                          d_min=float(getattr(spec, "clash_d_min", 4.0)))
                r = r + _cw * _cl
            _fw = _facing_weight_now(spec, state)
            if _fw > 0.0:
                _fv, _ = bs.facing_shell(reps, paratope_idx, bs.epitope_sites(epitope_idx)[0])
                r = r + _fw * _fv
            _ow = _ds.orient_weight_of(spec)
            if _ow > 0.0:
                _stamp_term("orient", _ow, tol_deg=float(getattr(spec, "orient_tol_deg", 0.0)))
                _ov, _ = bs.orientation_reward(reps, paratope_idx,
                                               bs.epitope_sites(epitope_idx)[0],
                                               bs.antibody_tokens(paratope_idx, _chain_of),
                                               tol_deg=float(getattr(spec, "orient_tol_deg", 0.0)))
                r = r + _ow * bs.relative_scale(r, _ov) * _ov
            _dw = _ds.direction_weight_of(spec)
            if _dw > 0.0:
                _stamp_term("approach", _dw)
                _dv, _ = bs.direction_reward(reps, paratope_idx, epitope_idx,
                                             chain_of=_chain_of,
                                             r_local=float(getattr(spec, "direction_r_local", 15.0)))
                r = r + _dw * _dv
            _rf = _ds.reserve_frac_of(spec)
            if _rf > 0.0:
                r = np.where(_reserve_of(len(r)), 0.0, r)
            # |epi| is per SITE: a multi-site map makes epitope_idx a LIST of patches, which has no
            # .size (that call crashed the first multi-site run before it reached a single step).
            logger.info("epitope reward: P=%d mean=%.3f (|para|=%d,|epi|=%s, d*=%s)", len(r),
                        float(r.mean()), int(paratope_idx.size),
                        "+".join(str(int(s.size)) for s in bs.epitope_sites(epitope_idx)),
                        (spec.umbrella_sweep) if sweep else f"{float(spec.cdr_target_rmsd):.2f}")
            return r
        if state["ref"] is None:
            state["ref"] = reps[0].copy()
            logger.info("cdr reference anchored on a sharp mini-rollout structure (|paratope|=%d)",
                        int(paratope_idx.size))
            try:    # persist the anchor so evaluation can score against the SAME reference the
                    # objective uses (scoring against a different anchor is what made the last
                    # calibration look like a failure when the objective was in fact improving)
                np.save(os.path.join(_dump_dir(model),
                                     f"cdr_anchor_{sample_name}_seed{_diffusion_seed()}.npy"), state["ref"])
            except Exception as e:                       # noqa: BLE001, never break sampling
                logger.warning("could not save cdr anchor (%s)", e)
        # per-particle d* when sweeping (round-robin umbrellas), else the single scalar target
        tgt = bs.sweep_targets(len(reps), sweep) if sweep else target
        r = bs.cdr_rmsd_reward(reps, state["ref"], paratope_idx, target_rmsd=tgt)
        logger.info("cdr reward: P=%d mean=%.3f max=%.3f (|paratope|=%d, target=%s)",
                    len(r), float(r.mean()), float(r.max()), int(paratope_idx.size),
                    ((spec.umbrella_sweep) if sweep else f"{float(target):.2f}"))
        return r

    def guidance(coords, sigma):
        """Twisted proposal: the displacement eps_t * grad_x r to ADD to this step's update."""
        state["sigma"] = float(sigma) if sigma is not None else None   # for the staged facing gate
        if epitope_idx is not None:
            c = coords.detach().float().cpu().numpy() if hasattr(coords, "detach") else np.asarray(coords)
            c = c.reshape(-1, c.shape[-2], 3)
            reps = bs.residue_reps_from_atoms(c, atom_to_token, n_token)
            tg = bs.sweep_targets(len(reps), sweep) if sweep else float(spec.cdr_target_rmsd)
            g_tok = bs.site_contact_grad(reps, paratope_idx, epitope_idx, target_dist=tg,
                             contact_weight=_ds.contact_weight_of(spec),
                             contact_saturate_tau=float(getattr(spec, "contact_saturate_tau", 0.0)),
                             r_contact=float(getattr(spec, "contact_r", 8.0)),
                             contact_sigma=float(getattr(spec, "contact_sigma", 2.0)),
                             hinge=bool(getattr(spec, "site_hinge", False)),
                             huber_delta=float(getattr(spec, "site_huber_delta", 0.0)),
                             contact_gate=float(getattr(spec, "site_contact_gate", 0.0)),
                             site_floor=float(getattr(spec, "site_floor", 0.0)),
                             softmin_tau=float(getattr(spec, "site_softmin_tau", 0.0)),
                             dev_cap=float(getattr(spec, "site_dev_cap", 0.0)))
            _cw = _ds.clash_weight_of(spec)
            if _cw > 0.0:
                _stamp_term("clash", _cw)
                _, _cg = bs.clash_penalty(reps, paratope_idx, bs.epitope_sites(epitope_idx)[0],
                                          d_min=float(getattr(spec, "clash_d_min", 4.0)))
                g_tok = g_tok + _cw * _cg
            _fw = _facing_weight_now(spec, state)
            if _fw > 0.0:
                _, _fg = bs.facing_shell(reps, paratope_idx, bs.epitope_sites(epitope_idx)[0])
                g_tok = g_tok + _fw * _fg
            _ow = _ds.orient_weight_of(spec)
            if _ow > 0.0:
                _stamp_term("orient", _ow, tol_deg=float(getattr(spec, "orient_tol_deg", 0.0)))
                _, _og = bs.orientation_reward(reps, paratope_idx,
                                               bs.epitope_sites(epitope_idx)[0],
                                               bs.antibody_tokens(paratope_idx, _chain_of),
                                               tol_deg=float(getattr(spec, "orient_tol_deg", 0.0)))
                g_tok = g_tok + _ow * bs.relative_scale(g_tok, _og) * _og
            _dw = _ds.direction_weight_of(spec)
            if _dw > 0.0:
                _stamp_term("approach", _dw)
                _, _dg = bs.direction_reward(reps, paratope_idx, epitope_idx,
                                             chain_of=_chain_of,
                                             r_local=float(getattr(spec, "direction_r_local", 15.0)))
                g_tok = g_tok + _dw * _dg
            _rf = _ds.reserve_frac_of(spec)
            if _rf > 0.0:
                g_tok = np.where(_reserve_of(len(g_tok))[:, None, None], 0.0, g_tok)
            g_atom = gd.scatter_token_grad_to_atoms(g_tok, atom_to_token, c.shape[-2])
            _terms = [gd.GuidanceTerm("site", getattr(spec, "site_route", "coord"),
                                      lambda _x, _g=g_atom: _g)]
            _disp, _ = gd.apply_terms(
                _terms, c, sigma, spec.guidance_scale,
                max_norm=spec.guidance_max_norm,
                max_rms_a=float(getattr(spec, 'guidance_max_rms_a', 0.0)),
                schedule=getattr(spec, "guidance_schedule", "sigma2"),
                latent_eta=float(getattr(spec, "latent_eta", 0.01)),
                latent_reg_scale=float(getattr(spec, "latent_reg_scale", 0.0)))
            return _disp
        if state["ref"] is None:
            return None            # anchor not established yet -> no steering (the early high-sigma
                                   # phase, which the reward's schedule_start_frac also skips).
        c = coords.detach().float().cpu().numpy() if hasattr(coords, "detach") else np.asarray(coords)
        c = c.reshape(-1, c.shape[-2], 3)
        reps = bs.residue_reps_from_atoms(c, atom_to_token, n_token)
        tgt = bs.sweep_targets(len(reps), sweep) if sweep else target
        if getattr(spec, "reward", "") == "cdr_distogram":
            _, g_tok = bs.cdr_distogram_reward(reps, state["ref"], paratope_idx, target_dev=tgt,
                                               delta=float(getattr(spec, "distogram_delta", 2.0)))
        else:
            g_tok = bs.cdr_rmsd_grad(reps, state["ref"], paratope_idx, target_rmsd=tgt)
        g_atom = gd.scatter_token_grad_to_atoms(g_tok, atom_to_token, c.shape[-2])
        if str(getattr(spec, "conf_route", "coord")).lower() == "latent":
            _EMBEDOPT_GRAD_FN[0] = lambda _c, _g=g_atom: _g
            return None
        _terms = [gd.GuidanceTerm("conf", getattr(spec, "conf_route", "coord"),
                                  lambda _x, _g=g_atom: _g)]
        _disp, _ = gd.apply_terms(
            _terms, c, sigma, spec.guidance_scale,
            max_norm=spec.guidance_max_norm,
            max_rms_a=float(getattr(spec, 'guidance_max_rms_a', 0.0)),
            schedule=getattr(spec, "guidance_schedule", "sigma2"),
            latent_eta=float(getattr(spec, "latent_eta", 0.01)),
            latent_reg_scale=float(getattr(spec, "latent_reg_scale", 0.0)))
        return _disp

    logger.info("cdr_rmsd reward active for %r (|paratope|=%d, %s%s)", sample_name, int(paratope_idx.size),
                f"sweep d*=[{spec.umbrella_sweep}]" if sweep else f"target={float(target):.2f}",
                f", GUIDED scale={spec.guidance_scale}" if spec.guidance_scale else ", gradient-free")
    reward.uses_confidence_head = False
    reward.steering_groups = steering_groups     # read by the SMC loop's stratified resample
    reward.n_sites = n_sites
    return reward, (guidance if spec.guidance_scale else None)


def _make_reward_fn(model, kw: dict, spec: DiffusionSearchSpec, chunk_size):
    """Return ``reward(coords) -> np.ndarray[P]``: the scalar the search maximises."""
    import numpy as np
    import torch
    sample_confidence = _cofold_module("protenix.model.sample_confidence")
    from method.verifier.ipsae import compute_ipsae
    from method.diffusion_side.base_search import soft_pae_interface

    if getattr(spec, "reward", "ipsae") == "none":
        raise RuntimeError(
            "reward='none' reached the reward builder: this arm declares no objective, so scoring "
            "should have been skipped by DiffusionSearchSpec.reward_affects_sampling(). Falling "
            "through would silently score ipSAE instead. Fix the caller, do not relax this check.")
    if getattr(spec, "reward", "ipsae") in tuple(_ds.CONF_REWARDS) + tuple(_ds.SITE_REWARDS):
        cdr = _make_cdr_reward_fn(model, kw, spec)
        if cdr is not None:
            return cdr                                   # (reward, guidance)
        logger.warning("cdr_rmsd reward unavailable, using ipSAE (set $PROTENIX_CDR_MAP)")

    ifd_net = kw["input_feature_dict"]
    ifd = _map_space_ifd(ifd_net)
    s_inputs, s_trunk = kw["s_inputs"], kw["s_trunk"]
    z_trunk = _trunk_z(model, kw)
    tri_mult = getattr(model.configs, "triangle_multiplicative", "torch")
    tri_attn = getattr(model.configs, "triangle_attention", "torch")
    pae_bins = sample_confidence.get_bin_params(model.configs.loss.pae)
    asym = ifd["asym_id"].reshape(-1).long().tolist()
    chain_ids = [str(a) for a in asym]
    n_token = len(chain_ids)
    ca_dummy = np.zeros((n_token, 3), dtype=np.float64)   # canonical ipSAE ignores coords (len-check only)
    ipk = spec.ipsae_kwargs()

    pae_cut = float(ipk.get("pae_cutoff", 10.0))

    def reward(coords):
        P = coords.shape[-3]
        scores = np.empty(P, dtype=np.float64)
        n_fallback = 0
        with torch.no_grad():
            for i in range(P):
                ci = coords[..., i:i + 1, :, :]           # keep the N_sample axis = 1
                out = model.run_confidence_head(
                    input_feature_dict=ifd_net, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
                    pair_mask=None, x_pred_coords=ci, triangle_multiplicative=tri_mult,
                    triangle_attention=tri_attn, inplace_safe=False, chunk_size=chunk_size)
                pae_logits = out[1]                        # [..., 1, N_token, N_token, bins]
                pae = sample_confidence.logits_to_score(pae_logits, **pae_bins)
                pae_np = pae.reshape(n_token, n_token).float().cpu().numpy().astype(np.float64)
                r = compute_ipsae(pae_np, ca_dummy, chain_ids, plddt=None, **ipk)
                if r is not None and r.ipsae is not None:
                    scores[i] = float(r.ipsae)             # primary: canonical (Dunbrack) ipSAE
                else:
                    # hard ipSAE undefined (no interface residue passes the cutoff), fall back to
                    # the always-defined PAE-only soft interface score so the search keeps a signal.
                    scores[i] = soft_pae_interface(pae_np, chain_ids, pae_cutoff=pae_cut)
                    n_fallback += 1
        logger.info("reward: P=%d mean=%.4f min=%.4f max=%.4f (%d/%d ipSAE-undefined→soft fallback)",
                    P, float(scores.mean()), float(scores.min()), float(scores.max()), n_fallback, P)
        return scores

    reward.uses_confidence_head = True          # one run_confidence_head call per particle, above
    return reward, None            # ipSAE has no closed-form gradient -> gradient-free (culling only)


def _search_sample_diffusion(model, spec: DiffusionSearchSpec, kw: dict):
    """Coupled replacement for Protenix.sample_diffusion. Returns coords [..., N_sample, N_atom, 3]."""
    import time
    import numpy as np
    import torch
    centre_random_augmentation = _cofold_module("protenix.model.utils").centre_random_augmentation
    from method.diffusion_side import base_search as bs
    from method.diffusion_side import guidance as gd
    from method.diffusion_side import interacting_particles as ip
    from method.diffusion_side import diffusion_search as ds

    denoise_net = kw["denoise_net"]
    ifd_net = kw["input_feature_dict"]
    ifd = _map_space_ifd(ifd_net)
    s_inputs, s_trunk, z_trunk = kw["s_inputs"], kw["s_trunk"], _trunk_z(model, kw)
    pair_z, p_lm, c_l = kw.get("pair_z"), kw.get("p_lm"), kw.get("c_l")
    noise_schedule = kw["noise_schedule"]
    N_sample = int(kw["N_sample"])

    _K = int(getattr(spec, "subsample_pool", 1) or 1)
    _joint_cond = None
    _jpool = {"sub": None, "sub0": None, "groups": None}
    # OUTER scope on purpose: incremented inside _run_population, read by the stats block at the
    # end of _search_sample_diffusion. A list holder is mutated through the closure.
    _lam_never = [0]        # scoring events that ran with an UNCALIBRATED lam, i.e. no reward tilt
    if _K > 1:
        _ws = int(os.environ.get("WORLD_SIZE", "1") or 1)
        if _ws > 1:
            raise RuntimeError(
                f"joint cross-subsample pool (subsample_pool={_K}) requires WORLD_SIZE=1, got {_ws}. "
                "The K conditionings are split across ranks by DistributedSampler, so no rank ever "
                "holds all K and the deferred diffusion never fires -- the run would emit only "
                f"{_K} placeholder structures and still exit 0. Set nproc_per_node=1; two GPUs means "
                "two independent runs.")
        _cpu = os.environ.get("TTS_JOINT_CACHE_CPU", "") == "1"
        if _cpu and (str(getattr(spec, "conf_route", "coord")).lower() == "latent"
                     or os.environ.get("PROTENIX_FORCE_DENOISER_GRAD", "") == "1"):
            raise RuntimeError(
                "TTS_JOINT_CACHE_CPU=1 stages the joint conditioning cache through host RAM and "
                "detaches it, which severs the denoiser gradient a latent route depends on. Unset "
                "one of the two. The cache is a MEMORY knob and must never change what is sampled.")
        _entry = dict(s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
                      pair_z=pair_z, p_lm=p_lm, c_l=c_l)
        if _cpu:
            _entry = {k: (v.detach().to("cpu", copy=True).pin_memory()
                          if torch.is_tensor(v) and v.is_cuda else v)
                      for k, v in _entry.items()}
            torch.cuda.empty_cache()
        _JOINT["cache"].append(_entry)
        _JOINT["names"].append(_CURRENT_SAMPLE_NAME)
        if len(_JOINT["cache"]) < _K:
            logger.info("joint pool: cached conditioning %d/%d (%s), deferring the diffusion",
                        len(_JOINT["cache"]), _K, _CURRENT_SAMPLE_NAME)
            N_sample = 1
        else:
            _joint_cond = list(_JOINT["cache"])
            logger.info("joint pool: %d conditionings in ONE diffusion of %d particles (%s)",
                        _K, N_sample, ",".join(str(n) for n in _JOINT["names"]))
            _JOINT["cache"], _JOINT["names"] = [], []
    _EFFECTIVE_N_SAMPLE[0] = int(N_sample)
    # ---------------------------------------------------------------------------------------------
    inplace_safe = kw.get("inplace_safe", False)
    eff = kw.get("enable_efficient_fusion", False)

    sd = model.configs.sample_diffusion
    gamma0, gamma_min = float(sd.get("gamma0")), float(sd.get("gamma_min"))
    noise_scale_lambda, step_scale_eta = float(sd.get("noise_scale_lambda")), float(sd.get("step_scale_eta"))
    attn_chunk_size = model.configs.infer_setting.chunk_size if not model.training else None
    # particle-axis chunk for the DENOISER CALL ONLY (never the trajectory -- see _denoise)
    diff_chunk = getattr(model.configs.infer_setting, "sample_diffusion_chunk_size", None)

    batch_shape = s_inputs.shape[:-2]
    device, dtype = s_inputs.device, s_inputs.dtype
    N_atom = ifd["atom_to_token_idx"].size(-1)
    n_steps = len(noise_schedule) - 1
    start = int(spec.schedule_start_frac * n_steps)
    _end_frac = float(getattr(spec, "schedule_end_frac", 1.0))
    stop = n_steps if _end_frac >= 1.0 else max(int(_end_frac * n_steps), start + 1)
    if stop < n_steps:
        logger.info("guidance GATE: steering active on steps [%d, %d) of %d, base kernel below "
                    "(schedule_end_frac=%.3f)", start, stop, n_steps, _end_frac)
    n_token = int(ifd["asym_id"].reshape(-1).shape[0])
    n_token_net = int(ifd_net["asym_id"].reshape(-1).shape[0])
    if n_token_net != n_token:
        logger.info("token space: reward/maps at %d residue tokens, denoiser at %d structural "
                    "sub-tokens (%.2fx)", n_token, n_token_net, n_token_net / max(n_token, 1))
    _reward_needed = spec.reward_affects_sampling()
    if not _reward_needed:
        logger.info("reward scoring SKIPPED: %s cannot act on it (no gradient, no tilt, no selection)",
                    spec.algorithm)
    stats = {"denoiser_particle_steps": 0, "confidence_passes": 0, "t_denoise": 0.0,
             "t_confidence": 0.0, "t_reward": 0.0, "t_guidance": 0.0, "t_resample": 0.0,
             "n_resample_events": 0, "n_score_events": 0}
    _t_all = time.perf_counter()
    if _reward_needed:
        _reward_impl, _guidance_impl = _make_reward_fn(model, kw, spec, attn_chunk_size)
    else:
        _reward_impl, _guidance_impl = None, None
    _algo = getattr(spec, "algorithm", "tds")
    if _algo in ("fk", "af3_red", "svgd") and _guidance_impl is not None:
        _guidance_impl = None
    logger.info("algorithm=%s: reward-gradient=%s, fk-weights/resampling=%s (%s)", _algo,
                ("OFF" if (_algo == "svgd" and bool(getattr(spec, "svgd_repulsion_only", True)))
                 else ("ON" if _guidance_impl is not None else "OFF")),
                "OFF" if _algo in ("svgd", "af3_red") else "ON",
                {"fk": "Singhal 2501.06848, reference has no gradient term",
                 "tds": "Wu 2306.17775, twisted proposal + weight correction",
                 "cast": "CAST: TDS over K subsample latents in one filter, site-tilted",
                 "svgd": "Liu & Wang 2016, transport only",
                 "af3_red": "Okazaki 2025, repulsive bias force only"}.get(_algo, "unknown"))

    _res_f = _ds.reserve_frac_of(spec)
    _res_rep = bool(getattr(spec, "reserve_repulsion", False))
    # EFFECTIVE, not requested. Reporting the requested flag would have the line assert "repulsion=ON"
    # on a tds arm that cannot run it, contradicting the warning immediately below it.
    _res_rep_eff = _res_rep and _algo == "svgd"
    logger.info("reserve: %s (frac=%.3f, reserve_repulsion=%s)",
                "ON" if _res_f > 0.0 else "OFF", _res_f,
                ("ON" if _res_rep_eff else "OFF (pristine Best-of-N)") if _res_f > 0.0 else "n/a")
    _ra_f = _ds.realloc_frac_of(spec)
    logger.info("site reallocation: %s (max_move_frac=%.3f, floor=%d/cell, contact_r=%.1f A)",
                "ON" if _ra_f > 0.0 else "OFF", _ra_f,
                int(getattr(spec, "realloc_min_per_cell", 6)),
                float(getattr(spec, "realloc_contact_r", 10.0)))
    _rep_b = float(getattr(spec, "repulsion_beta", 0.0) or 0.0)
    logger.info("within-cell repulsion: %s (repulsion_beta=%.3f%s)",
                "ON" if _rep_b > 0.0 else "OFF", _rep_b,
                ", reserve EXCLUDED from the penalty" if _rep_b > 0.0 and _res_f > 0.0 else "")
    if _res_rep and _algo != "svgd":
        logger.warning("reserve_repulsion=true has NO EFFECT under algorithm=%s -- the repulsion "
                       "term exists only under svgd, so the reserve is pristine regardless", _algo)
    if _res_rep and _res_f <= 0.0:
        logger.warning("reserve_repulsion=true but the reserve is OFF "
                       "(reserve_enabled=%s, reserve_frac=%s) -- nothing to apply it to",
                       bool(getattr(spec, "reserve_enabled", False)),
                       getattr(spec, "reserve_frac", 0.0))

    _atom_to_token = ifd["atom_to_token_idx"].reshape(-1).long().cpu().numpy()

    # ---- SVGD transport. STALE AND UNSUPPORTED: algorithm=svgd is selected by nothing that ships
    # and no reported result uses it, but this block stays because `_svgd_moved` / `_svgd_n` below
    # are also the af3_red displacement counters and `_svgd_par` is its paratope resolution. See
    # the note at DIFFUSION_FAMILIES in method/diffusion_side/diffusion_search.py.
    # Built here rather than inside the reward factory because SVGD has no reward, it is a
    # transport map over the particle cloud, so it needs only the paratope indices.
    _svgd_opt = ip.AdaGrad(stepsize=float(getattr(spec, "svgd_stepsize", 1e-3)),
                               alpha=float(getattr(spec, "svgd_adagrad_alpha", 0.9)))
    _svgd_par = _paratope_tokens(ifd, spec) if _algo == "svgd" else None
    _svgd_ag = None
    _site_path = [False]
    _want_site_kernel = bool(getattr(spec, "svgd_site_kernel", False))
    if _want_site_kernel and not (_algo == "svgd" and _svgd_par is not None):
        raise RuntimeError(
            f"svgd_site_kernel=true but it cannot run: algorithm={_algo!r} (needs 'svgd') and "
            f"paratope map {'MISSING' if _svgd_par is None else 'present'}. Refusing to run this "
            f"arm unsteered under a site-kernel label; set svgd_site_kernel=false for a control.")
    if _want_site_kernel:
        try:
            _cid = ifd["asym_id"].reshape(-1).long().cpu().numpy()
            _ab_chains = bs.antibody_chains(_svgd_par[0], _cid)
            if len(_ab_chains) > 2:
                raise RuntimeError(
                    f"paratope map spans {len(_ab_chains)} chains ({sorted(_ab_chains)}), but an "
                    f"antibody has at most two. The map and the model's token order disagree, so "
                    f"the site kernel would align on the wrong chains")
            _svgd_ag = np.flatnonzero(~np.isin(_cid, list(_ab_chains)))
            if _svgd_ag.size < 3:
                raise RuntimeError(
                    f"svgd site kernel: only {int(_svgd_ag.size)} antigen tokens resolved, too few "
                    f"to superpose on; the arm would silently run the SITE-BLIND paratope kernel")
            else:
                logger.info("svgd site kernel ON: %d antigen tokens vs %d paratope tokens",
                            int(_svgd_ag.size), int(len(_svgd_par[0])))
        except RuntimeError:
            raise                       # our own refusals must not be swallowed by the guard below
        except Exception as e:          # noqa: BLE001
            raise RuntimeError(
                f"svgd site kernel: could not resolve antigen tokens ({e}); refusing to fall back "
                f"to the site-blind paratope kernel under a site-kernel label")
    _site_path[0] = _svgd_ag is not None      # derived, never set independently

    _site_name = _extract_sample_name(ifd)
    _site_div_archive = [_load_site_archive(_site_name)]

    _bp_spec = bpmod.BoltzPerturbSpec(
        enabled=bool(getattr(spec, "boltz_perturb_enabled", False)),
        tcp=bool(getattr(spec, "boltz_perturb_tcp", False)),
        tbp=bool(getattr(spec, "boltz_perturb_tbp", False)),
        noise_scale=float(getattr(spec, "boltz_perturb_noise_scale", 0.3)),
        psi=float(getattr(spec, "boltz_perturb_psi", 0.7)),
        rescale=bool(getattr(spec, "boltz_perturb_rescale", True)),
        norm_proj=bool(getattr(spec, "boltz_perturb_norm_proj", True)),
        mode=str(getattr(spec, "boltz_perturb_mode", "cross_and_ll")),
        tads=bool(getattr(spec, "boltz_perturb_tads", True)),
        tads_floor=float(getattr(spec, "boltz_perturb_tads_floor", 0.0)))
    _bp_t = [1.0]                         # schedule fraction, refreshed once per denoise step
    _bp_ab_mask = None                    # (N_token,) bool: True on ANTIBODY tokens (their `ligand`)
    _bp_target = [None]                   # which conditioning got perturbed, logged once per run
    _grad_probe = [None]                  # one-shot latch: did autograd actually reach the denoiser
    if _bp_spec.enabled:
        if not (_bp_spec.tcp or _bp_spec.tbp):
            raise RuntimeError("boltz_perturb_enabled=true but neither boltz_perturb_tcp nor "
                               "boltz_perturb_tbp is set, so nothing would be perturbed")
        if _bp_spec.mode not in ("cross_only", "cross_and_ll", "all", "ll_only"):
            raise RuntimeError(f"unknown boltz_perturb_mode {_bp_spec.mode!r} "
                               "(cross_only|cross_and_ll|all|ll_only)")
        try:
            _cid_c = ifd["asym_id"].reshape(-1).long().cpu().numpy()
            _par_c = _svgd_par or _paratope_tokens(ifd, spec)
            if not _par_c:
                raise RuntimeError("boltz_perturb needs a paratope map to identify the antibody "
                                   "chains (their `ligand` role), and none resolved")
            _abc = bs.antibody_chains(_par_c[0], _cid_c)
            _bp_ab_mask = np.isin(_cid_c, list(_abc))
            _npair = int(bpmod.pair_mask_for_mode(_bp_ab_mask, _bp_spec.mode).sum())
            logger.info("boltz_perturb: %d/%d antibody tokens, mode=%s -> %d/%d pair entries",
                        int(_bp_ab_mask.sum()), int(_bp_ab_mask.size), _bp_spec.mode,
                        _npair, int(_bp_ab_mask.size ** 2))
        except Exception as e:      # noqa: BLE001 -- an antibody label must never mean "global"
            raise RuntimeError(f"boltz_perturb requested but the antibody mask failed: {e!r}")
        logger.info("boltz_perturb ON: tcp=%s tbp=%s noise_scale=%.3g psi=%.2f rescale=%s "
                    "norm_proj=%s mode=%s tads=%s floor=%.3g",
                    _bp_spec.tcp, _bp_spec.tbp, _bp_spec.noise_scale, _bp_spec.psi,
                    _bp_spec.rescale, _bp_spec.norm_proj, _bp_spec.mode, _bp_spec.tads,
                    _bp_spec.tads_floor)
    _guidance_raw = _make_raw_reward_grad(model, kw, spec) if (
        _algo == "svgd" and not bool(getattr(spec, "svgd_repulsion_only", True))
        and not _site_path[0]) else None
    if _site_path[0] and not bool(getattr(spec, "svgd_repulsion_only", True)):
        logger.info("svgd site kernel: reward gradient DISABLED (the site path is repulsion-only, "
                    "so a reward would be computed and dropped)")
    _svgd_moved, _svgd_n = [0.0], [0]     # cumulative |dx| and update count: makes a no-op visible
    _guid_moved, _guid_n = [0.0], [0]

    _red_hist, _red_chain, _red_n = None, None, [0]
    if _algo == "af3_red":
        _red_chain = ifd["asym_id"].reshape(-1).long().cpu().numpy()
        # paper-exact: whole-protein C-alpha set, independent of the conformation-CV subset
        import copy as _copy
        _red_spec = _copy.copy(spec)
        object.__setattr__(_red_spec, "rmsd_subset", getattr(spec, "red_rmsd_subset", "all"))
        _red_par = _paratope_tokens(ifd, _red_spec)
        _red_idx = None if _red_par is None else _red_par[0]
        _red_tid = str(_extract_sample_name(ifd) or _CURRENT_SAMPLE_NAME or "unknown").split("__")[0]
        _red_root = os.environ.get("PROTENIX_AF3RED_HIST_DIR") or os.path.join(
            os.path.dirname(os.path.normpath(_dump_dir(model))), "af3red_history")
        os.makedirs(_red_root, exist_ok=True)
        _red_mask_path = os.path.join(_red_root, f"af3red_plddt_{_red_tid}.npy")
        _red_plddt_applied = None
        if _red_idx is not None and os.path.exists(_red_mask_path):
            try:                                     # intersect the subset with pLDDT>60, as the paper does
                _m = np.load(_red_mask_path)
                _kept = _red_idx[_m[_red_idx]] if _m.shape[0] == len(_red_chain) else _red_idx
                if _kept.size >= 3:
                    _red_plddt_applied = (len(_red_idx), len(_kept))
                    _red_idx = _kept
            except (OSError, ValueError, IndexError):
                pass
        logger.info("af3_red: RMSD over %s (%d residues), C-alpha atoms, pLDDT filter %s",
                    getattr(spec, "red_rmsd_subset", "all"),
                    0 if _red_idx is None else len(_red_idx),
                    ("APPLIED, %d -> %d residues" % _red_plddt_applied) if _red_plddt_applied
                    else "NOT APPLIED (no %s; build it with `token_maps af3red-plddt-mask`)"
                    % os.path.basename(_red_mask_path))
        _red_path = os.path.join(_red_root, f"af3red_hist_{_red_tid}.npy")
        if os.path.exists(_red_path):
            try:
                _red_hist = np.load(_red_path)
            except (OSError, ValueError):
                _red_hist = None
        _red_n[0] = 0 if _red_hist is None else len(_red_hist)
        logger.info("af3_red: %d previously completed structures in the bias history (%s)",
                    _red_n[0], "unbiased reference run" if _red_n[0] == 0 else "biased")
        _red_bias = _red_hist
        if (getattr(spec, "red_bias_restrict", True) and _red_hist is not None
                and len(_red_hist) > N_sample and _red_idx is not None):
            _keep = ip.bias_restrict_mask(_red_hist, _red_hist[:N_sample], _red_idx,
                                          float(getattr(spec, "red_cut_rmsd", 0.5)))
            logger.info("af3_red: bias_restrict cut_rmsd=%.2f A keeps %d/%d history structures "
                        "for the bias (stored history stays %d)",
                        float(getattr(spec, "red_cut_rmsd", 0.5)), int(_keep.sum()),
                        len(_red_hist), len(_red_hist))
            if _keep.any():
                _red_bias = _red_hist[_keep]
            else:
                logger.warning("af3_red: bias_restrict excluded EVERY history structure -- biasing "
                               "against the full set, else the bias would be silently off")
    if _algo == "svgd" and _svgd_par is None:
        raise RuntimeError(
            "algorithm=svgd but no paratope map resolved from $PROTENIX_CDR_MAP: transport would be "
            "disabled and the arm would silently produce an UNSTEERED pool under a steering label")

    def _svgd_transport(x_cur, score_atoms, _t_frac=1.0):
        """phi* over the particle cloud, mapped atoms -> residues -> atoms."""
        if _svgd_par is None:
            return None
        par_idx, a2t, n_tok = _svgd_par
        c = x_cur.detach().float().cpu().numpy().reshape(-1, x_cur.shape[-2], 3)
        s = np.asarray(score_atoms).reshape(-1, x_cur.shape[-2], 3)
        reps = bs.residue_reps_from_atoms(c, a2t, n_tok)
        score_res = bs.residue_reps_from_atoms(s, a2t, n_tok)
        if _svgd_ag is not None:
            phi_res, _rho = ip.site_repulsion(
                reps, par_idx, _svgd_ag,
                bandwidth=float(getattr(spec, "svgd_bandwidth", 0.0)),
                archive=(_site_div_archive[0]
                         if float(getattr(spec, "site_archive_weight", 0.0)) > 0 else None),
                archive_weight=float(getattr(spec, "site_archive_weight", 0.0)))
            scale = bs.anneal_scale(_t_frac, float(getattr(spec, "site_anneal_tau1", 0.2)),
                                    float(getattr(spec, "site_anneal_tau2", 0.9)))
            phi_res = bs.normalise_step(
                phi_res, scale * float(getattr(spec, "site_step_a", 0.0)), subset_idx=par_idx)
        else:
            phi_res = ip.svgd_phi(reps, score_res, par_idx,
                                  bandwidth=float(getattr(spec, "svgd_bandwidth", 0.0)),
                                  repulsion_only=bool(getattr(spec, "svgd_repulsion_only", True)),
                                  drive_self_norm=bool(getattr(spec, "svgd_drive_self_norm", False)),
                                  drive_only=bool(getattr(spec, "svgd_drive_only", False)),
                                  recv_mask=bs.repulsion_recipients(
                                      reps.shape[0],
                                      _ds.reserve_frac_of(spec),
                                      bool(getattr(spec, "reserve_repulsion", True))))
        # DISPLACEMENT, not gradient: the site path sets an Angstrom budget, so the token
        # vector must reach every atom intact rather than divided by the atom count.
        if _svgd_ag is not None:
            return gd.scatter_token_step_to_atoms(phi_res, a2t, c.shape[1])
        return gd.scatter_token_grad_to_atoms(phi_res, a2t, c.shape[1])

    _ref_plddt = [None]                 # ConforMix's "default prediction" reference, captured once

    def _particle_plddt(coords):
        """Per-token pLDDT for every particle, for the ConforMix pLDDT criterion."""
        c = coords if coords.dim() >= 4 else coords[None]
        P = int(c.shape[-3])
        _t = time.perf_counter()
        out = []
        with torch.no_grad():
            for i in range(P):
                _pl = model.run_confidence_head(
                    input_feature_dict=ifd_net, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
                    pair_mask=None, x_pred_coords=c[..., i:i + 1, :, :],
                    triangle_multiplicative=getattr(model.configs, "triangle_multiplicative", "torch"),
                    triangle_attention=getattr(model.configs, "triangle_attention", "torch"),
                    inplace_safe=False, chunk_size=attn_chunk_size)[0]
                out.append(_token_plddt(_pl, ifd, n_token, model))
        _cuda_sync()                      # GPU work above, sync before stopping the clock
        stats["confidence_passes"] += P
        stats["t_confidence"] += time.perf_counter() - _t
        return np.stack(out)

    def _physical_mask(coords):
        """Per-particle physicality on x̂₀ (ConforMix's filtering step, arXiv 2512.03312 §3.2)."""
        c = coords.detach().float().cpu().numpy()
        c = c.reshape(-1, c.shape[-2], 3)
        reps = bs.residue_reps_from_atoms(c, _atom_to_token, n_token)
        _pl_pool = _ref = None
        if bool(getattr(spec, "phys_use_plddt", False)):
            _pl_pool = _particle_plddt(coords)
            if _ref_plddt[0] is None:
                _ref_plddt[0] = _pl_pool.mean(0)
            _ref = _ref_plddt[0]
        return bs.physicality_mask(
            plddt_pool=_pl_pool, ref_plddt=_ref,
            ca_pool=reps, clash_r=spec.clash_r, max_clashes=spec.max_clashes,
            use_clash=bool(getattr(spec, "phys_use_clash", True)),
            use_plddt=bool(getattr(spec, "phys_use_plddt", False)),
            window=int(getattr(spec, "plddt_window", 10)),
            max_drop=float(getattr(spec, "plddt_max_drop", 0.20)))
    _sweep = spec.sweep_list()
    _cells_fn = getattr(_reward_impl, "steering_groups", None)
    _n_sites = int(getattr(_reward_impl, "n_sites", 1))
    _cell_warned = [False]          # the budget needs P, which is only known at the first resample

    # Default True: an unlabelled new reward over-reports rather than silently under-reporting.
    _reward_uses_head = bool(getattr(_reward_impl, "uses_confidence_head", True))

    def reward_fn(coords):
        _t = time.perf_counter()
        r = _reward_impl(coords)
        _dt = time.perf_counter() - _t
        if _reward_uses_head:
            stats["t_confidence"] += _dt
            stats["confidence_passes"] += int(coords.shape[-3])   # one confidence-head pass / particle
        else:
            stats["t_reward"] += _dt      # pure geometry: real time, but NOT confidence-head time
        return r

    def _denoise(x_noisy, t_hat_e):
        """Denoise the whole population, CHUNKING ONLY THE NETWORK CALL."""
        _t = time.perf_counter()
        n_p = int(x_noisy.shape[-3])
        if _joint_cond and _jpool["groups"] is None:
            _jpool["sub"] = bs.subsample_ids(n_p, len(_joint_cond),
                                             n_umbrellas=max(len(_sweep), 1), n_sites=max(_n_sites, 1))
            bs.check_subsample_spread(
                bs.steering_group_ids(n_p, bs.sweep_targets(n_p, _sweep) if _sweep else None,
                                      max(_n_sites, 1), _ds.reserve_frac_of(spec)), _jpool["sub"])
            _jpool["groups"] = gd.index_groups(_jpool["sub"])
            _jpool["sub0"] = np.asarray(_jpool["sub"]).copy()
        step = n_p if not diff_chunk else max(1, int(diff_chunk))
        _zt, _pz, _tgt = z_trunk, pair_z, None
        _st = s_trunk
        if _bp_spec.enabled:
            if _bp_spec.tbp:                 # Token Bias Perturbation -> the PAIR track
                if pair_z is not None:
                    _pz, _tgt = bpmod.tbp_perturb_torch(
                        pair_z, _bp_ab_mask, _bp_spec, t=_bp_t[0]), "pair_z"
                elif z_trunk is not None:
                    _zt, _tgt = bpmod.tbp_perturb_torch(
                        z_trunk, _bp_ab_mask, _bp_spec, t=_bp_t[0]), "z_trunk"
                else:
                    raise RuntimeError("boltz_perturb_tbp=true but the denoiser was given neither "
                                       "pair_z nor z_trunk, so there is no pair track to perturb")
            if _bp_spec.tcp:                 # Token Conditioning Perturbation -> the SINGLE track
                # A missing s_trunk under this flag is an error, not a skip, or the arm runs
                # pair-only under a label that says both tracks were perturbed.
                if _st is None:
                    raise RuntimeError("boltz_perturb_tcp=true but the denoiser was given no "
                                       "s_trunk, so the single track cannot be perturbed")
                _st = bpmod.tcp_perturb_torch(_st, _bp_ab_mask, _bp_spec, t=_bp_t[0])
                _tgt = "s_trunk" if _tgt is None else _tgt + "+s_trunk"
            if _bp_target[0] is None:
                _bp_target[0] = _tgt
                logger.info("boltz_perturb: perturbing %s %s", _tgt,
                            tuple((_st if _tgt == "s_trunk" else
                                   (_pz if _tgt.startswith("pair_z") else _zt)).shape))
        _latent_on = str(getattr(spec, "conf_route", "coord")).lower() == "latent"
        # MEASUREMENT-ONLY SWITCH: turn the graph on WITHOUT selecting a route, so an otherwise
        # normal run reports the memory delta. Changes no numerics, nothing reads the graph.
        if os.environ.get("PROTENIX_FORCE_DENOISER_GRAD", "") == "1":
            _latent_on = True
        _gctx = torch.enable_grad() if _latent_on else contextlib.nullcontext()
        outs = []
        with _gctx:
            if _latent_on and _grad_probe[0] is None:
                _grad_probe[0] = bool(torch.is_grad_enabled())
                logger.info("embedopt: autograd at the denoiser (is_grad_enabled=%s), chunk=%d of %d",
                            _grad_probe[0], step, n_p)
            # EMBEDOPT: the denoiser reads base + delta. Deltas are created once, zeros, and then
            # accumulate across diffusion steps. SHARED across particles, matching rs-station's code.
            _eo = _EMBEDOPT_GRAD_FN[0] is not None
            if _eo:
                if _EMBEDOPT_DELTAS[0] is None:
                    _EMBEDOPT_DELTAS[0] = [
                        None if _st is None else torch.zeros_like(_st, requires_grad=True),
                        None if _pz is None else torch.zeros_like(_pz, requires_grad=True)]
                _d_s, _d_z = _EMBEDOPT_DELTAS[0]
                _st = _st if _d_s is None else _st + _d_s
                _pz = _pz if _d_z is None else _pz + _d_z
            _grp = _jpool["groups"]
            if _grp is None:
                for a in range(0, n_p, step):
                    b = min(a + step, n_p)
                    outs.append(denoise_net(
                        x_noisy=x_noisy[..., a:b, :, :], t_hat_noise_level=t_hat_e[..., a:b],
                        input_feature_dict=ifd_net, s_inputs=s_inputs, s_trunk=_st, z_trunk=_zt,
                        pair_z=_pz, p_lm=p_lm, c_l=c_l, chunk_size=attn_chunk_size,
                        inplace_safe=inplace_safe, enable_efficient_fusion=eff))
            else:
                _acc = None
                for _idx, _si in _grp:
                    _c = _joint_cond[_si]
                    # Stage a host-resident conditioning back to the card for this group only.
                    # non_blocking pairs with the pinned buffers written at cache time.
                    if any(torch.is_tensor(v) and not v.is_cuda for v in _c.values()):
                        _c = {k: (v.to(x_noisy.device, non_blocking=True)
                                  if torch.is_tensor(v) and not v.is_cuda else v)
                              for k, v in _c.items()}
                    _sel_all = torch.as_tensor(_idx, dtype=torch.long, device=x_noisy.device)
                    for a in range(0, _sel_all.numel(), step):
                        _sel = _sel_all[a:a + step]
                        _d = denoise_net(
                            x_noisy=x_noisy.index_select(-3, _sel),
                            t_hat_noise_level=t_hat_e.index_select(-1, _sel),
                            input_feature_dict=ifd_net, s_inputs=_c["s_inputs"], s_trunk=_c["s_trunk"],
                            z_trunk=_c["z_trunk"], pair_z=_c["pair_z"],
                            p_lm=_c["p_lm"], c_l=_c["c_l"],
                            chunk_size=attn_chunk_size, inplace_safe=inplace_safe,
                            enable_efficient_fusion=eff)
                        if _acc is None:
                            _acc = torch.empty(x_noisy.shape[:-3] + (n_p,) + tuple(_d.shape[-2:]),
                                               dtype=_d.dtype, device=_d.device)
                        _acc.index_copy_(-3, _sel, _d)
                outs.append(_acc)
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-3)
        if _eo:
            # ONE VJP carries the conformational reward's closed-form dR/dx back to the deltas.
            # Guarded so a failure here can never silently degrade to an unsteered run.
            _dr = _EMBEDOPT_GRAD_FN[0](out)
            _dr = torch.as_tensor(np.asarray(_dr), dtype=out.dtype, device=out.device) \
                if not torch.is_tensor(_dr) else _dr
            if _dr.shape != out.shape:
                raise RuntimeError("embedopt: reward gradient %s does not match denoiser output %s"
                                   % (tuple(_dr.shape), tuple(out.shape)))
            _EMBEDOPT_DELTAS[0] = gd.embedopt_step(
                out, _EMBEDOPT_DELTAS[0], _dr,
                [float(getattr(spec, "latent_eta", 0.01))],
                reg_scale=float(getattr(spec, "latent_reg_scale", 0.0)))
            out = out.detach()      # downstream sampling must not keep the graph alive
        _cuda_sync()
        stats["t_denoise"] += time.perf_counter() - _t
        stats["denoiser_particle_steps"] += n_p            # one denoiser pass / particle / step
        return out

    def _is_event(step_i):
        return (step_i >= start) and (step_i < stop) \
            and ((step_i - start) % max(1, spec.schedule_every) == 0) \
            and (step_i < n_steps - 1)

    n_mini = int(spec.mini_rollout_steps)

    def _score_coords(x_l_now, x_hat0, step_i):
        """(b) The structure the reward scores: a short DETERMINISTIC rollout down the REMAINING
                noise schedule (current σ → 0, already EDM-spaced) so ipSAE sees a sharp structure.
                n_mini=0 → the raw one-step x̂₀.
        """
        if n_mini <= 0:
            return x_hat0
        tail = noise_schedule[step_i + 1:]                        # remaining σ (ends at ~0)
        if tail.numel() < 2:
            return x_hat0
        if tail.numel() > n_mini + 1:                             # subsample, keep both endpoints
            idx = torch.linspace(0, tail.numel() - 1, n_mini + 1, device=device).round().long()
            tail = tail[idx]
        x = x_l_now
        for s_last, s in zip(tail[:-1], tail[1:]):
            t_e = s_last.reshape((1,) * (len(batch_shape) + 1)).expand(*batch_shape, x.shape[-3]).to(dtype)
            xd = _denoise(x, t_e)                                 # counted in the compute budget
            x = x + (s - s_last) * ((x - xd) / t_e[..., None, None])   # Euler, no churn
        return x

    @torch.no_grad()
    def _run_population(P, init_x=None, steer=True, deterministic=False):
        """Step P particles through the full schedule. Returns final coords [..., P, N_atom, 3]."""
        x_l = init_x if init_x is not None else \
            noise_schedule[0] * torch.randn((*batch_shape, P, N_atom, 3), device=device, dtype=dtype)
        logw = np.zeros(x_l.shape[-3], dtype=np.float64)   # FK/TDS importance weights
        _lam_fixed = [None]        # lambda is calibrated at the first event, then held
        r_prev = None
        _r_first, _lvl_done = [None], [False]   # r_0 and its one-shot level correction
        g_prev = None                                      # lag-1 guidance shift for the churn
        ev = 0
        for step_i, (c_tau_last, c_tau) in enumerate(zip(noise_schedule[:-1], noise_schedule[1:])):
            _bp_t[0] = 1.0 - (float(step_i) / max(n_steps - 1, 1))   # 1 -> 0, the TADS convention
            # PAST THE GATE: drop the lag-1 guidance shift so the base kernel below `stop` is not
            # carrying a stale displacement from the last steered step. No-op at end_frac=1.0.
            if step_i >= stop:
                g_prev = None
            cur = x_l.shape[-3]
            if g_prev is not None and g_prev.shape == tuple(x_l.shape):
                g_t = torch.as_tensor(g_prev, device=device, dtype=dtype)
                pair = torch.cat([x_l, x_l + g_t], dim=-2)               # share one random rotation
                pair = centre_random_augmentation(x_input_coords=pair, N_sample=1).squeeze(dim=-3).to(dtype)
                x_l, g_rot = pair[..., :N_atom, :], pair[..., N_atom:, :] - pair[..., :N_atom, :]
            else:
                x_l = centre_random_augmentation(x_input_coords=x_l, N_sample=1).squeeze(dim=-3).to(dtype)
                g_rot = None
            gamma = 0.0 if deterministic else (gamma0 if c_tau > gamma_min else 0)
            t_hat = c_tau_last * (gamma + 1)
            delta_noise = torch.sqrt(t_hat ** 2 - c_tau_last ** 2)
            s_noise = float(noise_scale_lambda * delta_noise)
            eps_draw = torch.randn(x_l.shape, device=device, dtype=dtype)
            if g_rot is not None and s_noise > 0:
                x_noisy = x_l + g_rot + s_noise * eps_draw
                logw = logw + gd.proposal_log_correction_torch(
                    g_rot, eps_draw, s_noise).detach().cpu().numpy()
                # accumulate the liveness counter ON DEVICE too: float() here forced a sync EVERY
                # step. Read once at the end instead.
                _guid_moved[0] = _guid_moved[0] + g_rot.abs().mean().detach()
                _guid_n[0] += 1
                if step_i % 40 == 0:
                    _gp = torch.as_tensor(g_prev, device=device, dtype=dtype).reshape(g_rot.shape)
                    _c = torch.nn.functional.cosine_similarity(
                        _gp.reshape(_gp.shape[-3], -1), g_rot.reshape(g_rot.shape[-3], -1), dim=1)
                    _flat = g_rot.reshape(g_rot.shape[-3], -1)
                    _nF = _flat.norm(dim=1)
                    _cap = float(spec.guidance_max_norm)
                    _rms = float(getattr(spec, 'guidance_max_rms_a', 0.0) or 0.0)
                    if _rms > 0:
                        _cap = _rms * (int(_flat.shape[1]) ** 0.5)
                    logger.info("guidance step %d: sigma=%.3f s=%.3f mean|dx|=%.4f A |dx|_F=%.3f "
                                "(clip=%.1f, saturated=%s) cos(g,Rg)=%.3f [random by design]",
                                step_i, float(c_tau_last), s_noise, float(g_rot.abs().mean()),
                                float(_nF.mean()), _cap,
                                bool(_cap > 0 and float(_nF.max()) >= 0.99 * _cap),
                                float(_c.mean()))
            else:
                x_noisy = x_l + s_noise * eps_draw
            t_hat_e = t_hat.reshape((1,) * (len(batch_shape) + 1)).expand(*batch_shape, cur).to(dtype)
            x_denoised = _denoise(x_noisy, t_hat_e)                          # x̂₀
            delta = (x_noisy - x_denoised) / t_hat_e[..., None, None]
            if _algo == "af3_red" and _red_bias is not None and len(_red_bias):
                try:
                    _c = x_noisy.detach().float().cpu().numpy().reshape(-1, N_atom, 3)
                    _reps = _ca_token_coords(_c, ifd, len(_red_chain))   # paper uses C-alpha
                    _bg = ip.af3_red_bias_grad_batch(
                        _reps, _red_bias, _red_chain, float(c_tau), spec.red_n_smooth,
                        spec.red_sigma, spec.red_weight, spec.red_bias_tmin, subset_idx=_red_idx)
                    if np.abs(_bg).max() > 0:
                        _ba = gd.scatter_token_grad_to_atoms(_bg, _atom_to_token, N_atom)
                        delta = delta + torch.as_tensor(_ba, device=device, dtype=dtype
                                                        ).reshape(tuple(delta.shape))
                        _svgd_moved[0] += float(np.abs(_ba).mean())
                        _svgd_n[0] += 1
                except Exception as e:                   # noqa: BLE001, bias must not kill a run
                    logger.warning("af3_red bias skipped at step %d (%s)", step_i, e)
            dt = c_tau - t_hat
            x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta
            if _guidance_impl is not None and steer and spec.algorithm != "noise_search":
                try:                                     # gradient for the NEXT step's churn shift
                    _g = _guidance_impl(x_denoised, float(c_tau))
                    g_prev = None if _g is None else _g.reshape(tuple(x_l.shape))
                except Exception as e:                   # noqa: BLE001, guidance must not kill a run
                    logger.warning("guidance skipped at step %d (%s)", step_i, e)
                    g_prev = None
            if _algo == "svgd" and steer and spec.algorithm != "noise_search":
                try:
                    sig2 = max(float(t_hat) ** 2, 1e-8)
                    score_atoms = ((x_denoised - x_noisy) / sig2).detach().float().cpu().numpy()
                    if not bool(getattr(spec, "svgd_repulsion_only", True)):
                        try:
                            _gr = _guidance_raw(x_denoised) if _guidance_raw is not None else None
                            if _gr is not None:
                                _add = float(spec.guidance_scale) * _gr.reshape(score_atoms.shape)
                                score_atoms = score_atoms + _add
                                _guid_moved[0] += float(np.abs(_add).mean())
                                _guid_n[0] += 1
                        except Exception as e:      # noqa: BLE001
                            logger.warning("svgd reward tilt skipped at step %d (%s)", step_i, e)
                    # t runs 1 -> 0 over the schedule, the convention the anneal expects
                    phi = _svgd_transport(x_l, score_atoms,
                                          1.0 - (float(step_i) / max(n_steps - 1, 1)))
                    if phi is not None:
                        _st = phi if _site_path[0] else _svgd_opt.step(phi)
                        x_l = x_l + torch.as_tensor(_st, device=device, dtype=dtype
                                                    ).reshape(tuple(x_l.shape))
                        _svgd_moved[0] += float(np.abs(_st).mean())
                        _svgd_n[0] += 1
                except Exception as e:                   # noqa: BLE001, transport must not kill a run
                    logger.warning("svgd transport skipped at step %d (%s)", step_i, e)
            if not steer or spec.algorithm == "noise_search" or not _is_event(step_i):
                continue
            if _algo in ("svgd", "af3_red"):
                if _algo == "svgd":
                    logger.info("svgd step %d: %d updates, cumulative mean|dx|=%.4f A",
                                step_i, _svgd_n[0], _svgd_moved[0])
                ev += 1
                continue
            if not _reward_needed:
                # Nothing downstream reads it (see spec.reward_affects_sampling), and scoring costs a
                # mini-rollout per particle. Deterministic rollout => skipping cannot move a coordinate.
                ev += 1
                continue
            r_cur = reward_fn(_score_coords(x_l, x_denoised, step_i))         # reward on (mini-rollout of) x̂₀
            if spec.algorithm in ("fk", "tds", "cast"):
                if r_prev is None:
                    r_prev = r_cur
                    _r_first[0] = np.asarray(r_cur).copy()   # r_0, for the level correction below
                _spread = float(np.std(np.asarray(r_cur) - np.asarray(r_prev)))
                if spec.lam_adaptive and _lam_fixed[0] is None and _spread > 1e-9:
                    _lam_fixed[0] = min(float(spec.lam),
                                        gd.lambda_for_target_ess(r_prev, r_cur, spec.potential,
                                                                 len(r_cur), spec.target_ess_frac,
                                                                 logw_prev=np.zeros(len(r_cur))))
                    if _lam_fixed[0] <= 1e-9:
                        logger.warning("%s: calibrated lambda=0 -- the reward tilt is OFF; this "
                                       "arm is NOT the method it claims to be", _algo)
                    logger.info("%s step %d: calibrated lambda=%.4f ONCE (reward spread=%.3f), "
                                "held fixed for the rest of the trajectory", _algo, step_i,
                                _lam_fixed[0], _spread)
                # until calibration happens, apply NO tilt rather than a possibly-wrong one
                _lam = (0.0 if _lam_fixed[0] is None else _lam_fixed[0]) if spec.lam_adaptive \
                    else float(spec.lam)
                if spec.lam_adaptive and _lam_fixed[0] is None:
                    _lam_never[0] += 1
                    logger.warning("%s step %d: lam_adaptive is ON but calibration has NEVER run "
                                   "(reward spread=%.3g <= 1e-9 at every event so far), so lam=0 and "
                                   "this arm has NO reward tilt -- it is NOT the method it claims",
                                   _algo, step_i, _spread)
                if (bool(getattr(spec, "weight_level_correction", False))
                        and not _lvl_done[0] and _lam != 0.0 and _r_first[0] is not None):
                    logw = logw + _lam * _r_first[0]
                    _lvl_done[0] = True
                    logger.info("%s step %d: level correction applied, +lam*r_0 (lam=%.4f, "
                                "mean r_0=%.3f) -- accumulated weight now tracks r_T, not r_T-r_0",
                                _algo, step_i, _lam, float(np.mean(_r_first[0])))
                logw = logw + gd.fk_log_increment(spec.potential, r_prev, r_cur, _lam)
                if spec.physicality_filter:
                    bad = ~_physical_mask(x_denoised)
                    if bad.any():
                        logw = logw - _PHYS_PENALTY * bad
                        logger.info("%s step %d: %d/%d particles down-weighted as unphysical",
                                    _algo, step_i, int(bad.sum()), len(bad))
                if getattr(spec, "repulsion_beta", 0.0) > 0.0 and _PARATOPE_IDX[0] is not None:
                    try:
                        _reps_rep = bs.residue_reps_from_atoms(
                            x_denoised.detach().float().cpu().numpy(), _atom_to_token, n_token)
                        _cells_rep = (_cells_fn(len(logw)) if _cells_fn is not None
                                      else np.zeros(len(logw), dtype=int))
                        _skip_rep = (_RESERVE_FN[0](len(logw)) if _RESERVE_FN[0] is not None
                                     else None)
                        _pen = gd.crowding_log_penalty(_reps_rep, _PARATOPE_IDX[0], _cells_rep,
                                                       float(spec.repulsion_beta), skip=_skip_rep)
                        logw = logw - _pen
                        if step_i % 40 == 0:
                            logger.info("%s step %d: within-cell repulsion beta=%.3f "
                                        "penalty mean=%.3f max=%.3f", _algo, step_i,
                                        spec.repulsion_beta, float(_pen.mean()), float(_pen.max()))
                    except Exception as e:                  # noqa: BLE001 -- never break a run
                        logger.warning("within-cell repulsion skipped at step %d: %s", step_i, e)
                _t_rs = time.perf_counter()
                w = gd.fk_clamp_weights(gd.normalize_log_weights(logw))     # FK reference guard
                _cells_now = (_cells_fn(len(w)) if _cells_fn is not None
                              else gd.group_ids(len(w), len(_sweep))) if _sweep else None
                ess_now = gd.ess(w)
                stats["n_score_events"] += 1
                _ess_cells = (gd.ess_stratified(w, _cells_now)
                              if _cells_now is not None and len(np.unique(_cells_now)) > 1 else None)
                _per = len(w) / float(len(_ess_cells)) if _ess_cells is not None else float(len(w))
                if _ess_cells is not None:
                    stats["ess_cell_median"] = float(np.median(_ess_cells))
                do_resample = (spec.resample == "ess" and ess_now < spec.ess_fraction * len(w)) or \
                              (spec.resample == "every_k" and (ev % max(1, spec.resample_every_k) == 0))
                # OPT-IN trigger on the correct statistic. Off by default so this change cannot
                # silently reinterpret any arm already run against the global criterion.
                if bool(getattr(spec, "ess_per_cell_trigger", False)) and _ess_cells is not None:
                    do_resample = bool(np.median(_ess_cells) < spec.ess_fraction * _per)
                _win = gd.parse_resample_windows(getattr(spec, "resample_windows", ""))
                if _win is None:
                    _lo, _hi = spec.resample_start_frac, spec.resample_end_frac
                else:
                    _lo, _hi = _win
                if not gd.in_resample_window(step_i, n_steps, _lo, _hi):
                    do_resample = False
                if do_resample:
                    u = float(((step_i + 1) * 0.6180339887) % 1.0)          # deterministic systematic offset
                    # With a d* sweep, resample WITHIN each umbrella: a global resample would let the
                    # best-scoring umbrella take over and collapse the sweep back to one mode.
                    if _sweep:
                        cells = _cells_now
                        if not _cell_warned[0]:
                            _cell_warned[0] = True
                            _lv, _bm = bs.check_cell_budget(
                                len(w), len(_sweep), _n_sites,
                                reserve_frac=_ds.reserve_frac_of(spec),
                                n_subsamples=int(getattr(spec, "subsample_pool", 1) or 1))
                            if _lv:
                                logger.warning("steering cell budget (%s): %s", _lv, _bm)
                        _rf = _ds.realloc_frac_of(spec)
                        if ((_rf > 0.0 or _REALLOC_PROBE[0]) and _PARATOPE_IDX[0] is not None
                                and _epi_idx_for_realloc[0] is not None):
                            try:
                                _reps_ra = bs.residue_reps_from_atoms(
                                    x_denoised.detach().float().cpu().numpy(), _atom_to_token, n_token)
                                _sc = bs.contact_consistency(
                                    _reps_ra, _PARATOPE_IDX[0], _epi_idx_for_realloc[0], cells,
                                    r_contact=float(getattr(spec, "realloc_contact_r", 10.0)))
                                if _REALLOC_PROBE[0]:
                                    try:
                                        _cc = np.asarray(cells)
                                        with open(_REALLOC_PROBE[0], "a") as _pf:
                                            _pf.write(json.dumps({
                                                "target": str(_probe_name[0]), "step": int(step_i),
                                                "n_sites": int(_n_sites), "n_umb": len(_sweep),
                                                "cells": {str(int(c)): [int((_cc == int(c)).sum()),
                                                                        float(_sc[int(c)])]
                                                          for c in sorted(_sc)},
                                            }) + "\n")
                                    except Exception as _pe:      # noqa: BLE001 -- never break a run
                                        logger.warning("realloc probe write failed: %s", _pe)
                                if _sc and _rf > 0.0:
                                    _res_ra = (_RESERVE_FN[0](len(w))
                                               if _RESERVE_FN[0] is not None else None)
                                    _before = cells
                                    cells = bs.reallocate_cells(
                                        cells, _sc,
                                        min_per_cell=int(getattr(spec, "realloc_min_per_cell", 6)),
                                        reserve=_res_ra, max_move_frac=_rf)
                                    _n_moved = int((np.asarray(cells) != np.asarray(_before)).sum())
                                    stats["n_realloc_moved"] = stats.get("n_realloc_moved", 0) + _n_moved
                                    stats["n_realloc_events"] = stats.get("n_realloc_events", 0) + 1
                                    if step_i % 40 == 0:
                                        logger.info("%s step %d: site realloc moved %d slot(s), "
                                                    "cell consistency %.3f-%.3f over %d scored cell(s)",
                                                    _algo, step_i, _n_moved, min(_sc.values()),
                                                    max(_sc.values()), len(_sc))
                            except Exception as e:          # noqa: BLE001 -- never break a run
                                logger.warning("site realloc skipped at step %d: %s", step_i, e)
                        idx = gd.stratified_resample(w, cells, u)
                        stats["n_resample_events"] += 1
                    elif spec.resample_scheme == "multinomial":
                        idx = gd.multinomial_resample(w, u)   # the FK reference's torch.multinomial
                    else:
                        idx = gd.systematic_resample(w, u)    # TDS default (lower variance)
                    logger.info("%s step %d: ESS=%.2f/%d%s RESAMPLE%s ancestors=%s "
                                "reward(kept)=%.3f vs(all)=%.3f (steering ON)", _algo, step_i, ess_now, len(w),
                                _ess_cell_str(_ess_cells, len(w)),
                                (f" (stratified, {len(_sweep)} umbrellas"
                                 + (f" x {_n_sites} sites" if _n_sites > 1 else "") + ")")
                                if _sweep else "",
                                list(np.asarray(idx)), float(r_cur[idx].mean()), float(r_cur.mean()))
                    x_l = x_l[..., idx, :, :]
                    if _jpool["groups"] is not None:
                        _jpool["sub"] = gd.permute_groups(_jpool["sub"], idx)
                        _jpool["groups"] = gd.index_groups(_jpool["sub"])
                    logw = np.zeros(len(idx))
                    r_prev = r_cur[idx]     # reindex the Difference-potential baseline by ancestry (x_t^i=x_t^{a_i})
                    if g_prev is not None:  # ...and the guidance shift, for the SAME reason: after
                        g_prev = g_prev[..., idx, :, :]   # resampling particle i IS ancestor a_i, so it
                                            # must carry a_i's gradient, not the gradient of whatever
                else:
                    logger.info("%s step %d: ESS=%.2f/%d%s no-resample w=%s", _algo, step_i,
                                ess_now, len(w), _ess_cell_str(_ess_cells, len(w)),
                                np.round(w, 3).tolist())
                    r_prev = r_cur
                # covers weight normalisation, ESS, cell assignment, ancestor draw and the reindex --
                # the SMC machinery that the earlier instrumentation left unattributed
                _cuda_sync()              # weight/ESS/reindex tensor ops are GPU work
                stats["t_resample"] += time.perf_counter() - _t_rs
            ev += 1
        if _algo == "af3_red":
            try:
                _fin = _ca_token_coords(
                    x_l.detach().float().cpu().numpy().reshape(-1, N_atom, 3), ifd, len(_red_chain))
                if not os.path.exists(_red_mask_path):
                    try:
                        with torch.no_grad():
                            _pl = model.run_confidence_head(
                                input_feature_dict=ifd_net, s_inputs=s_inputs, s_trunk=s_trunk,
                                z_trunk=z_trunk, pair_mask=None,
                                x_pred_coords=x_l[..., :1, :, :],
                                triangle_multiplicative=getattr(
                                    model.configs, "triangle_multiplicative", "torch"),
                                triangle_attention=getattr(
                                    model.configs, "triangle_attention", "torch"),
                                inplace_safe=False, chunk_size=attn_chunk_size)[0]
                        if _pl is None:                      # never observed in the reference source
                            raise RuntimeError(
                                "confidence head returned plddt=None (head=%s, out[0] is None)"
                                % type(getattr(model, "confidence_head", None)).__name__)
                        _plddt = _token_plddt(_pl, ifd, len(_red_chain), model)
                        np.save(_red_mask_path, _plddt > float(spec.red_plddt_min))
                        logger.info("af3_red: pLDDT mask built from this run (%d/%d residues kept "
                                    "at pLDDT>%.0f)", int((_plddt > spec.red_plddt_min).sum()),
                                    len(_plddt), spec.red_plddt_min)
                    except Exception as e:               # noqa: BLE001
                        logger.warning("af3_red: pLDDT mask unavailable (%s), filter not applied", e)
                _all = _fin if _red_hist is None or not len(_red_hist) else np.concatenate([_red_hist, _fin])
                np.save(_red_path, _all)
                logger.info("af3_red: history %d -> %d structures; bias applied at %d steps "
                            "(cumulative mean|dx|=%.4f A)", _red_n[0], len(_all), _svgd_n[0],
                            _svgd_moved[0])
            except Exception as e:                       # noqa: BLE001, never break a run
                logger.warning("af3_red: could not persist history (%s)", e)
        if _guidance_impl is not None or _guidance_raw is not None:   # TDS path OR SVGD tilt path
            if _guid_n[0]:
                _u = ("score-term, NOT a displacement; see the svgd step line for real transport"
                      if _guidance_raw is not None else "displacement in Angstrom")
                # _guid_moved is a device tensor on the TDS path (accumulated without a per-step
                # sync) and a plain float on the SVGD path. Resolve once, here, at the only read.
                _gm = _guid_moved[0]
                _gm = float(_gm.item()) if hasattr(_gm, "item") else float(_gm)
                logger.info("reward gradient APPLIED at %d steps (cumulative mean|d|=%.4f [%s])",
                            _guid_n[0], _gm, _u)
            else:
                logger.warning("reward gradient NEVER APPLIED (guidance_scale=%s): this arm is "
                               "reweighting only -- it is NOT the gradient method it is labelled",
                               spec.guidance_scale)
        if _ds.realloc_frac_of(spec) > 0.0 and not stats.get("n_realloc_moved"):
            logger.error("site reallocation ENABLED but moved 0 slots across %d event(s): this arm "
                         "is plain uniform multi-site steering under a reallocating label",
                         stats.get("n_realloc_events", 0))
        elif stats.get("n_realloc_moved"):
            logger.info("site reallocation: %d slot(s) moved over %d event(s)",
                        stats["n_realloc_moved"], stats.get("n_realloc_events", 0))
        if _svgd_ag is not None and float(getattr(spec, "site_archive_weight", 0.0)) > 0:
            try:
                _pi, _a2t, _ntok = _svgd_par
                _c = x_l.detach().float().cpu().numpy().reshape(-1, x_l.shape[-2], 3)
                _reps_a = bs.residue_reps_from_atoms(_c, _a2t, _ntok)
                _plc = _reps_a[:, _pi, :] - _reps_a[:, _svgd_ag, :].mean(1, keepdims=True)
                _site_div_archive[0] = (_plc if _site_div_archive[0] is None
                                        else np.concatenate([_site_div_archive[0], _plc], 0))
                _save_site_archive(_site_name, _site_div_archive[0])
                logger.info("site archive: +%d placements (total %d)",
                            _plc.shape[0], _site_div_archive[0].shape[0])
            except Exception as e:      # noqa: BLE001 -- archiving must not kill a finished run
                logger.warning("site archive update failed (%s)", e)
        return x_l

    if spec.algorithm == "noise_search":
        # Outer ES over the initial standardized noise z, each generation is one batched
        # sampling run of `popsize` particles (no in-loop steering). Keep the best N_sample.
        rng = np.random.default_rng(0)
        d = int(np.prod(batch_shape)) * N_atom * 3 if batch_shape else N_atom * 3
        _algo_ns = str(getattr(spec, "noise_search_algo", "es") or "es").lower()
        mean = np.zeros(d, dtype=np.float64)
        sigma = float(spec.sigma0)
        best_x = None

        def _evaluate(Z):
            """Decode a [n, d] latent batch into structures and score it. One full sampling run each."""
            n = int(Z.shape[0])
            zt = torch.tensor(Z, device=device, dtype=dtype).reshape(*batch_shape, n, N_atom, 3)
            xf = _run_population(n, init_x=noise_schedule[0] * zt, steer=False, deterministic=True)
            return xf, np.asarray(reward_fn(xf), dtype=np.float64)

        # POPULATION INITIALISATION is shared by both variants: isotropic draws on the noise shell.
        pop = ds.es_ask(mean, sigma, int(spec.popsize), rng, on_shell=spec.on_shell)
        pop_x, pop_f = _evaluate(pop)
        best_x = [(pop_x[..., i:i + 1, :, :], float(pop_f[i])) for i in range(int(spec.popsize))]
        best_x = sorted(best_x, key=lambda t: t[1], reverse=True)[:N_sample]
        _nfe = int(spec.popsize)

        for g in range(int(spec.generations)):
            if _algo_ns == "cosyne":
                P = int(spec.popsize)
                n_elite = int(P * float(getattr(spec, "elitism_ratio", 0.0) or 0.0))
                n_rel = max(n_elite, P // 4)                       # "top 25% or the elites"
                n_kids = int(getattr(spec, "num_children", 0) or 0) or 2 * P   # rate 2.0
                n_kids -= n_kids % 2                               # cross-over works on pairs
                top = np.asarray(pop)[np.argsort(-pop_f, kind="stable")[:max(2, n_rel)]]
                parents = ds.cosyne_tournament(top, pop_f[np.argsort(-pop_f, kind="stable")[:max(2, n_rel)]],
                                               int(spec.tournament_size), n_kids, rng)
                _eta = float(getattr(spec, "cosyne_eta", 0.0) or 0.0)
                kids = ds.sbx_crossover(parents, _eta, rng) if _eta > 0 else ds.one_point_crossover(parents, rng)
                _sd = float(getattr(spec, "mutation_stdev", 0.0) or 0.0)
                if _sd > 0:
                    _mp = float(getattr(spec, "mutation_probability", 0.0) or 0.0)
                    kids = ds.gaussian_mutation(kids, _sd, rng, probability=(_mp if _mp > 0 else None))
                perm = ds.cosyne_permute(pop, pop_f, rng, permute_all=bool(spec.permute_all))
                if spec.on_shell:      # the DNO guard applies to every candidate that is decoded
                    kids = np.stack([ds.project_to_shell(z) for z in kids])
                    perm = np.stack([ds.project_to_shell(z) for z in perm])
                cand = np.concatenate([kids, perm], axis=0)
                cand_x, cand_f = _evaluate(cand)
                _nfe += int(cand.shape[0])
                # extended population: already-evaluated elites carry over, no re-decode
                ext_Z = np.concatenate([pop[np.argsort(-pop_f, kind="stable")[:n_elite]], cand], axis=0) \
                    if n_elite else cand
                ext_f = np.concatenate([pop_f[np.argsort(-pop_f, kind="stable")[:n_elite]], cand_f]) \
                    if n_elite else cand_f
                keep = np.argsort(-ext_f, kind="stable")[:P]
                pop, pop_f = ext_Z[keep], ext_f[keep]
                xr = [(cand_x[..., i:i + 1, :, :], float(cand_f[i])) for i in range(int(cand.shape[0]))]
                r = cand_f
            else:
                Z = ds.es_ask(mean, sigma, int(spec.popsize), rng, on_shell=spec.on_shell)
                x_final, r = _evaluate(Z)
                _nfe += int(spec.popsize)
                mean, sigma = ds.es_tell(Z, r, sigma, elite_frac=spec.elite_frac, sigma_decay=spec.sigma_decay)
                pop, pop_f = Z, r
                xr = [(x_final[..., i:i + 1, :, :], float(r[i])) for i in range(int(spec.popsize))]
            merged = sorted(best_x + xr, key=lambda t: t[1], reverse=True)[:N_sample]
            best_x = merged
            logger.info("noise_search[%s] gen %d/%d: best %s=%.4f (sigma=%.3f, NFE=%d)",
                        _algo_ns, g + 1, spec.generations, spec.reward, merged[0][1], sigma, _nfe)
        result = torch.cat([x for x, _ in best_x], dim=-3)
    else:
        x_l = _run_population(N_sample, steer=True)
        result = x_l

    trunk_wall = float(getattr(model, "_tts_trunk_wall", 0.0))   # timed in the get_pairformer_output wrapper
    model._tts_trunk_wall = 0.0                                  # reset for the next target
    # patch_confidence_timing): the tracker reports ~0 for work the FLOPs budget puts at 2.5x trunk.
    stats["t_confidence_output"] = float(getattr(model, "_tts_conf_wall", 0.0))
    stats["n_confidence_output_calls"] = int(getattr(model, "_tts_conf_calls", 0))
    # NON-ZERO MEANS THE ARM HAD NO REWARD TILT. Recorded as a number, not only a log line, so a
    # pool health gate can refuse the run instead of a human having to notice a warning.
    stats["n_events_uncalibrated_lam"] = int(_lam_never[0])
    model._tts_conf_wall = 0.0
    model._tts_conf_calls = 0
    _record_diffusion_timing(spec, kw, stats, n_token, time.perf_counter() - _t_all, trunk_wall,
                             n_emitted=int(result.shape[-3]))
    _record_cell_map(spec, kw, _sweep, _n_sites, int(result.shape[-3]),
                     sub_ids=_jpool["sub"], sub_ids0=_jpool["sub0"], coords=result)
    return result


def install_diffusion() -> None:
    """Wrap ``Protenix.sample_diffusion`` so a coupled search replaces the independent draws."""
    try:
        Protenix = _cofold_model_class()
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Could not locate protenix.model.protenix.Protenix for the diffusion hook: {e!r}")

    patch_trunk_timing()

    orig_gpo = Protenix.get_pairformer_output

    def _gpo_capture(self, *a, **k):
        out = orig_gpo(self, *a, **k)
        try:
            self._tts_diff_z = out[2]                     # (s_inputs, s, z) → z = pair trunk rep
        except (TypeError, IndexError):
            pass
        return out

    Protenix.get_pairformer_output = _gpo_capture

    orig_sd = Protenix.sample_diffusion

    def _wrapped_sd(self, **kwargs):
        if _ACTIVE_DIFF is None:
            _n = int(kwargs.get("N_sample", 0) or 0)
            _steps = len(kwargs.get("noise_schedule", [])) - 1
            out = orig_sd(self, **kwargs)
            if _n and _steps > 0:
                _cfg_steps = int(getattr(self.configs.sample_diffusion, "N_step", _steps))
                logger.info("BoN nfe cross-check: %d particles x %d steps = %d denoiser passes "
                            "(analytic assumes N_step=%d)", _n, _steps, _n * _steps, _cfg_steps)
                if _steps != _cfg_steps:
                    logger.warning("BoN nfe cross-check: schedule gives %d steps but N_step=%d -- the "
                                   "analytic FLOPs assumption does not hold", _steps, _cfg_steps)
            return out
        return _search_sample_diffusion(self, _ACTIVE_DIFF, kwargs)

    Protenix.sample_diffusion = _wrapped_sd
    # CODE STAMP. Twice today a sync/submit race nearly produced results from stale code with no way
    # to tell from the output. Hash the method sources so every run log self-certifies its code.
    try:
        import hashlib
        _h = hashlib.sha256()
        for _f in ("models/protenix_hooks.py", "method/diffusion_side/base_search.py",
                   "method/diffusion_side/guidance.py", "method/diffusion_side/diffusion_search.py"):
            _p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _f)
            if os.path.exists(_p):
                _h.update(open(_p, "rb").read())
        logger.info("CODE STAMP diffusion_side=%s", _h.hexdigest()[:16])
    except Exception:                                    # noqa: BLE001, never break a run
        logger.warning("CODE STAMP unavailable")
    # Name the ACTUAL class (see the stage-timing note): under TTS_COFOLD_PKG=opendde this is
    # OpenDDE.sample_diffusion, and logging "Protenix" there would hide a mis-targeted attach.
    logger.info("diffusion-search hook installed on %s.sample_diffusion (%s)",
                Protenix.__name__, _ACTIVE_DIFF.label() if _ACTIVE_DIFF else "inactive")


# ---------------------------------------------------------------- OOM -> skip target
def apply_standard_hooks() -> None:
    """The full instrumentation sequence for the active backend. Every launcher calls THIS, not a list."""
    patch_sample_name_capture()      # keys every per-target write below
    patch_dumper_pae_npz()           # PAE -> compact .npz (uniform dump format across backends)
    patch_confidence_offload()       # opt-in: PAE -> CPU for big targets
    patch_trunk_timing()             # synchronised trunk wall, ALL runs (must precede stage timing)
    patch_stage_timing()             # per-sample stage_wall.json
    patch_stage_memory()             # per-stage GPU peak -> stage_mem.json (ALWAYS ON)
    patch_confidence_timing()        # the vendor tracker reports ~0 for the confidence head
    patch_chunked_confidence()       # confidence in sample sub-batches
    patch_diffusion_chunk_auto()     # opt-in: per-target diffusion chunk
    patch_oom_skip()                 # opt-in: GPU-OOM -> log + skip remaining seeds
    patch_flop_profile()             # opt-in: measured FLOPs
    patch_mem_trace()                # opt-in: host-RAM trace
    # VENDOR QUIRKS. Each self-gates on the active package and no-ops elsewhere, so this list is the
    # place a future backend's own workaround goes -- it does not fork the sequence above.
    patch_opendde_trunk_chunk()
    patch_opendde_sample_order()
    patch_opendde_sc_pair_chunk()
    if enable_from_env():
        install()                    # trunk latent modulation (z_pre)
    if enable_diffusion_from_env():
        install_diffusion()          # the diffusion-side steering hook
        logger.info("steering hook installed (pkg=%s)", cofold_pkg())
    else:
        logger.info("no %s -- running UNSTEERED (pkg=%s)", ENV_DIFF_SPEC, cofold_pkg())


def patch_opendde_trunk_chunk() -> None:
    """Expose OpenDDE's trunk chunk as a measurement knob, and record which value was used."""
    if cofold_pkg() != "opendde":
        return
    try:
        OpenDDE = _cofold_model_class()
    except Exception as e:                                  # noqa: BLE001
        logger.warning("opendde trunk-chunk hook: model class unavailable (%s); skipping", e)
        return
    if getattr(OpenDDE, "_tts_trunk_chunk_patched", False):
        return
    orig = getattr(OpenDDE, "_get_dynamic_chunk_size", None)
    if orig is None:
        logger.warning("opendde trunk-chunk hook: _get_dynamic_chunk_size absent; skipping")
        return

    forced = os.environ.get("TTS_ODDE_TRUNK_CHUNK")

    def _chunk(self, n_token: int):
        try:
            if forced is not None:
                v = None if str(forced).lower() in ("0", "none", "") else int(forced)
                logger.info("opendde trunk chunk: N_token=%d -> %s (FORCED via TTS_ODDE_TRUNK_CHUNK)",
                            n_token, v)
                self._tts_trunk_chunk_used = v              # picked up by patch_stage_memory
                return v
            v = orig(self, n_token)                        # vendor table, see the docstring
            self._tts_trunk_chunk_used = v                 # picked up by patch_stage_memory
            return v
        except Exception as e:                              # noqa: BLE001, never break a run
            logger.warning("opendde trunk-chunk hook failed (%s); using vendor value", e)
            return orig(self, n_token)

    OpenDDE._get_dynamic_chunk_size = _chunk
    OpenDDE._tts_trunk_chunk_patched = True
    logger.info("opendde trunk-chunk hook installed on %s._get_dynamic_chunk_size", OpenDDE.__name__)


def patch_opendde_sc_pair_chunk() -> None:
    """Shrink the shape-complementarity token chunk so a large joint pool fits on the card.

    ``compute_shape_complementarity_fields`` materialises ``[n_sample, pair_chunk, n_atom, 3]`` in
    ``_density_gradient_chunk``. The vendor default of 128 is sized for a handful of samples. A
    K=16 joint pool of 2880 particles on a 1171-token target asks for 37.33 GiB on an 80 GiB card,
    and it OOMs AFTER the 12 h diffusion has already finished, so the whole unit is lost. Chunking
    is EXACT, the same sum over the same atoms taken in smaller token blocks, so this moves cost
    and never the result. Opt-in, so no run that already fits is touched.
    """
    if cofold_pkg() != "opendde":
        return
    forced = os.environ.get("TTS_ODDE_SC_PAIR_CHUNK")
    if not forced:
        return
    try:
        k = int(forced)
    except ValueError:
        logger.warning("TTS_ODDE_SC_PAIR_CHUNK=%r is not an integer; ignoring", forced)
        return
    if k <= 0:
        return
    try:
        from opendde.model import opendde as _odm
    except Exception as e:                                  # noqa: BLE001
        logger.warning("opendde shape-comp hook: module unavailable (%s); skipping", e)
        return
    if getattr(_odm, "_tts_sc_pair_chunk_patched", False):
        return
    # PATCH THE IMPORTED NAME, not the source module: opendde/model/opendde.py does
    # `from .shape_complementarity import compute_shape_complementarity_fields` at import time,
    # so rebinding the source module would leave the live reference untouched.
    orig = getattr(_odm, "compute_shape_complementarity_fields", None)
    if orig is None:
        logger.warning("opendde shape-comp hook: compute_shape_complementarity_fields absent; skipping")
        return

    def _sc(*a, **kw):
        kw["pair_chunk_size"] = k
        return orig(*a, **kw)

    _odm.compute_shape_complementarity_fields = _sc
    _odm._tts_sc_pair_chunk_patched = True
    logger.info("opendde shape-comp hook: pair_chunk_size forced to %d", k)


def patch_opendde_sample_order() -> None:
    """Make OpenDDE's ``sample_{k}`` the k-th DRAW, as it is on Protenix, not the k-th best."""
    if cofold_pkg() != "opendde":
        return
    if os.environ.get("TTS_ODDE_RANK_SORT"):
        logger.info("opendde sample order: vendor rank sort kept (TTS_ODDE_RANK_SORT set)")
        return
    try:
        from runner.dumper import DataDumper
    except Exception as e:                                  # noqa: BLE001
        logger.warning("opendde sample-order hook: DataDumper unavailable (%s); skipping", e)
        return
    if getattr(DataDumper, "_tts_sample_order_patched", False):
        return
    orig_init = DataDumper.__init__

    def _init(self, *a, **kw):
        orig_init(self, *a, **kw)
        self.sorted_by_ranking_score = False               # sample_k is the k-th draw

    DataDumper.__init__ = _init
    DataDumper._tts_sample_order_patched = True
    logger.info("opendde sample order: sample_k is the k-th DRAW (matches Protenix), "
                "not the k-th best by ranking score")


def patch_oom_skip() -> None:
    """When ``$PROTENIX_OOM_SKIP`` is set, a target that hits a GPU OOM is LOGGED and its remaining
        seeds are SKIPPED (fast-fail), so the run moves on to the next target instead of re-attempting
        every seed of a target that cannot fit.
    """
    if not os.environ.get("PROTENIX_OOM_SKIP"):
        return
    try:
        import torch
        Protenix = _cofold_model_class()
    except (ImportError, AttributeError) as e:
        logger.warning("oom-skip: attach point not found (%s); skipping", e)
        return
    if getattr(Protenix, "_tts_oom_skip_patched", False) or not hasattr(Protenix, "main_inference_loop"):
        return
    orig_loop = Protenix.main_inference_loop
    oomed: set = set()

    def _is_oom(e: BaseException) -> bool:
        return isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())) or "out of memory" in str(e).lower()

    def _ctx(self, ifd) -> dict:
        """The memory decision in effect for this target: N_token, diffusion chunk, offload y/n."""
        c = {"n_token": None, "chunk": None, "offload": None, "n_samples": None}
        try:
            c["n_token"] = int(ifd["residue_index"].shape[-1])
        except Exception:                       # noqa: BLE001
            pass
        try:
            c["chunk"] = int(getattr(self.configs.infer_setting, "sample_diffusion_chunk_size", None))
        except Exception:                       # noqa: BLE001
            pass
        try:
            c["n_samples"] = int(self.configs.sample_diffusion["N_sample"])
        except Exception:                       # noqa: BLE001
            pass
        thr = os.environ.get("PROTENIX_CONF_OFFLOAD_TOKENS")   # resolved to a number by run()'s _resolve_offload_tokens
        try:
            c["offload"] = (c["n_token"] is not None and int(thr) < 10**8 and c["n_token"] > int(thr))
        except (TypeError, ValueError):
            pass
        return c

    def _record(self, tid, ctx, err) -> None:
        """Append a structured OOM event (target + chunk + offload) to <dump_dir>/oom_events.jsonl."""
        try:
            dd = _dump_dir(self)
            os.makedirs(dd, exist_ok=True)
            rec = {"target": tid, "n_token": ctx["n_token"], "diffusion_chunk": ctx["chunk"],
                   "offload": ctx["offload"], "n_samples": ctx["n_samples"], "error": str(err)[:200]}
            with open(os.path.join(dd, "oom_events.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:                  # noqa: BLE001, recording must never break the run
            logger.warning("oom-skip: could not write oom_events.jsonl (%s)", e)

    def _loop(self, *a, **k):
        ifd = k.get("input_feature_dict") or (a[0] if a else None)
        name = str(_extract_sample_name(ifd) or "")
        tid = name.split("__")[0] or name or "unknown"   # base target id (drop the __random@... seed suffix)
        if tid in oomed:                        # a prior seed of this target already OOM'd -> skip fast
            raise RuntimeError(f"OOM-skip: {tid} previously OOM'd on this GPU; skipping its remaining seeds")
        try:
            return orig_loop(self, *a, **k)
        except Exception as e:                  # noqa: BLE001
            if _is_oom(e):
                oomed.add(tid)
                ctx = _ctx(self, ifd)
                logger.error("OOM-skip: %s hit a GPU OOM at N_token=%s chunk=%s offload=%s n_samples=%s; "
                             "logging + skipping its remaining seeds", tid, ctx["n_token"], ctx["chunk"],
                             ctx["offload"], ctx["n_samples"])
                _record(self, tid, ctx, e)
                try:
                    torch.cuda.empty_cache()
                except Exception:               # noqa: BLE001
                    pass
            raise                               # re-raise so the runner's per-sample except logs + continues

    Protenix.main_inference_loop = _loop
    Protenix._tts_oom_skip_patched = True
    logger.info("oom-skip: GPU-OOM targets will be logged and their remaining seeds skipped")
