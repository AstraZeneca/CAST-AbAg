r"""OpenDDE backend, Aureka's open-source all-atom co-folding model.

Install::

    source      https://github.com/aurekaresearch/OpenDDE          (Apache-2.0)
    package     https://pypi.org/project/opendde/                  pinned to 1.1.0
    weights     https://huggingface.co/aurekaresearch/OpenDDE      opendde_abag.pt
    paper       https://arxiv.org/abs/2607.03787

    python3 -m venv $OPENDDE_ROOT_DIR/.venv
    $OPENDDE_ROOT_DIR/.venv/bin/pip install \
        --extra-index-url https://download.pytorch.org/whl/cu126 "opendde[gpu]==1.1.0"

OpenDDE needs its OWN venv, since it pins numpy 2.4 and torch 2.7.1+cu126 alongside the
cuEquivariance kernels, so this backend shells out to that interpreter instead of importing the
package. The checkpoint and the CCD go under ``$OPENDDE_ROOT_DIR`` as ``checkpoint/`` and
``common/``, the same shape Protenix uses, and the upstream default is ``~/.cache/opendde``.
PIN THE VERSION: upstream calls this a preview release whose CLI flags, JSON fields and checkpoints
may change, which moves the attach points the hooks depend on.

Shared with the Protenix backend: the input JSON schema (``sequences`` / ``proteinChain`` /
``sequence`` / ``count`` / ``id`` / ``pairedMsaPath`` / ``unpairedMsaPath``), the dump layout
(``.../seed_{seed}/predictions/``) and the output filenames (``{name}_sample_{k}.cif``,
``{name}_summary_confidence_sample_{k}.json``). The schema-facing methods delegate to
:class:`~models.protenix2.ProtenixBackend` static helpers, so wired inputs and an MSA database are
reusable across the two backends unchanged. ``run`` is written from scratch against the OpenDDE
CLI, because Protenix's checkpoint provisioning, ``PROTENIX_*`` wiring and memory constants do not
transfer; memory is predicted by scaling the Protenix curve (``models.base.peak_gib_for``).

CLI surface, read from the shipped ``runner/batch_inference.py``::

    -i/--input  -o/--out_dir  -s/--seeds (comma-separated)  -c/--cycle  -p/--step  -e/--sample
    -n/--model_name  --load_checkpoint_path  --use_msa  --use_template  --need_atom_confidence
    --trimul_kernel  --triatt_kernel  --enable_tf32  --enable_fusion  --enable_cache --deterministic
    --dtype  --device  --foldcp_mode {single,distributed}  --foldcp_size_dp  --foldcp_size_cp

Flag names this backend translates from Protenix:
``--sample_diffusion.N_sample`` -> ``--sample``, ``--sample_diffusion.N_step`` -> ``--step``,
``--model.N_cycle`` -> ``--cycle``, ``--input_json_path`` -> ``-i``, ``--dump_dir`` -> ``-o``,
``--triangle_multiplicative``/``--triangle_attention`` -> ``--trimul_kernel``/``--triatt_kernel``,
``--enable_efficient_fusion`` -> ``--enable_fusion``.

Four things a caller must know:

(1) TOKEN SPACE. OpenDDE runs its diffusion in an expanded "structural sub-token" space, roughly
    1.9x the residue count, while its pair and confidence stack stay in residue space. Every token
    map in this project is residue space (see experiments/scripts/token_maps.py), so anything
    reading indices, chain ids or atom-to-token bins out of the diffusion's feature dict must go
    through ``models.protenix_hooks._map_space_ifd`` first. A bounds check cannot catch a miss,
    because a residue-space index is always a valid sub-token index and simply addresses the wrong
    residue. Unsteered runs never touch a map.
(2) ``need_atom_confidence`` DEFAULTS TO False, and ``_save_confidence`` writes
    ``*_full_data_sample_N.json`` (which carries ``token_pair_pae``) only when it is true. Any
    config that scores a PAE-derived verifier (ipSAE, pDockQ2, LIS, iLIS) must set it true.
(3) NEVER force ``trimul_kernel``/``triatt_kernel`` to ``torch``. They default to ``auto``, which
    resolves to cuEquivariance; the ``torch`` fallback materialises the whole triangle-attention
    tensor and OOMs where ``auto`` fits. ``doctor`` reports which kernel ``auto`` resolved to.
(4) SAMPLE INDEX IS A RANK, NOT A DRAW ORDER. ``DataDumper.sorted_by_ranking_score`` defaults True
    and the CLI exposes no flag to disable it, so ``sample_{k}`` is the k-th best sample by
    ranking score, unlike Protenix where it is the k-th draw. Order-invariant statistics are
    unaffected; anything pairing a ``sample_idx`` across backends, or taking a first-m prefix, is
    not comparable.

Dump layout note: ``DataDumper._get_dump_dir`` builds
``{base}/{group_name}/{name}/seed_{seed}/predictions``, but the only call site passes
``group_name=""``, so it collapses to the Protenix layout. :meth:`iter_predictions` globs
recursively, which is defensive against a future non-empty group.

The backend is chosen by the run config's backend and checkpoint, never by how inputs were
prepared: ``prepare_dataset`` and ``generate_msa --wire-out`` produce byte-identical inputs under
either backend name.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from models import register_backend
from models.base import BackendRunSpec, PredictionSample, Target, stamp_dump_dir
from models.protenix2 import ProtenixBackend, should_use_bf16, _detect_gpu_gib
from models import protenix_hooks          # for ENV_DIFF_SPEC only; the hooks attach in the LAUNCHER

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "opendde_v1"
ABAG_CHECKPOINT = "opendde_abag.pt"
GENERAL_CHECKPOINT = "opendde.pt"





_ODDE_CONF_BASE_GIB = 0.54          # sample- and token-independent working set
_ODDE_CONF_PER_TOKEN_GIB = 0.01715  # grows with N alone (trunk/pair carried into the stage)
_ODDE_CONF_PER_SAMPLE_TOKEN2 = 6.609e-07   # the retained [S, N, N, bins] stack
# The share of the peak that CPU-offload actually moves: PAE+PDE only, which is what the offload
# hook relocates (models/protenix_hooks.py swaps those two tensors to host). Everything else stays.
_ODDE_CONF_OFFLOADABLE_TOKEN2 = 2 * 64 * 4 / 2 ** 30    # 4.768e-7 GiB per sample per token^2


def opendde_confidence_peak_gib(n_token: int, n_samples: int) -> float:
    """Predicted peak GPU memory of OpenDDE's confidence stage, in GiB, with NO offload."""
    n, s = float(n_token), float(n_samples)
    return (_ODDE_CONF_BASE_GIB + _ODDE_CONF_PER_TOKEN_GIB * n
            + _ODDE_CONF_PER_SAMPLE_TOKEN2 * n * n * s)


def opendde_offload_host_gib(n_token: int, n_samples: int) -> float:
    """Host RAM the PAE/PDE stack needs once offloaded. Offload only helps if the HOST can hold it."""
    return _ODDE_CONF_OFFLOADABLE_TOKEN2 * float(n_token) ** 2 * float(n_samples)


def opendde_max_ongpu_tokens(gpu_gib: float, n_samples: int, margin_gib: float = 8.0) -> int:
    """Largest token count whose confidence stage fits on the card unoffloaded, at this sample count."""
    budget = float(gpu_gib) - float(margin_gib) - _ODDE_CONF_BASE_GIB
    if budget <= 0:
        return 0
    a = _ODDE_CONF_PER_SAMPLE_TOKEN2 * float(n_samples)
    b = _ODDE_CONF_PER_TOKEN_GIB
    disc = b * b + 4.0 * a * budget
    if a <= 0 or disc < 0:
        return 0
    return max(int((-b + disc ** 0.5) / (2.0 * a)), 0)


def opendde_host_ram_gib():
    """This pod's host-RAM ceiling in GiB from cgroup v2, or None when unset/unreadable."""
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(p) as f:
                raw = f.read().strip()
        except OSError:
            continue
        if raw in ("", "max"):
            return None
        try:
            v = int(raw)
        except ValueError:
            continue
        if v <= 0 or v > (1 << 50):        # an unset limit shows as an absurd sentinel
            return None
        return v / 2 ** 30
    return None


def opendde_should_offload(n_token: int, n_samples: int, gpu_gib: float = 80.0,
                           margin_gib: float = 8.0, host_gib=None) -> bool:
    """Confidence CPU-offload decision for ONE OpenDDE target. Protenix's rails are not consulted."""
    peak = opendde_confidence_peak_gib(n_token, n_samples)
    if peak <= float(gpu_gib) - float(margin_gib):
        return False                                  # fits on the card, keep it there (no PCIe cost)
    if host_gib is not None and opendde_offload_host_gib(n_token, n_samples) > float(host_gib):
        logger.warning(
            "opendde %d tok x %d samples: confidence peak %.1f GiB exceeds the card, and the "
            "offloaded PAE/PDE stack needs %.1f GiB of host against %.1f GiB available -- NOT "
            "offloading, because a pod OOM-kill is worse than a contained GPU OOM. The on-GPU "
            "ceiling here is %d tokens at %d samples; set TTS_ODDE_CONF_CHUNK=1 to chunk the "
            "confidence head (keeps the budget matched) or use a host with more RAM.",
            n_token, n_samples, peak, opendde_offload_host_gib(n_token, n_samples), host_gib,
            opendde_max_ongpu_tokens(gpu_gib, n_samples), n_samples)
        return False
    return True


def opendde_resolve_offload_tokens(tokens, n_samples, gpu_gib=None):
    """Resolve $PROTENIX_CONF_OFFLOAD_TOKENS for an OpenDDE run into a concrete token threshold."""
    raw = os.environ.get("PROTENIX_CONF_OFFLOAD_TOKENS")
    if raw is None:
        return None
    if raw.strip().lower() != "auto":
        return raw
    gib = float(gpu_gib) if gpu_gib else _detect_gpu_gib()
    host = opendde_host_ram_gib()
    toks = [int(t) for t in (tokens or [])] or [1200]
    ns = int(n_samples)
    needing = [t for t in toks if opendde_should_offload(t, ns, gpu_gib=gib, host_gib=host)]
    thr = (min(needing) - 1) if needing else 10 ** 9   # hook offloads on N_token > threshold (strict)
    logger.info("opendde auto confidence-offload: %d/%d target(s) offload (n_samples=%d, gpu=%.0f "
                "GiB, host=%s GiB, ceiling=%d tok) -> threshold=%d",
                len(needing), len(toks), ns, gib,
                f"{host:.0f}" if host else "?", opendde_max_ongpu_tokens(gib, ns), thr)
    return str(thr)


class OpenDDEBackend:
    """OpenDDE co-folding backend. See the module docstring for what is and is not shared."""

    name = "opendde"
    input_suffix = ".json"          # same schema as Protenix (it is a fork)

    _as_jobs = staticmethod(ProtenixBackend._as_jobs)
    _iter_protein_chains = staticmethod(ProtenixBackend._iter_protein_chains)
    _target_to_job = staticmethod(ProtenixBackend._target_to_job)
    _parse_confidence = staticmethod(ProtenixBackend._parse_confidence)
    _merge_inputs = staticmethod(ProtenixBackend._merge_inputs)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        c = dict(config or {})
        self._config = c
        self.model_name: str = c.get("model_name", DEFAULT_MODEL_NAME)
        # OPENDDE_ROOT_DIR holds the downloaded checkpoints and runtime data (their default is
        # ~/.cache/opendde, which is NOT durable in a pod -- point it at the PVC).
        self.root_dir: Optional[str] = c.get("root_dir") or os.environ.get("OPENDDE_ROOT_DIR")
        # abag=True selects the antibody-antigen-tuned checkpoint. It is the whole reason this
        # backend is here, so it defaults ON; set abag=false for the general model.
        self.abag: bool = bool(c.get("abag", True))
        self.checkpoint_path: Optional[str] = c.get("checkpoint_path")
        self.executable: str = c.get("executable", "opendde")
        self.python: str = c.get("python", "python")
        self.need_atom_confidence: bool = bool(c.get("need_atom_confidence", False))
        self.enable_tf32: bool = bool(c.get("enable_tf32", True))
        self.enable_fusion: bool = bool(c.get("enable_fusion", True))
        self.enable_cache: bool = bool(c.get("enable_cache", True))
        self.deterministic: bool = bool(c.get("deterministic", False))
        self.trimul_kernel: str = c.get("trimul_kernel", "auto")
        self.triatt_kernel: str = c.get("triatt_kernel", "auto")
        for _which, _k in (("trimul", self.trimul_kernel), ("triatt", self.triatt_kernel)):
            if str(_k).lower() == "torch":
                logger.warning(
                    "opendde: %s_kernel=torch materialises the full N^3 triangle tensor and OOMs "
                    "beyond ~700 tokens even on 80GB. Use 'auto' unless cuEquivariance is broken.",
                    _which)
        self.dtype: Optional[str] = c.get("dtype")
        self.device: Optional[str] = c.get("device")
        self.extra_args: List[str] = list(c.get("extra_args", []))

    def emit_input(self, target: Target, out_dir: Path) -> Path:
        return ProtenixBackend.emit_input(self, target, out_dir)  # type: ignore[arg-type]

    @staticmethod
    def get_msa_sources(input_path: Path) -> Dict[str, Optional[str]]:
        return ProtenixBackend.get_msa_sources(input_path)

    @staticmethod
    def get_chain_sequences(input_path: Path) -> Dict[str, str]:
        return ProtenixBackend.get_chain_sequences(input_path)

    def set_msa_sources(self, input_path: Path, chain_to_a3m: Mapping[str, str], out_path: Path) -> Path:
        return ProtenixBackend.set_msa_sources(self, input_path, chain_to_a3m, out_path)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ checkpoint
    def _resolve_checkpoint(self) -> Optional[str]:
        """Explicit path wins; otherwise the abag checkpoint under ``root_dir/checkpoint``."""
        if self.checkpoint_path:
            return self.checkpoint_path
        if not self.abag:
            return None                      # let the CLI resolve the general model by name
        if self.root_dir:
            cand = Path(self.root_dir) / "checkpoint" / ABAG_CHECKPOINT
            if cand.is_file():
                return str(cand)
            logger.warning("opendde: abag=True but %s is missing; falling back to --model_name %s",
                           cand, self.model_name)
        else:
            logger.warning("opendde: abag=True but no root_dir/OPENDDE_ROOT_DIR set; "
                           "cannot locate %s, falling back to --model_name", ABAG_CHECKPOINT)
        return None

    # ------------------------------------------------------------------ steering plumbing
    def _venv_python(self) -> str:
        """The interpreter of the venv that OWNS the opendde console script."""
        if self._config.get("venv_python"):
            return str(self._config["venv_python"])
        exe = Path(self.executable)
        cand = exe.with_name("python")
        if cand.is_file():
            return str(cand)
        logger.warning("opendde: could not derive a venv python from executable %s; falling back to "
                       "%r, which will fail if it cannot import runner.cli", exe, self.python)
        return self.python

    # ------------------------------------------------------------------ command
    def _build_command(self, merged_json: Path, out_dir: Path, spec: BackendRunSpec) -> List[str]:
        seeds = ",".join(str(s) for s in spec.seeds)
        nproc = int(getattr(spec, "nproc_per_node", 1) or 1)
        if nproc > 1:
            _ds_spec = spec.extras.get("diffusion_search") or {}
            if _ds_spec:
                _k = int(_ds_spec.get("subsample_pool", 1) or 1)
                raise RuntimeError(
                    f"opendde nproc_per_node={nproc} cannot carry a steered run. The multi-GPU path "
                    "goes through runner.batch_inference, which our launcher never wraps, so the "
                    "diffusion-search hook is not installed and the pool would come out UNSTEERED "
                    "with a zero exit code. Use nproc_per_node=1"
                    + (f"; the joint pool (subsample_pool={_k}) requires it anyway." if _k > 1
                       else " and run one process per GPU for parallelism."))
            head = [self.python, "-m", "torch.distributed.run", "--standalone",
                    "--nproc_per_node", str(nproc), "-m", "runner.batch_inference", "pred"]
            tail = ["--foldcp_mode", "distributed", "--foldcp_size_dp", "1",
                    "--foldcp_size_cp", str(nproc)]
        else:
            head = [self._venv_python(), str(Path(__file__).with_name("_opendde_launch.py")), "pred"]
            tail = []
        cmd = head + [
            "-i", str(merged_json),
            "-o", str(out_dir),
            "-s", seeds,
            "-c", str(spec.n_cycles),
            "-p", str(spec.n_diffusion_steps),
            "-e", str(spec.n_samples_per_seed),
            "-n", self.model_name,
            # click's BOOL parses "True"/"False" case-insensitively, so str(bool) is accepted.
            "--use_msa", str(bool(spec.use_msa)),
            "--use_template", str(bool(spec.use_template)),
            "--need_atom_confidence", str(self.need_atom_confidence),
            "--enable_tf32", str(self.enable_tf32),
            "--enable_fusion", str(self.enable_fusion),
            "--enable_cache", str(self.enable_cache),
            "--deterministic", str(self.deterministic),
            "--trimul_kernel", str(self.trimul_kernel),
            "--triatt_kernel", str(self.triatt_kernel),
        ]
        ckpt = self._resolve_checkpoint()
        if ckpt:
            cmd += ["--load_checkpoint_path", ckpt]
        dtype = self._resolve_dtype(merged_json)
        if dtype:
            cmd += ["-d", str(dtype)]
        if self.device:
            cmd += ["--device", str(self.device)]
        return cmd + tail + self.extra_args

    def _resolve_dtype(self, merged_json_path: Path) -> Optional[str]:
        """Pick the inference precision for this invocation."""
        if self.dtype and str(self.dtype).strip().lower() != "auto":
            return self.dtype
        forced = os.environ.get("PROTENIX_FORCE_DTYPE")
        if forced:
            logger.info("opendde: dtype=%s forced by $PROTENIX_FORCE_DTYPE", forced)
            return forced
        toks = ProtenixBackend._count_tokens(merged_json_path)
        if not toks:
            return self.dtype                      # cannot size the input: leave the default alone
        gib = _detect_gpu_gib()
        big = max(toks)
        if should_use_bf16(big, gpu_gib=gib):
            logger.warning(
                "opendde: N_token=%d exceeds the fp32 rail for this %.0f GiB card, using bf16. fp32 "
                "cannot fit this target at ANY chunk/offload setting (measured: 1155 tok OOM'd at "
                "chunk 5 with full offload). Structures for this target are bf16 while smaller "
                "targets in the same pool are fp32.", big, gib)
            return "bf16"
        return self.dtype

    # ------------------------------------------------------------------ run
    def run(self, inputs_dir: Path, dump_dir: Path, spec: BackendRunSpec) -> Dict[str, Any]:
        """Execute OpenDDE inference over every input json in ``inputs_dir``."""
        inputs_dir, dump_dir = Path(inputs_dir), Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        # Claim the dump dir. Protenix and OpenDDE write identical filenames, so a shared dir would
        # interleave two models' structures indistinguishably. This raises rather than allowing it.
        stamp_dump_dir(dump_dir, {"backend": self.name, "model_name": self.model_name,
                                  "checkpoint": self._resolve_checkpoint(), "abag": self.abag})
        stage = dump_dir / f"_stage_{uuid.uuid4().hex[:12]}"
        stage.mkdir(parents=True, exist_ok=True)
        merged = stage / "merged_input.json"
        n_jobs = ProtenixBackend._merge_inputs(inputs_dir, merged)

        cmd = self._build_command(merged, dump_dir, spec)
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if self.root_dir:
            env["OPENDDE_ROOT_DIR"] = self.root_dir

        repo_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        # Read at protenix_hooks IMPORT time -- setting it later leaves the hooks bound to
        # `protenix`, which is absent in this venv, so every hook silently no-ops.
        env["TTS_COFOLD_PKG"] = "opendde"
        if not Path(self._venv_python()).is_file():
            raise RuntimeError(
                f"opendde needs its venv interpreter, but {self._venv_python()!r} does not exist. "
                "Set model.opendde.venv_python explicitly. Refusing to fall back to the unhooked "
                "console script: that loses the trunk-chunk memory hook (large targets then OOM) "
                "and, on a steered arm, would silently produce an UNSTEERED pool.")

        diffusion_search = spec.extras.get("diffusion_search")
        if diffusion_search:
            ds_path = stage / "diffusion_search.json"       # per-invocation, like the merged input
            with open(ds_path, "w") as f:
                json.dump(dict(diffusion_search), f, indent=2)
            env[protenix_hooks.ENV_DIFF_SPEC] = str(ds_path)

        if spec.extras.get("dry_run"):
            logger.info("opendde DRY RUN: %s", " ".join(cmd))
            return {"returncode": 0, "cmd": cmd, "n_jobs": n_jobs, "dry_run": True}

        _resolved_off = opendde_resolve_offload_tokens(
            ProtenixBackend._count_tokens(merged),
            int(getattr(spec, "n_samples_per_seed", 1) or 1))
        if _resolved_off is not None:
            env["PROTENIX_CONF_OFFLOAD_TOKENS"] = _resolved_off

        env.setdefault("PROTENIX_DIFFUSION_CHUNK_AUTO", "1")

        _k = int((diffusion_search or {}).get("subsample_pool", 1) or 1)
        _want = ((int(spec.n_samples_per_seed) + _k - 1) if _k > 1
                 else len(spec.seeds) * int(spec.n_samples_per_seed))
        if _k > 1:
            logger.info("opendde: %d job(s), joint pool of %d subsamples x %d samples -> %d "
                        "structure(s) expected -> %s", n_jobs, _k, spec.n_samples_per_seed,
                        _want, dump_dir)
        else:
            logger.info("opendde: %d job(s), %d seed(s) x %d samples -> %s",
                        n_jobs, len(spec.seeds), spec.n_samples_per_seed, dump_dir)
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, cwd=str(stage))
        made = len(list(dump_dir.rglob("*.cif"))) if dump_dir.exists() else 0
        rec = {
            "returncode": proc.returncode,
            "cmd": cmd,
            "n_jobs": n_jobs,
            "n_structures_written": made,
            "n_structures_expected": _want,
            "seconds": round(time.time() - t0, 1),
            "checkpoint": self._resolve_checkpoint(),
            "model_name": self.model_name,
            "abag": self.abag,
        }

        try:
            from models.base import OPENDDE_ARCH
            rec["compute_budget"] = ProtenixBackend._compute_budget(
                n_jobs, spec, ProtenixBackend._count_tokens(merged), arch=OPENDDE_ARCH)
            if _k > 1:
                rec["compute_budget"]["n_structures_is_analytic_separate_model"] = True
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("opendde: compute budget not recorded (%s)", exc)
            rec["compute_budget_error"] = str(exc)
        if proc.returncode != 0:
            logger.error("opendde exited %s after %.1fs", proc.returncode, rec["seconds"])
        else:
            if made < _want:
                logger.warning("opendde: %.1fs, rc=0, but only %d of %d expected structure(s) were "
                               "written -- the run FAILED for this target", rec["seconds"], made, _want)
            else:
                logger.info("opendde: %.1fs for %d structure(s) (rc=0)", rec["seconds"], made)
            for _name in ("diffusion_timing.jsonl", "cell_map.jsonl"):
                _src = stage / _name
                if _src.is_file():
                    try:
                        shutil.copy2(_src, dump_dir / _name)
                    except OSError as _e:                           # noqa: BLE001, never fail a unit
                        logger.warning("opendde: could not preserve %s (%s)", _name, _e)
            shutil.rmtree(stage, ignore_errors=True)
        return rec

    # ------------------------------------------------------------------ predictions
    def iter_predictions(self, dump_dir: Path, target_id: str) -> List[PredictionSample]:
        """Enumerate samples for ``target_id``."""
        base = Path(dump_dir) / target_id
        patterns = [
            base / "seed_*" / "predictions" / "*_summary_confidence_sample_*.json",
            base / "seed_*" / "*_summary_confidence_sample_*.json",
            # Defensive only, for a future non-empty group level. The target_id stays a DIRECTORY
            # component so it can never reach into a sibling target.
            Path(dump_dir) / "**" / target_id / "seed_*" / "predictions" / "*_summary_confidence_sample_*.json",
        ]
        files: List[str] = []
        for p in patterns:
            files.extend(glob.glob(str(p), recursive=True))
        files = sorted(set(files))
        if not files:
            logger.warning("opendde: no confidence JSONs for %s under %s", target_id, dump_dir)
        ckpt = self._resolve_checkpoint()
        out: List[PredictionSample] = []
        for f in files:
            ps = ProtenixBackend._parse_confidence(f, target_id)
            ps.backend, ps.model_name, ps.checkpoint = self.name, self.model_name, ckpt
            ps.extras["backend"] = self.name
            out.append(ps)
        return out

    def load_pae_and_coords(
        self, dump_dir: Path, target_id: str, seed: int, sample_idx: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]], Optional[np.ndarray], Optional[str]]:
        """Delegate to the Protenix loader."""
        try:
            return ProtenixBackend.load_pae_and_coords(self, dump_dir, target_id, seed, sample_idx)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001, an unverified format must degrade, not crash a run
            return None, None, None, None, f"opendde PAE load failed (format unverified): {e}"


register_backend("opendde", OpenDDEBackend)


# --------------------------------------------------------------- diffusion sample chunk (OpenDDE)
def opendde_auto_diffusion_chunk(n_token: int, n_samples: int, gpu_gib: Optional[float] = None,
                                 offload: bool = False, margin_gib: float = 8.0) -> int:
    """``sample_diffusion_chunk_size`` for ONE OpenDDE target, sized by OpenDDE's OWN peak curve."""
    from models.base import peak_gib_for
    ns = max(1, int(n_samples))
    gib = float(gpu_gib) if gpu_gib else _detect_gpu_gib()

    _m = os.environ.get("OPENDDE_CHUNK_MARGIN_GIB")
    if _m:
        try:
            margin_gib = float(_m)
        except ValueError:
            logger.warning("OPENDDE_CHUNK_MARGIN_GIB=%r is not a number; using %.1f", _m, margin_gib)

    _cap = os.environ.get("OPENDDE_DIFFUSION_CHUNK_MAX")
    cap = None
    if _cap:
        try:
            cap = max(1, int(_cap))
        except ValueError:
            logger.warning("OPENDDE_DIFFUSION_CHUNK_MAX=%r is not an integer; ignoring", _cap)
    _floor = os.environ.get("OPENDDE_DIFFUSION_CHUNK_MIN")
    floor = None
    if _floor:
        try:
            floor = max(1, int(_floor))
        except ValueError:
            logger.warning("OPENDDE_DIFFUSION_CHUNK_MIN=%r is not an integer; ignoring", _floor)
    usable = gib - float(margin_gib)
    if usable <= 0:
        return max(1, floor or 1)
    best = 0
    for c in range(1, ns + 1):
        if peak_gib_for("opendde", int(n_token), c, ns, offload) <= usable:
            best = c
        else:
            break
    if best <= 0:
        got = max(1, floor or 1)
        logger.warning(
            "opendde diffusion chunk: %d tok x %d samples does not fit even at chunk 1 "
            "(predicted %.1f GiB vs %.1f GiB usable). Proceeding at %d (predicted %.1f GiB of %.1f "
            "GiB on the card) -- if this OOMs, THIS is why.",
            int(n_token), ns, peak_gib_for("opendde", int(n_token), 1, ns, offload), usable, got,
            peak_gib_for("opendde", int(n_token), got, ns, offload), gib)
        return got
    if cap is not None and best > cap:
        logger.info("opendde diffusion chunk: capping %d -> %d (OPENDDE_DIFFUSION_CHUNK_MAX)",
                    best, cap)
        return cap
    return best
