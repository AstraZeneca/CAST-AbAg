"""Protenix-v2 backend, the reference implementation of :class:`ModelBackend`.

Wraps the Protenix ``runner/inference.py`` (invoked via ``torch.distributed.run``)
and its per-sample confidence + PAE outputs.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import subprocess
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

import numpy as np

from data.target import Target
from models import register_backend
from models.base import stamp_dump_dir
from models.base import BackendRunSpec, PredictionSample

logger = logging.getLogger(__name__)


CONFIDENCE_METRICS = ["ranking_score", "plddt", "iptm", "ptm", "gpde", "disorder"]
ALL_METRICS = CONFIDENCE_METRICS + ["has_clash"]


_MEM_ANCHORS = (
    (303, 4.3, 0.10),
    (666, 15.2, 0.119),
    (1155, 31.6, 0.103),
)

_OFFLOAD_SLOPE_FACTOR = 0.30


_ATOM_ATTN_ANCHOR = (1155.0, 7.66, 0.3276)      # (n_token, base_GiB, GiB per sample of chunk)


def _atom_attn_base_slope(n_token: int) -> tuple[float, float]:
    """(base_GiB, GiB_per_chunk_sample) of the atom-attention transient at ``n_token``."""
    ref, base, slope = _ATOM_ATTN_ANCHOR
    f = max(1.0, float(n_token)) / ref
    return base * f, slope * f


def _mem_base_slope(n_token: int) -> tuple[float, float]:
    """Interpolate (base_GiB, slope_GiB_per_chunk) of the peak-memory model at ``n_token``."""
    a = _MEM_ANCHORS
    if n_token <= a[0][0]:
        return a[0][1], a[0][2]
    lo, hi = (a[-2], a[-1]) if n_token >= a[-1][0] else next(
        (p, q) for p, q in zip(a, a[1:]) if p[0] <= n_token <= q[0])
    t = (n_token - lo[0]) / (hi[0] - lo[0])
    return lo[1] + t * (hi[1] - lo[1]), lo[2] + t * (hi[2] - lo[2])


def _ceil_to_5(x: int) -> int:
    """Round UP to a multiple of 5 (>=1). Used for the pass-count target, where rounding DOWN would
    add an extra diffusion pass and cancel the saving it was chosen for."""
    x = max(1, int(x))
    return x if x % 5 == 0 else x + (5 - x % 5)


def _floor_to_5(x: int) -> int:
    """Round ``x`` DOWN to a multiple of 5 (min 1). Floors, never rounds up, so the result is
    always <= the true limit (extra OOM margin) and lands on the sample counts users actually
    specify (5 / 10 / 50 / ...), which divide the diffusion pool cleanly."""
    return max(1, (int(x) // 5) * 5)


def infer_chunk_for_tokens(n_token: int, budget_gib: float = 3.0,
                           choices=(256, 128, 64, 32, 16)) -> int:
    """Largest ``infer_setting.chunk_size`` whose triangle-attention logits stay under ``budget_gib``."""
    n = max(int(n_token), 1)
    for c in sorted(choices, reverse=True):
        if c * 32 * n * n / float(2 ** 30) <= float(budget_gib):
            return int(c)
    return int(min(choices))


def max_diffusion_chunk(n_token: int, gpu_gib: float = 80.0, offload: bool = False,
                        margin_gib: float = 4.0, n_samples: Optional[int] = None) -> int:
    """Largest OOM-safe ``sample_diffusion_chunk_size`` for one target of ``n_token`` tokens, or
        ``0`` if NOT EVEN chunk 1 fits (caller must then offload).
    """
    base, slope = _mem_base_slope(int(n_token))
    if offload:
        slope *= _OFFLOAD_SLOPE_FACTOR
    # the chosen chunk stays clear of the real OOM point (not just the modelled one).
    usable = max(1.0, float(gpu_gib) - float(margin_gib) - _MODEL_OVERHEAD_GIB)
    ns = int(n_samples) if n_samples else 1
    acc = _ACCUM_FRAC * slope                        # retained-output cost per already-done sample
    # peak(chunk) = base + chunk*slope + (ns - chunk)*acc  <= usable  (chunk in-flight + retained)
    #  => chunk <= (usable - base - ns*acc) / (slope - acc)
    numer = usable - base - ns * acc
    denom = max(slope - acc, 1e-6)
    if numer < denom:                                # not even chunk 1 fits on-GPU
        return 0
    limit = int(numer / denom)
    a_base, a_slope = _atom_attn_base_slope(int(n_token))
    atom_limit = int((usable - a_base) / a_slope) if usable > a_base else 0
    if atom_limit < 1:
        return 0
    return _floor_to_5(min(512, ns, limit, atom_limit))


_HOST_ANCHORS = ((666, 11.0), (1155, 13.5))


def host_ram_gib(n_token: int, n_samples: int = 50, offload: bool = True) -> float:
    """Predicted PEAK HOST RSS in GiB. Interpolates the anchors, extrapolating ~N^2 above the top
        anchor (the pair stack drives it) and holding flat below the bottom one.
    """
    (n0, h0), (n1, h1) = _HOST_ANCHORS
    n = max(1, int(n_token))
    if n <= n0:
        h = h0 * (n / float(n0)) ** 2
    elif n <= n1:
        h = h0 + (h1 - h0) * (n - n0) / float(n1 - n0)
    else:
        h = h1 * (n / float(n1)) ** 2                 # conservative above the measured range
    if int(n_samples) <= 25:
        h *= 0.75                                     # 4.2 vs 5.8 GiB at 666 tok, 25 vs 200 samples
    if not offload:
        h *= 0.6
    return round(h, 2)


def recommended_memory_limit_gib(n_token: int, n_samples: int = 50, headroom: float = 4.0) -> int:
    """Container memory LIMIT to request for one inference process, rounded up to a multiple of 16."""
    want = host_ram_gib(n_token, n_samples) * float(headroom)
    return int(max(16, 16 * -(-want // 16)))


def _target_passes() -> int:
    """How many diffusion passes we are willing to run. Env: PROTENIX_TARGET_DIFFUSION_PASSES."""
    try:
        v = int(os.environ.get("PROTENIX_TARGET_DIFFUSION_PASSES", ""))
        return v if v >= 1 else _TARGET_PASSES
    except (TypeError, ValueError):
        return _TARGET_PASSES


def _predicted_peak_gib(n_token: int, chunk: int, n_samples: int, offload: bool,
                        accum_frac: float = None) -> float:
    """The model's own peak estimate for a given chunk: the LARGER of the two measured paths."""
    base, slope = _mem_base_slope(int(n_token))
    if offload:
        slope *= _OFFLOAD_SLOPE_FACTOR
    frac = _ACCUM_FRAC if accum_frac is None else float(accum_frac)
    inflight = base + chunk * slope + max(0, int(n_samples) - chunk) * frac * slope
    a_base, a_slope = _atom_attn_base_slope(int(n_token))
    return max(inflight, a_base + chunk * a_slope)    # offload does not touch the atom-attention term


def optimal_diffusion_chunk(n_token: int, n_samples: int, gpu_gib: float = 80.0,
                            offload: bool = False, margin_gib: float = 4.0) -> int:
    """The chunk to ACTUALLY USE: the SMALLEST that reaches the speed knee, not the largest that fits."""
    ns = max(1, int(n_samples))
    fits = max_diffusion_chunk(n_token, gpu_gib, offload=offload,
                               margin_gib=margin_gib, n_samples=ns)
    if fits <= 0:
        return 0
    want = _ceil_to_5(-(-ns // _target_passes()))      # the knee: ceil(n_samples / target_passes)
    chunk = min(ns, fits, want)
    if chunk > 0 and -(-ns // chunk) > _target_passes():
        return 0
    return max(1, chunk)


def _container_memory_limit_gib():
    """The cgroup memory LIMIT in GiB, or None when unlimited or unreadable (v2 then v1)."""
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(path).read().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            v = int(raw)
        except ValueError:
            continue
        if v <= 0 or v >= (1 << 62):            # v1 spells "unlimited" as a huge sentinel
            return None
        return v / float(1 << 30)
    return None


def _warn_if_host_limit_tight(n_token: int, n_samples: int, offload: bool) -> None:
    """Warn when the container's memory limit looks too small for the biggest target in the run."""
    lim = _container_memory_limit_gib()
    if lim is None:
        return
    need = host_ram_gib(n_token, n_samples, offload=offload)
    if lim < need * 2.0:
        logger.warning(
            "container memory limit %.0f GiB is tight for the largest target in this run "
            "(N_token=%d, n_samples=%d, offload=%s -> predicted peak %.1f GiB). The memguard fires "
            "at 92%% of the limit and its kill is SILENT: partial structures, no row, arm missing "
            "from RESULT.txt. Recommended limit %d GiB.",
            lim, int(n_token), int(n_samples), offload, need,
            recommended_memory_limit_gib(n_token, n_samples))


def _detect_gpu_gib(default: float = 80.0) -> float:
    """Total memory (GiB) of the current CUDA device, via torch then nvidia-smi, else ``default``."""
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            return torch.cuda.get_device_properties(dev).total_memory / 1024 ** 3
    except Exception:                           # noqa: BLE001, detection must never break a run
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        return float(out.strip().splitlines()[0].strip()) / 1024.0      # MiB -> GiB
    except Exception:                           # noqa: BLE001
        return default


_OFFLOAD_SPEED_CHUNK = 60


def _offload_chunk_or_warn(n_token: int, n_samples: int, gib: float) -> int:
    """Offloaded chunk, or the floor WITH A LOUD WARNING when the model says nothing fits."""
    chunk = optimal_diffusion_chunk(n_token, n_samples, gib, offload=True)
    if chunk > 0:
        return min(chunk, _OFFLOAD_SPEED_CHUNK)
    usable = max(1.0, gib - 4.0 - _MODEL_OVERHEAD_GIB)
    need = _predicted_peak_gib(n_token, _MIN_ONGPU_CHUNK, n_samples, True)
    logger.warning(
        "diffusion-chunk: model says NO on-GPU chunk fits (N_token=%d, n_samples=%d, %.0f GiB card, "
        "offload ON): predicted %.1f GiB at the chunk floor of %d vs %.1f GiB usable, short by "
        "%.1f GiB. Proceeding at the floor -- if this OOMs, THIS is the cause. The peak model is "
        "conservative at high token counts, so it may still run.",
        int(n_token), int(n_samples), gib, need, _MIN_ONGPU_CHUNK, usable, need - usable)
    return _MIN_ONGPU_CHUNK


def auto_diffusion_chunk(n_token: int, n_samples: int, gpu_gib: Optional[float] = None,
                         offload: bool = False) -> int:
    """Optimal ``sample_diffusion_chunk_size`` for ONE target. FREE-GPU-FIRST: prefer the largest
        on-GPU chunk (incl. the retained-output accumulation) that fits; ``offload`` is passed only when
        the caller has already decided even chunk 1 won't fit on-GPU (:func:`should_offload`).
    """
    gib = gpu_gib if gpu_gib is not None else _detect_gpu_gib()
    ns = int(n_samples)
    if offload:
        return max(1, min(ns, _offload_chunk_or_warn(n_token, ns, gib)))
    # FIT first (does anything fit on-GPU at all?), then OPTIMISE (how much chunk is worth using).
    chunk = max_diffusion_chunk(n_token, gib, offload=False, n_samples=ns)
    if chunk < _MIN_ONGPU_CHUNK:                     # 0 or below the floor -> not safe on-GPU, offload
        return max(1, min(ns, _offload_chunk_or_warn(n_token, ns, gib)))
    opt = optimal_diffusion_chunk(n_token, ns, gib, offload=False)
    if opt <= 0:
        return max(1, min(ns, _offload_chunk_or_warn(n_token, ns, gib)))
    return max(1, min(ns, opt))


def _accum_frac_opendde(n_token: int) -> float:
    """Retained-sample fraction of `slope` for the OPENDDE backend, by TOKEN COUNT."""
    return min(_ACCUM_FRAC, 3.75e-06 * float(max(1, int(n_token))) ** 1.5)


_ACCUM_FRAC = 0.144      # retained-sample accumulation, as a fraction of the per-chunk slope

_MAX_ONGPU_TOKENS_BY_CARD = (
    (48.0, 800),        # ~40 GB A100 and smaller
    (float("inf"), 950),   # ~80 GB A100 and larger
)


def _max_ongpu_tokens(gpu_gib: float) -> int:
    """On-GPU token ceiling for the DETECTED card (its total memory in GiB): above it, offload
        regardless of predicted chunk. Per-card because the model's top-end under-estimate is absorbable
        on a big card but not a small one. See the block comment above for calibration + the caveat.
    """
    is_40g = float(gpu_gib) <= 48.0
    env = os.environ.get("PROTENIX_MAX_ONGPU_TOKENS_40G" if is_40g else "PROTENIX_MAX_ONGPU_TOKENS_80G")
    if env:
        try:
            return int(env)
        except ValueError:
            logger.warning("ignoring non-int %s=%r; using table default", "40G" if is_40g else "80G", env)
    for upper, ceil in _MAX_ONGPU_TOKENS_BY_CARD:
        if float(gpu_gib) <= upper:
            return ceil
    return _MAX_ONGPU_TOKENS_BY_CARD[-1][1]

_MODEL_OVERHEAD_GIB = 3.0


_MIN_USEFUL_CHUNK = 10   # below this, offload instead of running many tiny passes (see

_TARGET_PASSES = 8       # the speed knee: more passes than this buys no throughput

_SAFE_PEAK_FRACTION = 0.70

_MIN_ONGPU_CHUNK = 10

_MAX_ONGPU_SAMPLES = 500            # small cards, and the floor for any card this table omits
_MAX_ONGPU_SAMPLES_80G = 500

def _max_ongpu_samples(gpu_gib: float = 80.0) -> int:
    """On-GPU sample ceiling for the DETECTED card. Above it, offload regardless of token count."""
    env = os.environ.get("PROTENIX_MAX_ONGPU_SAMPLES")
    if env:
        try:
            return int(env)
        except ValueError:
            logger.warning("ignoring non-int PROTENIX_MAX_ONGPU_SAMPLES=%r; using default", env)
    return _MAX_ONGPU_SAMPLES_80G if float(gpu_gib) >= 60.0 else _MAX_ONGPU_SAMPLES


def should_offload(n_token: int, n_samples: int, gpu_gib: float = 80.0,
                   margin_gib: float = 4.0) -> bool:
    """Auto-decide confidence CPU-offload for ONE target. FREE-GPU-FIRST: offload only when the target
        cannot safely stay on-GPU. THREE hard rails (each a regime where the model UNDER-predicts, so we
        offload regardless of the chunk math), then the chunk check:
          1. token ceiling (_max_ongpu_tokens, card-aware: 40 GB >800 tok, 80 GB >950) -- large targets;
          2. sample ceiling (_max_ongpu_samples, 500 on BOTH cards since 2026-08-20) -- high n_samples,
             accumulation OOMs even SMALL targets (live pb500: N=453/503/556 OOM'd on-GPU at 500 samples);
          3. the <_MIN_ONGPU_CHUNK floor -- a target scraping the memory floor at any size. Evaluated at
             at least _MIN_ONGPU_CHUNK samples, because the chunk is capped by n_samples and a run with
             few samples is not a run that is short of memory.
        Otherwise a reduced on-GPU chunk is preferred (faster, no PCIe). Calibrated to live OOMs: 40 GB
        offloads 813/857 tok; 80 GB keeps 857 at 50 samples; both cards offload >500 samples.
    """
    if int(n_token) > _max_ongpu_tokens(gpu_gib):    # rail 1: card-aware token ceiling
        return True
    if int(n_samples) > _max_ongpu_samples(gpu_gib):
        return True
    probe_samples = max(int(n_samples), _MIN_ONGPU_CHUNK)
    chunk = max_diffusion_chunk(n_token, gpu_gib, offload=False,
                                margin_gib=margin_gib, n_samples=probe_samples)
    return chunk == 0 or chunk < _MIN_ONGPU_CHUNK    # rail 3: memory-floor


_BF16_TOKENS_BY_CARD = (
    (1e9, 1100),     # measured on 80 GB: above the largest fp32 success (1095), below the failure (1155).
                     # job that needs a different value can set $PROTENIX_BF16_TOKENS_40G.
)


def should_use_bf16(n_token: int, gpu_gib: float = 80.0) -> bool:
    """Auto-decide inference precision for ONE target on the DETECTED card."""
    is_40g = float(gpu_gib) <= 48.0
    env = os.environ.get("PROTENIX_BF16_TOKENS_40G" if is_40g else "PROTENIX_BF16_TOKENS_80G")
    rail = None
    if env:
        try:
            rail = int(env)
        except ValueError:
            logger.warning("ignoring non-int PROTENIX_BF16_TOKENS_%s=%r; using the calibrated default",
                           "40G" if is_40g else "80G", env)
    if rail is None:
        rail = next(c for upper, c in _BF16_TOKENS_BY_CARD if float(gpu_gib) <= upper)
    return int(n_token) > rail


class ProtenixBackend:
    """Adapter around Protenix-v2's ``runner/inference.py``."""

    name = "protenix2"
    input_suffix = ".json"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        c = config or {}
        self._config: Dict[str, Any] = dict(c)
        self.model_name: str = c.get("model_name", "protenix-v2")
        self.component_path: Optional[str] = c.get("component_path") or os.environ.get("PROTENIX_COMPONENT_PATH")
        self.root_dir: Optional[str] = c.get("root_dir") or os.environ.get("PROTENIX_ROOT_DIR")
        # Checkpoint source is resolved lazily at run() time (private-first, public
        # fallback) so dry-runs / offline tests don't probe the network.
        self._model_source = None  # type: ignore[assignment]
        self.runner_path: Optional[str] = c.get("runner_path")
        self.enable_tf32: bool = bool(c.get("enable_tf32", True))
        self.deterministic: bool = bool(c.get("deterministic", False))
        self.enable_efficient_fusion: bool = bool(c.get("enable_efficient_fusion", True))
        self.need_atom_confidence: bool = bool(c.get("need_atom_confidence", False))
        self.triangle_kernel: Optional[str] = c.get("triangle_kernel") or os.environ.get("PROTENIX_TRIANGLE_KERNEL") or None
        self.infer_chunk_size: Optional[int | str] = (
            c.get("infer_chunk_size") or os.environ.get("PROTENIX_INFER_CHUNK_SIZE") or None)
        self.sample_diffusion_chunk_size: Optional[int | str] = (
            c.get("sample_diffusion_chunk_size")
            or os.environ.get("PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE") or None)
        self.layernorm_type: Optional[str] = c.get("layernorm_type") or os.environ.get("PROTENIX_LAYERNORM_TYPE") or None

    # ------------------------------------------------------------------ inputs
    def emit_input(self, target: Target, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        job = self._target_to_job(target)
        path = out_dir / f"{target.target_id}.json"
        with open(path, "w") as f:
            json.dump([job], f, indent=2)
        return path

    @staticmethod
    def _target_to_job(target: Target) -> Dict[str, Any]:
        sequences: List[Dict[str, Any]] = []
        for c in target.chains:
            if c.chain_type == "protein":
                pc: Dict[str, Any] = {"sequence": c.sequence, "count": 1, "id": [c.chain_id]}
                if c.msa_a3m_path:
                    pc["unpairedMsaPath"] = c.msa_a3m_path
                sequences.append({"proteinChain": pc})
            elif c.chain_type in {"rna", "dna"}:
                key = "rnaChain" if c.chain_type == "rna" else "dnaChain"
                sequences.append({key: {"sequence": c.sequence, "count": 1, "id": [c.chain_id]}})
            elif c.chain_type == "ligand":
                sequences.append({"ligand": {"ccdCode": c.sequence, "count": 1, "id": [c.chain_id]}})
        return {"name": target.target_id, "sequences": sequences}

    @staticmethod
    def _as_jobs(payload: Any) -> List[Dict[str, Any]]:
        """Normalise a loaded input (single job or list of jobs) to a job list."""
        return payload if isinstance(payload, list) else [payload]

    @staticmethod
    def _iter_protein_chains(payload: Any) -> Iterator[Dict[str, Any]]:
        """Yield every ``proteinChain`` dict in an input, skipping rna/dna/ligand."""
        for job in ProtenixBackend._as_jobs(payload):
            for entry in job.get("sequences", []):
                pc = entry.get("proteinChain")
                if pc is not None:
                    yield pc

    @staticmethod
    def get_msa_sources(input_path: Path) -> Dict[str, Optional[str]]:
        """Return ``{chain_id: unpairedMsaPath}`` for every protein chain in an input."""
        with open(input_path) as f:
            payload = json.load(f)
        out: Dict[str, Optional[str]] = {}
        for pc in ProtenixBackend._iter_protein_chains(payload):
            for cid in pc.get("id", []):
                out[cid] = pc.get("unpairedMsaPath")
        return out

    @staticmethod
    def get_chain_sequences(input_path: Path) -> Dict[str, str]:
        """Return ``{chain_id: sequence}`` for every protein chain in an input."""
        with open(input_path) as f:
            payload = json.load(f)
        out: Dict[str, str] = {}
        for pc in ProtenixBackend._iter_protein_chains(payload):
            for cid in pc.get("id", []):
                out[cid] = pc.get("sequence", "")
        return out

    def set_msa_sources(self, input_path: Path, chain_to_a3m: Mapping[str, str], out_path: Path) -> Path:
        """Copy the input to ``out_path`` with a **per-chain** MSA patched in."""
        with open(input_path) as f:
            payload = json.load(f)
        jobs = self._as_jobs(payload)
        out_path = Path(out_path)
        for job in jobs:
            job["name"] = out_path.stem
        for pc in self._iter_protein_chains(jobs):
            for cid in pc.get("id", []):
                if chain_to_a3m.get(cid):
                    pc["unpairedMsaPath"] = chain_to_a3m[cid]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(jobs, f, indent=2)
        return out_path

    # ------------------------------------------------------------------ inference
    def run(self, inputs_dir: Path, dump_dir: Path, spec: BackendRunSpec) -> Dict[str, Any]:
        inputs_dir = Path(inputs_dir)
        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)

        stamp_dump_dir(dump_dir, {"backend": self.name, "model_name": self.model_name})
        stage = dump_dir / f"_stage_{uuid.uuid4().hex[:12]}"
        stage.mkdir(parents=True, exist_ok=True)
        merged = stage / "merged_input.json"
        n_jobs = self._merge_inputs(inputs_dir, merged)

        # Resolve the checkpoint source (private-first, public fallback). Skipped for
        # dry-runs so offline command-preview never touches the network.
        dry_run = bool(spec.extras.get("dry_run"))
        source_record: Optional[Dict[str, Any]] = None
        if not dry_run:
            source_record = self._apply_model_source()

        cmd = self._build_command(merged, dump_dir, spec)

        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if self.component_path:
            env["PROTENIX_COMPONENT_PATH"] = self.component_path
            _vbin = os.path.join(self.component_path, "bin")
            if os.path.exists(os.path.join(_vbin, "python")):
                env["PATH"] = _vbin + os.pathsep + env.get("PATH", "")
        if self.root_dir:
            env["PROTENIX_ROOT_DIR"] = self.root_dir
        if self.layernorm_type:
            env["LAYERNORM_TYPE"] = self.layernorm_type
        if not dry_run:
            _resolved_off = self._resolve_offload_tokens(
                self._count_tokens(merged),
                int(getattr(spec, "n_samples_per_seed", 1) or 1))
            if _resolved_off is not None:
                env["PROTENIX_CONF_OFFLOAD_TOKENS"] = _resolved_off
            _ch = self.sample_diffusion_chunk_size
            if isinstance(_ch, str) and _ch.strip().lower() == "auto":
                env["PROTENIX_DIFFUSION_CHUNK_AUTO"] = "1"

        from models import protenix_hooks
        real_runner = self._resolve_runner_path()
        env[protenix_hooks.ENV_REAL_RUNNER] = real_runner
        latent_mod = spec.extras.get("latent_modulation")
        if latent_mod:
            spec_path = stage / "latent_mod.json"     # per-run scratch (see the staging note above)
            with open(spec_path, "w") as f:
                json.dump(dict(latent_mod), f, indent=2)
            env[protenix_hooks.ENV_SPEC] = str(spec_path)
            if str(latent_mod.get("kind")) == "conformets":
                cache = dump_dir.parent / "_conformets_cache"
                env[protenix_hooks.ENV_CONFORMETS_CACHE] = str(cache)
        diffusion_search = spec.extras.get("diffusion_search")
        if diffusion_search:
            ds_path = stage / "diffusion_search.json"     # per-run scratch (see the staging note above)
            with open(ds_path, "w") as f:
                json.dump(dict(diffusion_search), f, indent=2)
            env[protenix_hooks.ENV_DIFF_SPEC] = str(ds_path)
        launcher = str(Path(__file__).with_name("_protenix_launch.py"))
        cmd = [launcher if tok == real_runner else tok for tok in cmd]   # swap runner slot for our hooks
        if launcher not in cmd:                                          # fail loud, never silently skip
            raise RuntimeError(
                f"could not route the Protenix runner through {launcher}: real runner "
                f"{real_runner!r} not in the command. PAE-as-.npz + A2 hooks require this.")

        record: Dict[str, Any] = {
            "backend": self.name,
            "n_jobs": n_jobs,
            "merged_input": str(merged),
            "dump_dir": str(dump_dir),
            "command": cmd,
            "spec": asdict(spec),
        }
        if source_record is not None:
            record["model_source"] = source_record

        budget = self._compute_budget(n_jobs, spec, self._count_tokens(merged))
        record["compute_budget"] = budget

        if dry_run:
            record["dry_run"] = True
        else:
            logger.info("Running Protenix: %s", " ".join(cmd))
            t0 = time.perf_counter()
            record["started_at"] = _utcnow_iso()
            proc = subprocess.run(cmd, env=env, check=False)
            record["returncode"] = proc.returncode
            record["finished_at"] = _utcnow_iso()
            elapsed = round(time.perf_counter() - t0, 2)
            record["inference_sec"] = elapsed
            record["sec_per_structure"] = round(elapsed / budget["n_structures"], 3) \
                if budget["n_structures"] else None
            made = len(list(dump_dir.rglob("*.cif"))) if dump_dir.exists() else 0
            record["n_structures_written"] = made
            _k = int((spec.extras.get("diffusion_search") or {}).get("subsample_pool", 1) or 1)
            _want = (int(spec.n_samples_per_seed) + _k - 1) if _k > 1 else int(budget["n_structures"])
            if made < _want:
                logger.warning("Protenix inference: %ss, rc=%s, but only %d of %d expected "
                               "structure(s) were written -- the run FAILED for this target",
                               elapsed, proc.returncode, made, _want)
            else:
                logger.info("Protenix inference: %ss for %d structure(s) (%s s/structure, rc=%s)",
                            elapsed, made, record["sec_per_structure"], proc.returncode)

        with open(stage / "run_record.json", "w") as f:   # per-run scratch (concurrent runs sharing dump_dir would clobber a shared record)
            json.dump(record, f, indent=2, default=str)
        return record

    @staticmethod
    def _count_tokens(merged_path: Path) -> List[int]:
        """Token count (protein/NA residues) per job in a merged input, for FLOPs."""
        try:
            with open(merged_path) as f:
                jobs = json.load(f)
        except Exception:                                       # noqa: BLE001
            return []
        counts: List[int] = []
        for job in (jobs if isinstance(jobs, list) else [jobs]):
            n = 0
            for entry in job.get("sequences", []):
                for key in ("proteinChain", "rnaChain", "dnaChain"):
                    chain = entry.get(key)
                    if chain:
                        n += len(chain.get("sequence", "")) * int(chain.get("count", 1) or 1)
            counts.append(n)
        return counts

    @staticmethod
    def _compute_budget(n_jobs: int, spec: BackendRunSpec,
                        token_counts: Optional[List[int]] = None,
                        arch: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Deterministic compute budget for a run, split into trunk and diffusion."""
        n_seeds = len(spec.seeds)
        n_trunk_runs = n_jobs * n_seeds
        per_input = n_seeds * spec.n_samples_per_seed
        n_structures = n_jobs * per_input
        budget = {
            "n_input_jobs": n_jobs,
            "n_seeds": n_seeds,
            "n_samples_per_seed": spec.n_samples_per_seed,
            "n_structures": n_structures,
            "n_diffusion_steps": spec.n_diffusion_steps,
            "n_cycles": spec.n_cycles,
            # trunk (Pairformer, once per input×seed with N_cycle recycles)
            "n_trunk_runs": n_trunk_runs,
            "nfe_trunk": n_trunk_runs * spec.n_cycles,          # exact fwd-pass counts
            "nfe_diffusion": n_structures * spec.n_diffusion_steps,
            "token_counts": [int(t) for t in (token_counts or [])],
        }
        if token_counts:
            from models.base import estimate_protenix_flops
            budget.update(estimate_protenix_flops(
                token_counts, n_seeds, spec.n_samples_per_seed,
                spec.n_cycles, spec.n_diffusion_steps, arch=arch))
        budget["nfe_diffusion_is_analytic"] = True
        return budget

    @staticmethod
    def _check_msa_pointers(jobs, files):
        """Refuse to run an input whose MSA does not resolve. Set PROTENIX_ALLOW_NO_MSA=1 to bypass."""
        if os.environ.get("PROTENIX_ALLOW_NO_MSA") == "1":
            return
        bad = []
        for job, src in zip(jobs, list(files) + [None] * len(jobs)):
            name = (job or {}).get("name", "?")
            for entry in (job or {}).get("sequences", []):
                pc = entry.get("proteinChain")
                if not pc:
                    continue
                p = pc.get("unpairedMsaPath")
                if not p:
                    bad.append(f"{name}:{pc.get('id')} has NO unpairedMsaPath")
                elif not os.path.exists(p):
                    bad.append(f"{name}:{pc.get('id')} -> {p} does not exist")
        if bad:
            raise FileNotFoundError(
                "MSA pointers do not resolve, so this run would silently predict at N_msa 1 "
                "(query only). Re-wire from the MSA DB, or set PROTENIX_ALLOW_NO_MSA=1 if a "
                "sequence-only run is intended.\n  " + "\n  ".join(bad[:10])
                + (f"\n  ... and {len(bad) - 10} more" if len(bad) > 10 else ""))

    @staticmethod
    def _merge_inputs(inputs_dir: Path, merged_path: Path) -> int:
        files = sorted(inputs_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No Protenix input JSONs in {inputs_dir}")
        jobs: List[Dict[str, Any]] = []
        for p in files:
            with open(p) as f:
                payload = json.load(f)
            if isinstance(payload, list):
                jobs.extend(payload)
            else:
                jobs.append(payload)
        ProtenixBackend._check_msa_pointers(jobs, files)
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        with open(merged_path, "w") as f:
            json.dump(jobs, f, indent=2)
        return len(jobs)

    def _apply_model_source(self) -> Dict[str, Any]:
        """Resolve private-vs-public checkpoint once and apply it to this backend."""
        from models.protenix_provision import resolve_model_source

        if self._model_source is None:
            src = resolve_model_source(self._config)
            self.model_name = src.model_name
            if src.root_dir:
                self.root_dir = src.root_dir
            self._model_source = src
        return self._model_source.as_record()

    def _resolve_runner_path(self) -> str:
        if self.runner_path:
            return self.runner_path
        if not self.component_path:
            raise RuntimeError(
                "Cannot locate runner/inference.py. Set model.protenix.component_path "
                "or export PROTENIX_COMPONENT_PATH (mirrors the upstream Protenix distribution)."
            )
        return str(Path(self.component_path) / "lib" / "python3.11" / "site-packages" / "runner" / "inference.py")

    def _resolve_offload_tokens(self, tokens: List[int], n_samp: int) -> Optional[str]:
        """Resolve ``$PROTENIX_CONF_OFFLOAD_TOKENS`` for this run into a concrete token threshold."""
        raw = os.environ.get("PROTENIX_CONF_OFFLOAD_TOKENS")
        if raw is None:
            return None
        if raw.strip().lower() != "auto":
            return raw
        gib = _detect_gpu_gib()
        toks = [int(t) for t in tokens] or [1200]
        ns = int(n_samp)
        needing = [t for t in toks if should_offload(t, ns, gpu_gib=gib)]
        # Hook offloads when N_token > threshold (STRICT), so put the threshold ONE BELOW the smallest
        # needing target -- otherwise that target (N_token == threshold) is excluded and OOMs.
        thr = (min(needing) - 1) if needing else 10 ** 9
        logger.info("auto confidence-offload: %d/%d targets offload (n_samples=%d, gpu=%.0f GiB) -> threshold=%d (offloads N_token>=%s)",
                    len(needing), len(toks), ns, gib, thr, min(needing) if needing else "n/a")
        _warn_if_host_limit_tight(max(toks), ns, bool(needing))
        return str(int(thr))

    def _build_command(self, merged_json_path: Path, dump_dir: Path, spec: BackendRunSpec) -> List[str]:
        seeds = ",".join(str(s) for s in spec.seeds)
        master_port = os.environ.get("PROTENIX_MASTER_PORT")
        if not master_port:
            import socket
            with socket.socket() as _s:
                _s.bind(("127.0.0.1", 0))
                master_port = str(_s.getsockname()[1])
        cmd = [
            "python", "-m", "torch.distributed.run",
            "--nproc_per_node", str(spec.nproc_per_node),
            "--master_addr", "127.0.0.1",
            "--master_port", master_port,
            self._resolve_runner_path(),
            "--model_name", self.model_name,
            "--seeds", seeds,
            "--dump_dir", str(dump_dir),
            "--input_json_path", str(merged_json_path),
            "--model.N_cycle", str(spec.n_cycles),
            "--sample_diffusion.N_sample", str(spec.n_samples_per_seed),
            "--sample_diffusion.N_step", str(spec.n_diffusion_steps),
            "--use_msa", str(spec.use_msa),
            "--enable_tf32", str(self.enable_tf32),
            "--deterministic", str(self.deterministic),
            "--enable_efficient_fusion", str(self.enable_efficient_fusion),
            "--need_atom_confidence", str(self.need_atom_confidence),
            "--use_template", str(spec.use_template),
        ]
        if self.triangle_kernel:
            cmd += ["--triangle_multiplicative", str(self.triangle_kernel),
                    "--triangle_attention", str(self.triangle_kernel)]
        _ics = self.infer_chunk_size
        if isinstance(_ics, str) and _ics.strip().lower() == "auto":
            _tk = self._count_tokens(merged_json_path) or [1200]
            _ics = infer_chunk_for_tokens(max(int(t) for t in _tk))
            logger.info("infer_chunk_size=auto -> %d (largest target %d tokens, triangle logits "
                        "%.2f GiB)", _ics, max(int(t) for t in _tk),
                        _ics * 32 * max(int(t) for t in _tk) ** 2 / 2 ** 30)
        if _ics:                                        # force attention chunking (OOM control)
            cmd += ["--infer_setting.chunk_size", str(int(_ics)),
                    "--infer_setting.dynamic_chunk_size", "false"]
        chunk = self.sample_diffusion_chunk_size
        if isinstance(chunk, str) and chunk.strip().lower() == "auto":
            tokens = self._count_tokens(merged_json_path)
            all_tok = tokens or [1200]                                  # conservative if unreadable
            n_samp = int(getattr(spec, "n_samples_per_seed", 1) or 1)
            off_thr = self._resolve_offload_tokens(tokens, n_samp)      # expands 'auto' via the memory model
            gib = _detect_gpu_gib()
            thr = float(off_thr) if off_thr is not None else float("inf")
            on_gpu = [t for t in all_tok if t <= thr]                   # hook offloads N_token > thr
            if on_gpu:                                                  # size for the largest on-GPU target
                chunk = auto_diffusion_chunk(max(on_gpu), n_samp, gpu_gib=gib, offload=False)
            else:                                                       # everything offloads
                chunk = auto_diffusion_chunk(max(all_tok), n_samp, gpu_gib=gib, offload=True)
            logger.info("auto sample_diffusion_chunk_size=%d (%d/%d on-GPU, max on-GPU N_token=%d, "
                        "n_samples=%d, gpu=%.0f GiB, offload_thr=%s)",
                        chunk, len(on_gpu), len(all_tok), max(on_gpu) if on_gpu else 0, n_samp, gib, off_thr)
        if chunk:                                       # diffusion-sample batching (speed vs memory)
            cmd += ["--infer_setting.sample_diffusion_chunk_size", str(int(chunk))]
        return cmd

    # ------------------------------------------------------------------ predictions
    def iter_predictions(self, dump_dir: Path, target_id: str) -> List[PredictionSample]:
        base = Path(dump_dir) / target_id
        patterns = [
            base / "seed_*" / "predictions" / "*_summary_confidence_sample_*.json",
            base / "seed_*" / "*_summary_confidence_sample_*.json",
        ]
        files: List[str] = []
        for p in patterns:
            files.extend(glob.glob(str(p)))
        files = sorted(set(files))
        if not files:
            logger.warning("Protenix: no confidence JSONs for %s under %s", target_id, dump_dir)
        out: List[PredictionSample] = []
        for path in files:
            ps = self._parse_confidence(path, target_id)
            ps.backend, ps.model_name = self.name, self.model_name
            ps.extras["backend"] = self.name
            out.append(ps)
        return out

    @staticmethod
    def _parse_confidence(json_path: str, target_id: str) -> PredictionSample:
        p = Path(json_path)
        with open(p) as f:
            data = json.load(f)
        # The seed_* directory is either the parent (curated layout) or the grandparent
        # (fresh runs, where a `predictions/` subdir sits in between).
        seed_dir = p.parent.name if p.parent.name.startswith("seed_") else p.parent.parent.name
        seed = int(seed_dir.split("_", 1)[1]) if seed_dir.startswith("seed_") else 0
        stem = p.stem                                # "{sample_name}_summary_confidence_sample_{k}"
        tag = "_summary_confidence_sample_"
        sample_idx = int(stem.split(tag)[-1]) if tag in stem else 0
        sample_name = stem.split(tag)[0] if tag in stem else stem
        cif_candidate = p.parent / f"{sample_name}_sample_{sample_idx}.cif"
        cif_path = str(cif_candidate) if cif_candidate.is_file() else ""

        extras: Dict[str, Any] = {}
        for k, v in data.items():
            if k not in ALL_METRICS:
                extras[k] = v
        return PredictionSample(
            target_id=target_id, seed=seed, sample_idx=sample_idx, cif_path=cif_path,
            ranking_score=data.get("ranking_score"),
            plddt=data.get("plddt"),
            iptm=data.get("iptm"),
            ptm=data.get("ptm"),
            gpde=data.get("gpde"),
            disorder=data.get("disorder"),
            has_clash=data.get("has_clash"),
            extras=extras,
        )

    # ------------------------------------------------------------------ PAE
    def load_pae_and_coords(
        self,
        dump_dir: Path,
        target_id: str,
        seed: int,
        sample_idx: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]], Optional[np.ndarray], Optional[str]]:
        # Support both layouts: fresh Protenix (…/seed_N/predictions/) and curated layout (…/seed_N/).
        seed_dir = Path(dump_dir) / target_id / f"seed_{seed}"
        candidates_dirs: List[Path] = []
        pred_subdir = seed_dir / "predictions"
        if pred_subdir.is_dir():
            candidates_dirs.append(pred_subdir)
        if seed_dir.is_dir():
            candidates_dirs.append(seed_dir)
        if not candidates_dirs:
            return None, None, None, None, f"seed dir missing: {seed_dir}"

        pae: Optional[np.ndarray] = None
        pred_root: Optional[Path] = None
        stem: Optional[str] = None
        for root in candidates_dirs:
            npz = sorted(glob.glob(str(root / f"*_pae_sample_{sample_idx}.npz"))
                         + glob.glob(str(root / f"*_pae_sample_{sample_idx}.npy")))
            if npz:
                p = npz[0]
                try:
                    if p.endswith(".npz"):
                        with np.load(p) as z:
                            key = "pae" if "pae" in z.files else z.files[0]
                            pae = z[key].astype(np.float64)
                    else:
                        pae = np.load(p).astype(np.float64)
                except Exception as e:                          # noqa: BLE001
                    return None, None, None, None, f"failed to load PAE {p}: {e}"
                pred_root = root
                stem = Path(p).name.split("_pae_sample_")[0]
                break
            fd = sorted(glob.glob(str(root / f"*_full_data_sample_{sample_idx}.json")))
            if fd:
                try:
                    with open(fd[0]) as f:
                        data = json.load(f)
                    if "token_pair_pae" not in data:
                        return None, None, None, None, f"no token_pair_pae in {fd[0]}"
                    pae = np.asarray(data["token_pair_pae"], dtype=np.float64)
                except Exception as e:                          # noqa: BLE001
                    return None, None, None, None, f"failed to load full_data {fd[0]}: {e}"
                pred_root = root
                stem = Path(fd[0]).name.split("_full_data_sample_")[0]
                break
        if pae is None or pred_root is None:
            return None, None, None, None, f"no PAE (npz or full_data) for sample {sample_idx} in {seed_dir}"

        cif_candidates = sorted(glob.glob(str(pred_root / f"{stem}_sample_{sample_idx}.cif")))
        if not cif_candidates:
            return pae, None, None, None, f"no matching CIF for sample {sample_idx} in {pred_root}"
        coords, chains, plddt = _load_ca_and_chains(cif_candidates[0])
        # Guard the token↔residue alignment: ipSAE indexes PAE by residue.
        if pae.shape[0] != len(coords):
            return None, None, None, None, (
                f"PAE dim {pae.shape[0]} != {len(coords)} Cα residues for sample "
                f"{sample_idx} (non-protein tokens?); skipping ipSAE")
        return pae, coords, chains, plddt, None


def _load_ca_and_chains(cif_path: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Cα coords, chain ids (auth_asym_id → H/L/A), and pLDDT (B-factor) from a Protenix
    mmCIF by scanning the ``_atom_site`` loop directly. ~10x faster than
    ``Bio.PDB.MMCIFParser``, which builds the whole atom object tree only for the Cα
    subset. Column order is read from the loop header (not hard-coded); first model only,
    matching the previous parser."""
    cols: List[str] = []
    idx = None
    c_atom = c_chain = c_b = c_x = c_y = c_z = c_model = None
    model0 = None
    coords: List[Tuple[float, float, float]] = []
    chains: List[str] = []
    plddt: List[float] = []
    with open(cif_path) as fh:
        for line in fh:
            if line.startswith("_atom_site."):
                cols.append(line.split(".", 1)[1].strip())
                continue
            if cols and idx is None:                    # first row after the loop header
                idx = {name: i for i, name in enumerate(cols)}
                c_atom = idx.get("auth_atom_id", idx.get("label_atom_id"))
                c_chain = idx.get("auth_asym_id", idx.get("label_asym_id"))
                c_b = idx.get("B_iso_or_equiv")
                c_x, c_y, c_z = idx["Cartn_x"], idx["Cartn_y"], idx["Cartn_z"]
                c_model = idx.get("pdbx_PDB_model_num")
            if idx is None:
                continue
            is_atom = line.startswith("ATOM") or line.startswith("HETATM")
            if not is_atom:
                if coords:
                    break                               # end of the _atom_site loop
                continue
            p = line.split()
            if len(p) <= c_z or p[c_atom] != "CA":
                continue
            if c_model is not None:
                if model0 is None:
                    model0 = p[c_model]
                elif p[c_model] != model0:
                    break                               # first model only
            coords.append((float(p[c_x]), float(p[c_y]), float(p[c_z])))
            chains.append(p[c_chain])
            plddt.append(float(p[c_b]) if c_b is not None else 0.0)
    return np.asarray(coords, dtype=np.float64), chains, np.asarray(plddt, dtype=np.float64)


# Register with the model registry so `get_backend("protenix2")` works.
register_backend("protenix2", ProtenixBackend)
