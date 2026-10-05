"""ModelBackend protocol + shared data types.

Every model exposes the same five operations, so the trunk-side and diffusion-side methods drive
them uniformly:

1. **emit_input**, turn a backend-agnostic :class:`Target` into a concrete input file.
2. **MSA methods** (``get_msa_sources`` / ``set_msa_sources`` / ``get_chain_sequences``), read a
   per-chain MSA map, patch per-chain MSAs, and read chain sequences. Used to wire and resample
   MSAs into inputs.
3. **run**, execute inference on a directory of inputs, dumping predictions.
4. **iter_predictions**, enumerate a target's samples as :class:`PredictionSample`, with confidence
   fields normalised to a common vocabulary.
5. **load_pae_and_coords**, return PAE + Ca coords + chain IDs (+ pLDDT) for one sample, needed by
   ipSAE.

Cross-backend confidence vocabulary, whatever the model produces is normalised into this by
:meth:`iter_predictions`::

    ranking_score, plddt, iptm, ptm, gpde, disorder, has_clash

Missing fields are ``None``; unknown fields go into ``PredictionSample.extras``.

SEED vs SAMPLE. Every variable, config key, dataframe column and filename uses these two words with
exactly these meanings::

    seed        one distinct MSA SUBSAMPLE. Each seed re-draws the MSA subsample, hence the trunk
                and the conditioning latent z. The trunk-side axis.
                Fields: `seeds`, `n_seeds`, `seed`.
    sample      one diffusion DENOISING sample from a fixed seed's latent z (Protenix
                `--sample_diffusion.N_sample`). The diffusion-side axis.
                Fields: `n_samples_per_seed`, `sample_idx`.

Total structures per target is ``n_seeds * n_samples_per_seed``, except under a joint pool (CAST),
which shares one diffusion across K conditionings and emits ``N + K - 1``.

This module also carries the cross-backend GPU peak-memory prediction, :func:`peak_gib_for`, which
scales the Protenix-fitted curve per backend through ``_PEAK_SCALE_BY_BACKEND``. The chunk and
offload decisions built on it (``max_diffusion_chunk``, ``should_offload``, ``host_ram_gib``) live
in ``models/protenix2.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from data.target import Target


# ---------------------------------------------------------------------------- run spec
@dataclass
class BackendRunSpec:
    """Cross-backend inference knobs, the union of what everyone accepts."""

    seeds: List[int] = field(default_factory=lambda: [101])   # one MSA subsample per seed (trunk-side axis)
    n_cycles: int = 10
    n_samples_per_seed: int = 5          # diffusion samples per seed (diffusion-noise axis. Protenix N_sample)
    n_diffusion_steps: int = 200         # denoising steps per sample (Protenix N_step)
    use_msa: bool = True
    use_template: bool = False
    nproc_per_node: int = 1
    extras: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------- prediction sample
@dataclass
class PredictionSample:
    """One predicted structure for one target."""

    target_id: str
    seed: int          # MSA-subsample identity (see SEED vs SAMPLE in module docstring)
    sample_idx: int    # diffusion-sample index within this seed
    cif_path: str

    # Normalised confidence vocabulary
    ranking_score: Optional[float] = None
    plddt: Optional[float] = None
    iptm: Optional[float] = None
    ptm: Optional[float] = None
    gpde: Optional[float] = None
    disorder: Optional[float] = None
    has_clash: Optional[bool] = None

    # Post-hoc verifier scores (filled in by method.verifier.*)
    ipsae: Optional[float] = None
    pdockq2: Optional[float] = None    # Zhu 2023, ab-ag mean over interfaces
    lis: Optional[float] = None        # Kim 2024 Local Interaction Score, ab-ag mean
    ilis: Optional[float] = None       # Kim 2024 integrated LIS (contact-filtered)
    consensus: Optional[float] = None          # pool-based: antibody-pose cluster-population fraction
    consensus_density: Optional[float] = None  # pool-based: Gaussian-weighted antibody-pose agreement
    # Phase A coordinate-only interface signals (no PAE)
    i_plddt: Optional[float] = None            # mean interface-residue pLDDT, ab-ag mean
    n_interface_residues: Optional[int] = None
    n_interface_contacts: Optional[int] = None
    clash_ca: Optional[int] = None             # cross-chain Cα-Cα pairs < 3.5 A
    # Phase-0 zero-compute signals promoted from the summary-JSON chain_pair_iptm matrix
    # (free reads, ab-ag masked, see interface_scores.compute_summary_pair_signals)
    min_pair_iptm: Optional[float] = None      # weakest ab-ag pairwise ipTM
    avg_pair_iptm: Optional[float] = None      # mean ab-ag pairwise ipTM
    # Biophysical interface bundle (heavy-atom, biotite, post-hoc, see verifier/biophysical.py)
    interface_bsa: Optional[float] = None          # buried surface area (Ų)
    bsa_hydrophobic_frac: Optional[float] = None   # fraction of BSA from C/S atoms
    n_salt_bridges: Optional[int] = None
    n_hbond_contacts: Optional[int] = None
    # CDR / paratope focus (ANARCI IMGT, post-hoc, see verifier/cdr.py).
    # CONVENTION: 4 A HEAVY-ATOM paratope, NOT the 10 A CA convention the steering CV, token_maps.py and the report's "Paratope on CDR" all use.
    paratope_cdr_fraction: Optional[float] = None  # fraction of the paratope that is CDR
    n_cdr_contacts: Optional[int] = None
    n_paratope_residues: Optional[int] = None

    backend: Optional[str] = None
    model_name: Optional[str] = None      # e.g. "protenix-v2", "opendde_v1"
    checkpoint: Optional[str] = None      # resolved checkpoint path/name, when the backend knows it

    # Backend-specific fields go here so nothing is lost.
    extras: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------- dump provenance
BACKEND_STAMP = "_backend.json"


def stamp_dump_dir(dump_dir, record: Dict[str, Any]) -> None:
    """Record which backend owns ``dump_dir``, and REFUSE to write a second one into it."""
    import json as _json
    from pathlib import Path as _Path

    d = _Path(dump_dir)
    d.mkdir(parents=True, exist_ok=True)
    f = d / BACKEND_STAMP
    if f.is_file():
        try:
            prev = _json.loads(f.read_text())
        except Exception:                                    # unreadable stamp: rewrite it
            prev = {}
        if prev.get("backend") and prev["backend"] != record.get("backend"):
            raise RuntimeError(
                f"{d} already holds output from backend {prev['backend']!r} "
                f"(model {prev.get('model_name')!r}); refusing to write {record.get('backend')!r} "
                "into the same dump dir. Both backends use identical output filenames, so mixing "
                "them makes the structures indistinguishable. Use a separate dump dir per backend.")
        return
    f.write_text(_json.dumps(record, indent=2))


def read_dump_backend(dump_dir) -> Optional[str]:
    """The backend that owns ``dump_dir``, or None if unstamped (pre-existing runs)."""
    import json as _json
    from pathlib import Path as _Path
    f = _Path(dump_dir) / BACKEND_STAMP
    if not f.is_file():
        return None
    try:
        return _json.loads(f.read_text()).get("backend")
    except Exception:
        return None


def compose_run_dir(runs_root, dataset: str, backend: str, run_name: str):
    """The canonical output path for a run: ``<runs_root>/<dataset>/<backend>/<run_name>``."""
    from pathlib import Path as _Path
    return _Path(runs_root) / dataset / backend / run_name


def resolve_run_dir(runs_root, dataset: str, backend: str, run_name: str):
    """Locate an EXISTING run, preferring the backend-segmented path and falling back to flat."""
    seg = compose_run_dir(runs_root, dataset, backend, run_name)
    if seg.exists():
        return seg
    from pathlib import Path as _Path
    flat = _Path(runs_root) / dataset / run_name
    if flat.exists():
        owner = read_dump_backend(flat)
        if owner and owner != backend:
            raise RuntimeError(
                f"{flat} is a legacy flat run owned by backend {owner!r}, not {backend!r}. "
                f"Write the new run to {seg} instead.")
        return flat
    return seg


# ---------------------------------------------------------------------------- protocol
@runtime_checkable
class ModelBackend(Protocol):
    """Every model backend implements this five-method interface."""

    name: str

    def emit_input(self, target: Target, out_dir: Path) -> Path:
        """Serialise ``target`` into the model's native input format under ``out_dir``."""
        ...

    def get_msa_sources(self, input_path: Path) -> Dict[str, Optional[str]]:
        """Return ``{chain_id: MSA path}`` for the input (None where a chain has none)."""
        ...

    def set_msa_sources(self, input_path: Path, chain_to_a3m: Mapping[str, str], out_path: Path) -> Path:
        """Write a copy of ``input_path`` with a **per-chain** MSA patched in."""
        ...

    def get_chain_sequences(self, input_path: Path) -> Dict[str, str]:
        """Return ``{chain_id: sequence}`` for every protein chain in an input."""
        ...

    def run(self, inputs_dir: Path, dump_dir: Path, spec: BackendRunSpec) -> Dict[str, Any]:
        """Execute inference. Returns a small dict recording (returncode, cmd, spec)."""
        ...

    def iter_predictions(self, dump_dir: Path, target_id: str) -> List[PredictionSample]:
        """Enumerate every predicted sample for ``target_id`` under ``dump_dir``."""
        ...

    def load_pae_and_coords(
        self,
        dump_dir: Path,
        target_id: str,
        seed: int,
        sample_idx: int,
    ) -> Tuple[
        Optional[np.ndarray],   # (N, N) PAE array
        Optional[np.ndarray],   # (N, 3) Cα coords
        Optional[List[str]],    # length-N chain IDs
        Optional[np.ndarray],   # (N,)  pLDDT
        Optional[str],          # error message, or None on success
    ]:
        """Load the PAE + coords for a single sample. Used by ipSAE."""
        ...


# ---------------------------------------------------------------------------- convenience adapters
def build_run_spec_from_config(cfg: Any) -> BackendRunSpec:
    """Build a :class:`BackendRunSpec` from an OmegaConf-style config subtree."""
    run = cfg.get("run", {}) if hasattr(cfg, "get") else {}
    backend_name = cfg.get("backend") if hasattr(cfg, "get") else None
    extras = {}
    if backend_name and hasattr(cfg, "get"):
        extras = dict(cfg.get(backend_name, {}) or {})
    return BackendRunSpec(
        seeds=list(cfg.get("seeds", run.get("seeds", [101]))),
        n_cycles=int(run.get("n_cycles", 10)),
        n_samples_per_seed=int(run.get("n_samples_per_seed", 5)),
        n_diffusion_steps=int(run.get("n_diffusion_steps", 200)),
        use_msa=bool(run.get("use_msa", True)),
        use_template=bool(run.get("use_template", False)),
        nproc_per_node=int(run.get("nproc_per_node", 1)),
        extras=extras,
    )


_PEAK_SCALE_BY_BACKEND: Dict[str, float] = {
    "protenix2": 1.0,      # the reference the curve was fitted on
    "opendde": 1.03,
}


def peak_gib_for(backend: str, n_token: int, chunk: int, n_samples: int,
                 offload: bool = False) -> float:
    """Predicted GPU peak (GiB): the Protenix curve times ``backend``'s scale."""
    import logging

    from models.protenix2 import _accum_frac_opendde, _predicted_peak_gib
    scale = _PEAK_SCALE_BY_BACKEND.get(backend)
    if scale is None:
        logging.getLogger(__name__).warning(
            "no memory scale for backend %r; assuming Protenix-like (1.0)", backend)
        scale = 1.0
    frac = _accum_frac_opendde(n_token) if backend == "opendde" else None
    return scale * _predicted_peak_gib(n_token, chunk, n_samples, offload, accum_frac=frac)


PROTENIX_V2_ARCH: Dict[str, float] = {
    "c_z": 256, "c_s": 384, "c_m": 128, "c_atom": 128,
    "n_pairformer_blocks": 48, "n_msa_blocks": 4,
    "n_diff_token_blocks": 24, "n_atom_blocks": 6,   # encoder 3 + decoder 3
    "atom_window": 32, "atoms_per_token": 8,          # ~heavy atoms / residue
    "n_confidence_blocks": 4,                         # confidence head's pairformer-style blocks (rough)
}

OPENDDE_ARCH: Dict[str, float] = {
    "c_z": 384, "c_s": 384, "c_m": 128, "c_atom": 128,
    "n_pairformer_blocks": 48, "n_msa_blocks": 4,
    "n_diff_token_blocks": 24, "n_atom_blocks": 6,   # encoder 3 + decoder 3
    "atom_window": 32, "atoms_per_token": 8,
    "n_confidence_blocks": 4,
    "n_refiner_blocks": 4,
}

_ARCH_BY_BACKEND: Dict[str, Dict[str, float]] = {
    "protenix2": PROTENIX_V2_ARCH,
    "opendde": OPENDDE_ARCH,
}


def arch_for(backend: str) -> Dict[str, float]:
    """The FLOPs architecture for ``backend``. Unknown names warn and fall back to Protenix, so a
    new backend degrades to "Protenix-like" instead of breaking a run, exactly as peak_gib_for does."""
    import logging
    a = _ARCH_BY_BACKEND.get(backend)
    if a is None:
        logging.getLogger(__name__).warning(
            "no FLOPs architecture for backend %r; assuming Protenix-like", backend)
        return PROTENIX_V2_ARCH
    return a


def _pairformer_flops_per_cycle(n: int, a: Dict[str, float]) -> float:
    """Trunk FLOPs for one recycle over ``n`` tokens (dominant terms)."""
    cz = a["c_z"]
    tri = a["n_pairformer_blocks"] * 16 * (n ** 3) * cz          # triangle mult+attn, O(N³)
    other = a["n_pairformer_blocks"] * 8 * (n ** 2) * (cz ** 2)  # transitions/pair attn, O(N²)
    msa = a["n_msa_blocks"] * 4 * (n ** 2) * a["c_m"] * cz       # MSA module (rough, O(N²))
    return tri + other + msa


def _diffusion_flops_per_step(n: int, a: Dict[str, float]) -> float:
    """Diffusion FLOPs for one denoising step over ``n`` tokens (~8·n atoms)."""
    cs = a["c_s"]; m = n * a["atoms_per_token"]; w = a["atom_window"]; catom = a["c_atom"]
    token = a["n_diff_token_blocks"] * (2 * (n ** 2) * cs + 8 * n * (cs ** 2))  # token transformer
    atom = a["n_atom_blocks"] * (2 * m * w * catom + 8 * m * (catom ** 2))       # atom enc/dec (local attn)
    return token + atom


def estimate_protenix_flops(
    token_counts: Sequence[int],
    n_seeds: int,
    n_samples_per_seed: int,
    n_cycle: int,
    n_step: int,
    arch: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Estimate trunk / diffusion / total FLOPs for a run."""
    a = arch or PROTENIX_V2_ARCH
    flops_trunk = 0.0
    flops_diff = 0.0
    flops_out_c = 0.0
    n_struct_per_target = n_seeds * n_samples_per_seed
    n_ref = float(a.get("n_refiner_blocks", 0) or 0)
    refiner = dict(a, n_pairformer_blocks=n_ref, n_msa_blocks=0) if n_ref else None
    for n in token_counts:
        flops_trunk += n_seeds * n_cycle * _pairformer_flops_per_cycle(int(n), a)
        if refiner is not None:
            flops_trunk += n_seeds * _pairformer_flops_per_cycle(int(n), refiner)
        flops_diff += n_struct_per_target * n_step * _diffusion_flops_per_step(int(n), a)
        flops_out_c += n_struct_per_target * _confidence_flops_per_pass(int(n), a)
    return {
        "flops_trunk": float(flops_trunk),
        "flops_diffusion": float(flops_diff),
        "flops_total": float(flops_trunk + flops_diff),   # homogeneous → sum is meaningful
        "confidence_output_nfe": int(n_struct_per_target * len(token_counts)),
        "flops_confidence_output": float(flops_out_c),
        "flops_total_with_confidence": float(flops_trunk + flops_diff + flops_out_c),
    }


def estimate_conformets_opt_flops(
    token_counts: Sequence[int],
    k_nets: int,
    n_steps: int,
    arch: Optional[Dict[str, float]] = None,
    bwd_factor: float = 2.0,
) -> Dict[str, float]:
    """EXTRA FLOPs of the A2.c ConforNets unsupervised optimisation, on top of the base
        trunk+diffusion (:func:`estimate_protenix_flops`).
    """
    a = arch or PROTENIX_V2_ARCH
    tc = [int(n) for n in token_counts]
    passes_per_target = k_nets * n_steps
    flops = sum(passes_per_target * (1.0 + bwd_factor) * _pairformer_flops_per_cycle(n, a) for n in tc)
    return {
        "conformets_opt_pairformer_passes": passes_per_target * len(tc),
        "flops_conformets_opt": float(flops),
    }


def flops_totals(trunk: float, denoiser: float, conf_search: float, conf_output: float) -> Dict[str, float]:
    """The run's two totals, defined ONCE."""
    search = float(trunk) + float(denoiser) + float(conf_search)
    return {"flops_total_search": search, "flops_total": search + float(conf_output)}


def _confidence_flops_per_pass(n: int, a: Dict[str, float]) -> float:
    """One confidence-head pass over ``n`` tokens, the plug-in-ipSAE verifier cost on x̂₀."""
    share = a.get("n_confidence_blocks", 4) / a["n_pairformer_blocks"]
    return _pairformer_flops_per_cycle(int(n), a) * share


def estimate_diffusion_search_flops(records: Sequence[Dict[str, float]], n_cycle: int = 10,
                                    trunk_flops: Optional[float] = None,
                                    trunk_nfe: Optional[int] = None,
                                    arch: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Aggregate a MEASURED diffusion-side search run into the COMPLETE three-lens compute budget, **NFE, FLOPs, and wall-time**, split across the three cost stages (trunk / denoiser /
        confidence-head), per the plan §6.1.
    """
    a = arch or PROTENIX_V2_ARCH
    denoiser_nfe = conf_nfe = 0
    flops_d = flops_c = 0.0
    out_conf_nfe = 0
    flops_out_c = 0.0
    flops_t_calc = 0.0
    trunk_nfe_calc = 0
    wall_search = wall_t = wall_d = wall_c = 0.0
    for r in records:
        n = int(r.get("n_token", 0))
        ds, cp = int(r.get("denoiser_particle_steps", 0)), int(r.get("confidence_passes", 0))
        denoiser_nfe += ds
        conf_nfe += cp
        flops_d += ds * _diffusion_flops_per_step(n, a)
        flops_c += cp * _confidence_flops_per_pass(n, a)
        flops_t_calc += n_cycle * _pairformer_flops_per_cycle(n, a)   # fallback only
        trunk_nfe_calc += int(n_cycle)                                # fallback only
        n_out = int(r.get("n_structures_emitted") or r.get("n_sample", 0))
        out_conf_nfe += n_out
        flops_out_c += n_out * _confidence_flops_per_pass(n, a)
        wall_search += float(r.get("wall_s", 0.0))
        wall_t += float(r.get("wall_trunk_s", 0.0))
        wall_d += float(r.get("wall_denoise_s", 0.0))
        wall_c += float(r.get("wall_confidence_s", 0.0))
    # Trunk FLOPs/NFE: reuse the shared compute_budget when given (do NOT double-compute).
    flops_t = float(trunk_flops) if trunk_flops is not None else flops_t_calc
    tk_nfe = int(trunk_nfe) if trunk_nfe is not None else trunk_nfe_calc
    wall_total = wall_search + wall_t
    n_calls = max(1, len(records))
    return {
        "n_calls": len(records),
        "trunk_from_compute_budget": trunk_flops is not None,
        "trunk_nfe": tk_nfe, "denoiser_nfe": denoiser_nfe,
        # search-time head passes (reward only; 0 for a geometry reward, by design)
        "confidence_nfe": conf_nfe,
        # output head passes: one per finished structure, paid by every run
        "confidence_output_nfe": out_conf_nfe,
        "flops_trunk": flops_t, "flops_denoiser": float(flops_d),
        "flops_confidence": float(flops_c),
        "flops_confidence_output": float(flops_out_c),
        # total now includes the per-structure output confidence pass, flops_total_search excludes it
        # so the two questions ("what did steering cost" vs "what did the run cost") stay separable.
        **flops_totals(flops_t, flops_d, flops_c, flops_out_c),
        "wall_total_s": float(wall_total),
        "wall_mean_per_target_s": float(wall_total / n_calls),
        "wall_by_stage_s": {"trunk": float(wall_t), "denoise": float(wall_d), "confidence": float(wall_c)},
    }
