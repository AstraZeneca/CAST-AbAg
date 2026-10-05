"""Trunk-side MSA input builders: attach MSAs to, and resample MSAs into, backend inputs.

Two backend-agnostic operations, which know no JSON or YAML schema, that being the backend's job:
  * :func:`wire_msa_from_dump`, attach each chain's full MSA from an MSA database to a base input,
    matching the a3m query to the chain sequence.
  * :func:`build_msa_subsample_inputs`, emit K per-seed inputs, subsampling every MSA-bearing chain
    coherently with a shared seed.

Layering: generation lives in ``data/msa/generate.py``, the subsampling ALGORITHM in
``data/msa/resample.py``, and this module wires the database's MSAs into inputs and applies that
algorithm to build the backend inputs.

NAMING (see models/base.py): each MSA subsample IS one "seed". This realises the seed axis
explicitly, K subsamples = K seeds, rather than relying on a backend's internal per-seed MSA
resampling. Each emitted input is one seed, so the backend's own ``--seeds`` flag is then just a
diffusion RNG.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data.msa import MSASubsampleSpec, read_a3m, subsample_a3m
from models.base import ModelBackend


def wire_msa_from_dump(
    backend: ModelBackend,
    input_path: str | Path,
    msa_root: str | Path,
    out_path: str | Path,
    prefer_unpaired: bool = True,
) -> Tuple[Path, Dict[str, Optional[str]]]:
    """Gap-A: populate a base input's per-chain ``unpairedMsaPath`` from a generated
        MSA dump, matching each ``*.a3m`` by its **query row** to the chain it belongs to.
    """
    input_path, msa_root, out_path = Path(input_path), Path(msa_root), Path(out_path)
    chain_seqs = backend.get_chain_sequences(input_path)              # {cid: seq}
    # Index every a3m under msa_root by its query sequence (first record).
    by_query: Dict[str, List[Path]] = {}
    for a3m in sorted(msa_root.rglob("*.a3m")):
        try:
            recs = read_a3m(a3m)          # raises on empty/header-only files (common in real dumps)
        except ValueError:
            continue
        if recs:
            by_query.setdefault(recs[0][1].replace("-", "").upper(), []).append(a3m)

    def _pick(cands: List[Path]) -> Path:
        if prefer_unpaired:
            non = [p for p in cands if "pairing" not in p.name or "non_pairing" in p.name]
            if non:
                cands = non
        return max(cands, key=lambda p: len(read_a3m(p)))            # deepest match

    chain_to_a3m: Dict[str, str] = {}
    wired: Dict[str, Optional[str]] = {}
    for cid, seq in chain_seqs.items():
        cands = by_query.get(seq.replace("-", "").upper(), [])
        path = str(_pick(cands)) if cands else None
        wired[cid] = path
        if path is not None:
            chain_to_a3m[cid] = path
    backend.set_msa_sources(input_path, chain_to_a3m, out_path)
    return out_path, wired


def build_msa_subsample_inputs(
    backend: ModelBackend,
    base_input_path: str | Path,
    target_id: str,
    spec: MSASubsampleSpec,
    k: int,
    out_dir: str | Path,
    chain_filter: Optional[Sequence[str]] = None,
) -> Tuple[List[Path], Dict[str, Any]]:
    """Emit K per-seed inputs, subsampling **every MSA-bearing chain** coherently."""
    base_input_path = Path(base_input_path)
    out_dir = Path(out_dir)
    (out_dir / "msa").mkdir(parents=True, exist_ok=True)

    chain_msa = backend.get_msa_sources(base_input_path)          # {chain_id: a3m|None}
    selected = {c: a for c, a in chain_msa.items()
                if a and (chain_filter is None or c in chain_filter)}
    depths = {c: len(read_a3m(a)) for c, a in selected.items()}   # incl. query row
    any_deep = any(d > spec.depth for d in depths.values())
    effective_k = k if any_deep else 1

    import time
    emitted: List[Path] = []
    t_build = 0.0                                        # wall spent in the subsample/cluster op
    for i in range(effective_k):
        # one spec (and label) per seed, shared across all chains → coherent state
        s = MSASubsampleSpec(policy=spec.policy, depth=spec.depth,
                             seed=spec.seed + i, identity_threshold=spec.identity_threshold)
        label = s.label()
        chain_sub: Dict[str, str] = {}
        for cid, a3m in selected.items():
            sub = out_dir / "msa" / f"{target_id}__{label}__{cid}.a3m"
            _t = time.perf_counter()
            subsample_a3m(a3m, s, sub)                    # cluster policy's O(rows·centroids) build lives here
            t_build += time.perf_counter() - _t
            chain_sub[cid] = str(sub)
        out_input = out_dir / f"{target_id}__{label}{base_input_path.suffix or ''}"
        backend.set_msa_sources(base_input_path, chain_sub, out_input)
        emitted.append(out_input)
    info = {
        "chain_depths": depths,
        "n_msa_rows": max(depths.values()) if depths else 0,
        "subsampled": any_deep,
        "effective_k": effective_k,
        # EXTRA compute of the MSA-subsample build (CPU, one-off per target, not in the GPU/FLOPs
        # budget). Material only for the cluster policy (O(rows·centroids) Hamming); ~0 for random.
        "msa_build_policy": spec.policy,
        "msa_build_wall_s": round(t_build, 4),
    }
    return emitted, info
