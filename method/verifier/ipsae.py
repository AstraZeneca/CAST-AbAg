"""ipSAE, Interface predicted Structure Alignment Error (Dunbrack Lab).

Reference: https://github.com/DunbrackLab/ipSAE
Preprint:  https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1

Two implementations:

- :func:`compute_ipsae`, canonical numpy reference. Non-differentiable (hard PAE mask, piecewise
  ``d0``, argmax over residues). The default verifier for sample selection.
- :func:`soft_ipsae`, differentiable surrogate (PyTorch, optional). Each hard non-smoothness is
  replaced::

    hard mask ``pae < c``            -> sigmoid((c - pae)/tau)
    ``d0`` clamp (max 26 / floor 1)  -> softplus / clamp
    ``argmax_i ptm_i``               -> temperature-scaled LogSumExp

  torch is imported at call time, not module load, so the numpy path works without it.

Formula (``ipsae_d0res_asym``, the headline ipSAE). For an ordered chain pair (A, B) and residues
``i in A``::

    V_i  = { j in B :  pae[i,j] < pae_cutoff  and optional pLDDT filter }
    n_i  = |V_i|
    d0(n) = max(1.0, 1.24 * (max(n, 26) - 15)^(1/3) - 1.8)
    ptm_i = (1 / n_i) * sum_{j in V_i}  1 / (1 + (pae[i,j] / d0(n_i))^2)   if n_i > 0
    ipSAE_{A->B} = max over ALL i in A of ptm_i

The interface is defined by the PAE cutoff alone, with no 3D distance cutoff, which is what
distinguishes ipSAE from ipTM. ``dist_cutoff`` is kept for API compatibility and does not affect
the score. The symmetric report is ``max(ipSAE_{A->B}, ipSAE_{B->A})``.

Scalar aggregation is antibody-antigen aware: the mean over the antibody-antigen interfaces only,
each interface being the max of its two directions. The intra-antibody VH-VL interface is excluded,
since it is nearly constant across samples and would dominate a plain max. This matches the DockQ
metric. Antibody chains default to ``ANTIBODY_LABELS`` (H/L); pass ``antibody_chains`` to override.
Non-antibody complexes fall back to the max over all pairs, and ``per_chain_pair`` returns every
direction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from method import ANTIBODY_CHAINS


# ---------------------------------------------------------------------------- constants
DEFAULT_PAE_CUTOFF_A = 10.0
DEFAULT_DIST_CUTOFF_A = 10.0
DEFAULT_PLDDT_CUTOFF = 0.0    # 0 = no pLDDT filter, canonical repo has a variant that filters at 50

ANTIBODY_LABELS = frozenset(ANTIBODY_CHAINS)   # the one definition, see method/__init__.py


@dataclass
class IpsaeResult:
    """Result of a single-sample ipSAE evaluation."""
    ipsae: Optional[float]                             # scalar for verifier ranking
    per_chain_pair: Dict[str, float] = field(default_factory=dict)  # "A_B": ipSAE_{A→B}
    n_interface_residues: Dict[str, int] = field(default_factory=dict)
    pae_cutoff: float = DEFAULT_PAE_CUTOFF_A
    dist_cutoff: float = DEFAULT_DIST_CUTOFF_A
    plddt_cutoff: float = DEFAULT_PLDDT_CUTOFF
    error: Optional[str] = None


# ---------------------------------------------------------------------------- canonical numpy
def _d0(n: np.ndarray | int) -> np.ndarray | float:
    """d0 from ipSAE (Dunbrack ``calc_d0_array``): ``max(1.0, 1.24·(max(26,n)−15)^(1/3)−1.8)``."""
    L = np.maximum(np.asarray(n, dtype=np.float64), 26.0)
    return np.maximum(1.24 * np.cbrt(L - 15.0) - 1.8, 1.0)


def _ipsae_one_direction(
    pae_sub: np.ndarray,
    plddt_j: Optional[np.ndarray],
    pae_cutoff: float,
    plddt_cutoff: float,
) -> Tuple[Optional[float], int]:
    """Compute ipSAE_{A→B} from the (|A|, |B|) PAE slice."""
    if pae_sub.size == 0:
        return None, 0

    # Valid partner mask V_ij = 1 iff pae_ij < cutoff (strict, matching the reference)
    valid = pae_sub < pae_cutoff
    if plddt_j is not None and plddt_cutoff > 0:
        valid = valid & (plddt_j[None, :] >= plddt_cutoff)

    n_i = valid.sum(axis=1).astype(np.int64)       # (|A|,) per-residue valid-partner count
    if not (n_i > 0).any():
        return None, 0                             # no PAE interface in this direction
    d0_i = _d0(n_i)                                # (|A|,) per-residue d0
    # 1 / (1 + (pae/d0)^2), zero-out invalid partners
    term = 1.0 / (1.0 + (pae_sub / d0_i[:, None]) ** 2)
    term = np.where(valid, term, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ptm_i = np.where(n_i > 0, term.sum(axis=1) / np.maximum(n_i, 1), 0.0)

    # ipSAE_{A→B} = max over ALL A residues (no distance interface).
    return float(ptm_i.max()), int((n_i > 0).sum())


def _abag_scalar(per_pair: Dict[str, float], unique_chains: Sequence[str],
                 antibody_chains: Optional[Sequence[str]]) -> Optional[float]:
    """Scalar ipSAE restricted to antibody-antigen interfaces, aggregated as the
    mean over interfaces (each interface = max of its two directions), to match the
    project DockQ metric (mean over ab-ag interfaces). Excludes the intra-antibody
    VH-VL pair. Falls back to max over all pairs if the antibody/antigen split
    cannot be identified (so non-antibody complexes still get a score)."""
    ab = set(antibody_chains) if antibody_chains else {c for c in unique_chains if c in ANTIBODY_LABELS}
    # need a genuine split: some antibody chains and some antigen chains
    identifiable = bool(ab) and not ab.issuperset(unique_chains)
    if not per_pair:
        return 0.0 if identifiable else None
    if not identifiable:
        return max(per_pair.values())             # not an identifiable antibody complex, fall back
    interfaces: Dict[tuple, float] = {}
    for key, val in per_pair.items():
        a, b = key.split("__")
        if (a in ab) ^ (b in ab):                     # cross-role (antibody <-> antigen) only
            uk = tuple(sorted((a, b)))
            interfaces[uk] = max(interfaces.get(uk, 0.0), val)
    return float(np.mean(list(interfaces.values()))) if interfaces else 0.0


def compute_ipsae(
    pae: np.ndarray,
    ca_coords: np.ndarray,
    chain_ids: Sequence[str],
    plddt: Optional[np.ndarray] = None,
    pae_cutoff: float = DEFAULT_PAE_CUTOFF_A,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF_A,
    plddt_cutoff: float = DEFAULT_PLDDT_CUTOFF,
    antibody_chains: Optional[Sequence[str]] = None,
) -> IpsaeResult:
    """Compute ipSAE for a single sample. Non-differentiable (numpy reference)."""
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        return IpsaeResult(None, error=f"pae must be square, got {pae.shape}")
    if ca_coords.shape[0] != pae.shape[0]:
        return IpsaeResult(None, error=f"ca_coords/pae length mismatch: {ca_coords.shape[0]} vs {pae.shape[0]}")
    if len(chain_ids) != pae.shape[0]:
        return IpsaeResult(None, error=f"chain_ids length {len(chain_ids)} != N={pae.shape[0]}")

    chain_arr = np.asarray(chain_ids)
    unique_chains = list(dict.fromkeys(chain_arr.tolist()))   # preserve order, dedupe
    if len(unique_chains) < 2:
        # Monomer target: ipSAE is only defined for interfaces.
        return IpsaeResult(None, error="ipSAE requires ≥ 2 chains")

    # pLDDT normalisation: some pipelines write 0-1, others 0-100.
    plddt_norm = None
    if plddt is not None:
        p = np.asarray(plddt, dtype=np.float64)
        plddt_norm = p * 100.0 if p.max() <= 1.5 else p
        # Interpret plddt_cutoff on the 0-100 scale if it's obviously that.
        plddt_cutoff_use = plddt_cutoff * (100.0 if plddt_cutoff <= 1.0 and plddt_cutoff > 0 else 1.0)
    else:
        plddt_cutoff_use = plddt_cutoff

    per_pair: Dict[str, float] = {}
    n_iface: Dict[str, int] = {}
    for a in unique_chains:
        for b in unique_chains:
            if a == b:
                continue
            idx_a = np.where(chain_arr == a)[0]
            idx_b = np.where(chain_arr == b)[0]
            if idx_a.size == 0 or idx_b.size == 0:
                continue
            pae_sub = pae[np.ix_(idx_a, idx_b)]
            plddt_j = plddt_norm[idx_b] if plddt_norm is not None else None
            score, n = _ipsae_one_direction(pae_sub, plddt_j, pae_cutoff, plddt_cutoff_use)
            key = f"{a}__{b}"
            if score is not None:
                per_pair[key] = score
            n_iface[key] = n

    # scalar = mean over the antibody-antigen interfaces (excludes intra-antibody
    # VH-VL); per_chain_pair keeps every direction for debugging.
    scalar = _abag_scalar(per_pair, unique_chains, antibody_chains)
    return IpsaeResult(
        ipsae=scalar,
        per_chain_pair=per_pair,
        n_interface_residues=n_iface,
        pae_cutoff=pae_cutoff,
        dist_cutoff=dist_cutoff,
        plddt_cutoff=plddt_cutoff,
    )


# ---------------------------------------------------------------------------- differentiable surrogate
def soft_ipsae(
    pae,                     # torch.Tensor (N, N)
    ca_coords,               # torch.Tensor (N, 3)
    chain_ids: Sequence[str],
    plddt=None,              # torch.Tensor (N,) | None
    *,
    pae_cutoff: float = DEFAULT_PAE_CUTOFF_A,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF_A,
    plddt_cutoff: float = DEFAULT_PLDDT_CUTOFF,
    pae_temp: float = 0.5,
    dist_temp: float = 0.5,
    argmax_temp: float = 0.1,
    antibody_chains: Optional[Sequence[str]] = None,
):
    """Differentiable ipSAE surrogate. Requires PyTorch."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as e:
        raise ImportError("soft_ipsae requires PyTorch. Install with: pip install '.[guidance]'") from e

    device = pae.device
    dtype = pae.dtype
    chain_arr = np.asarray(chain_ids)
    unique_chains = list(dict.fromkeys(chain_arr.tolist()))
    if len(unique_chains) < 2:
        return torch.zeros((), device=device, dtype=dtype)

    if plddt is not None:
        plddt_norm = plddt * 100.0 if plddt.max().item() <= 1.5 else plddt
        plddt_cutoff_use = plddt_cutoff * (100.0 if 0 < plddt_cutoff <= 1.0 else 1.0)
    else:
        plddt_norm = None
        plddt_cutoff_use = plddt_cutoff

    # antibody-antigen restriction: keep only cross-role (antibody <-> antigen) pairs,
    # matching compute_ipsae. Excludes the intra-antibody VH-VL interface.
    ab = set(antibody_chains) if antibody_chains else {c for c in unique_chains if c in ANTIBODY_LABELS}
    restrict = bool(ab) and not ab.issuperset(unique_chains)

    per_pair_scores: List[torch.Tensor] = []
    for a in unique_chains:
        for b in unique_chains:
            if a == b:
                continue
            if restrict and ((a in ab) == (b in ab)):   # skip intra-antibody and intra-antigen pairs
                continue
            idx_a = torch.as_tensor(np.where(chain_arr == a)[0], device=device, dtype=torch.long)
            idx_b = torch.as_tensor(np.where(chain_arr == b)[0], device=device, dtype=torch.long)
            if idx_a.numel() == 0 or idx_b.numel() == 0:
                continue

            pae_sub = pae.index_select(0, idx_a).index_select(1, idx_b)             # (|A|,|B|)

            soft_valid = torch.sigmoid((pae_cutoff - pae_sub) / pae_temp)           # ∈ (0,1), PAE-only
            if plddt_norm is not None and plddt_cutoff_use > 0:
                pj = plddt_norm.index_select(0, idx_b)
                soft_valid = soft_valid * torch.sigmoid((pj[None, :] - plddt_cutoff_use) / max(pae_temp, 1e-3))

            # Smooth n_i = Σ_j soft_valid_ij  (differentiable count)
            n_i = soft_valid.sum(dim=1)                                             # (|A|,)
            L = 26.0 + F.softplus(n_i - 26.0)                                       # smooth max(n, 26), Dunbrack calc_d0_array
            d0_i = torch.clamp(1.24 * torch.pow(torch.clamp(L - 15.0, min=1e-6), 1.0 / 3.0) - 1.8, min=1.0)

            term = 1.0 / (1.0 + (pae_sub / d0_i[:, None]) ** 2)
            term = term * soft_valid
            ptm_i = term.sum(dim=1) / torch.clamp(n_i, min=1e-3)                    # (|A|,)

            # ipSAE_{A→B} = smooth max over ALL A residues (no distance interface).
            if ptm_i.numel() == 0:
                continue
            lse = argmax_temp * torch.logsumexp(ptm_i / argmax_temp, dim=0)
            per_pair_scores.append(lse)

    if not per_pair_scores:
        return torch.zeros((), device=device, dtype=dtype)

    stacked = torch.stack(per_pair_scores)
    return argmax_temp * torch.logsumexp(stacked / argmax_temp, dim=0)


# ---------------------------------------------------------------------------- IO helpers
def dump_result(result: IpsaeResult, path: str | Path) -> Path:
    from results_io import dump_dataclass_json
    return dump_dataclass_json(result, path)


