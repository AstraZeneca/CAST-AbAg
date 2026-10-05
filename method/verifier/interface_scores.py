"""pDockQ2 and LIS/iLIS interface-confidence scores, antibody-antigen aware.

Two PAE-based interface verifiers, alternatives to ipTM/ranking_score for Best-of-N selection. Both
reuse the model's PAE matrix, per-residue pLDDT and Ca coordinates, so they share the loader with
:mod:`method.verifier.ipsae`.

Per-direction values follow the reference implementations, cross-checked against
samuelmurail/af_analysis ``compute_pdockQ2`` / ``compute_LIS_matrix``. The scalar is aggregated as
for ipSAE and DockQ: per undirected chain pair take the max of its two directions, then average
over the antibody-antigen interfaces only, excluding the intra-antibody VH-VL interface (see
:func:`method.verifier.ipsae._abag_scalar`).

pDockQ2 (Zhu et al., Bioinformatics 2023)
    Per ordered chain pair a->b, interface contacts are Ca pairs closer than ``PD2_CONTACT_A``
    (8 A). With ipPAE the contact mean of ``1/(1+(PAE/d0)^2)``, d0 = 10, and ipLDDT the
    contact-weighted mean pLDDT (0-100) of chain a's contacting residues::

        x = ipLDDT * ipPAE
        pDockQ2 = L/(1+exp(-k*(x-x0))) + b

    with ``PD2_L, PD2_X0, PD2_K, PD2_B`` = 1.31034849, 84.7326239, 0.0747157696, 0.00501886443.
    Higher is better, about 0.23 marks a likely acceptable interface.

LIS / iLIS (Kim et al., bioRxiv 2024, https://github.com/flyark/AFM-LIS)
    Per ordered chain pair a->b, LIS keeps block entries with PAE < ``LIS_PAE_CUTOFF`` (12), maps
    each to ``1 - PAE/12`` in [0,1] and averages, or 0 if none. cLIS adds a Ca contact filter at
    ``LIS_CONTACT_A`` (8 A). ``iLIS = sqrt(LIS * cLIS)``, which collapses to 0 with no confident
    contact. Higher is better, iLIS >= 0.203 flags a confident interaction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ipsae import ANTIBODY_LABELS, _abag_scalar  # reuse the ab-ag aggregation

# -- pDockQ2 constants (Zhu et al. 2023, af_analysis.compute_pdockQ2 defaults) --
PD2_L, PD2_X0, PD2_K, PD2_B = 1.31034849, 84.7326239, 0.0747157696, 0.00501886443
PD2_PAE_D0 = 10.0            # PAE normalisation d0 in 1/(1+(PAE/d0)^2)
PD2_CONTACT_A = 8.0          # inter-chain Cα contact cutoff (strict <) defining the interface
# -- LIS constants (Kim et al., AFM-LIS) -------------------------------------
LIS_PAE_CUTOFF = 12.0        # keep inter-chain pairs with PAE < 12 (strict, matches af_analysis)
LIS_CONTACT_A = 8.0          # cLIS additional Cα contact cutoff


@dataclass
class InterfaceScores:
    """pDockQ2 / LIS / iLIS for a single sample (ab-ag scalars + per-direction detail)."""
    pdockq2: Optional[float]
    lis: Optional[float]
    ilis: Optional[float]
    per_pair_pdockq2: Dict[str, float] = field(default_factory=dict)
    per_pair_lis: Dict[str, float] = field(default_factory=dict)
    per_pair_ilis: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


def _sigmoid_pdockq2(x: float) -> float:
    return PD2_L / (1.0 + np.exp(-PD2_K * (x - PD2_X0))) + PD2_B


def _directional_scores(
    pae_ab: np.ndarray, dist_ab: np.ndarray, plddt_a: Optional[np.ndarray],
) -> Tuple[float, float, float]:
    """pDockQ2, LIS, iLIS for the ordered chain pair a->b."""
    if pae_ab.size == 0:
        return 0.0, 0.0, 0.0

    # -- LIS / cLIS / iLIS (directional block) -------------------------------
    mask = pae_ab < LIS_PAE_CUTOFF
    lis = float((1.0 - pae_ab[mask] / LIS_PAE_CUTOFF).mean()) if mask.any() else 0.0
    cmask = mask & (dist_ab < LIS_CONTACT_A)
    clis = float((1.0 - pae_ab[cmask] / LIS_PAE_CUTOFF).mean()) if cmask.any() else 0.0
    ilis = float(np.sqrt(lis * clis))

    # -- pDockQ2 (contacts a->b, contact-weighted chain-a pLDDT) --------------
    contact = dist_ab < PD2_CONTACT_A
    if contact.any() and plddt_a is not None:
        rows, cols = np.where(contact)                       # repeats -> contact weighting
        pae_sel = pae_ab[rows, cols]
        norm_if_interpae = np.mean(1.0 / (1.0 + (pae_sel / PD2_PAE_D0) ** 2))
        plddt_avg = float(np.mean(plddt_a[rows]))
        pdockq2 = float(_sigmoid_pdockq2(norm_if_interpae * plddt_avg))
    else:
        pdockq2 = 0.0                                         # no interface contact -> worst case
    return pdockq2, lis, ilis


def compute_interface_scores(
    pae: np.ndarray,
    ca_coords: np.ndarray,
    chain_ids: Sequence[str],
    plddt: Optional[np.ndarray] = None,
    antibody_chains: Optional[Sequence[str]] = None,
) -> InterfaceScores:
    """pDockQ2, LIS and iLIS for a single sample, each aggregated as the mean over
    the antibody-antigen interfaces (max of the two directions per interface, VH-VL
    excluded), matching the project ipSAE aggregation."""
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        return InterfaceScores(None, None, None, error=f"pae must be square, got {pae.shape}")
    if ca_coords.shape[0] != pae.shape[0] or len(chain_ids) != pae.shape[0]:
        return InterfaceScores(None, None, None, error="pae/coords/chain length mismatch")

    chain_arr = np.asarray(chain_ids)
    unique = list(dict.fromkeys(chain_arr.tolist()))
    if len(unique) < 2:
        return InterfaceScores(None, None, None, error="interface scores require >= 2 chains")

    plddt_norm = None
    if plddt is not None:
        p = np.asarray(plddt, dtype=np.float64)
        plddt_norm = p * 100.0 if p.max() <= 1.5 else p       # pDockQ2 expects 0-100

    coords = np.asarray(ca_coords, dtype=np.float64)
    idx = {c: np.where(chain_arr == c)[0] for c in unique}
    pp_pd2: Dict[str, float] = {}
    pp_lis: Dict[str, float] = {}
    pp_ilis: Dict[str, float] = {}
    for a in unique:
        for b in unique:
            if a == b:
                continue
            ia, ib = idx[a], idx[b]
            if ia.size == 0 or ib.size == 0:
                continue
            dist_ab = np.linalg.norm(coords[ia][:, None, :] - coords[ib][None, :, :], axis=-1)
            pd2, lis, ilis = _directional_scores(
                pae[np.ix_(ia, ib)], dist_ab,
                plddt_norm[ia] if plddt_norm is not None else None,
            )
            pp_pd2[f"{a}__{b}"] = pd2
            pp_lis[f"{a}__{b}"] = lis
            pp_ilis[f"{a}__{b}"] = ilis

    return InterfaceScores(
        pdockq2=_abag_scalar(pp_pd2, unique, antibody_chains),
        lis=_abag_scalar(pp_lis, unique, antibody_chains),
        ilis=_abag_scalar(pp_ilis, unique, antibody_chains),
        per_pair_pdockq2=pp_pd2,
        per_pair_lis=pp_lis,
        per_pair_ilis=pp_ilis,
    )


CONTACT_A = 8.0             # inter-chain Cα contact cutoff defining the interface
CLASH_CA_A = 3.5           # cross-chain Cα-Cα distance below which we count a clash


@dataclass
class CoordSignals:
    """Coordinate-only interface signals for one sample."""
    i_plddt: Optional[float]           # mean interface-residue pLDDT (0-100), ab-ag mean
    n_interface_residues: Optional[int]
    n_interface_contacts: Optional[int]
    clash_ca: Optional[int]           # cross-chain Cα-Cα pairs < 3.5 A (all chain pairs)
    error: Optional[str] = None


def compute_coord_signals(
    ca_coords: np.ndarray,
    chain_ids: Sequence[str],
    plddt: Optional[np.ndarray] = None,
    antibody_chains: Optional[Sequence[str]] = None,
    contact_A: float = CONTACT_A,
    clash_ca_A: float = CLASH_CA_A,
) -> CoordSignals:
    """i_plddt (ab-ag mean over interfaces), interface size (summed over ab-ag
    interfaces), and a Cα clash count. NaN-safe: undefined scalars come back None."""
    if ca_coords.ndim != 2 or ca_coords.shape[1] != 3 or len(chain_ids) != ca_coords.shape[0]:
        return CoordSignals(None, None, None, None, error="coords/chain length mismatch")
    chain_arr = np.asarray(chain_ids)
    unique = list(dict.fromkeys(chain_arr.tolist()))
    if len(unique) < 2:
        return CoordSignals(None, None, None, None, error="need >= 2 chains")
    coords = np.asarray(ca_coords, dtype=np.float64)
    p = None
    if plddt is not None:
        p = np.asarray(plddt, dtype=np.float64)
        p = p * 100.0 if p.max() <= 1.5 else p
    idx = {c: np.where(chain_arr == c)[0] for c in unique}

    ab = set(antibody_chains) if antibody_chains else {c for c in unique if c in ANTIBODY_LABELS}
    ident_ab = bool(ab) and not set(unique).issubset(ab)   # a real ab/ag split exists

    def is_abag(a, b):
        return ((a in ab) ^ (b in ab)) if ident_ab else True

    iface_plddt: List[float] = []                          # one both-side mean per ab-ag interface
    n_res = n_con = 0
    for i_a in range(len(unique)):
        for i_b in range(i_a + 1, len(unique)):
            a, b = unique[i_a], unique[i_b]
            if not is_abag(a, b):                          # ab-ag interfaces only (excludes VH-VL)
                continue
            ia, ib = idx[a], idx[b]
            if ia.size == 0 or ib.size == 0:
                continue
            d = np.linalg.norm(coords[ia][:, None, :] - coords[ib][None, :, :], axis=-1)
            con = d < contact_A
            if not con.any():
                continue
            rows = con.any(axis=1)                          # a-side interface residues
            cols = con.any(axis=0)                          # b-side interface residues
            n_con += int(con.sum())
            n_res += int(rows.sum() + cols.sum())
            if p is not None:                              # both-side interface-residue mean pLDDT
                iface_plddt.append(float(np.concatenate([p[ia][rows], p[ib][cols]]).mean()))

    # clash: cross-chain Cα-Cα pairs below the clash cutoff (any chain pair, incl. VH-VL,
    # since a clash anywhere signals a broken structure)
    from scipy.spatial import cKDTree
    pairs = cKDTree(coords).query_pairs(clash_ca_A, output_type="ndarray")
    clash = int((chain_arr[pairs[:, 0]] != chain_arr[pairs[:, 1]]).sum()) if len(pairs) else 0

    i_plddt = float(np.mean(iface_plddt)) if iface_plddt else None
    return CoordSignals(i_plddt, n_res, n_con, clash)


@dataclass
class SummaryPairSignals:
    """Ab-ag-masked ipTM scalars read from the summary-JSON chain_pair_iptm matrix."""
    min_pair_iptm: Optional[float]     # weakest ab-ag pairwise ipTM (worst-interface confidence)
    avg_pair_iptm: Optional[float]     # mean ab-ag pairwise ipTM
    error: Optional[str] = None


def compute_summary_pair_signals(
    summary: Dict,
    ordered_chains: Sequence[str],
    antibody_chains: Optional[Sequence[str]] = None,
) -> SummaryPairSignals:
    """Promote chain_pair_iptm to ab-ag-masked scalars."""
    cpi = summary.get("chain_pair_iptm")
    if cpi is None:
        return SummaryPairSignals(None, None, error="no chain_pair_iptm field in summary")
    chains = list(ordered_chains)
    n = len(chains)
    if len(cpi) != n or not all(len(r) == n for r in cpi):
        return SummaryPairSignals(None, None, error=f"chain_pair_iptm shape != n_chains ({n})")

    ab = set(antibody_chains) if antibody_chains else {c for c in chains if c in ANTIBODY_LABELS}
    ident = bool(ab) and not set(chains).issubset(ab)     # a real ab/ag split exists
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if (((chains[i] in ab) ^ (chains[j] in ab)) if ident else True)]
    if not pairs:
        return SummaryPairSignals(None, None, error="no ab-ag chain pairs")

    vals = [cpi[i][j] for i, j in pairs]
    return SummaryPairSignals(float(min(vals)), float(sum(vals) / len(vals)))
