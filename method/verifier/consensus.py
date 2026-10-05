"""Consensus interface-RMSD selection, the pool-based verifier.

Instead of scoring each sample alone, it uses the whole pool of N: a model that has found the
binding mode keeps returning to it, so the correct pose is the one the ensemble converges on while
wrong poses scatter. Ranking by how crowded a pose's neighbourhood is is the standard docking
selection rule (ClusPro, HADDOCK). It uses no model internals, so it still works when the
confidence heads are saturated.

Method, align on receptor and measure ligand agreement:
  1. Split chains into antibody (H/L or ``antibody_chains``) and antigen (the rest).
  2. Superpose every sample onto a reference by its antigen Ca (Kabsch), putting all antigens in a
     common frame, and carry the antibody Ca along.
  3. The pairwise antibody-Ca RMSD in that frame is the relative-pose difference. Two per-sample
     crowd scores:
       * ``consensus`` (fraction), the fraction of the pool within ``rmsd_threshold`` of it, a hard
         cluster-population count including self,
       * ``consensus_density``, the soft variant, the pool mean of
         ``exp(-rmsd^2 / density_sigma^2)``.

Both are higher when the pose is shared by more of the ensemble. They can diverge on bimodal
targets: the density is dominated by very tight clusters, whereas the fraction rewards raw
population within the cutoff.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .ipsae import ANTIBODY_LABELS

DEFAULT_RMSD_THRESHOLD_A = 8.0     # antibody-pose LRMSD within which two samples share a mode
DEFAULT_DENSITY_SIGMA_A = 4.0      # Gaussian width (Å) of the consensus-density kernel


def _superpose(mobile_fit: np.ndarray, ref_fit: np.ndarray, mobile_apply: np.ndarray) -> np.ndarray:
    """Kabsch: rotate/translate ``mobile_apply`` by the transform that best aligns
    ``mobile_fit`` onto ``ref_fit``. All are (·, 3) Cα arrays."""
    mc = mobile_fit.mean(axis=0)
    rc = ref_fit.mean(axis=0)
    H = (mobile_fit - mc).T @ (ref_fit - rc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return (mobile_apply - mc) @ R.T + rc


def consensus_scores(
    coords_list: Sequence[Optional[np.ndarray]],
    chains_list: Sequence[Optional[Sequence[str]]],
    antibody_chains: Optional[Sequence[str]] = None,
    rmsd_threshold: float = DEFAULT_RMSD_THRESHOLD_A,
    density_sigma: float = DEFAULT_DENSITY_SIGMA_A,
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Per-sample consensus signals, aligned with the input order, as two lists:"""
    n = len(coords_list)
    frac_out: List[Optional[float]] = [None] * n
    dens_out: List[Optional[float]] = [None] * n
    if n < 2:
        return frac_out, dens_out

    # Reference chain labelling from the first valid sample. antigen_frame_pool picks the SAME reference
    # by the same rule, so it is only needed here to size the antibody mask.
    ref_chains = None
    for ch, co in zip(chains_list, coords_list):
        if ch is not None and co is not None:
            ref_chains = np.asarray(ch)
            break
    if ref_chains is None:
        return frac_out, dens_out

    ab = set(antibody_chains) if antibody_chains else {c for c in ref_chains if c in ANTIBODY_LABELS}
    ab_mask = np.array([c in ab for c in ref_chains])
    if not ab_mask.any() or ab_mask.all():
        return frac_out, dens_out         # need both an antibody and an antigen to define a pose

    # Antibody of every valid sample in a common antigen frame. Shared with the diversity_axes
    # (imported here rather than at module scope: diversity imports _superpose back from this module).
    from .diversity import antigen_frame_pool
    framed, _ref_ag, valid_idx = antigen_frame_pool(coords_list, chains_list, ab_chains=ab)
    if framed is None or len(valid_idx) < 2:
        return frac_out, dens_out

    # Pairwise antibody-Cα squared RMSD via the ||a-b||^2 = |a|^2+|b|^2-2 a·b identity.
    A = framed.reshape(len(valid_idx), -1)                     # (m, n_ab*3)
    n_ab = ab_mask.sum()
    g = (A * A).sum(axis=1) / n_ab                             # mean sq-norm per sample
    cross = (A @ A.T) / n_ab
    d2 = np.maximum(g[:, None] + g[None, :] - 2.0 * cross, 0.0)
    rmsd = np.sqrt(d2)
    frac = (rmsd < rmsd_threshold).mean(axis=1)                # cluster population fraction (self included)
    dens = np.exp(-d2 / (density_sigma ** 2)).mean(axis=1)     # Gaussian-weighted crowd (self included)
    for k, i in enumerate(valid_idx):
        frac_out[i] = float(frac[k])
        dens_out[i] = float(dens[k])
    return frac_out, dens_out
