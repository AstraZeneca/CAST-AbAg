"""RMSD and geometry primitives for antibody-antigen prediction DIVERSITY and inference-time STEERING.

Two axes, each usable POOLED (a per-target diversity, the median over structure pairs) or
SINGLE-PAIR (a steering reward of one candidate against a reference, as in ConforMix,
arXiv 2512.03312):

  binding position  the epitope centroid (contacted antigen residues) and how far it moves, i.e.
                    WHERE the antibody binds, orientation-free.
  CDR conformation  Kabsch-superposed paratope Ca RMSD, i.e. the internal CDR-loop conformation
                    with global pose removed.

All functions take Ca coordinate arrays. cif loading and antigen framing are the caller's job: the
caller superposes each antigen onto a reference and carries the antibody, so antibody coords passed
here are in a common antigen frame (``antigen_frame_pool`` does this). The paratope/epitope index
set can come from a contact rule (``paratope_mask`` / ``epitope_mask`` here) or from IMGT CDR
numbering (``cdr.py``).

RMSD is the Kabsch-Umeyama optimal superposition with the determinant/reflection correction (Kabsch
1976 Acta Cryst A32:922; Umeyama 1991 IEEE PAMI 13:376; closed form from the 3x3 cross-covariance
singular values, Coutsias et al. 2004 J Comput Chem 25:1849). The as-placed pooled form uses
``|Xi - Xj|^2 = |Xi|^2 + |Xj|^2 - 2<Xi, Xj>``. For gradient-based guidance, port ``kabsch_rmsd``
to torch, the superposition being differentiable in the coordinates through the SVD.
"""
from __future__ import annotations

import numpy as np

from method import ANTIBODY_CHAINS

CONTACT_R = 10.0        # antibody/antigen Ca within this (Angstrom) across the interface counts as a contact


# --------------------------------------------------------------------------- core RMSD primitives
def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Optimal-superposition Ca RMSD between two point sets P, Q (both (M, 3)); a single pair, e.g. a steering
    reward of a candidate versus a reference. Kabsch-Umeyama with the reflection/determinant correction."""
    P = P - P.mean(0); Q = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ Q)                            # SVD of the 3x3 cross-covariance
    sgn = 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0           # proper rotation, not a reflection
    r2 = ((P ** 2).sum() + (Q ** 2).sum() - 2.0 * (S[0] + S[1] + sgn * S[2])) / len(P)
    return float(np.sqrt(max(r2, 0.0)))


def pairwise_superposed_rmsd(coords: np.ndarray) -> np.ndarray:
    """FULL (n, n) matrix of pairwise Kabsch-superposed Ca RMSD over a pool, coords (n, M, 3), via batched
        3x3 SVD. The per-pair superposition removes global position and orientation, so this is a frame-invariant
        measure of CONFORMATION spread (CDR, interface, whole-Fv, or epitope, depending on the atoms passed).
    """
    n, M = coords.shape[0], coords.shape[1]
    Q = coords - coords.mean(1, keepdims=True)
    ss = (Q ** 2).sum((1, 2))
    H = np.empty((n, n, 3, 3))
    for a in range(3):
        for b in range(3):
            H[:, :, a, b] = Q[:, :, a] @ Q[:, :, b].T
    U, S, Vt = np.linalg.svd(H.reshape(n * n, 3, 3))
    det = np.linalg.det(U @ Vt)
    Ssum = S.sum(1) - 2.0 * S[:, 2] * (det < 0)                  # reflection correction (S descending)
    d2 = (ss[:, None] + ss[None, :] - 2.0 * Ssum.reshape(n, n)) / M
    return np.sqrt(np.clip(d2, 0.0, None))


def median_pairwise_superposed_rmsd(coords: np.ndarray) -> float:
    """Median of :func:`pairwise_superposed_rmsd` over the upper triangle: the pooled conformation spread."""
    d = pairwise_superposed_rmsd(coords)
    return float(np.median(d[np.triu_indices(len(d), 1)]))


def median_pairwise_rmsd_in_frame(coords: np.ndarray) -> float:
    """Median pairwise as-placed Ca RMSD over a pool already in a common frame, coords (n, M, 3), via the Gram
    identity (no per-pair superposition). Keeps position + orientation, so it measures PLACEMENT/POSE spread."""
    n, M = coords.shape[0], coords.shape[1]
    Pf = coords.reshape(n, -1)
    G = Pf @ Pf.T
    s = np.einsum("ii->i", G)
    d2 = (s[:, None] + s[None, :] - 2.0 * G) / M
    iu = np.triu_indices(n, 1)
    return float(np.median(np.sqrt(np.clip(d2[iu], 0.0, None))))


# --------------------------------------------------------------------------- binding position (epitope)
def _min_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each row of a (Na, 3), the distance to its nearest row of b (Nb, 3)."""
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(1)


def epitope_mask(ab_ca: np.ndarray, ag_ca: np.ndarray, contact_r: float = CONTACT_R) -> np.ndarray:
    """Boolean mask over antigen residues within ``contact_r`` of any antibody Ca (the epitope)."""
    return _min_dist(ag_ca, ab_ca) < contact_r


def paratope_mask(ab_ca: np.ndarray, ag_ca: np.ndarray, contact_r: float = CONTACT_R) -> np.ndarray:
    """Boolean mask over antibody residues within ``contact_r`` of any antigen Ca (the paratope / CDR contacts)."""
    return _min_dist(ab_ca, ag_ca) < contact_r


def epitope_centroid(ab_ca: np.ndarray, ag_ca: np.ndarray, contact_r: float = CONTACT_R) -> np.ndarray:
    """Centroid (3,) of the contacted antigen residues: the binding position, orientation-free. Falls back to
        the single closest antigen residue if nothing is within ``contact_r``.
    """
    dmin = _min_dist(ag_ca, ab_ca)
    m = dmin < contact_r
    return ag_ca[m].mean(0) if m.any() else ag_ca[dmin.argmin()]


AB_CHAINS = frozenset(ANTIBODY_CHAINS)  # the one definition, see method/__init__.py. "A" here
                                        # would read the antigen as antibody and silently make
                                        # both binding-site axes meaningless


def antigen_frame_pool(coords_list, chains_list, ab_chains=AB_CHAINS):
    """Antibody Cα of every structure in a SHARED antigen frame, plus that frame's antigen Cα."""
    ref_chains = ab_mask = ag_mask = ref_ag = None
    framed, kept = [], []
    for i, (co, ch) in enumerate(zip(coords_list, chains_list)):
        if co is None or ch is None:
            continue
        ch = np.asarray(ch)
        if ref_chains is None:
            ref_chains = ch
            ab_mask = np.array([c in ab_chains for c in ch])
            ag_mask = ~ab_mask
            if not ab_mask.any() or not ag_mask.any():
                return None, None, []
        c = np.asarray(co, float)
        if len(ch) != len(ref_chains) or len(c) != len(ref_chains) or not np.array_equal(ch, ref_chains):
            continue                # coords are length-checked too: a chain/coord mismatch would other-
                                    # wise index a boolean mask of the wrong length and raise
        if ref_ag is None:
            ref_ag = c[ag_mask]
        from method.verifier.consensus import _superpose      # inner: consensus imports this module
        framed.append(_superpose(c[ag_mask], ref_ag, c[ab_mask]))
        kept.append(i)
    return (np.stack(framed) if framed else None), ref_ag, kept


# --------------------------------------------------------------------------- the two Figure 10 diversity axes (pooled)
def binding_diversity(ab_ca_pool, ref_ag_ca: np.ndarray, contact_r: float = CONTACT_R) -> float:
    """Figure 10a: median pairwise epitope-centroid distance over a pool of antibodies in the shared antigen
    frame (``ab_ca_pool`` is an (n, n_ab, 3) array or list). Angstrom."""
    c = np.array([epitope_centroid(ab, ref_ag_ca, contact_r) for ab in ab_ca_pool])
    iu = np.triu_indices(len(c), 1)
    return float(np.median(np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))[iu]))


def cdr_conformation_diversity(ab_ca_pool, paratope_idx: np.ndarray) -> float:
    """Figure 10b: median pairwise Kabsch-superposed paratope Ca RMSD over a pool (``ab_ca_pool`` (n, n_ab, 3),
    ``paratope_idx`` selecting the CDR/paratope residues). Angstrom."""
    return median_pairwise_superposed_rmsd(np.asarray(ab_ca_pool)[:, paratope_idx, :])


# --------------------------------------------------------------------------- single-pair steering rewards
def cdr_conformation_rmsd(ab_ca: np.ndarray, ref_ab_ca: np.ndarray, paratope_idx: np.ndarray) -> float:
    """Steering CV: Kabsch RMSD of the paratope Ca of one structure versus a reference. Large = a novel CDR
    conformation (the recommended axis to steer AWAY from the default, cf. ConforMix)."""
    return kabsch_rmsd(ab_ca[paratope_idx], ref_ab_ca[paratope_idx])


def achieved_contact_distance(res_pool, paratope_idx, sites, site_ids=None) -> np.ndarray:
    """Per-particle CONTACT distance d = ||mu_paratope - mu_(that particle's candidate site)||."""
    pool = np.asarray(res_pool, dtype=np.float64)
    pi = np.asarray(paratope_idx, dtype=int)
    sites = [np.asarray(s, dtype=int) for s in (sites or [])]
    if pool.ndim != 3 or pi.size == 0 or not sites or int(pi.max()) >= pool.shape[1]:
        return np.full(len(pool), np.nan)
    gid = (np.zeros(len(pool), dtype=int) if site_ids is None
           else np.asarray(site_ids, dtype=int).reshape(-1))
    if gid.size != len(pool):
        return np.full(len(pool), np.nan)
    out = np.full(len(pool), np.nan)
    for g, ei in enumerate(sites):
        m = gid == g
        if not m.any() or ei.size == 0 or int(ei.max()) >= pool.shape[1]:
            continue
        out[m] = np.linalg.norm(pool[m][:, pi, :].mean(1) - pool[m][:, ei, :].mean(1), axis=-1)
    return out


def binding_shift(ab_ca: np.ndarray, ref_ab_ca: np.ndarray, ref_ag_ca: np.ndarray,
                  contact_r: float = CONTACT_R) -> float:
    """Steering CV: distance the epitope centroid moved versus a reference (both antibodies in the same
    antigen frame). Large = the antibody has walked to a new binding site."""
    return float(np.linalg.norm(epitope_centroid(ab_ca, ref_ag_ca, contact_r)
                                - epitope_centroid(ref_ab_ca, ref_ag_ca, contact_r)))
