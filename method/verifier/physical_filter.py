"""Post-hoc physical filtering of a finished prediction pool, reported on both diversity axes.

Filters after sampling ends, which is the placement ConforMix (arXiv 2512.03312) uses. The
criteria are a 10-residue sliding window rejected at a 20% pLDDT drop against the default
prediction (``plddt_window_drop``), plus a clash test. The in-trajectory ``physicality_filter`` in
``DiffusionSearchSpec`` folds the same criteria into the FK potential, where resampling replaces a
reject with a duplicate of a survivor, so it defaults off; filtering at the end only shrinks the
pool.

Reuses the primitives rather than restating them: ``physicality_mask`` / ``count_ca_clashes``
(method/diffusion_side/base_search.py) for the physical criteria, and ``pairwise_superposed_rmsd``
/ ``cdr_conformation_diversity`` / ``binding_diversity`` / ``antigen_frame_pool``
(method/verifier/diversity.py) for the geometry.

Every report carries BOTH axes, because conformation and binding site move independently::

    div_conf  median pairwise superposed paratope RMSD   -- CDR/interface CONFORMATION
    div_site  median pairwise epitope-centroid distance  -- WHERE the antibody binds

``max_clashes`` defaults to 10 here, not the 0 used by the in-trajectory path, so the filter
discards grossly broken structures without discarding an arm's best candidate. ``filter_pool``
reports what it removed rather than implying a torn pool was repaired: ``div_conf`` is a median, so
a pool only trips the distortion ceiling when the dispersion is pool-wide, and deleting members of
a uniformly dispersed set barely moves it. A torn arm is fixed by lowering the guidance strength.
The verdict also states descriptor COVERAGE, so a filter is never silently a no-op on a pool whose
descriptors were never computed.

All-atom clash validation, and exactly which part of each reference is implemented:

  * PoseBusters (Buttenschoen, Morris & Deane, Chem. Sci. 2024, 15, 3130; arXiv:2308.05777).
    Intermolecular test is a distance ratio on heavy atoms: a pair fails when
    d_ij / (vdw_i + vdw_j) < threshold, default 0.75. Implemented unchanged in
    :func:`clash_pairs_posebusters`. NOT implemented: the RDKit volume-overlap test and the
    ligand-chemistry tests, none of which applies to a two-protein interface.
  * MolProbity (Davis et al., Nucleic Acids Res. 2007, 35, W375; Chen et al., Acta Cryst. D 2010,
    66, 12). Clashscore = all-atom overlaps > 0.4 A per 1000 atoms. :func:`clashscore_heavy_atom`
    implements that counting convention on heavy atoms only, so it under-counts relative to
    MolProbity (which places hydrogens) and is named ``clashscore_heavy``, never ``clashscore``.
    :func:`_excluded_pairs` drops H-bond and disulfide pairs, without which the score is several
    times higher and unusable.
  * van der Waals radii: Bondi, J. Phys. Chem. 1964, 68, 441.

Not implemented: rotamer outliers, which need a rotamer library. The Ramachandran test uses coarse
boxes rather than MolProbity contours, so use it to compare arms, not against literature
thresholds.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from method.verifier.diversity import (binding_diversity, cdr_conformation_diversity,
                                       pairwise_superposed_rmsd)

TORN_CEILING = 7.88


def distortion_score(paratope_coords: np.ndarray) -> np.ndarray:
    """Per-structure distortion: median superposed paratope RMSD to the REST of the pool."""
    d = pairwise_superposed_rmsd(np.asarray(paratope_coords, float))
    return np.nanmedian(d + np.diag(np.full(len(d), np.nan)), axis=1)


def _col(df, name):
    """Column as float with unscorable entries as NaN, or None when the column is absent."""
    import pandas as pd
    if name not in df:
        return None
    return pd.to_numeric(df[name], errors="coerce").values.astype(float)


def physical_mask(df, ca_pool=None, plddt_quantile: float = 0.25,
                  plddt_min: float | None = None, max_clashes: int = 10, clash_r: float = 3.5):
    """Keep-mask over a per-sample frame, plus the per-criterion coverage that produced it."""
    n = len(df)
    cover, per = {}, {}          # coverage = inputs available, per = each criterion's own keep-mask

    pl = _col(df, "plddt")
    if pl is not None and np.isfinite(pl).any():
        cover["plddt"] = int(np.isfinite(pl).sum())
        m = np.ones(n, bool)
        if plddt_quantile and plddt_quantile > 0:
            m &= pl >= np.nanquantile(pl, plddt_quantile)
        if plddt_min is not None:
            m &= pl >= plddt_min
        per["plddt"] = m
    else:
        cover["plddt"] = 0

    if ca_pool is not None:
        from method.diffusion_side.base_search import physicality_mask   # inner: verifier -> method
        per["ca_clash"] = physicality_mask(ca_pool=ca_pool, use_clash=True, use_plddt=False,
                                           clash_r=clash_r, max_clashes=max_clashes)
        cover["ca_clash"] = len(ca_pool)
    else:
        cl = _col(df, "clash_ca")
        if cl is not None and np.isfinite(cl).any():
            cover["clash_ca"] = int(np.isfinite(cl).sum())
            per["clash_ca"] = cl <= max_clashes
        else:
            cover["clash_ca"] = 0
            hc = _col(df, "has_clash")
            cover["has_clash"] = int(np.isfinite(hc).sum()) if hc is not None else 0
            if cover["has_clash"] and max_clashes == 0:
                per["has_clash"] = ~(hc > 0)

    keep = np.ones(n, bool)
    for m in per.values():
        keep &= m
    cover["rejected"] = {k: int((~m).sum()) for k, m in per.items()}
    if keep.sum() < 5:
        return np.ones(n, bool), dict(
            cover, aborted="only %d would survive; filter NOT applied (pool kept whole)" % keep.sum())
    return keep, cover


def filter_pool(df, framed_ab, ref_ag, paratope_idx, dockq_col: str = "dockq_abag", **kw) -> dict:
    """Apply the filter and report its effect on BOTH diversity axes and on the best structure."""
    framed_ab = np.asarray(framed_ab, float)
    para = framed_ab[:, np.asarray(paratope_idx, int), :]
    keep, cover = physical_mask(df, ca_pool=framed_ab, **kw)

    dq = _col(df, dockq_col)
    if dq is None or not np.isfinite(dq).any():
        dq = _col(df, "dockq")

    def best_dockq(mask):
        """Max DockQ over ``mask``, NaN when nothing there is scorable."""
        if dq is None or not np.isfinite(dq[mask]).any():
            return float("nan")
        return float(np.nanmax(dq[mask]))

    all_rows = np.ones(len(df), bool)
    conf_b = cdr_conformation_diversity(framed_ab, paratope_idx)
    conf_a = cdr_conformation_diversity(framed_ab[keep], paratope_idx)
    site_b = binding_diversity(framed_ab, ref_ag)
    site_a = binding_diversity(framed_ab[keep], ref_ag)
    dq_b, dq_a = best_dockq(all_rows), best_dockq(keep)

    # the oracle bound: what dropping the worst half BY TRUE DISTORTION would achieve. If even this
    # stays torn, the pool is uniformly dispersed and no descriptor-based filter can rescue it.
    s = distortion_score(para)
    oracle = cdr_conformation_diversity(framed_ab[s <= np.nanquantile(s, 0.5)], paratope_idx)

    if conf_a < TORN_CEILING:
        verdict = "intact, best kept" if dq_a >= 0.98 * dq_b else "intact, BEST LOST"
    else:
        verdict = ("still torn (oracle also torn: dispersion is pool-wide, not outliers)"
                   if oracle >= TORN_CEILING else "still torn (oracle would fix it: weak descriptor)")
    # a filter that tidies conformation by collapsing the pool onto one binding site has made the
    # pool worse for selection, and only the site axis can see it
    if site_a < 0.75 * site_b:
        verdict += "; SITE DIVERSITY COLLAPSED %.1f -> %.1f A" % (site_b, site_a)
    return {"n": len(df), "n_kept": int(keep.sum()), "coverage": cover,
            "div_conf": conf_b, "div_conf_filtered": conf_a, "div_conf_oracle_half": oracle,
            "div_site": site_b, "div_site_filtered": site_a,
            "max_dockq": dq_b, "max_dockq_filtered": dq_a,
            "intact_before": conf_b < TORN_CEILING, "intact_after": conf_a < TORN_CEILING,
            "verdict": verdict, "keep": keep}


# Bondi 1964 heavy-atom radii, Angstrom. Element symbol -> radius.
VDW: Dict[str, float] = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
                         "SE": 1.90, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98}
VDW_DEFAULT = 1.70
POSEBUSTERS_RATIO = 0.75          # PoseBusters default intermolecular distance-ratio threshold
MOLPROBITY_OVERLAP = 0.4          # A, the overlap that counts as a clash in the clashscore


HBOND_MIN_A = 2.2       # shortest real N/O...N/O hydrogen bond, below this it is an overlap
HBOND_MAX_A = 3.5       # beyond this the pair is not close enough to be a false positive anyway
SS_MIN_A = 1.8          # disulfide S-S sits at ~2.05 A
SS_MAX_A = 2.5
BOND_MIN_A = 1.1       # shortest heavy-atom covalent bond here (C=O 1.23), below this it is an overlap
BOND_MAX_A = 1.9       # longest heavy-atom covalent bond here (C-S 1.81)


def _excluded_pairs(el_i: np.ndarray, el_j: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Pairs that are close by DESIGN and must not count as clashes."""
    polar = np.isin(el_i, ("N", "O")) & np.isin(el_j, ("N", "O"))
    hbond = polar & (d >= HBOND_MIN_A) & (d <= HBOND_MAX_A)
    ss = (el_i == "S") & (el_j == "S") & (d >= SS_MIN_A) & (d <= SS_MAX_A)
    return hbond | ss


def _radii(elements: Sequence[str]) -> np.ndarray:
    return np.asarray([VDW.get(str(e).strip().upper(), VDW_DEFAULT) for e in elements], float)


def _pairs_within(xyz: np.ndarray, cutoff: float) -> Tuple[np.ndarray, np.ndarray]:
    """Index pairs closer than ``cutoff``, via a uniform grid. O(N) for our densities."""
    n = len(xyz)
    if n < 2:
        return np.zeros(0, int), np.zeros(0, int)
    cell = max(cutoff, 1e-6)
    keys = np.floor(xyz / cell).astype(np.int64)
    buckets: Dict[Tuple[int, int, int], list] = {}
    for i, k in enumerate(map(tuple, keys)):
        buckets.setdefault(k, []).append(i)
    ii, jj = [], []
    for k, idx in buckets.items():
        neigh = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neigh += buckets.get((k[0] + dx, k[1] + dy, k[2] + dz), ())
        for a in idx:
            for b in neigh:
                if b > a:
                    ii.append(a); jj.append(b)
    if not ii:
        return np.zeros(0, int), np.zeros(0, int)
    ii = np.asarray(ii); jj = np.asarray(jj)
    d = np.linalg.norm(xyz[ii] - xyz[jj], axis=1)
    keep = d < cutoff
    return ii[keep], jj[keep]


def _within_n_bonds(xyz, chain_ids, ii, jj, d, n_bonds=3):
    """Mask of candidate pairs separated by at most ``n_bonds`` covalent bonds."""
    from scipy.sparse import coo_matrix
    n = len(xyz)
    bond = (np.asarray(chain_ids, dtype=object)[ii] == np.asarray(chain_ids, dtype=object)[jj]) & \
           (d >= BOND_MIN_A) & (d < BOND_MAX_A)
    if not bond.any():
        return np.zeros(len(ii), dtype=bool)
    bi, bj = ii[bond], jj[bond]
    A = coo_matrix((np.ones(len(bi)), (bi, bj)), shape=(n, n)).tocsr()
    A = A + A.T
    reach = A.copy()
    P = A.copy()
    for _ in range(int(n_bonds) - 1):
        P = (P @ A).tocsr()
        P.data[:] = 1.0
        reach = (reach + P).tocsr()
    reach.data[:] = 1.0
    return np.asarray(reach[ii, jj]).reshape(-1) > 0


def _clash_candidates(xyz, elements, chain_ids=None, residue_ids=None, intermolecular_only=False):
    """Neighbour pairs with their distances and radius sums, minus everything close BY DESIGN."""
    xyz = np.asarray(xyz, float)
    r = _radii(elements)
    if not len(r):
        return np.zeros(0), np.zeros(0), np.zeros(0, bool)
    ii, jj = _pairs_within(xyz, cutoff=2.0 * float(np.max(r)))
    if not len(ii):
        return np.zeros(0), np.zeros(0), np.zeros(0, bool)
    d = np.linalg.norm(xyz[ii] - xyz[jj], axis=1)
    e = np.asarray([str(x).strip().upper() for x in elements])
    keep = ~_excluded_pairs(e[ii], e[jj], d)
    if chain_ids is None:
        raise ValueError(
            "chain_ids is required: without it, covalently bonded atoms count as clashes "
            "(measured: backbone N-CA at 1.46 A -> clashscore 500). Pass chain_ids, and "
            "residue_ids too unless intermolecular_only=True.")
    ch = np.asarray(chain_ids, dtype=object)
    same_chain = ch[ii] == ch[jj]
    if intermolecular_only:
        keep &= ~same_chain
    elif residue_ids is None:
        raise ValueError("residue_ids is required when intermolecular_only=False, else bonded "
                         "atoms within a residue are counted as clashes.")
    else:
        res = np.asarray(residue_ids, int)
        keep &= ~(same_chain & (np.abs(res[ii] - res[jj]) <= 1))
        # THE BOND GRAPH, which is what MolProbity actually specifies. The numbering rule above is
        # kept because it is cheap and catches the common case; this catches everything it misses.
        keep &= ~_within_n_bonds(xyz, chain_ids, ii, jj, d, n_bonds=3)
        keep &= ~(same_chain & (d >= BOND_MIN_A) & (d < BOND_MAX_A))
    return d, r[ii] + r[jj], keep


def clash_pairs_posebusters(xyz, elements, chain_ids=None, residue_ids=None,
                            ratio: float = POSEBUSTERS_RATIO,
                            intermolecular_only: bool = False) -> int:
    """PoseBusters distance-ratio clash count: pairs with ``d / (vdw_i + vdw_j) < ratio``."""
    d, rsum, keep = _clash_candidates(xyz, elements, chain_ids, residue_ids, intermolecular_only)
    return int(((d < ratio * rsum) & keep).sum())


def clashscore_heavy_atom(xyz, elements, chain_ids=None, residue_ids=None,
                          overlap: float = MOLPROBITY_OVERLAP) -> float:
    """MolProbity clashscore CONVENTION on heavy atoms: overlaps > ``overlap`` A per 1000 atoms."""
    n = len(np.asarray(xyz, float))
    if not n:
        return float("nan")
    d, rsum, keep = _clash_candidates(xyz, elements, chain_ids, residue_ids)
    return float(1000.0 * (((rsum - d) > overlap) & keep).sum() / n)


_RAMA_BOXES = (
    (-180.0, -45.0, -70.0, 50.0),      # alpha / bridge, phi in [-180,-45], psi in [-70,50]
    (-180.0, -45.0, 90.0, 180.0),      # beta, upper
    (-180.0, -45.0, -180.0, -150.0),   # beta, wrapped lower
    (35.0, 90.0, -10.0, 90.0),         # left-handed alpha
)


def ramachandran_outlier_fraction(phi_psi: Iterable[Tuple[float, float]]) -> float:
    """Fraction of residues whose (phi, psi) falls in NO coarse allowed box. Degrees."""
    tot = out = 0
    for phi, psi in phi_psi:
        if phi is None or psi is None or math.isnan(phi) or math.isnan(psi):
            continue
        tot += 1
        if not any(a <= phi <= b and c <= psi <= d for a, b, c, d in _RAMA_BOXES):
            out += 1
    return float(out) / tot if tot else float("nan")
