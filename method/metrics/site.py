"""RIGHT-SITE metric: did the prediction bind the TRUE epitope patch?

GROUND TRUTH LIVES HERE, NOT IN method.verifier. The verifier package is the inference-time
signal side and must stay GT-free, or a measure can leak into selection and quietly become an
oracle. This measure needs the TRUE epitope, so it is a METRIC in the same sense as DockQ and
belongs in this package, scored by the metric stage in `method.metrics.score`.

The GT-free half of the computation stays where it was: `method.verifier.diversity.epitope_centroid`
derives the predicted binding patch from the structure alone and is legitimately usable as a signal.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from method.verifier.diversity import AB_CHAINS, CONTACT_R, epitope_centroid


def site_distance_a(x, chains, true_epitope_idx: Sequence[int],
                    contact_r: float = CONTACT_R,
                    antibody_chains: Sequence[str] = tuple(AB_CHAINS)) -> Optional[float]:
    """Predicted-epitope centroid to TRUE-epitope centroid, in Angstrom, for ONE structure.

    ``antibody_chains`` is an argument because `dockq_abag` takes one (`--antibody-chains`). With the
    set frozen here, a non-default flag split the complex one way for DockQ and another way for this
    metric, on the same row.
    """
    if chains is None:
        return None
    x = np.asarray(x, float)
    ei = np.asarray(true_epitope_idx, int)
    if x.ndim != 2 or ei.size == 0 or x.shape[0] <= int(ei.max()):
        return None
    ch = np.asarray(chains)
    m = np.isin(ch, list(antibody_chains))
    ab, ag = x[m], x[~m]
    if len(ab) == 0 or len(ag) == 0:
        return None
    return float(np.linalg.norm(epitope_centroid(ab, ag, contact_r) - x[ei].mean(0)))


def native_ca_by_chain_order(pdb_path, chain_ids):
    """CA coords of a native .pdb, chains concatenated in `chain_ids` order so the token indices of
    the CDR and epitope maps apply. `cif_io.load_ca_coords_chains` is CIF-only and returns None on
    the FoldBench natives, which is why this exists."""
    per, seen = {}, set()
    for ln in open(pdb_path):
        if not ln.startswith("ATOM") or ln[12:16].strip() != "CA":
            continue
        if ln[16] not in (" ", "A"):
            continue
        key = (ln[21], ln[22:27])
        if key in seen:
            continue
        seen.add(key)
        per.setdefault(ln[21], []).append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    order = [c for c in (chain_ids.split(";") if isinstance(chain_ids, str) else chain_ids) if c]
    return np.concatenate([np.asarray(per[c], float) for c in order], axis=0)
