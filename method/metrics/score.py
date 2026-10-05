"""Parallel evaluation-metric scoring, the metric stage, separate from the verifier stage.

* **Verifier stage** (``method.verifier.score_samples``) ranks samples at inference time. It is
  parallelised over targets with a process pool, being CPU-bound on the PAE and CIF parse.
* **Metric stage** (this module) scores finished structures against ground truth. It is
  parallelised over samples with a THREAD pool, because DockQ shells out to a binary that releases
  the GIL, so threads run concurrently with no pickling overhead.

Every metric is cached per (metric, sample) as JSON, so re-runs skip recompute. Keeping the two
stages apart means the selected verifier and the selected metric are chosen and computed
independently, and ``evaluate.py`` composes them.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from method.metrics import compute_dockq
from method.metrics.dockq import DEFAULT_ANTIBODY_CHAINS, antigen_mean_dockq_terms
from method.metrics.dockq import dump_result as dump_dockq_result
from method.metrics.dockq import load_result as load_dockq_result

logger = logging.getLogger(__name__)

ALL_METRICS: Tuple[str, ...] = ("dockq", "site", "contact")


@dataclass
class MetricSpec:
    metrics: Sequence[str] = ALL_METRICS
    dockq_bin: Optional[str] = None
    antibody_chains: Sequence[str] = DEFAULT_ANTIBODY_CHAINS
    results_dir: Optional[Path] = None
    force_recompute: bool = False
    # TRUE-epitope token indices for the target being scored, for the "site" metric. None (the
    # default) leaves site_dist_a null, so every existing caller is unchanged.
    true_epitope_idx: Optional[Sequence[int]] = None
    true_epitope_map: Optional[Dict[str, Sequence[int]]] = None
    cdr_map: Optional[Dict[str, Sequence[int]]] = None
    native_contact: Optional[Dict[str, float]] = None


def _target_of(tag: str) -> str:
    """Target id from a score tag. Tags are built as f"{target_id}__seed{seed}__s{idx}"."""
    return tag.split("__seed")[0]


def _target_idx(spec: "MetricSpec", tag: str, attr: str):
    """Per-target token indices from a map on the spec, or None."""
    m = getattr(spec, attr, None)
    return m.get(_target_of(tag)) if isinstance(m, dict) else None


def _cache_path(spec: MetricSpec, kind: str, tag: str) -> Optional[Path]:
    return (Path(spec.results_dir) / "metrics" / kind / f"{tag}.json") if spec.results_dir else None


def _cached_scalar(path: Optional[Path], key: str, force: bool) -> Optional[float]:
    if force or path is None or not path.is_file():
        return None
    try:
        return json.load(open(path)).get(key)
    except (json.JSONDecodeError, OSError):
        return None


def score_one(tag: str, cif: str, gt: str, spec: MetricSpec) -> Dict[str, Optional[float]]:
    """All selected ground-truth metrics for ONE sample, cache-aware."""
    out: Dict[str, Optional[float]] = {}
    if "dockq" in spec.metrics:
        cache = _cache_path(spec, "dockq", tag)
        r = None if (spec.force_recompute or cache is None) else load_dockq_result(cache, spec.antibody_chains)
        if r is None:
            r = compute_dockq(cif, gt, dockq_bin=spec.dockq_bin, antibody_chains=spec.antibody_chains)
            if cache is not None:
                dump_dockq_result(r, cache)
        out["dockq"], out["dockq_abag"] = r.dockq, r.dockq_abag
        # DockQ decomposition terms (fnat/irms/lrms) are a free byproduct of the DockQ stdout
        # (persisted in the cache), so they cost nothing extra on a fresh run or a cache hit.
        terms = antigen_mean_dockq_terms(r.stdout or "", spec.antibody_chains)
        out["fnat_term"] = terms["fnat_term"] if terms else None
        out["irms_term"] = terms["irms_term"] if terms else None
        out["lrms_term"] = terms["lrms_term"] if terms else None
    if "site" in spec.metrics:
        idx = _target_idx(spec, tag, "true_epitope_map") or getattr(spec, "true_epitope_idx", None)
        v = None
        if idx is not None and len(idx):
            try:
                from method.verifier.cif_io import load_ca_coords_chains
                from method.metrics.site import site_distance_a
                xyz, chains, _ = load_ca_coords_chains(cif)
                v = None if xyz is None else site_distance_a(
                    xyz, chains, idx, antibody_chains=spec.antibody_chains)
            except Exception:                      # a provenance column must never fail a scoring run
                v = None
        out["site_dist_a"] = v
    if "contact" in spec.metrics:
        # CONTACT distance: paratope centroid -> TRUE-epitope centroid, ANTIBODY side to ANTIGEN side.
        # Reported as the signed gap to the native value, see MetricSpec.native_contact.
        ei = _target_idx(spec, tag, "true_epitope_map")
        pi = _target_idx(spec, tag, "cdr_map")
        cd = dl = None
        if ei is not None and pi is not None and len(ei) and len(pi):
            try:
                import numpy as _np
                from method.verifier.cif_io import load_ca_coords_chains
                xyz, _c, _ = load_ca_coords_chains(cif)
                if xyz is not None:
                    x = _np.asarray(xyz, float)
                    need = max(int(max(pi)), int(max(ei)))
                    if x.ndim == 2 and x.shape[0] > need:
                        cd = float(_np.linalg.norm(x[list(pi)].mean(0) - x[list(ei)].mean(0)))
                        dn = (spec.native_contact or {}).get(_target_of(tag))
                        if dn is not None:
                            dl = cd - float(dn)
            except Exception:                        # a provenance column must never fail a run
                cd = dl = None
        out["contact_dist_a"] = cd
        out["contact_delta_native"] = dl
    return out


def score_metrics(items: List[Tuple[str, str, str]], spec: MetricSpec,
                  max_workers: int = 16) -> Dict[str, Dict[str, Optional[float]]]:
    """Score many samples in parallel. ``items`` = list of (tag, cif_path, gt_path).
    Returns ``{tag: {metric: value}}``. Thread pool (external metric binaries release the
    GIL); serial when a single worker or item. Cache makes already-scored samples cheap."""
    if not items:
        return {}
    n = max(1, min(max_workers, len(items)))

    def _one(it: Tuple[str, str, str]):
        tag, cif, gt = it
        return tag, score_one(tag, cif, gt, spec)

    if n == 1:
        return dict(_one(it) for it in items)
    with ThreadPoolExecutor(max_workers=n) as ex:
        return dict(ex.map(_one, items))
