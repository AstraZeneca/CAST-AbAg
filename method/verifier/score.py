"""Parallel verifier scoring, the verifier stage, run after inference.

Only ipSAE is actually computed here, from ``token_pair_pae`` plus Ca coords. ``ranking_score``,
``iptm``, ``ptm`` and ``plddt`` are confidence-head outputs the model already wrote during
inference, so those verifiers are read from the summary JSON at no compute cost.

``score_samples`` parallelises across targets with a process pool, the per-sample work being
CPU-bound and independent, and each worker writes only its own target's ipSAE JSONs. It falls back
to serial on a single target or a pool failure. The metric stage against ground truth is separate,
see ``method.metrics.score``.
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.base import ModelBackend
from results_io import SampleRecord

from .base import Verifier
from .biophysical import compute_biophys
from .cdr import compute_cdr_paratope_4a_heavy
from .cif_io import load_ca_coords_chains
from .confidence import ConfidenceVerifier
from .consensus import DEFAULT_DENSITY_SIGMA_A, DEFAULT_RMSD_THRESHOLD_A, consensus_scores
from .interface_scores import compute_coord_signals, compute_summary_pair_signals
from .interface_verifier import INTERFACE_SCORES, InterfaceScoreVerifier

logger = logging.getLogger(__name__)

# per_sample columns the cif/JSON augmentation path fills (the non-PAE signal groups).
AUGMENT_SIGNAL_COLS = ("i_plddt", "n_interface_residues", "n_interface_contacts", "clash_ca",
                       "min_pair_iptm", "avg_pair_iptm", "consensus", "consensus_density",
                       "interface_bsa", "bsa_hydrophobic_frac", "n_salt_bridges", "n_hbond_contacts",
                       "paratope_cdr_fraction", "n_cdr_contacts", "n_paratope_residues")

_GROUP_PRIMARY_COL = {
    "coord": "i_plddt",
    "pair_iptm": "avg_pair_iptm",
    "consensus": "consensus",
    "biophys": "interface_bsa",
    "cdr": "paratope_cdr_fraction",
    "interface": "ipsae",
}


def _group_already_computed(df, group: str) -> bool:
    """True if ``group``'s primary column is present with >=1 non-null value (attempted before)."""
    col = _GROUP_PRIMARY_COL.get(group)
    return bool(col) and col in df.columns and df[col].notna().any()


def make_verifier(
    backend: ModelBackend,
    predictions_dir: str | Path,
    rank_by: str,
    ipsae_kwargs: Optional[Dict[str, Any]] = None,
    results_dir: Optional[Path] = None,
) -> Verifier:
    """Return the verifier implied by ``rank_by``. Backend-agnostic."""
    kw = dict(ipsae_kwargs or {})
    groups = kw.get("compute_groups")
    if groups:
        groups = set(groups)
        return InterfaceScoreVerifier(
            backend, predictions_dir,
            rank_by=rank_by if rank_by in INTERFACE_SCORES else "ipsae",
            pae_cutoff=kw.get("pae_cutoff_A", 10.0),
            dist_cutoff=kw.get("dist_cutoff_A", 10.0),
            plddt_cutoff=kw.get("plddt_cutoff", 0.0),
            fallback_rank_by=kw.get("fallback_rank_by", "ranking_score"),
            results_dir=results_dir,
            antibody_chains=kw.get("antibody_chains"),
            compute_interface="interface" in groups,
            compute_coord="coord" in groups,
            compute_pair_iptm="pair_iptm" in groups,
            compute_consensus="consensus" in groups,
            compute_biophys="biophys" in groups,
            compute_cdr="cdr" in groups,
            hmmer_path=kw.get("hmmer_path"),
            density_sigma=kw.get("density_sigma_A", DEFAULT_DENSITY_SIGMA_A),
        )
    if rank_by in INTERFACE_SCORES:
        return InterfaceScoreVerifier(
            backend, predictions_dir, rank_by=rank_by,
            pae_cutoff=kw.get("pae_cutoff_A", 10.0),
            dist_cutoff=kw.get("dist_cutoff_A", 10.0),
            plddt_cutoff=kw.get("plddt_cutoff", 0.0),
            fallback_rank_by=kw.get("fallback_rank_by", "ranking_score"),
            results_dir=results_dir,
            antibody_chains=kw.get("antibody_chains"),
        )
    if rank_by in ("consensus", "consensus_density"):
        # pool consensus (fraction + density), served by a consensus-only
        # InterfaceScoreVerifier (PAE-optional), so there is a single consensus code path.
        return InterfaceScoreVerifier(
            backend, predictions_dir, rank_by=rank_by,
            fallback_rank_by=kw.get("fallback_rank_by", "ranking_score"),
            results_dir=results_dir,
            antibody_chains=kw.get("antibody_chains"),
            compute_interface=False, compute_coord=False, compute_pair_iptm=False,
            compute_consensus=True,
            rmsd_threshold=kw.get("rmsd_threshold_A", DEFAULT_RMSD_THRESHOLD_A),
            density_sigma=kw.get("density_sigma_A", DEFAULT_DENSITY_SIGMA_A),
        )
    return ConfidenceVerifier(backend, predictions_dir, rank_by=rank_by)


def _score_workers() -> int:
    """Worker count for the verifier pool. ``PROTEIN_TTS_COLLECT_WORKERS`` overrides
    (set 1 to force serial); else min(16, cpu_count)."""
    env = os.environ.get("PROTEIN_TTS_COLLECT_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return min(16, os.cpu_count() or 2)


def _score_one_target(payload) -> List[SampleRecord]:
    """Verifier-score every sample of ONE target. Top-level (picklable) so it can run in a
    ProcessPool worker; rebuilds the verifier from picklable primitives."""
    backend, predictions_dir, rank_by, ipsae_kwargs, results_dir, label, tid = payload
    v = make_verifier(backend, predictions_dir, rank_by, ipsae_kwargs, results_dir)
    return [
        SampleRecord(
            target_id=tid, method=label, seed=s.seed, sample_idx=s.sample_idx,
            cif_path=s.cif_path, ipsae=s.ipsae, pdockq2=s.pdockq2, lis=s.lis, ilis=s.ilis,
            i_plddt=s.i_plddt, n_interface_residues=s.n_interface_residues,
            n_interface_contacts=s.n_interface_contacts, clash_ca=s.clash_ca,
            min_pair_iptm=s.min_pair_iptm, avg_pair_iptm=s.avg_pair_iptm,
            consensus=s.consensus, consensus_density=s.consensus_density,
            interface_bsa=s.interface_bsa, bsa_hydrophobic_frac=s.bsa_hydrophobic_frac,
            n_salt_bridges=s.n_salt_bridges, n_hbond_contacts=s.n_hbond_contacts,
            paratope_cdr_fraction=s.paratope_cdr_fraction, n_cdr_contacts=s.n_cdr_contacts,
            n_paratope_residues=s.n_paratope_residues,
            plddt=s.plddt, iptm=s.iptm, ptm=s.ptm,
            ranking_score=s.ranking_score, gpde=s.gpde, disorder=s.disorder,
            has_clash=s.has_clash, notes=s.notes,
        )
        for s in v.iter_samples(tid)
    ]


def score_samples(
    backend: ModelBackend,
    target_ids: List[str],
    predictions_dir: str | Path,
    rank_by: str = "ranking_score",
    ipsae_kwargs: Optional[Dict[str, Any]] = None,
    results_dir: Optional[Path] = None,
    method_label: str = "unspecified",
    max_workers: Optional[int] = None,
) -> List[SampleRecord]:
    """Score every sample of every target with the ``rank_by`` verifier, in parallel.
    Does not invoke the model, call after inference. Returns one SampleRecord per sample."""
    payloads = [
        (backend, str(predictions_dir), rank_by, ipsae_kwargs, results_dir, method_label, tid)
        for tid in target_ids
    ]
    n = max_workers if max_workers is not None else _score_workers()

    if n <= 1 or len(target_ids) <= 1:
        return [rec for p in payloads for rec in _score_one_target(p)]

    records: List[SampleRecord] = []
    try:
        with ProcessPoolExecutor(max_workers=n) as ex:
            for recs in ex.map(_score_one_target, payloads):
                records.extend(recs)
    except Exception as e:                              # noqa: BLE001, never fail scoring
        logger.warning("verifier pool failed (%s: %s); falling back to serial", type(e).__name__, e)
        records = [rec for p in payloads for rec in _score_one_target(p)]
    return records



def _summary_json_from_cif(cif_path: str, sample_idx) -> Optional[str]:
    """``{name}_sample_{k}.cif`` -> ``{name}_summary_confidence_sample_{k}.json`` (same dir)."""
    suffix = f"_sample_{sample_idx}.cif"
    if not cif_path.endswith(suffix):
        return None
    d, base = os.path.split(cif_path)
    return os.path.join(d, base[: -len(suffix)] + f"_summary_confidence_sample_{sample_idx}.json")


def _augment_signals_target(payload):
    """Top-level (picklable) worker: the selected cif/JSON signal groups for one target's pool."""
    tid, rows, antibody_chains, density_sigma, groups, cdr_by_chain = payload
    want_coord = "coord" in groups
    want_pair = "pair_iptm" in groups
    want_cons = "consensus" in groups
    want_bio = "biophys" in groups
    want_cdr = "cdr" in groups and bool(cdr_by_chain)
    cdr_by_chain = {c: set(v) for c, v in (cdr_by_chain or {}).items()}
    coords_list, chains_list, loaded = [], [], []
    for seed, samp, cif in rows:
        co, ch, pl = load_ca_coords_chains(cif) if cif else (None, None, None)
        coords_list.append(co); chains_list.append(ch)
        loaded.append((seed, samp, cif, co, ch, pl))
    if want_cons:
        frac, dens = consensus_scores(coords_list, chains_list, antibody_chains=antibody_chains,
                                      density_sigma=density_sigma)
    else:
        frac = dens = [None] * len(loaded)
    out = []
    for (seed, samp, cif, co, ch, pl), f, d in zip(loaded, frac, dens):
        rec = {"target_id": tid, "seed": seed, "sample_idx": samp}
        if want_cons:
            rec["consensus"] = f
            rec["consensus_density"] = d
        if co is not None:
            if want_coord:
                cs = compute_coord_signals(co, ch, plddt=pl, antibody_chains=antibody_chains)
                rec.update(i_plddt=cs.i_plddt, n_interface_residues=cs.n_interface_residues,
                           n_interface_contacts=cs.n_interface_contacts, clash_ca=cs.clash_ca)
            if want_pair:
                jp = _summary_json_from_cif(cif, samp) if cif else None
                if jp and os.path.isfile(jp):
                    try:
                        sp = compute_summary_pair_signals(
                            json.load(open(jp)), list(dict.fromkeys(ch)), antibody_chains=antibody_chains)
                        rec["min_pair_iptm"] = sp.min_pair_iptm
                        rec["avg_pair_iptm"] = sp.avg_pair_iptm
                    except Exception:                  # noqa: BLE001, a bad JSON just leaves NaN
                        pass
        if want_bio and cif:                           # heavy-atom biophysical bundle (reads the cif)
            bp = compute_biophys(cif, antibody_chains=antibody_chains)
            if bp.error is None:
                rec.update(interface_bsa=bp.interface_bsa, bsa_hydrophobic_frac=bp.bsa_hydrophobic_frac,
                           n_salt_bridges=bp.n_salt_bridges, n_hbond_contacts=bp.n_hbond_contacts)
        if want_cdr and cif:                           # CDR/paratope focus (precomputed per-target map)
            cd = compute_cdr_paratope_4a_heavy(cif, antibody_chains, cdr_by_chain)
            if cd.error is None:
                rec.update(paratope_cdr_fraction=cd.paratope_cdr_fraction,
                           n_cdr_contacts=cd.n_cdr_contacts, n_paratope_residues=cd.n_paratope_residues)
        out.append(rec)
    return out


def augment_signals(
    df,
    groups,
    antibody_chains,
    density_sigma: float = DEFAULT_DENSITY_SIGMA_A,
    threads: int = 8,
    cdr_map: Optional[Dict[str, Dict[str, list]]] = None,
    force: bool = False,
):
    """Fill the selected cif/JSON verifier-signal columns on an existing per_sample DataFrame."""
    import pandas as pd
    requested = set(groups)
    # 'interface' can only come from the PAE path, report its state rather than silently dropping it.
    if "interface" in requested:
        if _group_already_computed(df, "interface"):
            logger.info("augment_signals: 'interface' already computed (ipsae present) -> skip")
        else:
            logger.warning("augment_signals: 'interface' needs the PAE (score_samples / "
                           "--from-predictions); not computable here -> skip")
    work = {g for g in requested if g in ("coord", "pair_iptm", "consensus", "biophys", "cdr")}
    if "cdr" in work and not cdr_map:
        logger.warning("augment_signals: 'cdr' requested but no cdr_map given; skipping cdr")
        work.discard("cdr")
    if not force:                                          # skip groups already computed (idempotent)
        already = {g for g in work if _group_already_computed(df, g)}
        for g in sorted(already):
            logger.info("augment_signals: '%s' already computed (%s present) -> skip "
                        "(use force=True to recompute)", g, _GROUP_PRIMARY_COL[g])
        work -= already
    if not work:
        logger.info("augment_signals: no cif/JSON group left to compute (all present or interface-only)")
        return df
    cdr_map = cdr_map or {}
    payloads = []
    for tid, sub in df.groupby("target_id"):
        rows = [(int(r.seed), int(r.sample_idx), r.cif_path)
                for r in sub.itertuples(index=False) if isinstance(r.cif_path, str) and r.cif_path]
        if rows:
            payloads.append((tid, rows, list(antibody_chains), density_sigma, work, cdr_map.get(tid)))
    if not payloads:
        return df
    all_rows: List[dict] = []
    if threads <= 1:
        for p in payloads:
            all_rows += _augment_signals_target(p)
    else:
        with ProcessPoolExecutor(max_workers=threads) as ex:
            for rows in ex.map(_augment_signals_target, payloads):
                all_rows += rows
    sig = pd.DataFrame(all_rows).set_index(["target_id", "seed", "sample_idx"])
    df = df.set_index(["target_id", "seed", "sample_idx"])
    filled = [c for c in AUGMENT_SIGNAL_COLS if c in sig.columns]
    for col in filled:
        df[col] = sig[col]
    logger.info("augment_signals: filled %s for %d samples", ", ".join(filled), len(sig))
    return df.reset_index()
