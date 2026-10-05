"""Standardised on-disk layout for results.

All ablation cells write into the same schema so blocks can be joined with a
plain SQL / pandas merge. The bookkeeping is the "one row per (target, method,
seed, sample_idx)" Parquet table.

Directory conventions::

    {results_dir}/
        config.yaml                          # dump of the effective config used
        per_sample.parquet                   # long-form: target x method x seed x sample_idx
        per_target.csv                       # aggregate: one row per (target, method)
        metrics/                             # raw metric outputs
            dockq/{target}.json
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd


STRUCTURE_GLOB = "*_sample_*.cif"
"""What counts as ONE finished structure on disk.

ONE DEFINITION BECAUSE FOUR PLACES COMPARE THEIR COUNTS AGAINST EACH OTHER: the attempt-aware
timing-record selection in `run_ablation`, the `compute-check` gate and the `compute-rederive`
rewrite in `jobs/tts_job_lib.py`, and the M4 table in `experiments/scripts/compute_table.py`. All
four test the identity `confidence_output_nfe == structure count`, so if any of them counted a
different set of files the gate would pass or fail for reasons unrelated to the data. Other globs in
the tree that happen to use the same pattern are unrelated lookups and are left alone.
"""


def structure_count(pred_dir) -> int:
    """Number of finished structures under a unit's predictions dir, counted the ONE way."""
    import glob as _glob
    import os as _os
    return len(_glob.glob(_os.path.join(str(pred_dir), "**", STRUCTURE_GLOB), recursive=True))


def pool_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The rows that are DRAWS OF THE METHOD, with the joint pool's trunk by-products dropped."""
    if "seed" not in df.columns or df.empty:
        return df
    sizes = df.groupby("seed").size()
    if len(sizes) < 2 or sizes.max() <= 1:
        return df
    stubs = set(sizes[sizes == 1].index)
    return df[~df["seed"].isin(stubs)] if stubs else df


def dump_dataclass_json(result, path: "str | Path", indent: int = 2, default=None) -> Path:
    """Persist a dataclass instance as JSON at ``path`` (creating parent dirs). Overwrites.
    Shared by the per-metric result dumpers (ipsae / dockq / lddt / tm), one place, not four."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=indent, default=default)
    return path


# NAMING: seed = MSA subsample (trunk-side), sample_idx = diffusion sample per
# seed (diffusion-noise side). Authoritative definition in models/base.py.
PER_SAMPLE_SCHEMA = [
    "target_id",
    "method",           # e.g. "best_of_n", "a1_msa_resample:cluster@32", "m2_tds:lambda=1.0"
    "seed",             # MSA-subsample identity
    "sample_idx",       # diffusion-sample index within this (target, method, seed)
    "cif_path",
    "ipsae",            # ipSAE interface verifier (Dunbrack Lab); computed post-hoc from PAE
    "pdockq2",          # PAE interface verifier (Zhu 2023), ab-ag mean over interfaces
    "lis",              # Local Interaction Score (Kim 2024), ab-ag mean over interfaces
    "ilis",             # integrated LIS (Kim 2024), contact-filtered
    "consensus",          # pool consensus: antibody-pose cluster-population fraction
    "consensus_density",  # pool consensus: Gaussian-weighted antibody-pose agreement
    "i_plddt",          # Phase A: mean interface-residue pLDDT (no PAE)
    "n_interface_residues",
    "n_interface_contacts",
    "clash_ca",         # cross-chain Cα-Cα clashes
    "min_pair_iptm",    # Phase 0: weakest ab-ag pairwise ipTM (free, from chain_pair_iptm)
    "avg_pair_iptm",    # Phase 0: mean ab-ag pairwise ipTM
    "interface_bsa",    # biophys: buried surface area (Ų, biotite)
    "bsa_hydrophobic_frac",
    "n_salt_bridges",
    "n_hbond_contacts",
    "paratope_cdr_fraction",
    "n_cdr_contacts",
    "n_paratope_residues",
    "plddt",
    "iptm",
    "ptm",
    "ranking_score",    # alternate verifier (Protenix native composite)
    "gpde",
    "disorder",
    "has_clash",
    "dockq",            # NaN if no ground truth: Total DockQ over all native interfaces
    "dockq_abag",       # mean DockQ over antibody-antigen interfaces only (doc metric)
    "fnat_term",        # DockQ decomposition (ab-ag mean): Fnat contribution
    "irms_term",        # DockQ decomposition (ab-ag mean): scaled iRMS term 1/(1+(iRMS/1.5)^2)
    "lrms_term",        # DockQ decomposition (ab-ag mean): scaled LRMS term 1/(1+(LRMS/8.5)^2)
    "lddt",
    "tm_score",
    "ca_rmsd",
    "notes",
]


@dataclass
class SampleRecord:
    target_id: str
    method: str
    seed: int          # MSA-subsample identity (trunk-side axis)
    sample_idx: int    # diffusion-sample index within this seed (diffusion-noise axis)
    cif_path: str
    ipsae: Optional[float] = None
    pdockq2: Optional[float] = None
    lis: Optional[float] = None
    ilis: Optional[float] = None
    consensus: Optional[float] = None
    consensus_density: Optional[float] = None
    i_plddt: Optional[float] = None
    n_interface_residues: Optional[int] = None
    n_interface_contacts: Optional[int] = None
    clash_ca: Optional[int] = None
    min_pair_iptm: Optional[float] = None
    avg_pair_iptm: Optional[float] = None
    interface_bsa: Optional[float] = None
    bsa_hydrophobic_frac: Optional[float] = None
    n_salt_bridges: Optional[int] = None
    n_hbond_contacts: Optional[int] = None
    # 4 A HEAVY-ATOM paratope, NOT the 10 A CA convention the steering CV, token_maps.py and the report's "Paratope on CDR" all use. See method/verifier/cdr.py.
    paratope_cdr_fraction: Optional[float] = None
    n_cdr_contacts: Optional[int] = None
    n_paratope_residues: Optional[int] = None
    plddt: Optional[float] = None
    iptm: Optional[float] = None
    ptm: Optional[float] = None
    ranking_score: Optional[float] = None
    gpde: Optional[float] = None
    disorder: Optional[float] = None
    has_clash: Optional[bool] = None
    dockq: Optional[float] = None
    dockq_abag: Optional[float] = None
    fnat_term: Optional[float] = None
    irms_term: Optional[float] = None
    lrms_term: Optional[float] = None
    lddt: Optional[float] = None
    tm_score: Optional[float] = None
    ca_rmsd: Optional[float] = None
    site_dist_a: Optional[float] = None
    contact_dist_a: Optional[float] = None
    contact_delta_native: Optional[float] = None
    notes: str = ""

    def as_row(self) -> Dict[str, Any]:
        return asdict(self)


def write_per_sample(records: Iterable[SampleRecord], out_path: str | Path) -> Path:
    """Append/replace the long-form Parquet. Existing rows for (target, method, seed, sample_idx) are overwritten."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([r.as_row() for r in records], columns=PER_SAMPLE_SCHEMA)

    if out_path.is_file():
        df_old = pd.read_parquet(out_path)
        key = ["target_id", "method", "seed", "sample_idx"]
        df_old = df_old.merge(df_new[key], on=key, how="left", indicator=True)
        df_old = df_old[df_old["_merge"] == "left_only"].drop(columns=["_merge"])
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(out_path, index=False)
    return out_path


def write_per_target(df_per_sample: pd.DataFrame, out_path: str | Path,
                     rank_by: str = "ranking_score") -> Path:
    """One row per (target, method): agg of the sample-level scores. The deployed
    (``selected_*``) structure is the Best-of-N pick, argmax of the verifier score."""
    from method.diffusion_side.best_of_n import select_argmax  # lazy: avoid import cycle

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = df_per_sample.groupby(["target_id", "method"], dropna=False)
    rows: List[Dict[str, Any]] = []
    for (target_id, method), g in grouped:
        # DROP THE JOINT POOL'S TRUNK BY-PRODUCTS. CAST emits N + K - 1, the K - 1 extras being
        # single unsteered trunk passes, not draws of the method. Applied PER GROUP, not to the
        # whole frame: a stub seed id in one target can coincide with a full seed id in another,
        # and a global pass would then keep it. Every aggregate below is over the 720, not the 723.
        g = pool_rows(g)
        # Best-of-N selection: argmax of the chosen verifier, ranking_score as fallback so
        # a sample missing the primary score (PAE not written / monomer) still lands somewhere.
        best = select_argmax(g, rank_by=rank_by, fallback="ranking_score")
        rows.append({
            "target_id": target_id,
            "method": method,
            "n_samples": len(g),
            # Best-of-K on the selector
            "best_ipsae": g["ipsae"].max(),
            "best_ranking_score": g["ranking_score"].max(),
            "best_iptm": g["iptm"].max(),
            "best_plddt": g["plddt"].max(),
            "best_dockq": g["dockq"].max(),
            "best_dockq_abag": g["dockq_abag"].max() if "dockq_abag" in g else None,
            "best_lddt": g["lddt"].max(),
            "best_tm": g["tm_score"].max(),
            "min_ca_rmsd": g["ca_rmsd"].min(),
            # ipSAE of the Best-of-N selected sample (argmax of rank_by, the *deployed* structure)
            "selected_ipsae": best.get("ipsae"),
            "selected_dockq": best.get("dockq"),
            "selected_dockq_abag": best.get("dockq_abag"),
            "selected_lddt": best.get("lddt"),
            "selected_tm": best.get("tm_score"),
            "selected_ca_rmsd": best.get("ca_rmsd"),
            "selected_cif": best.get("cif_path"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return out_path


def dump_metric(target_id: str, kind: str, payload: Mapping[str, Any], results_dir: str | Path) -> Path:
    """Persist a raw metric JSON, one file per (metric, target) so re-runs are cheap."""
    out = Path(results_dir) / "metrics" / kind / f"{target_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(payload), f, indent=2, default=str)
    return out
