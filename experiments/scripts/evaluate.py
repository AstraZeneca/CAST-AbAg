#!/usr/bin/env python
r"""Score predictions against ground truth. Backend-agnostic.

Scoring runs as two separate, independently-parallel stages:

* **Verifier stage**, ranks samples (ipSAE / ranking_score / ...) via
  ``method.verifier.score_samples``, a process pool over targets. Chosen with ``--rank-by``.
* **Metric stage**, scores structures against ground truth (DockQ) via
  ``method.metrics.score.score_metrics``, a thread pool over samples, since the metric binary
  releases the GIL. Chosen with ``--metrics``. Every metric is cached per (metric, sample), so
  re-runs skip recompute.

Two entry paths:

1. **From predictions** (default), both stages from a backend's raw outputs, writing
   ``per_sample.parquet`` + ``per_target.csv``. Needs ``--predictions-dir``.
2. **From an existing per_sample.parquet** (``--from-per-sample``), the verifier stage is already
   done, so only the metric stage runs and augments the parquet in place. This is the path the
   ``run_ablation`` pipeline uses, because its subsample directories carry compound ids that path 1
   cannot match.

``--summary`` adds the DockQ rank/oracle summary: mean and %acceptable for the verifier-selected
(rank) and best-sampled (oracle) structures, over all targets, an optional subset
(``--summary-subset hard.json``) and seed or sample prefixes (``--summary-seed-subsets 5,10,20``,
``--summary-sample-subsets``).

    python -m experiments.scripts.evaluate --from-per-sample \
        --results-dir  runs/<dataset>/<arm>/res \
        --target-index runs/<dataset>/target_index.csv \
        --backend protenix2 --rank-by ranking_score --metrics dockq \
        --dockq-bin /path/to/DockQ --threads 16 --summary
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from method.diffusion_side.best_of_n import select_argmax
from method.metrics import diversity_summary
from method.metrics.dockq import DEFAULT_ANTIBODY_CHAINS
from method.metrics.score import ALL_METRICS, MetricSpec, score_metrics
from models import get_backend
from results_io import SampleRecord, dump_metric, write_per_sample, write_per_target

# Verifiers selectable with --rank-by. Interface scores (computed together from PAE)
# + the pool-based consensus selector + native confidence fields off the summary JSON.
VERIFIER_CHOICES = ["ipsae", "pdockq2", "lis", "ilis", "consensus", "consensus_density",
                    "ranking_score", "iptm", "ptm", "plddt", "gpde"]

ACCEPTABLE, MEDIUM = 0.23, 0.49
# verifier signal groups selectable with --verifiers (each is computed as a unit)
_VERIFIER_GROUPS = ("interface", "coord", "pair_iptm", "consensus", "biophys", "cdr")
_IFACE_ALIASES = {"ipsae", "pdockq2", "lis", "ilis", "interface"}


def _rank_by_group(rank_by: str) -> Optional[str]:
    """The verifier signal group that owns the --rank-by column, or None for the native
    confidence fields (ranking_score/iptm/ptm/plddt/gpde) which ride on the sample and need
    no compute. Used so the selection column is always populated."""
    if rank_by in _IFACE_ALIASES:
        return "interface"
    if rank_by in ("consensus", "consensus_density"):
        return "consensus"
    return None


def _resolve_verifier_groups(args) -> Optional[set]:
    """Parse --verifiers / --all-verifiers into a set of signal groups, or None for the
    classic single-verifier (rank_by-driven) behaviour."""
    if getattr(args, "all_verifiers", False):
        return set(_VERIFIER_GROUPS)
    spec = getattr(args, "verifiers", None)
    if not spec:
        return None
    groups = set()
    for tok in spec.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t == "all":
            return set(_VERIFIER_GROUPS)
        if t in _IFACE_ALIASES:
            groups.add("interface")
        elif t in ("coord", "pair_iptm", "consensus", "biophys", "cdr"):
            groups.add(t)
        else:
            raise SystemExit(f"--verifiers: unknown group '{t}' (choose from "
                             "interface,coord,pair_iptm,consensus,biophys,cdr,all or ipsae/pdockq2/lis/ilis)")
    return groups or None
# per-sample metric name → per_sample.parquet column it fills
_METRIC_COL = {"dockq": "dockq_abag",
               "site": "site_dist_a", "contact": "contact_dist_a"}
_SAMPLE_COLS = ("dockq", "dockq_abag", "fnat_term", "irms_term", "lrms_term",
                "site_dist_a", "contact_dist_a", "contact_delta_native")
_DOCKQ_TERM_COLS = ("fnat_term", "irms_term", "lrms_term")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions-dir", help="Backend raw outputs (from-predictions mode).")
    p.add_argument("--target-index", required=True,
                   help="CSV from prepare_dataset (needs target_id, ground_truth_cif).")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--from-per-sample", action="store_true",
                   help="Skip the verifier stage; run only the metric stage on the existing "
                        "<results-dir>/per_sample.parquet (for run_ablation-collected runs).")
    p.add_argument("--method-label", default="unspecified")
    p.add_argument("--metrics", nargs="*", default=["dockq"],
                   choices=["dockq", "diversity", "site", "contact"])
    p.add_argument("--verifiers", default=None,
                   help="Comma list of verifier signal GROUPS to compute together in one "
                        "PAE-optional pass: interface (ipsae/pdockq2/lis/ilis), coord "
                        "(i_plddt/n_interface/clash), pair_iptm (min/avg_pair_iptm), consensus "
                        "(consensus_density), biophys (BSA/salt_bridges/hbonds, needs biotite), or "
                        "'all'. Individual names ipsae/pdockq2/lis/ilis "
                        "select the interface group. --rank-by then only picks the selection "
                        "column among the computed ones. Omit to keep the classic single-verifier "
                        "behaviour driven by --rank-by.")
    p.add_argument("--all-verifiers", action="store_true",
                   help="Alias for --verifiers all (compute every signal group).")
    p.add_argument("--rank-by", default="ranking_score", choices=VERIFIER_CHOICES,
                   help="Verifier for argmax selection. Default: ranking_score. Interface scores "
                        "(ipsae/pdockq2/lis/ilis) are computed together in one PAE pass; "
                        "consensus is the pool-based interface-RMSD selector; the rest are "
                        "native confidence fields read from the summary JSON.")
    p.add_argument("--ipsae-pae-cutoff", type=float, default=10.0)
    p.add_argument("--ipsae-dist-cutoff", type=float, default=10.0)
    p.add_argument("--ipsae-plddt-cutoff", type=float, default=0.0)
    p.add_argument("--ipsae-fallback-rank-by", default="ranking_score")
    p.add_argument("--consensus-density-sigma", type=float, default=4.0,
                   help="Gaussian width (Å) of the consensus-density kernel over antibody-pose "
                        "Cα RMSD (the consensus_density verifier).")
    # GROUND-TRUTH geometry maps. Both optional: without them `site_dist_a`, `contact_dist_a` and
    # `contact_delta_native` are simply null, so every existing invocation is byte-identical.
    p.add_argument("--true-epitope-map", default=None,
                   help="JSON {target_id: [token indices]} of the TRUE epitope. Enables the 'site' "
                        "and 'contact' metrics. ANALYSIS ONLY -- never wire this into a steering or "
                        "selection path.")
    p.add_argument("--paratope-map", default=None,
                   help="JSON {target_id: [token indices]} of the paratope/CDR. Needed for 'contact' "
                        "in addition to --true-epitope-map.")
    p.add_argument("--cdr-map", default=None,
                   help="JSON {target_id: {chain: [cdr_res_ids]}} for the 'cdr' group in "
                        "--from-per-sample mode (precompute with cdr.cdr_residues_from_cif where "
                        "ANARCI is installed). Not needed for --from-predictions (numbers inline).")
    p.add_argument("--hmmer-path", default=None,
                   help="Directory containing hmmscan for ANARCI, if not on PATH (cdr group, "
                        "from-predictions).")
    p.add_argument("--dockq-bin", default=None)
    p.add_argument("--antibody-chains", default=",".join(DEFAULT_ANTIBODY_CHAINS),
                   help="Comma-separated antibody chain ids; interfaces touching any other chain "
                        "count as antibody-antigen (dockq_abag). Default: H,L.")
    p.add_argument("--threads", type=int, default=16, help="Parallel workers for the metric stage.")
    p.add_argument("--force-recompute", action="store_true",
                   help="Ignore cached metric JSONs and recompute verifier groups even when "
                        "already present (default: skip already-computed metrics and verifiers).")
    p.add_argument("--only", nargs="*", default=None, help="Only score these target IDs.")
    # --- DockQ rank/oracle summary ---
    p.add_argument("--summary", action="store_true")
    p.add_argument("--summary-dockq", default="abag", choices=["abag", "total"])
    p.add_argument("--summary-subset", default=None, help="JSON list of target_ids → extra summary row.")
    p.add_argument("--summary-subset-name", default="subset")
    p.add_argument("--summary-seed-subsets", default=None,
                   help="Comma list of seed counts (e.g. '5,10,20') → one row per MSA-subsample prefix.")
    p.add_argument("--summary-sample-subsets", default=None,
                   help="Comma list of diffusion-sample counts (e.g. '1,5,10,50,100'). With "
                        "--summary-seed-subsets this emits the full seeds×samples grid (B0.b).")
    p.add_argument("--verbose", "-v", action="count", default=0)
    return p


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose > 0 else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.antibody_chains = tuple(c.strip() for c in args.antibody_chains.split(",") if c.strip())

    idx = pd.read_csv(args.target_index)
    if "target_id" not in idx.columns:
        raise SystemExit(f"--target-index missing 'target_id' column: {args.target_index}")
    idx = idx.set_index("target_id")
    if args.only:
        idx = idx.loc[[t for t in args.only if t in idx.index]]

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.from_per_sample:
        df = score_from_per_sample(idx, results_dir, args)
    else:
        if not args.predictions_dir:
            raise SystemExit("--predictions-dir is required unless --from-per-sample is set")
        df = score_from_predictions(idx, results_dir, args)

    if args.summary and "dockq" in args.metrics:
        summarize_dockq(df, results_dir, args)
    return 0


def _load_map(path):
    if not path:
        return None
    try:
        return json.load(open(path))
    except (OSError, ValueError) as e:
        logging.getLogger(__name__).warning("could not read map %s: %s", path, e)
        return None


def _native_contacts(idx, epi_map, par_map):
    """target_id -> paratope-to-true-epitope centroid distance ON THE NATIVE structure."""
    if not (epi_map and par_map):
        return None
    import numpy as np
    from method.metrics.site import native_ca_by_chain_order
    out = {}
    for t, row in idx.iterrows():
        ei, pi = epi_map.get(t), par_map.get(t)
        if not ei or not pi:
            continue
        nat = None
        for k in ("ground_truth_pdb", "native_pdb", "ground_truth_cif"):
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                nat = v.strip(); break
        if not nat:
            continue
        try:
            x = native_ca_by_chain_order(nat, row.get("chain_ids", ""))
            out[t] = float(np.linalg.norm(x[list(pi)].mean(0) - x[list(ei)].mean(0)))
        except Exception as e:                       # a reference we cannot read is simply absent
            logging.getLogger(__name__).warning("native contact for %s: %s", t, e)
    return out or None


def _metric_spec(args, results_dir: Path, idx=None) -> MetricSpec:
    epi = _load_map(getattr(args, "true_epitope_map", None))
    par = _load_map(getattr(args, "paratope_map", None) or getattr(args, "cdr_map", None))
    return MetricSpec(
        metrics=[m for m in args.metrics if m in ALL_METRICS],
        dockq_bin=args.dockq_bin,
        antibody_chains=args.antibody_chains, results_dir=results_dir,
        force_recompute=args.force_recompute,
        true_epitope_map=epi, cdr_map=par,
        native_contact=(_native_contacts(idx, epi, par) if idx is not None else None),
    )


def _gt_map(idx: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {t: (str(row.get("ground_truth_cif") or "").strip() or None) for t, row in idx.iterrows()}



def score_from_predictions(idx: pd.DataFrame, results_dir: Path, args) -> pd.DataFrame:
    from method.verifier import score_samples
    predictions_dir = Path(args.predictions_dir).resolve()
    backend = get_backend(args.backend, {})
    target_ids = list(idx.index)

    groups = set(_resolve_verifier_groups(args) or ())
    groups.add("interface")
    rbg = _rank_by_group(args.rank_by)
    if rbg:
        groups.add(rbg)
    ipsae_kwargs = {"pae_cutoff_A": args.ipsae_pae_cutoff, "dist_cutoff_A": args.ipsae_dist_cutoff,
                    "plddt_cutoff": args.ipsae_plddt_cutoff, "fallback_rank_by": args.ipsae_fallback_rank_by,
                    # shared by the interface + consensus verifiers so they use the SAME
                    # antibody/antigen split as the dockq_abag metric:
                    "antibody_chains": list(args.antibody_chains),
                    "density_sigma_A": args.consensus_density_sigma,
                    "compute_groups": groups}
    records: List[SampleRecord] = score_samples(
        backend, target_ids, predictions_dir, rank_by=args.rank_by,
        ipsae_kwargs=ipsae_kwargs, results_dir=results_dir, method_label=args.method_label)
    if not records:
        logging.warning("no predictions parsed under %s", predictions_dir)

    if "diversity" in args.metrics:
        by_target: Dict[str, List[str]] = {}
        for r in records:
            if r.cif_path:
                by_target.setdefault(r.target_id, []).append(r.cif_path)
        for tid, cifs in by_target.items():
            dump_metric(tid, "diversity", diversity_summary(cifs), results_dir)

    # --- metric stage (thread pool over samples) ---
    _attach_metrics(records, _gt_map(idx), results_dir, args, idx)

    write_per_sample(records, results_dir / "per_sample.parquet")
    df = pd.read_parquet(results_dir / "per_sample.parquet")
    write_per_target(df, results_dir / "per_target.csv")
    print(f"Wrote {len(records)} records → {results_dir}")
    return df


def _attach_metrics(records: List[SampleRecord], gt: Dict[str, Optional[str]],
                    results_dir: Path, args, idx=None) -> None:
    spec = _metric_spec(args, results_dir, idx)
    if not spec.metrics:
        return
    items = []
    for r in records:
        g = gt.get(r.target_id)
        if g and r.cif_path:
            items.append((f"{r.target_id}__seed{r.seed}__s{r.sample_idx}", r.cif_path, g))
    scored = score_metrics(items, spec, max_workers=args.threads)
    for r in records:
        m = scored.get(f"{r.target_id}__seed{r.seed}__s{r.sample_idx}")
        if not m:
            continue
        r.dockq = m.get("dockq", r.dockq)
        r.dockq_abag = m.get("dockq_abag", r.dockq_abag)
        r.site_dist_a = m.get("site_dist_a", r.site_dist_a)
        r.contact_dist_a = m.get("contact_dist_a", r.contact_dist_a)
        r.contact_delta_native = m.get("contact_delta_native", r.contact_delta_native)



def _backfill_dockq_terms(df: pd.DataFrame, spec, threads: int = 8) -> pd.DataFrame:
    """Fill fnat/irms/lrms terms for already-DockQ-scored rows that lack them, from the cached
    DockQ stdout only (never a DockQ re-run). The terms are a pure function of the persisted
    stdout, so a parquet scored before the term columns existed is upgraded for free."""
    from concurrent.futures import ThreadPoolExecutor
    from method.metrics.score import _cache_path
    from method.metrics.dockq import antigen_mean_dockq_terms
    if spec.results_dir is None:
        return df
    for col in _DOCKQ_TERM_COLS:
        if col not in df.columns:
            df[col] = np.nan
    needy = df["dockq_abag"].notna() & df[list(_DOCKQ_TERM_COLS)].isna().any(axis=1)
    todo = df[needy]
    if not len(todo):
        return df
    logging.info("dockq-terms backfill: %d rows from cached stdout (no DockQ rerun)", len(todo))

    def _one(row):
        tag = f"{row.target_id}__seed{int(row.seed)}__s{int(row.sample_idx)}"
        p = _cache_path(spec, "dockq", tag)
        if p is None or not p.is_file():
            return None
        try:
            stdout = json.load(open(p)).get("stdout") or ""
        except (OSError, json.JSONDecodeError):
            return None
        t = antigen_mean_dockq_terms(stdout, spec.antibody_chains)
        return (row.Index, t) if t else None

    filled = 0
    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        for res in ex.map(_one, todo.itertuples()):
            if res:
                idx_, t = res
                for col in _DOCKQ_TERM_COLS:
                    df.at[idx_, col] = t[col]
                filled += 1
    logging.info("dockq-terms backfill: filled %d/%d (rest: no cache)", filled, len(todo))
    return df


def score_from_per_sample(idx: pd.DataFrame, results_dir: Path, args) -> pd.DataFrame:
    ps = results_dir / "per_sample.parquet"
    if not ps.is_file():
        raise SystemExit(f"--from-per-sample: no per_sample.parquet under {results_dir}")
    df = pd.read_parquet(ps)
    for col in _SAMPLE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    spec = _metric_spec(args, results_dir, idx)
    if spec.metrics:
        gt = _gt_map(idx)
        valid = [t for t, g in gt.items() if g]
        need = df.cif_path.notna() & df.target_id.isin(valid)
        cols = [_METRIC_COL[m] for m in spec.metrics if m in _METRIC_COL]
        if not args.force_recompute and cols:
            missing = None
            for c in cols:
                if c not in df.columns:
                    missing = need.copy() if missing is None else missing
                    missing = missing | need           # column absent entirely -> every row needs it
                    continue
                m_c = df[c].isna()
                missing = m_c if missing is None else (missing | m_c)
            if missing is not None:
                need &= missing
        primary = "dockq_abag" if "dockq" in spec.metrics else _METRIC_COL.get(spec.metrics[0], "dockq_abag")
        todo = df[need][["target_id", "seed", "sample_idx", "cif_path"]].drop_duplicates("cif_path")
        logging.info("from-per-sample: %d/%d samples need metrics (rest cached)", len(todo), len(df))
        if len(todo):
            items = [(f"{r.target_id}__seed{r.seed}__s{r.sample_idx}", r.cif_path, gt[r.target_id])
                     for r in todo.itertuples(index=False)]
            scored = score_metrics(items, spec, max_workers=args.threads)
            tag2cif = {it[0]: it[1] for it in items}
            cif2m = {tag2cif[tag]: m for tag, m in scored.items()}
            for col in _SAMPLE_COLS:
                df[col] = [cif2m[c][col] if (c in cif2m and cif2m[c].get(col) is not None) else cur
                           for c, cur in zip(df.cif_path, df[col])]
        # DockQ decomposition terms are free from the cached DockQ stdout: backfill them for any
        # already-scored row that predates the term columns, reading the cache only (never a rerun).
        if "dockq" in spec.metrics and not args.force_recompute:
            df = _backfill_dockq_terms(df, spec)

    groups = _resolve_verifier_groups(args)
    if groups:
        from method.verifier import augment_signals   # cif/JSON verifier signals (PAE-optional)
        cdr_map = json.load(open(args.cdr_map)) if (args.cdr_map and "cdr" in groups) else None
        df = augment_signals(df, groups, list(args.antibody_chains),
                             density_sigma=args.consensus_density_sigma, threads=args.threads,
                             cdr_map=cdr_map, force=args.force_recompute)

    df.to_parquet(ps, index=False)
    write_per_target(df, results_dir / "per_target.csv")
    print(f"Augmented {len(df)} records → {results_dir}")
    return df



def _subsample_index(cif_path, seed) -> int:
    """Trunk-latent index for the seed-subset sweep. Prefer the record's canonical
    ``seed``, run_ablation sets it to the MSA-subsample index (A1) or the latent-variant
    index (A2). The cif_path ``_s{N}`` is only the MSA-subsample suffix, so for A2 pure
    cells (one MSA subsample × K latent variants) it is ``_s0`` on every path and would
    collapse the whole K_latent sweep to one bucket. Fall back to parsing it only when
    ``seed`` is missing."""
    if seed is not None:
        try:
            return int(seed)
        except (TypeError, ValueError):
            pass
    m = re.search(r"_s(\d+)\b", str(cif_path) or "")
    return int(m.group(1)) if m else 0


def _rank_oracle(sub: pd.DataFrame, rank_col: str, dockq_col: str) -> Optional[Dict[str, float]]:
    ranks, oracles = [], []
    for _, g in sub.groupby("target_id"):
        g = g.dropna(subset=[dockq_col])
        if g.empty:
            continue
        # rank = the Best-of-N pick's DockQ (pure verifier argmax, NaN last, no fallback).
        # oracle = best DockQ sampled.
        best = select_argmax(g, rank_by=rank_col, fallback=None)
        ranks.append(best[dockq_col])
        oracles.append(g[dockq_col].max())
    if not ranks:
        return None
    r, o = np.asarray(ranks, float), np.asarray(oracles, float)
    return {"n": len(r),
            "rank_mean": r.mean(), "rank_acc": (r >= ACCEPTABLE).mean(), "rank_med": (r >= MEDIUM).mean(),
            "oracle_mean": o.mean(), "oracle_acc": (o >= ACCEPTABLE).mean(), "oracle_med": (o >= MEDIUM).mean()}


def summarize_dockq(df: pd.DataFrame, results_dir: Path, args) -> Path:
    dockq_col = "dockq_abag" if args.summary_dockq == "abag" else "dockq"
    if dockq_col not in df or df[dockq_col].notna().sum() == 0:
        if dockq_col == "dockq_abag" and "dockq" in df and df["dockq"].notna().any():
            logging.warning("no dockq_abag values, falling back to total dockq for the summary")
            dockq_col = "dockq"
        else:
            logging.warning("no DockQ values to summarise, skipping summary")
            return results_dir / "dockq_summary.csv"
    rank_col = args.rank_by if args.rank_by in df else "ranking_score"

    rows: List[Dict] = []

    def _add(label: str, sub: pd.DataFrame) -> None:
        st = _rank_oracle(sub, rank_col, dockq_col)
        if st:
            rows.append({"subset": label, **st})

    subset_ids = None
    if args.summary_subset:
        if os.path.isfile(args.summary_subset):
            subset_ids = set(json.load(open(args.summary_subset)))
        else:
            logging.warning("--summary-subset %s does not exist: skipping the %r summary row. It is "
                            "an extra row, not a metric, and every other result is unaffected.",
                            args.summary_subset, args.summary_subset_name)

    _add("all", df)
    if subset_ids is not None:
        _add(args.summary_subset_name, df[df.target_id.isin(subset_ids)])

    def _parse(s):
        return [int(x) for x in s.split(",") if x.strip()] if s else []
    seed_ks = _parse(args.summary_seed_subsets)
    sample_ms = _parse(args.summary_sample_subsets)
    if seed_ks or sample_ms:
        df = df.copy()
        df["_k"] = [_subsample_index(c, s) for c, s in zip(df.cif_path, df.seed)]
        ks = sorted(df["_k"].unique())
        has_sidx = "sample_idx" in df.columns

        def _sub(S, M):
            d = df
            if S is not None:
                d = d[d["_k"].isin(set(ks[:S]))]           # first S MSA-subsample seeds
            if M is not None and has_sidx:
                d = d[d["sample_idx"] < M]                 # first M diffusion samples/seed
            return d

        # seed-only ('seeds<=S'), sample-only ('samp<=M'), or the full seeds×samples grid
        # ('seeds<=S x samp<=M', B0.b). Each cell is also crossed with the hard subset if given.
        for S in (seed_ks or [None]):
            for M in (sample_ms or [None]):
                parts = ([f"seeds<={S}"] if S is not None else []) + ([f"samp<={M}"] if M is not None else [])
                label = " x ".join(parts)
                sub = _sub(S, M)
                _add(label, sub)
                if subset_ids is not None:
                    _add(f"{args.summary_subset_name}+{label}", sub[sub.target_id.isin(subset_ids)])

    out = results_dir / "dockq_summary.csv"
    summ = pd.DataFrame(rows, columns=["subset", "n", "rank_mean", "rank_acc", "rank_med",
                                       "oracle_mean", "oracle_acc", "oracle_med"])
    summ.to_csv(out, index=False)
    print(f"\nDockQ summary ({dockq_col}, rank by {rank_col}) → {out}")
    for _, r in summ.iterrows():
        print(f"  [{r['subset']:28s} n={int(r['n']):2d}] "
              f"rank {r['rank_mean']:.3f} · {100*r['rank_acc']:.0f}% acc · {100*r['rank_med']:.0f}% med  |  "
              f"oracle {r['oracle_mean']:.3f} · {100*r['oracle_acc']:.0f}% acc · {100*r['oracle_med']:.0f}% med")
    return out


if __name__ == "__main__":
    sys.exit(main())
