#!/usr/bin/env python
"""Run one ablation cell end-to-end.

Loads a YAML config, resolves the backend via the model registry, invokes
inference, collects per-sample records, writes them to Parquet + CSV.

Usage:
    python -m experiments.scripts.run_ablation --config experiments/configs/baseline/b0b.yaml
    python -m experiments.scripts.run_ablation --config ... --set model.backend=opendde
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _write_timing(results_dir: Path, label: str, phases: Dict[str, float], total_sec: float,
                  compute_budget: Optional[Dict[str, Any]] = None) -> None:
    """Persist per-phase + total wall-clock **and compute budget** for one cell."""
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cell": label,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_sec": round(total_sec, 2),
        "phases_sec": {k: round(v, 2) for k, v in phases.items()},
        "compute_budget": compute_budget,      # {trunk, diffusion, total}, None if not run
    }
    with open(results_dir / "timing.json", "w") as f:
        json.dump(payload, f, indent=2)
    log = logging.getLogger(__name__)
    log.info("timing[%s]: total=%.1fs (%s)", label, total_sec,
             ", ".join(f"{k}={v:.1f}s" for k, v in phases.items()))
    if compute_budget:
        b = compute_budget
        log.info("compute[%s]: trunk=%d runs (%.2e FLOP)  |  diffusion=%d structures (%.2e FLOP)  |  total=%.2e FLOP",
                 label, b.get("n_trunk_runs", 0), b.get("flops_trunk", 0.0),
                 b.get("n_structures", 0), b.get("flops_diffusion", 0.0), b.get("flops_total", 0.0))

from config import dump_effective_config, load_config, resolve_paths
from data.msa import MSASubsampleSpec
from method.diffusion_side.diffusion_search import DIFFUSION_FAMILIES
from method.verifier import score_samples
from method.trunk_side import build_msa_subsample_inputs
from method.trunk_side.latent_modulation import LatentModulationSpec
from models import get_backend
from models.base import build_run_spec_from_config
from results_io import write_per_sample, write_per_target


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="YAML config for this ablation cell.")
    p.add_argument("--set", dest="overrides", nargs="*", action="append", default=None,
                   help="Dotlist overrides, e.g. model.backend=opendde paths.results_dir=/x. "
                        "Repeatable: multiple --set flags accumulate (each may carry several key=val).")
    p.add_argument("--dry-run", action="store_true", help="Build inputs + print backend cmd; skip inference.")
    p.add_argument("--skip-inference", action="store_true", help="Assume predictions already exist.")
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose > 0 else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # --set is action="append" with nargs="*" → list of lists, flatten to a dotlist.
    overrides = [kv for group in (args.overrides or []) for kv in group]
    cfg = load_config(args.config, overrides=overrides or None)
    t_start = time.perf_counter()          # cell wall-clock (see timing.json)
    phases: Dict[str, float] = {}
    paths = resolve_paths(cfg)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    dump_effective_config(cfg, paths.results_dir / "config.yaml")

    # ---------------------------------------------------- backend
    backend_name = cfg.model.backend
    backend_cfg = dict(cfg.model.get(backend_name, {}) or {})
    backend = get_backend(backend_name, backend_cfg)

    inputs_dir = paths.inputs_dir
    latent_mod = cfg.method.get("latent_modulation")
    diffusion_side = str(cfg.method.get("diffusion_side", "best_of_n"))
    method_label = _method_label(cfg)

    if latent_mod is not None and str(latent_mod.get("kind", "none")) != "none":
        _t = time.perf_counter()
        rc = _run_latent_modulation(cfg, paths, backend, method_label, args)
        if not args.dry_run:
            _write_timing(paths.results_dir, method_label,
                          {"latent_modulation": time.perf_counter() - _t},
                          time.perf_counter() - t_start)
        return rc
    diffusion_search_spec = None
    if diffusion_side in DIFFUSION_FAMILIES and diffusion_side != "best_of_n":
        from method.diffusion_side.diffusion_search import spec_from_dict
        dsd = dict(cfg.method.get("diffusion_search", {}) or {})
        algo = str(dsd.get("algorithm", "") or "")
        allowed = DIFFUSION_FAMILIES[diffusion_side]
        if algo not in allowed:
            raise ValueError(
                f"method.diffusion_search.algorithm={algo!r} is not in family "
                f"{diffusion_side!r} ({' | '.join(allowed)}). The family groups by mechanism and the "
                "algorithm names the method; a mismatch would run a different method than the label "
                "claims.")
        diffusion_search_spec = spec_from_dict(dsd)
        _k = int(getattr(diffusion_search_spec, "subsample_pool", 1) or 1)
        if _k > 1:
            _ms = cfg.model.run.setdefault("msa_subsample", {}) \
                if hasattr(cfg.model.run, "setdefault") else None
            if _ms is None:                       # attribute-style config object
                _ms = getattr(cfg.model.run, "msa_subsample", None)
            if _ms is not None:
                _cur = int((_ms.get("n_seeds") if hasattr(_ms, "get") else
                            getattr(_ms, "n_seeds", 0)) or 0)
                if _cur != _k:
                    if hasattr(_ms, "__setitem__"): _ms["n_seeds"] = _k
                    else: setattr(_ms, "n_seeds", _k)
                    logging.getLogger(__name__).warning(
                        "joint pool: model.run.msa_subsample.n_seeds %d -> %d to match "
                        "method.diffusion_search.subsample_pool. They MUST agree: the diffusion is "
                        "deferred to the K-th subsample and a mismatch never fires it.", _cur, _k)
    elif diffusion_side != "best_of_n":
        raise ValueError(
            f"method.diffusion_side={diffusion_side!r} not wired "
            f"({' | '.join(DIFFUSION_FAMILIES)}).")
    n_seeds, _, _, _, subsample_enabled = _seed_setup(cfg)
    _t = time.perf_counter()
    msa_manifest, did_subsample, inputs_dir = _build_seed_inputs(cfg, paths, backend)
    phases["build_inputs"] = time.perf_counter() - _t

    base_rng = int(cfg.model.get("seeds", [101])[0])
    rng_seeds = [base_rng] if subsample_enabled else [base_rng + i for i in range(max(1, n_seeds))]

    # ---------------------------------------------------- inference
    compute_budget: Optional[Dict[str, Any]] = None
    if not args.skip_inference:
        run_spec = build_run_spec_from_config(cfg.model)
        run_spec.seeds = rng_seeds        # explicit subsample ⇒ [1 fixed RNG], full MSA ⇒ n_seeds RNGs
        if diffusion_search_spec is not None:
            run_spec.extras["diffusion_search"] = diffusion_search_spec.to_dict()
        if args.dry_run:
            run_spec.extras["dry_run"] = True
        _t = time.perf_counter()
        record = backend.run(inputs_dir, paths.predictions_dir, run_spec)
        phases["inference"] = time.perf_counter() - _t
        compute_budget = record.get("compute_budget")   # trunk/diffusion/total split
        if not args.dry_run:
            _persist_inference_wall(paths.predictions_dir, inputs_dir, phases["inference"])
        if compute_budget is not None:                   # reuse the trunk numbers from compute_budget
            _report_diffusion_compute(paths, int(cfg.model.run.get("n_cycles", 10)), compute_budget,
                                      backend=str(cfg.model.get("backend", "protenix2")),
                                      measured_wall_s=phases["inference"])
        if args.dry_run:
            print(f"[DRY RUN] Command written to {paths.predictions_dir}/run_record.json")
            return 0
        if record.get("returncode", 0) != 0:
            print(f"Backend {backend.name} returned {record.get('returncode')}", file=sys.stderr)
            return int(record.get("returncode") or 1)
    else:
        try:
            run_spec = build_run_spec_from_config(cfg.model)
            run_spec.seeds = rng_seeds
            run_spec.extras["dry_run"] = True            # analytical only -- builds the budget, runs nothing
            compute_budget = backend.run(inputs_dir, paths.predictions_dir, run_spec).get("compute_budget")
        except Exception as e:                           # noqa: BLE001 -- budget must never break the merge
            logging.getLogger(__name__).warning(
                "skip-inference: could not rebuild compute_budget (%s); results will omit it", e)

    if compute_budget is not None:
        inf_wall = _read_inference_wall(paths.predictions_dir)   # reliable measured device-seconds (total)
        if inf_wall:
            compute_budget.update(inf_wall)
        stage_wall, stage_counts = _read_stage_wall(paths.predictions_dir)
        if stage_wall:
            fwd = float(stage_wall.get("model_forward_cpu_span_s") or 0.0)
            total = float(compute_budget.get("inference_device_s") or 0.0)
            if total > 0 and fwd > 0:
                stage_wall["outside_forward_s"] = round(total - fwd, 2)
                stage_wall["_note"] = (
                    "*_cpu_span_s are UNSYNCHRONISED CPU intervals, not stage attributions. "
                    "trunk_measured_s and confidence_measured_s ARE synchronised and are the ones "
                    "to compare across methods (both are emitted for steered and unsteered runs "
                    "alike). Identity: model_forward_cpu_span_s + outside_forward_s = "
                    "inference_device_s")
            compute_budget["wall_by_stage_s"] = stage_wall
        if stage_counts:
            compute_budget["nfe_by_stage"] = stage_counts        # counts, NOT seconds

    # ---------------------------------------------------- collect
    _t = time.perf_counter()
    target_ids = _list_target_ids(inputs_dir, backend)
    ipsae_kwargs = _ipsae_kwargs_from_cfg(cfg)
    records = score_samples(
        backend, target_ids, paths.predictions_dir, rank_by=cfg.verifier.rank_by,
        ipsae_kwargs=ipsae_kwargs, results_dir=paths.results_dir, method_label=method_label,
    )
    if did_subsample:
        _remap_a1_records(records, method_label, msa_manifest)
    else:
        # Full-MSA mode: seeds are diffusion RNG (kept as-is), just stamp the label.
        for r in records:
            r.method = method_label

    write_per_sample(records, paths.results_dir / "per_sample.parquet")
    df = pd.read_parquet(paths.results_dir / "per_sample.parquet")
    write_per_target(df, paths.results_dir / "per_target.csv")
    phases["collect_score"] = time.perf_counter() - _t
    if compute_budget is not None:
        compute_budget = dict(compute_budget)
        if did_subsample:
            k = _seed_setup(cfg)[0]
            n_targets = len(msa_manifest) if msa_manifest \
                else max(1, compute_budget["n_input_jobs"] // max(k, 1))
            compute_budget["n_targets"] = n_targets
            compute_budget["n_seeds"] = k                 # k MSA subsamples = trunk seeds
            compute_budget["seeds_are"] = "msa_subsamples(k)"
        else:
            compute_budget["n_targets"] = compute_budget["n_input_jobs"]
            compute_budget["seeds_are"] = "diffusion_rng"
    _write_timing(paths.results_dir, method_label, phases, time.perf_counter() - t_start,
                  compute_budget=compute_budget)
    print(f"Wrote {len(records)} sample records to {paths.results_dir}")
    return 0


def _ipsae_kwargs_from_cfg(cfg) -> Dict[str, Any]:
    """Verifier kwargs for make_verifier. Shared across the interface scores
    (ipsae/pdockq2/lis/ilis) and the consensus selector; keys not relevant to the
    chosen rank_by are ignored downstream."""
    v = cfg.get("verifier", {}) if hasattr(cfg, "get") else {}
    sub = v.get("ipsae") or {}
    kw: Dict[str, Any] = {
        "pae_cutoff_A": float(sub.get("pae_cutoff_A", 10.0)),
        "dist_cutoff_A": float(sub.get("dist_cutoff_A", 10.0)),
        "plddt_cutoff": float(sub.get("plddt_cutoff", 0.0)),
        "fallback_rank_by": str(sub.get("fallback_rank_by", "ranking_score")),
        "density_sigma_A": float(v.get("consensus", {}).get("density_sigma_A", 4.0)),
        "compute_groups": list(v.get("compute_groups", ["interface"])),
    }
    ab = v.get("antibody_chains")
    if ab:
        kw["antibody_chains"] = list(ab)
    return kw


def _method_label(cfg) -> str:
    """Descriptor over the composed axes: (inherent MSA resampling [+ latent mod]) × diffusion."""
    lm = cfg.method.get("latent_modulation")
    if lm is not None and str(lm.get("kind", "none")) != "none":
        return str(cfg.method.get("label", f"a2_{lm.kind}"))
    # trunk = inherent MSA resampling ("fullMSA" when disabled), diffusion = best_of_n / guidance.
    diffusion_side = str(cfg.method.get("diffusion_side", "best_of_n"))
    n_seeds, spec, _, _, enabled = _seed_setup(cfg)
    n_sample = int(cfg.model.run.n_samples_per_seed)
    trunk = f"{spec.policy}@d{spec.depth}" if enabled else "fullMSA"
    return f"msa_resample[{trunk},seeds={n_seeds}]+{diffusion_side}[samples={n_sample}]"


def _latent_modulation_spec_from_cfg(cfg) -> LatentModulationSpec:
    m = cfg.method.latent_modulation
    kind = str(m.kind)
    common = dict(kind=kind, seed=int(m.get("seed", 0)))
    if kind == "rescale":
        return LatentModulationSpec(
            beta=float(m.get("beta", 0.0)),
            beta_grid=list(m.get("beta_grid", [-0.6, -0.3, 0.0, 0.3, 0.6])),
            **common,
        )
    if kind == "conformets":
        return LatentModulationSpec(
            channel_dim=int(m.get("channel_dim", 128)),
            init_noise_std=float(m.get("init_noise_std", 1e-3)),
            k_nets=int(m.get("k_nets", 2)),
            lr=float(m.get("lr", 1e-3)),
            lr_halve_every=int(m.get("lr_halve_every", 5)),
            grad_clip_norm=float(m.get("grad_clip_norm", 10.0)),
            n_steps=int(m.get("n_steps", 20)),
            snapshot_steps=list(m.get("snapshot_steps", [5, 10, 15, 20])),
            **common,
        )
    return LatentModulationSpec(**common)   # "none"


def _run_latent_modulation(cfg, paths, backend, method_label: str, args) -> int:
    """A2 driver: run K trunk-latent variants (one z modulation each) + collect."""
    from dataclasses import asdict
    spec = _latent_modulation_spec_from_cfg(cfg)
    k_variants = int(cfg.method.get("k", 10))     # fallback; both A2 kinds derive K below
    if spec.kind == "rescale":
        # Pair-Representation Scaling: the K trunk latents are the β grid (Boltz-sample sweep).
        k_variants = len(spec.beta_grid)
        logging.info("A2.b rescale: K_latent = |beta_grid| = %d  (grid=%s)",
                     k_variants, spec.beta_grid)
    elif spec.kind == "conformets":
        k_variants = spec.k_nets * len(spec.snapshot_steps)
        logging.info("A2.c ConforNets: K_latent = k_nets(%d) × |snapshots|(%d) = %d",
                     spec.k_nets, len(spec.snapshot_steps), k_variants)

    # Record the concrete per-variant modulations for reproducibility.
    variants = [spec.resolve_for_seed(i) for i in range(k_variants)]
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.results_dir / "latent_modulation_spec.json", "w") as f:
        json.dump({"base_label": spec.label(), "base": asdict(spec), "k_variants": k_variants,
                   "variants": [{"seed": i, "label": v.label(), **asdict(v)}
                                for i, v in enumerate(variants)]}, f, indent=2)

    base_rng = int(cfg.model.get("seeds", [101])[0])
    _msa_manifest, did_subsample, inputs_dir = _build_seed_inputs(cfg, paths, backend)
    target_ids = _list_target_ids(inputs_dir, backend)
    ipsae_kwargs = _ipsae_kwargs_from_cfg(cfg)

    def _relabel(recs_i, i, variant):
        """Stamp the variant's records with the K_msa × K_latent seed convention (in place)."""
        for r in recs_i:
            r.method = method_label
            msa_k = 0
            if did_subsample and "__" in r.target_id:
                base, suffix = r.target_id.rsplit("__", 1)
                m = re.search(r"_s(\d+)", suffix)                # MSA-subsample index
                msa_k = int(m.group(1)) if m else 0
                r.target_id = base                               # drop the MSA-subsample suffix
            r.seed = msa_k * k_variants + i     # enumerate the K_msa × K_latent trunk latents
            r.notes = (r.notes + f"; msa_s{msa_k}+lat={variant.label()}").lstrip("; ")
        return recs_i

    all_records = []
    with ThreadPoolExecutor(max_workers=1) as scorer:
        pending = []  # (i, variant, Future[List[SampleRecord]])
        for i, variant in enumerate(variants):
            run_spec = build_run_spec_from_config(cfg.model)
            run_spec.seeds = [base_rng]     # hold diffusion RNG to one value (variant = the seed axis)
            run_spec.extras["latent_modulation"] = asdict(variant)
            if args.dry_run:
                run_spec.extras["dry_run"] = True
            variant_dump = paths.predictions_dir / f"variant_{i}"
            if not args.skip_inference:
                rec = backend.run(inputs_dir, variant_dump, run_spec)   # GPU (blocking)
                if args.dry_run:
                    logging.info("[DRY RUN] variant %d (%s) command → %s/run_record.json",
                                 i, variant.label(), variant_dump)
                    continue
                if rec.get("returncode", 0) != 0:
                    print(f"Backend {backend.name} variant {i} returned {rec.get('returncode')}",
                          file=sys.stderr)
                    return int(rec.get("returncode") or 1)
            # queue this variant's scoring; it runs while the NEXT variant infers on the GPU
            pending.append((i, variant, scorer.submit(
                score_samples, backend, target_ids, variant_dump,
                rank_by=cfg.verifier.rank_by, ipsae_kwargs=ipsae_kwargs,
                results_dir=paths.results_dir, method_label=method_label)))

        if args.dry_run:
            print(f"[DRY RUN] A2 {method_label}: {k_variants} variant commands written under "
                  f"{paths.predictions_dir}")
            return 0

        for i, variant, fut in pending:
            all_records.extend(_relabel(fut.result(), i, variant))

    write_per_sample(all_records, paths.results_dir / "per_sample.parquet")
    df = pd.read_parquet(paths.results_dir / "per_sample.parquet")
    write_per_target(df, paths.results_dir / "per_target.csv")
    if spec.kind == "conformets":
        _report_conformets_opt(paths, spec)            # aggregate the optimisation FLOPs + wall time
    print(f"Wrote {len(all_records)} A2 sample records ({k_variants} variants) to {paths.results_dir}")
    return 0


def _report_conformets_opt(paths, spec) -> None:
    """Aggregate the per-target ConforNets optimisation cost and write results_dir/conformets_opt.json."""
    jl = paths.predictions_dir / "_conformets_cache" / "opt_timing.jsonl"
    if not jl.exists():
        return
    recs = [json.loads(ln) for ln in open(jl) if ln.strip()]
    if not recs:
        return
    from models.base import estimate_conformets_opt_flops
    walls = [float(r["wall_s"]) for r in recs]
    fl = estimate_conformets_opt_flops([r["n_token"] for r in recs], spec.k_nets, spec.n_steps)
    report = {
        "n_targets_optimised": len(recs),
        "decode_passes_per_target": spec.k_nets * spec.n_steps,
        "opt_wall_total_s": sum(walls),
        "opt_wall_mean_per_target_s": sum(walls) / len(walls),
        **fl,
        "note": ("ConforNets optimisation OVERHEAD, extra FLOPs/time on top of the base "
                 "trunk+diffusion in timing.json; runs once per target (snapshot-cached). "
                 "rescale (A2.b) has no such term."),
    }
    with open(paths.results_dir / "conformets_opt.json", "w") as f:
        json.dump(report, f, indent=2)
    logging.info("ConforNets opt: %d targets, %.0fs total (%.1fs/target), +%.3e FLOPs",
                 len(recs), report["opt_wall_total_s"], report["opt_wall_mean_per_target_s"],
                 fl["flops_conformets_opt"])


def _persist_inference_wall(predictions_dir, inputs_dir, call_wall_s: float) -> None:
    """Record this call's MEASURED inference wall PER TARGET, written INSIDE each target's prediction
        dir (``<predictions_dir>/<tid>__*/inference_wall.json``). Living with the predictions makes wall
        RERUN-SAFE by construction: the checkpoint's ``rm -rf <tid>__*`` on a partial target deletes its
        wall too (so a rerun REPLACES it, no double-count), a completed target skipped on resubmit keeps
        it, and shard consolidation (which moves ``<tid>__*`` dirs) moves the wall with it.
    """
    import glob
    pd = str(predictions_dir)
    weights: Dict[str, float] = {}
    for f in glob.glob(os.path.join(str(inputs_dir), "*.json")):
        tid = os.path.basename(f)[:-5].split("__")[0]          # base target id (drop __random@..._sN)
        if tid in weights:
            continue
        try:
            d = json.load(open(f))
            seqs = d[0]["sequences"] if isinstance(d, list) else d["sequences"]
            n_tok = sum(len(v["sequence"]) for s in seqs if isinstance(s, dict)
                        for k, v in s.items() if isinstance(v, dict) and "sequence" in v)
        except Exception:                                       # noqa: BLE001
            n_tok = 1
        weights[tid] = float(max(1, n_tok)) ** 2
    tot = sum(weights.values()) or 1.0
    for tid, w in weights.items():
        dirs = sorted(glob.glob(os.path.join(pd, tid + "__*")))
        if not dirs:                                            # target produced nothing (OOM) -> no wall
            continue
        try:
            with open(os.path.join(dirs[0], "inference_wall.json"), "w") as wf:
                json.dump({"target": tid, "inference_sec": round(call_wall_s * w / tot, 3)}, wf)
        except Exception as e:                                  # noqa: BLE001 -- wall log must never break a run
            logging.getLogger(__name__).warning("could not persist wall for %s (%s)", tid, e)


def _read_inference_wall(predictions_dir) -> Dict[str, float]:
    """Sum the MEASURED per-target inference wall (``<tid>__*/inference_wall.json``) under
    ``predictions_dir`` -> total DEVICE-seconds, each target counted ONCE (its last successful run,
    since the file lives with -- and is cleaned/rewritten with -- the target's predictions). The
    reliable measured-wall source for the merged results; independent of Protenix internals."""
    total = 0.0
    n = 0
    for jf in Path(predictions_dir).rglob("inference_wall.json"):
        try:
            total += float(json.load(open(jf)).get("inference_sec", 0) or 0)
            n += 1
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            continue
    return {"inference_device_s": round(total, 2), "n_target_walls": n} if n else {}


def _read_stage_wall(predictions_dir) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Sum Protenix's per-(target,seed) stage timer into ``(stages, counts)``."""
    _VENDOR_SPANS = ("pairformer", "diffusion", "confidence", "model_forward")
    agg: Dict[str, float] = {}
    for jf in Path(predictions_dir).rglob("stage_wall.json"):
        try:
            r = json.load(open(jf))
        except (json.JSONDecodeError, OSError):
            continue
        for k, v in r.items():
            if k == "sample":
                continue
            try:
                agg[k] = agg.get(k, 0.0) + float(v)
            except (TypeError, ValueError):
                continue
    counts = {k: int(v) for k, v in agg.items() if k.endswith("_calls")}
    stages = {(f"{k}_cpu_span_s" if k in _VENDOR_SPANS else f"{k}_s"): round(v, 2)
              for k, v in agg.items() if k not in counts}
    return stages, counts


def _timing_records_for_this_attempt(paths) -> list:
    """The timing records belonging to the structures currently on disk, not every attempt's."""
    pred = Path(paths.predictions_dir)
    from results_io import structure_count
    n_cif = structure_count(pred)
    logs = sorted(pred.rglob("diffusion_timing.jsonl"), key=lambda q: q.stat().st_mtime, reverse=True)
    if not logs:
        return []
    per_log = []
    for jl in logs:
        rs = [json.loads(ln) for ln in open(jl) if ln.strip()]
        per_log.append((jl, rs, sum(int(r.get("n_structures_emitted") or 0) for r in rs)))
    total_all = sum(n for _j, _r, n in per_log)
    if len(per_log) == 1 or not n_cif or not total_all:
        return [r for _j, rs, _n in per_log for r in rs]
    kept, got = [], 0
    for jl, rs, n in per_log:
        if got >= n_cif:
            logging.info("compute: ignoring %s (%d record(s), %d structure(s)) -- a previous attempt; "
                         "the %d structures on disk are already accounted for", jl.parent.name,
                         len(rs), n, n_cif)
            continue
        kept += rs
        got += n
    if got != n_cif:
        logging.warning("compute: kept %d record-structure(s) against %d cif(s) across %d stage dir(s)"
                        " -- the compute record may not match what is on disk", got, n_cif, len(per_log))
    return kept


def _report_diffusion_compute(paths, n_cycle: int = 10, compute_budget: Optional[dict] = None,
                              backend: str = "protenix2",
                              measured_wall_s: Optional[float] = None) -> None:
    """Aggregate the MEASURED diffusion-side search run into the COMPLETE three-lens compute budget
        → results_dir/diffusion_compute.json.
    """
    recs = _timing_records_for_this_attempt(paths)
    from models.base import arch_for, estimate_diffusion_search_flops, flops_totals
    cb = compute_budget or {}
    agg = estimate_diffusion_search_flops(recs, n_cycle=n_cycle, arch=arch_for(backend),
                                          trunk_flops=cb.get("flops_trunk"), trunk_nfe=cb.get("nfe_trunk"))
    report = {
        "kind": recs[0].get("kind") if recs else "best_of_n (no search records)",
        "n_targets": len(recs),
        **agg,
        "note": ("MEASURED three-lens compute budget (NFE / FLOPs / wall), split trunk vs denoiser "
                 "vs confidence-head (verifier-on-x̂₀). Confidence usually dominates FLOPs. Report "
                 "on the matched-NFE / matched-FLOPs axis (plan §0/§6.1)."),
        "wall_inference_phase_s": (float(measured_wall_s) if measured_wall_s is not None else None),
        "wall_total_s_scope": "stage-attributed (hook records only); compare on wall_inference_phase_s",
    }
    if compute_budget is not None:
        analytic_nfe = int(compute_budget.get("nfe_diffusion", 0))
        if recs:
            compute_budget["nfe_diffusion_measured"] = int(agg["denoiser_nfe"])
            compute_budget["nfe_diffusion_analytic"] = analytic_nfe
            compute_budget["confidence_output_nfe"] = int(agg["confidence_output_nfe"])
            compute_budget["flops_confidence_output"] = float(agg["flops_confidence_output"])
            compute_budget["flops_total_with_confidence"] = float(agg["flops_total"])
            compute_budget["flops_total_source"] = "analytic (separate-model n_structures)"
            compute_budget["flops_total_with_confidence_source"] = "measured (hook records)"
            compute_budget["flops_totals_are_not_subtractable"] = True
            _emitted = sum(int(r.get("n_structures_emitted") or 0) for r in recs)
            if _emitted:
                compute_budget["n_structures_emitted"] = _emitted
        else:
            compute_budget["nfe_diffusion_measured"] = analytic_nfe
            compute_budget["nfe_diffusion_analytic"] = analytic_nfe
            agg["denoiser_nfe"] = analytic_nfe
            agg["flops_denoiser"] = float(compute_budget.get("flops_diffusion", 0.0))
            agg["confidence_output_nfe"] = int(compute_budget.get("confidence_output_nfe", 0))
            agg["flops_confidence_output"] = float(compute_budget.get("flops_confidence_output", 0.0))
            # Re-total through the SHARED definition, never a second copy of the sum here.
            agg.update(flops_totals(agg["flops_trunk"], agg["flops_denoiser"],
                                    agg["flops_confidence"], agg["flops_confidence_output"]))
            compute_budget["flops_total_with_confidence"] = float(agg["flops_total"])
            report.update({k: agg[k] for k in (
                "denoiser_nfe", "flops_denoiser", "confidence_output_nfe",
                "flops_confidence_output", "flops_total_search", "flops_total")})

    with open(paths.results_dir / "diffusion_compute.json", "w") as f:
        json.dump(report, f, indent=2)
    logging.info("diffusion compute [%s]: %d tgt | NFE trunk=%d denoiser=%d conf=%d | "
                 "FLOPs=%.3e (conf %.0f%%) | wall=%.0fs", report["kind"], report["n_targets"],
                 agg["trunk_nfe"], agg["denoiser_nfe"], agg["confidence_nfe"], agg["flops_total"],
                 100 * agg["flops_confidence"] / max(agg["flops_total"], 1e-9), agg["wall_total_s"])


def _seed_setup(cfg) -> Tuple[int, MSASubsampleSpec, Optional[List[str]], Optional[Path], bool]:
    """Unified MSA-resampling knob for the whole sampling family (B0 / BoN / A1.*)."""
    run = cfg.model.run
    ms = (run.get("msa_subsample", {}) or {}) if hasattr(run, "get") else {}
    policy = str(ms.get("policy", "random"))
    enabled = policy != "none"           # policy: none ⇒ full MSA (no subsampling)
    spec = MSASubsampleSpec(
        policy=policy, depth=int(ms.get("depth", 1000)), seed=int(ms.get("seed", 0)),
        identity_threshold=float(ms.get("identity_threshold", 0.9)),
    )
    cf = list(ms.get("chain_filter")) if ms.get("chain_filter") else None
    base_dir = None
    if ms.get("source_inputs_dir"):
        p = Path(str(ms.source_inputs_dir)).expanduser()
        if not p.is_absolute():                          # resolve relative to the config dir, like paths:
            p = (Path(cfg["_meta"]["config_dir"]) / p).resolve()
        base_dir = p
    return int(ms.get("n_seeds", 1)), spec, cf, base_dir, enabled


def _build_workers() -> int:
    """Worker count for the parallel input-build. ``PROTEIN_TTS_BUILD_WORKERS`` overrides
    (set 1 to force serial); else min(16, cpu_count). Subsampling per target is independent
    and CPU-bound (e.g. cluster policy is expensive), so a process pool is a big win."""
    env = os.environ.get("PROTEIN_TTS_BUILD_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return min(16, os.cpu_count() or 2)


def _build_one_target(payload):
    """Build the k MSA subsamples for ONE target (top-level → picklable for the pool)."""
    backend, base_str, target_id, spec, k, out_str, chain_filter = payload
    _, info = build_msa_subsample_inputs(
        backend=backend, base_input_path=Path(base_str), target_id=target_id,
        spec=spec, k=k, out_dir=Path(out_str), chain_filter=chain_filter,
    )
    return target_id, info


def _build_seed_inputs(cfg, paths, backend) -> Tuple[Dict[str, Dict[str, Any]], bool, Path]:
    """Emit the per-seed inputs for the unified sampling family."""
    import csv
    n_seeds, spec, chain_filter, base_dir, enabled = _seed_setup(cfg)
    src = Path(base_dir) if base_dir else paths.inputs_dir
    if not enabled:
        return {}, False, src                                # full-MSA path (RNG seed scaling)

    out = paths.inputs_dir / "_seeded"
    out.mkdir(parents=True, exist_ok=True)
    sfx = _input_suffix(backend)
    for stale in out.glob(f"*{sfx}"):                    # avoid cross-cell pollution of _list_target_ids
        stale.unlink()
    inputs = [b for b in sorted(src.iterdir()) if b.is_file() and b.suffix == sfx]
    payloads = [(backend, str(base), base.stem, spec, n_seeds, str(out), chain_filter) for base in inputs]
    nw = _build_workers()
    if nw <= 1 or len(inputs) <= 1:
        results = [_build_one_target(p) for p in payloads]
    else:
        logging.info("building %d targets × k=%d subsamples with %d workers", len(inputs), n_seeds, nw)
        try:
            with ProcessPoolExecutor(max_workers=nw) as ex:
                results = list(ex.map(_build_one_target, payloads))   # order preserved
        except Exception as e:                                        # noqa: BLE001
            logging.warning("build pool failed (%s: %s); falling back to serial", type(e).__name__, e)
            results = [_build_one_target(p) for p in payloads]

    manifest: Dict[str, Dict[str, Any]] = {}
    for target_id, info in results:
        if not info["subsampled"]:
            logging.warning("MSA for %s ≤ depth %d (or none), subsampling no-op; 1 input "
                            "(K seeds identical). Flagged.", target_id, spec.depth)
        manifest[target_id] = {
            "target_id": target_id, "policy": spec.policy, "depth": spec.depth,
            "n_msa_rows": info["n_msa_rows"], "subsampled": info["subsampled"],
            "requested_k": n_seeds, "effective_k": info["effective_k"],
            "seeds_identical": (not info["subsampled"]),
        }

    _emitted = sorted(out.glob("*" + _input_suffix(backend))) if enabled else []
    if os.environ.get("TTS_RESUME_SEEDS") == "1" and _emitted:
        want = int(getattr(cfg.model.run, "n_samples_per_seed", 0) or 0)
        keep, skip = [], []
        for p in _emitted:
            d = paths.predictions_dir / Path(p).stem
            have = len(list(d.rglob("*.cif"))) if d.is_dir() else 0
            (skip if (want and have >= want) else keep).append((p, have))
        if skip and keep:
            for p, have in skip:
                Path(p).unlink(missing_ok=True)
                logging.info("resume: seed %s already has %d cifs, skipping its inference",
                             Path(p).stem, have)
            logging.info("resume: %d of %d seeds already complete, running the remaining %d",
                         len(skip), len(skip) + len(keep), len(keep))
        elif skip and not keep:
            logging.info("resume: every seed already complete (%d), running them anyway because an "
                         "empty input set would abort the unit", len(skip))

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    man_path = paths.results_dir / "msa_manifest.csv"
    with open(man_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target_id", "policy", "depth", "n_msa_rows",
                                          "subsampled", "requested_k", "effective_k", "seeds_identical"])
        w.writeheader()
        for row in manifest.values():
            w.writerow(row)
    n_flat = sum(1 for v in manifest.values() if not v["subsampled"])
    logging.info("MSA manifest → %s (%d/%d targets had no subsampling: seeds identical)",
                 man_path, n_flat, len(manifest))

    # EXTRA compute of the MSA-subsample build (CPU one-off, separate from the GPU/FLOPs budget).
    # Material only for the cluster policy (O(rows·centroids) Hamming per target); ~0 for random.
    build_walls = [float(info.get("msa_build_wall_s", 0.0)) for _, info in results]
    total_build = sum(build_walls)
    report = {
        "policy": spec.policy, "depth": spec.depth, "requested_k": n_seeds,
        "n_targets": len(results),
        "msa_build_wall_total_s": round(total_build, 3),
        "msa_build_wall_mean_per_target_s": round(total_build / max(1, len(results)), 4),
        "note": ("EXTRA one-off CPU cost of building the MSA subsamples (cluster = O(rows·centroids) "
                 "Hamming; random ≈ 0). NOT part of the GPU/FLOPs budget; report alongside it."),
    }
    with open(paths.results_dir / "msa_build_compute.json", "w") as f:
        json.dump(report, f, indent=2)
    logging.info("MSA build compute [%s]: %.1fs total (%.3fs/target) over %d targets",
                 spec.policy, total_build, report["msa_build_wall_mean_per_target_s"], len(results))
    return manifest, True, out


def _remap_a1_records(records, method_label: str, msa_manifest: Dict[str, Dict[str, Any]]) -> None:
    """Rewrite A1 records to the seed=MSA-subsample convention (in place)."""
    s_re = re.compile(r"_s(\d+)$")
    for r in records:
        r.method = method_label
        if "__" in r.target_id:
            base, label = r.target_id.rsplit("__", 1)   # rsplit: label has no "__", robust to "__" in base
            m = s_re.search(label)
            r.target_id = base
            if m:
                r.seed = int(m.group(1))          # MSA-subsample index = convention seed
        info = msa_manifest.get(r.target_id)
        if info and not info["subsampled"]:
            note = f"msa_not_subsampled(n_rows={info['n_msa_rows']}<=depth={info['depth']}): seeds_identical"
            r.notes = (r.notes + "; " + note).lstrip("; ") if r.notes else note


def _input_suffix(backend) -> str:
    """The input file extension THIS backend emits and reads."""
    return getattr(backend, "input_suffix", ".json")


def _list_target_ids(inputs_dir: Path, backend=None) -> List[str]:
    # The backend's own suffix only, matching its _merge_inputs glob, so colocated a3m/cif files
    # or a subdir (e.g. _seeded/msa/) never produce phantom target ids.
    return sorted(p.stem for p in inputs_dir.glob(f"*{_input_suffix(backend)}"))


if __name__ == "__main__":
    sys.exit(main())
