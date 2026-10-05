#!/usr/bin/env python
r"""Load a dataset -> emit backend-native input files + a target index CSV.

The dataset loader and the backend are both selected by name, so a new
combination is a config change and not a code change.

Output paths are **auto-named from ``--dataset``**: unless overridden, inputs go
to ``{runs_root}/{dataset}/inputs`` and the index to
``{runs_root}/{dataset}/target_index.csv`` (runs_root defaults to ``runs``). This
mirrors the run configs, whose paths namespace by ``${env:PROTEIN_TTS_DATASET}``.

    python -m experiments.scripts.prepare_dataset \
        --dataset foldbench \
        --dataset-args '{"targets_dir": "$FOLDBENCH_ROOT/FoldBench/targets",
                         "tasks": ["interface_antibody_antigen"],
                         "ground_truth_dir": "$FOLDBENCH_ROOT/ground_truths"}' \
        --backend protenix2
        # -> writes runs/foldbench/inputs/*.json + runs/foldbench/target_index.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

from data.datasets import (CFA50Dataset, CoFoldArenaDataset, CollectedBadCasesDataset,
                           FoldBenchDataset, OpenDDE40Dataset, dump_target_index)
from data.datasets.base import DatasetLoader, check_biological_assembly
from models import get_backend


_LOADERS: Dict[str, type] = {
    "foldbench": FoldBenchDataset,
    "collected_bad_cases": CollectedBadCasesDataset,
    "cofoldarena": CoFoldArenaDataset,
    "cfa50": CFA50Dataset,
    "opendde40": OpenDDE40Dataset,
}


def _expand_env(v: Any) -> Any:
    if isinstance(v, str):
        return os.path.expandvars(v)
    if isinstance(v, list):
        return [_expand_env(x) for x in v]
    if isinstance(v, dict):
        return {k: _expand_env(x) for k, x in v.items()}
    return v


_RETIRED = {
    "cfa52": "cfa50 (12qj_ab1 and 8s4u_nb1 were removed on 2026-09-05, so the set is 50 targets, "
             "28 CoFold Arena + 22 collected bad cases). This is a membership change and not a "
             "rename, so the old key is refused rather than aliased: silently returning 50 targets "
             "under a name that says 52 would corrupt every count derived from it.",
}


def _make_loader(name: str, args_json: str) -> DatasetLoader:
    if name in _RETIRED:
        raise SystemExit(f"Dataset {name!r} no longer exists. Use {_RETIRED[name]}")
    if name not in _LOADERS:
        raise SystemExit(
            f"Unknown dataset: {name!r}. Registered: {sorted(_LOADERS)}. "
            f"Add a new loader under data/datasets/ and register it here."
        )
    args = _expand_env(json.loads(args_json)) if args_json else {}
    return _LOADERS[name](**args)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Retired keys stay in `choices` so _make_loader can explain what replaced them. Without this
    # argparse rejects them first, with a bare "invalid choice" that names no successor.
    p.add_argument("--dataset", required=True, choices=sorted(set(_LOADERS) | set(_RETIRED)))
    p.add_argument("--dataset-args", default="{}",
                   help="JSON dict of kwargs for the loader (env-vars expanded).")
    p.add_argument("--backend", required=True, help="Model backend name (e.g. protenix2, opendde).")
    p.add_argument("--backend-args", default="{}",
                   help="JSON dict of backend-specific config (e.g. model_name, component_path).")
    p.add_argument("--runs-root", default="runs",
                   help="Root for auto-named outputs (default: runs). Ignored if --out-dir is given.")
    p.add_argument("--out-dir", default=None,
                   help="Override: where to write per-target inputs. Default: {runs_root}/{dataset}/inputs.")
    p.add_argument("--index-csv", default=None,
                   help="Override: where to write target_index.csv. Default: {runs_root}/{dataset}/target_index.csv.")
    p.add_argument("--verify-assembly", action="store_true",
                   help="refuse any reference whose chains come from different biological assemblies")
    p.add_argument("--assembly-cache", default="~/.cache/rcsb_cif",
                   help="where the depositions fetched for --verify-assembly are cached")
    p.add_argument("--assemble", default=None, metavar="DATA_ROOT",
                   help="also materialise ground truth, wired inputs and merged maps under this "
                        "root (loaders that support it only)")
    p.add_argument("--gt-root", default=None,
                   help="where --assemble writes ground_truths/ (default: <DATA_ROOT>_GT)")
    p.add_argument("--targets-dir-out", default=None, metavar="DIR",
                   help="also write <DIR>/<target>/<target>.pdb for every loaded target, with "
                        "SEQRES. That tree is what --dataset-args '{\"targets_dir\": DIR}' reads, "
                        "so a later build needs no manifest and no per-half source dirs")
    p.add_argument("--gt-out-dir", default=None,
                   help="where a relabelled ground-truth tree is written, for loaders whose "
                        "references keep their deposited chain ids (default: {runs_root}/{dataset}"
                        "/ground_truths)")
    p.add_argument("--no-relabel-gt", action="store_true",
                   help="use the references as deposited instead of writing a relabelled tree")
    p.add_argument("--wired-dir", action="append", default=[],
                   help="source dir searched for <target>.json wired inputs; repeatable, in order")
    p.add_argument("--gt-dir", action="append", default=[],
                   help="source dir searched for an EXISTING reference before the loader's own; "
                        "repeatable, in order. Use it to keep the relabelled copy a result set was "
                        "actually scored against.")
    p.add_argument("--map", action="append", default=[], metavar="NAME=JSON[,JSON...]",
                   help="merge these maps across the halves into <DATA_ROOT>/maps/NAME.json; "
                        "repeatable")
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose > 0 else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    loader = _make_loader(args.dataset, args.dataset_args)
    backend_cfg = _expand_env(json.loads(args.backend_args)) if args.backend_args else {}
    backend = get_backend(args.backend, backend_cfg)

    # Auto-name outputs from --dataset unless explicitly overridden.
    run_dir = Path(args.runs_root).expanduser() / args.dataset
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir \
        else (run_dir / "inputs").resolve()
    index_csv = Path(args.index_csv).expanduser().resolve() if args.index_csv \
        else (run_dir / "target_index.csv").resolve()

    # ONE CHAIN-ID CONVENTION ACROSS DATASETS. A loader that exposes gt_out_dir has references
    # still carrying their deposited chain ids, which every antibody-by-chain-id consumer reads
    # wrongly (dockq_abag, true-epitope). Writing the relabelled H/L/A tree by default puts them in
    # the same shape the other datasets already ship, `<dir>/<target>/<target>.pdb`.
    if hasattr(loader, "gt_out_dir") and not args.no_relabel_gt and loader.gt_out_dir is None:
        loader.gt_out_dir = (Path(args.gt_out_dir).expanduser().resolve() if args.gt_out_dir
                             else (run_dir / "ground_truths").resolve())

    targets = loader.load()
    if args.targets_dir_out:
        from data.datasets.cofoldarena import write_targets_dir
        write_targets_dir(targets, Path(args.targets_dir_out).expanduser().resolve())
    if not targets:
        print("No targets loaded, check --dataset-args.", file=sys.stderr)
        return 1

    if args.verify_assembly:
        check_biological_assembly(targets, cache_dir=args.assembly_cache, strict=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    for t in targets:
        backend.emit_input(t, out_dir)
    if args.assemble:
        if not hasattr(loader, "assemble"):
            raise SystemExit(f"--assemble: the {args.dataset!r} loader does not define assemble()")
        data_root = Path(args.assemble).expanduser().resolve()
        if args.gt_root:
            gt_root = Path(args.gt_root).expanduser().resolve()
        else:
            # Strip ONE trailing "_data" only. `.replace("_data", "")` rewrote every occurrence
            # anywhere in the name, so a root called my_data_dir became my_dir_GT.
            stem = data_root.name[:-5] if data_root.name.endswith("_data") else data_root.name
            gt_root = data_root.with_name(stem + "_GT")
        maps = {}
        for spec in args.map:
            if "=" not in spec:
                raise SystemExit(f"--map {spec!r}: expected NAME=JSON[,JSON...]")
            name, paths = spec.split("=", 1)
            maps[name] = [os.path.expandvars(x) for x in paths.split(",") if x]
        rep = loader.assemble(targets, data_root, gt_root,
                              wired_dirs=[os.path.expandvars(d) for d in args.wired_dir],
                              gt_dirs=[os.path.expandvars(d) for d in args.gt_dir], maps=maps)
        print(f"  assembled   → {data_root}  (ground truth → {gt_root})")
        for k, v in rep.items():
            print(f"    {k:22s} {v}")

    # AFTER assemble, never before: assemble repoints each target at the reference it copied, and
    # the index has to name that copy for the assembled folder to be self-contained.
    dump_target_index(targets, index_csv)

    print(f"Prepared {len(targets)} targets ({args.dataset} → {args.backend})")
    print(f"  inputs      → {out_dir}")
    print(f"  target_index→ {index_csv}")
    print(f"  (run cells with PROTEIN_TTS_DATASET={args.dataset} so configs read/write runs/{args.dataset}/…)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
