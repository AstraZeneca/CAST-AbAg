#!/usr/bin/env python
r"""Build a full-MSA database for a dataset's inputs (MMseqs2 / ColabFold).

MSA generation is CPU/API-bound and SEPARATE from inference (the Protenix inference
runner does NOT generate MSAs, an input without an MSA predicts at N_msa=1). Run
this once to populate the DB. With ``--wire-out`` it also performs the DB->inputs step:
attach each input's per-chain MSA from the DB (query-match, via
``method.trunk_side.wire_msa_from_dump``) -> ready-to-subsample "wired" inputs that
experiments feed to ``build_msa_subsample_inputs``.

    python -m experiments.scripts.generate_msa \
        --inputs-dir runs/collected_bad_cases/inputs \
        --db-dir     runs/collected_bad_cases/msa_db \
        --wire-out   runs/collected_bad_cases/inputs_wired \
        --server-url https://api.colabfold.com \
        --jobs 10 --exclude '*-update-msa*'

Requires the ``protenix`` CLI on PATH (the Protenix venv) and a reachable MSA
server. On a CPU-only node keep the default ``LAYERNORM_TYPE=torch`` (set inside
``data.msa.generate``) so the CLI import does not trigger a CUDA-JIT compile.
"""
from __future__ import annotations

import argparse
import fnmatch
import logging
import sys
from pathlib import Path

from data.msa import DEFAULT_COLABFOLD_URL, generate_msa_db


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs-dir", required=True, help="Dir of Protenix input JSONs to search.")
    p.add_argument("--db-dir", required=True, help="Output MSA database dir ({db}/{target}/msa/…).")
    p.add_argument("--server-url", default=DEFAULT_COLABFOLD_URL,
                   help=f"MMseqs2/ColabFold server URL (default: {DEFAULT_COLABFOLD_URL}).")
    p.add_argument("--mode", default="colabfold", choices=["colabfold", "protenix"],
                   help="protenix mode hardwires its own server; use colabfold for a custom URL.")
    p.add_argument("--jobs", type=int, default=1, help="Concurrent protenix-msa subprocesses (API-bound).")
    p.add_argument("--exclude", default=None, help="Glob of input stems to skip (e.g. '*-update-msa*').")
    p.add_argument("--no-skip-existing", action="store_true", help="Regenerate even if already in the DB.")
    p.add_argument("--wire-out", default=None,
                   help="After building the DB, wire each input's per-chain MSA from it (query-match) "
                        "into ready-to-subsample inputs written here, the DB→inputs_wired step. "
                        "Omit to only build the a3m DB.")
    p.add_argument("--backend", default="protenix2", help="Backend used for MSA wiring (--wire-out).")
    p.add_argument("--verbose", "-v", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose == 0 else logging.DEBUG,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    inputs = sorted(Path(args.inputs_dir).glob("*.json"))
    if args.exclude:
        inputs = [f for f in inputs if not fnmatch.fnmatch(f.stem, args.exclude)]
    if not inputs:
        print(f"No input JSONs in {args.inputs_dir} (after --exclude).", file=sys.stderr)
        return 1

    results = generate_msa_db(
        inputs, args.db_dir, server_url=args.server_url, mode=args.mode,
        skip_existing=not args.no_skip_existing, max_workers=max(1, args.jobs),
    )
    ok = sum(1 for r in results if r.get("skipped") or r.get("returncode") == 0)
    fail = [r["target"] for r in results if not r.get("skipped") and r.get("returncode") != 0]
    print(f"MSA DB → {args.db_dir}: {ok}/{len(results)} ok, {len(fail)} failed")
    if fail:
        print("  failed:", " ".join(fail), file=sys.stderr)

    if args.wire_out:
        _wire_inputs(inputs, args.db_dir, args.wire_out, args.backend)

    return 0 if not fail else 1


def _wire_inputs(inputs, db_dir, wire_out, backend_name) -> None:
    """DB → inputs_wired: attach each input's per-chain MSA from the generated DB via
    query-match (``method.trunk_side.wire_msa_from_dump``). Lazy imports so the plain
    DB-build path never pulls in the backend/torch stack."""
    from method.trunk_side import wire_msa_from_dump
    from models import get_backend

    backend = get_backend(backend_name, {})
    wire_out, db_dir = Path(wire_out), Path(db_dir)
    wire_out.mkdir(parents=True, exist_ok=True)
    n = 0
    for inp in inputs:
        msa_root = db_dir / inp.stem
        if not msa_root.exists():
            logging.warning("wire: no MSA dir for %s in DB, skipping", inp.stem)
            continue
        _, wired = wire_msa_from_dump(backend, inp, msa_root, wire_out / inp.name)
        got = sum(1 for v in wired.values() if v)
        logging.info("wired %s: %d/%d chains from DB", inp.stem, got, len(wired))
        n += 1
    print(f"Wired {n} inputs → {wire_out}")


if __name__ == "__main__":
    sys.exit(main())
