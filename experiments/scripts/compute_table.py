"""Measured FLOPs and wall-clock per ARM, the compute table::

    python -m experiments.scripts.compute_table RUN_DIR [RUN_DIR ...] [--ref-arm hybk12]

One row per arm, aggregated over the targets that arm covers. Numbers come from each unit's
``res/diffusion_compute.json``, the measured three-lens budget (trunk / denoiser / confidence).

Two rules this applies so a hand-rolled aggregation does not have to:

  1. It reports ``flops_total``, never ``flops_total_search``. Only the first is comparable across
     methods, because it includes the one output-confidence pass per finished structure that every
     run pays, steered or not.
  2. It EXCLUDES a unit that retried. The record sums every ``diffusion_timing.jsonl`` under the
     unit's predictions dir, and a retry writes a second ``_stage_<hash>`` directory with its own
     log while overwriting the structures, so the record would sum both attempts against one set of
     cifs. The identity ``confidence_output_nfe == cif count`` detects it exactly. Excluded units
     are named in the output, never dropped silently.

Wall-clock is reported as context, not as the comparison: it depends on the card the unit landed on
and the contention on that node.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

from results_io import structure_count


def _unit_record(unit_dir):
    """(arm, target, record) for one unit, or None when it has no usable compute record."""
    f = os.path.join(unit_dir, "res", "diffusion_compute.json")
    if not os.path.exists(f):
        return None
    try:
        with open(f) as fh:
            j = json.load(fh)
    except (OSError, ValueError):
        return None
    n_cif = structure_count(os.path.join(unit_dir, "pred"))
    if not n_cif:
        return None
    base = os.path.basename(unit_dir)
    if "__" not in base:
        return None
    target, arm = base.split("__", 1)
    j["_n_cif"] = n_cif
    w = float(j.get("wall_inference_phase_s") or 0.0)
    if w <= 0:
        tot = 0.0
        for f2 in glob.glob(os.path.join(unit_dir, "pred", "**", "inference_wall.json"), recursive=True):
            try:
                with open(f2) as fh2:
                    tot += float(json.load(fh2).get("inference_sec") or 0.0)
            except (OSError, ValueError):
                pass
        w = tot
    j["_wall_s"] = w if w > 0 else None
    j["_ok"] = not (int(j.get("confidence_output_nfe") or 0)
                    and int(j["confidence_output_nfe"]) != n_cif)
    j["_seeds"], j["_samples"], _l = _cfg_budget(os.path.join(unit_dir, "res", "config.yaml"))
    return arm, target, j


def _cfg_budget(path):
    """(n_seeds, n_samples_per_seed, label) from a config.yaml, any of them None if absent."""
    try:
        import yaml
        c = yaml.safe_load(open(path)) or {}
    except Exception:                                   # noqa: BLE001
        return None, None, None
    run_c = (c.get("model") or {}).get("run") or {}
    sub = run_c.get("msa_subsample") or {}
    return (sub.get("n_seeds"), run_c.get("n_samples_per_seed"),
            (c.get("method") or {}).get("label"))


def _return_records(run):
    """(arm, target, record) for a HANDOFF RETURN: ``<run>/<target>/diffusion_compute.json`` with no
    structures beside it. The cif count falls back to the record's own confidence_output_nfe, which
    makes the retry check vacuous, so the caller is told."""
    out, fallback = [], os.path.basename(run.rstrip("/")) or "return"
    for f in sorted(glob.glob(os.path.join(run, "*", "diffusion_compute.json"))):
        d = os.path.dirname(f)
        target = os.path.basename(d)
        if "__" in target:                              # a steering unit, _unit_record owns it
            continue
        try:
            j = json.load(open(f))
        except (OSError, ValueError):
            continue
        n_cif = structure_count(d) or int(j.get("confidence_output_nfe") or 0)
        if not n_cif:
            continue
        j["_no_structures"] = not structure_count(d)
        j["_n_cif"] = n_cif
        j["_wall_s"] = (float(j.get("wall_inference_phase_s") or 0.0) or None)
        j["_ok"] = not (int(j.get("confidence_output_nfe") or 0)
                        and int(j["confidence_output_nfe"]) != n_cif)
        j["_seeds"], j["_samples"], _l = _cfg_budget(os.path.join(d, "config.yaml"))
        # These labels are per target (`bon_<target>`), so strip the target or every target becomes
        # its own arm and two halves of one ladder never merge.
        arm = (_l[:-len("_" + target)] if _l and _l.endswith("_" + target) else None) or fallback
        out.append((arm, target, j))
    return out


def _pool_arm(run):
    """Arm name for a pool run that has no per-unit dirs to read it from."""
    try:
        import yaml
        with open(os.path.join(run, "results", "config.yaml")) as fh:
            return (yaml.safe_load(fh) or {}).get("method", {}).get("label") or os.path.basename(run)
    except Exception:                                   # noqa: BLE001 -- the name is cosmetic
        return os.path.basename(run.rstrip("/")) or "pool"


def _pool_records(run):
    """(arm, target, record) for a POOL run, whose records sit in ``<run>/compute/`` and whose
    structures sit together under ``<run>/predictions/<target>__*/``."""
    out = []
    arm = None
    for f in sorted(glob.glob(os.path.join(run, "compute", "*__diffusion_compute.json"))):
        target = os.path.basename(f)[:-len("__diffusion_compute.json")]
        try:
            with open(f) as fh:
                j = json.load(fh)
        except (OSError, ValueError):
            continue
        n_cif = sum(structure_count(d)
                    for d in glob.glob(os.path.join(run, "predictions", target + "__*")))
        if not n_cif:
            continue
        if arm is None:
            arm = _pool_arm(run)
        j["_n_cif"] = n_cif
        j["_wall_s"] = (float(j.get("wall_inference_phase_s") or 0.0) or None)
        j["_ok"] = not (int(j.get("confidence_output_nfe") or 0)
                        and int(j["confidence_output_nfe"]) != n_cif)
        j["_seeds"], j["_samples"], _l = _cfg_budget(os.path.join(run, "results", "config.yaml"))
        out.append((arm, target, j))
    return out


def _reprice(arms, K, M):
    """Each unit re-costed at K conditionings x M samples, from its own per-trunk and per-structure
    rate. Units whose config.yaml gives no budget cannot be re-priced and are returned separately."""
    out, bad = {}, []
    for arm, rows in arms.items():
        for t, j in rows:
            S, N = j.get("_seeds"), j.get("_samples")
            if not S or not N:
                bad.append(t)
                continue
            per_trunk = j["flops_trunk"] / S
            per_struct = (j["flops_denoiser"] + j["flops_confidence_output"]) / (S * N)
            k = dict(j)
            k["flops_trunk"] = K * per_trunk
            k["flops_denoiser"] = K * M * (j["flops_denoiser"] / (S * N))
            k["flops_confidence_output"] = K * M * (j["flops_confidence_output"] / (S * N))
            k["flops_total"] = K * per_trunk + K * M * per_struct
            k["_n_cif"] = K * M
            k["_wall_s"] = None
            out.setdefault(arm, []).append((t, k))
    return out, bad


def collect(run_dirs):
    arms, dropped, seen_units = {}, [], 0
    seen_pairs, duplicates = set(), []
    for run in run_dirs:
        records = [(os.path.basename(u), _unit_record(u))
                   for u in sorted(glob.glob(os.path.join(run, "*__*")))]
        seen_units += len(records)
        records = [(n, r) for n, r in records if r is not None]
        for arm, target, j in list(_pool_records(run)) + list(_return_records(run)):
            seen_units += 1
            records.append((target, (arm, target, j)))
        for name, (arm, target, j) in records:
            if not j["_ok"]:
                dropped.append((os.path.basename(run) + "/" + name,
                                int(j["confidence_output_nfe"]), j["_n_cif"]))
                continue
            # One record per (arm, target): a target present in several run dirs would otherwise
            # be weighted twice. First run dir on the command line wins.
            if (arm, target) in seen_pairs:
                duplicates.append((arm, target, os.path.basename(run)))
                continue
            seen_pairs.add((arm, target))
            arms.setdefault(arm, []).append((target, j))
    return arms, dropped, seen_units, duplicates


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories to scan")
    ap.add_argument("--ref-arm", default=None,
                    help="arm to express the others as a ratio of (e.g. hybk12)")
    ap.add_argument("--targets", default=None,
                    help="restrict to these targets, a comma/space list or a file of names")
    ap.add_argument("--rung", default=None, metavar="KxM",
                    help="re-price each unit as K conditionings x M samples, e.g. 4x180. The trunk "
                         "is charged per conditioning, the denoiser and output confidence per "
                         "structure, both at the unit's own measured rate")
    a = ap.parse_args(argv)

    want = None
    if a.targets:
        txt = (open(a.targets).read() if os.path.exists(a.targets) else a.targets)
        want = set(txt.replace(",", " ").split())

    rung = None
    if a.rung:
        try:
            rung = tuple(int(x) for x in a.rung.lower().split("x"))
            assert len(rung) == 2 and all(v > 0 for v in rung)
        except (ValueError, AssertionError):
            print("--rung wants KxM, e.g. 4x180"); return 2

    arms, dropped, seen_units, duplicates = collect(a.runs)
    if rung:
        arms, unpriceable = _reprice(arms, *rung)
        if unpriceable:
            print("--rung: %d unit(s) have no n_seeds/n_samples in their config.yaml and were "
                  "dropped: %s\n" % (len(unpriceable), ", ".join(unpriceable[:6])))
    if duplicates:
        print("collapsed %d duplicate unit(s) already counted from an earlier run dir: %s\n"
              % (len(duplicates), ", ".join("%s/%s" % (t, r) for _a, t, r in duplicates[:6])))
    if want is not None:
        arms = {k: [(t, j) for t, j in v if t in want] for k, v in arms.items()}
        arms = {k: v for k, v in arms.items() if v}
        missing = want - {t for v in arms.values() for t, _j in v}
        if missing:
            print("--targets: no usable record for %d of %d: %s\n"
                  % (len(missing), len(want), sorted(missing)[:8]))
    if dropped:
        print("EXCLUDED -- compute record disagrees with the structures on disk (a retry summed two")
        print("attempts; see tts_job_lib.py compute-check):")
        for name, co, n in dropped:
            print("   %-46s confidence_output_nfe=%-6d cifs=%-5d (%.2fx)" % (name, co, n, co / n))
        print()
    if not arms:
        if not seen_units:
            print("no <target>__<arm> unit directories under: %s" % ", ".join(a.runs))
            print("This wants a STEERING pool run dir, whose units are <target>__<arm>/pred/. A b0b")
            print("pool run dir is shaped differently (predictions/, results/, queue/) and has no")
            print("per-unit compute records to tabulate.")
        else:
            print("no units with a usable compute record")
        return 1

    med = lambda arm, k: float(np.median([j[k] for _t, j in arms[arm]]))
    avg = lambda arm, k: float(np.mean([j[k] for _t, j in arms[arm]]))

    def wall(arm):
        v = [j["_wall_s"] for _t, j in arms[arm] if j["_wall_s"]]
        return float(np.median(v)) if v else None

    def ratio_vs_ref(arm):
        """Median of the PER-TARGET ratio, over targets where BOTH arms have a clean record."""
        if not a.ref_arm or a.ref_arm not in arms or arm == a.ref_arm:
            return None
        r = dict((t, j["flops_total"]) for t, j in arms[a.ref_arm])
        pairs = [j["flops_total"] / r[t] for t, j in arms[arm] if t in r and r[t]]
        return (float(np.median(pairs)), len(pairs)) if pairs else None

    ref = a.ref_arm if (a.ref_arm and a.ref_arm in arms) else None

    hdr = "%-10s %5s %6s  %10s %10s %10s %10s %10s" % (
        "arm", "units", "cifs", "trunk", "denoiser", "conf-out", "TOTAL", "TOTALmean")
    hdr += ("  %14s" % "vs ref (paired)") if ref else ""
    hdr += "  %9s %8s" % ("wall_s", "s/struct")
    print(hdr)
    print("-" * len(hdr))
    for arm in sorted(arms, key=lambda k: -med(k, "flops_total")):
        n_cif = float(np.median([j["_n_cif"] for _t, j in arms[arm]]))
        row = "%-10s %5d %6d  %.3e %.3e %.3e %.3e %.3e" % (
            arm, len(arms[arm]), n_cif, med(arm, "flops_trunk"), med(arm, "flops_denoiser"),
            med(arm, "flops_confidence_output"), med(arm, "flops_total"), avg(arm, "flops_total"))
        if ref:
            rr = ratio_vs_ref(arm)
            row += "  %14s" % ("--" if rr is None else "%.3f (n=%d)" % rr)
        w = wall(arm)
        row += "  %9s %8s" % ("n/a" if w is None else "%.0f" % w,
                              "n/a" if w is None else "%.1f" % (w / max(n_cif, 1)))
        print(row)
    print()
    if any(j.get("_no_structures") for v in arms.values() for _t, j in v):
        print("Some units are handoff returns with no structures on disk, so their cif count is the")
        print("record's own confidence_output_nfe and the retry check could not run.")
        print()
    print("`TOTAL` is the MEDIAN over the arm's units and `TOTALmean` the MEAN over them. Both are")
    print("flops_total (trunk + denoiser +")
    print("confidence + the one output-confidence pass per finished structure) and is the ONLY")
    print("cross-method-comparable figure. Wall is the card's throughput on the day, not a property")
    print("of the method -- report it as context, never as the comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
