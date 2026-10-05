"""Screen every target's CDR token map against the residues that ACTUALLY touch the antigen.

A steering CV built on the CDR map only steers the interface if the antibody binds THROUGH its CDRs.
That is the normal case but not a law: a framework binder has a paratope the CDR map cannot see, and
steering it moves the wrong residues while the label still says "paratope". One cheap check per
target catches it: overlap between the CDR map and the contacting antibody residues.
"""
import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.scripts.token_maps import ca_by_chain, token_order

CONTACT_A = 10.0
FRAMEWORK_BINDER_FRAC = 0.15
PARTIAL_FRAC = 0.5


def _default_paths() -> Tuple[str, str]:
    """Return ``(run_root, ground_truth_dir)`` from the environment, with defaults."""
    tts_root = os.environ.get("TTS_ROOT", "/data/tts")
    run_root = os.environ.get("RUN_ROOT", os.path.join(tts_root, "runs", "collected_bad_cases"))
    gt_dir = os.environ.get("GT_DIR", os.path.join(tts_root, "ground_truths"))
    return run_root, gt_dir


def contact_tokens(target_id: str, run_root: str, gt_dir: str,
                   contact_a: float = CONTACT_A) -> Optional[np.ndarray]:
    """Return the antibody token indices within ``contact_a`` of the antigen, or None."""
    pdb = os.path.join(gt_dir, "%s.pdb" % target_id)
    if not os.path.exists(pdb):
        return None
    try:
        coords = ca_by_chain(pdb)
        order = token_order(os.path.join(run_root, "inputs_wired", target_id + ".json"))
    except Exception:
        return None
    ab_tokens: List[Tuple[int, np.ndarray]] = []
    antigen = None
    for chain_id, offset, length in order:
        if chain_id not in coords:
            continue
        if chain_id in ("H", "L"):
            ab_tokens += [(offset + i, coords[chain_id][i])
                          for i in range(min(length, len(coords[chain_id])))]
        else:
            antigen = (coords[chain_id] if antigen is None
                       else np.vstack([antigen, coords[chain_id]]))
    if antigen is None or not ab_tokens:
        return None
    idx = np.array([t for t, _ in ab_tokens])
    xyz = np.stack([c for _, c in ab_tokens])
    dist = np.linalg.norm(xyz[:, None, :] - antigen[None, :, :], axis=-1).min(axis=1)
    return idx[dist < contact_a]


def screen(cdr_map: Dict[str, Sequence[int]], run_root: str, gt_dir: str,
           contact_a: float = CONTACT_A) -> List[Tuple[str, int, int, int, float]]:
    """Return ``(target, n_contact, n_cdr, n_overlap, overlap_fraction)`` sorted by fraction."""
    rows = []
    for target_id in sorted(cdr_map):
        tokens = contact_tokens(target_id, run_root, gt_dir, contact_a)
        if tokens is None:
            continue
        contact = set(int(i) for i in tokens)
        cdr = set(int(i) for i in cdr_map[target_id])
        overlap = contact & cdr
        rows.append((target_id, len(contact), len(cdr), len(overlap),
                     len(overlap) / max(len(contact), 1)))
    rows.sort(key=lambda r: r[4])
    return rows


def _verdict(frac: float) -> str:
    """Classify a CDR-map overlap fraction."""
    if frac < FRAMEWORK_BINDER_FRAC:
        return "FRAMEWORK BINDER: CDR map misses the paratope"
    return "partial" if frac < PARTIAL_FRAC else "ok"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print the screen and optionally write contact-derived paratope overrides."""
    default_run_root, default_gt_dir = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=default_run_root)
    parser.add_argument("--gt-dir", default=default_gt_dir)
    parser.add_argument("--cdr-map", default=None,
                        help="CDR token map JSON (default: <run-root>/cdr_tokenmap_52.json)")
    parser.add_argument("--contact-a", type=float, default=CONTACT_A)
    parser.add_argument("--emit-override", metavar="OUT_JSON",
                        help="write {target: [contact-derived paratope tokens]} for flagged targets")
    args = parser.parse_args(argv)

    cdr_path = args.cdr_map or os.path.join(args.run_root, "cdr_tokenmap_52.json")
    with open(cdr_path) as fh:
        cdr_map = json.load(fh)

    rows = screen(cdr_map, args.run_root, args.gt_dir, args.contact_a)
    print("  CDR map vs the residues that actually contact the antigen (%.0f A)" % args.contact_a)
    print("  %-12s %9s %8s %8s %8s  %s"
          % ("target", "contacts", "cdr_map", "overlap", "frac", "verdict"))
    for target_id, n_contact, n_cdr, n_overlap, frac in rows:
        print("  %-12s %9d %8d %8d %8.2f  %s"
              % (target_id, n_contact, n_cdr, n_overlap, frac, _verdict(frac)))

    flagged = [r[0] for r in rows if r[4] < FRAMEWORK_BINDER_FRAC]
    print("\n  framework binders (overlap < %d%%): %s"
          % (FRAMEWORK_BINDER_FRAC * 100, ", ".join(flagged) if flagged else "none"))

    if args.emit_override and flagged:
        override = {}
        for target_id in flagged:
            tokens = contact_tokens(target_id, args.run_root, args.gt_dir, args.contact_a)
            if tokens is not None:
                override[target_id] = sorted(int(i) for i in tokens)
        with open(args.emit_override, "w") as fh:
            json.dump(override, fh)
        print("  wrote contact-derived paratope for %d target(s) -> %s"
              % (len(override), args.emit_override))
        for target_id, tokens in override.items():
            print("    %s: %s" % (target_id, tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
