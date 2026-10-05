"""OpenDDE-40 transfer set: 20 unusable + 20 solvable targets, for the cross-backbone experiment.

Targets are drawn from those with an existing unsteered OpenDDE Best-of-N pool, so the control arm
is already available and only the steered arm has to be run. Bands over that pool, using
``dockq_abag`` against the CAPRI-acceptable threshold of 0.23::

    unusable    oracle < 0.23            nothing acceptable in the pool
    fails       acc_frac <= 0.10         acceptable exists but is rare
    borderline  0.10 < acc_frac < 0.90
    solves      acc_frac >= 0.90

``arm`` is the two-way split the loader filters on, ``unusable`` or ``solvable``.

Layout on disk::

    {root}/targets.csv                      the 40, with band, arm, native_src and set membership
    {root}/sequences.json                   {target_id: [[chain_id, sequence], ...]}, from predictions
    {root}/ground_truths/{tid}/{tid}.pdb    H, L (paired antibodies only), then A, B, ... for antigens

:func:`build` materialises that tree from the scoring artifacts and needs the extracted
predictions; the loader only reads it. Two properties of ``build`` a caller should know:

  * Sequences come from a PREDICTION cif (:func:`_chain_sequences`), never from a deposition, so
    the chains recovered are exactly the ones the run was given.
  * Natives come from the single ``native_dir`` passed in, copied verbatim. There is deliberately
    no second source and no fallback.

Which targets are in the set is NOT computed here. ``build`` writes exactly the rows of the
``selection_csv`` it is handed, produced upstream by requiring the supplied native and a second
independent one to agree on band and oracle over the whole pool, then dropping small-interface
targets where the deposition offers a substantially larger antibody-antigen interface.

The target ids are in ``data/opendde40_ids.txt``. The 40 are drawn from the same curation as the collected bad cases, described in
``data/datasets/collected_bad_cases.py``.

Export: :class:`OpenDDE40Dataset`, and :func:`build` for the one-time materialisation.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from data.target import Chain, Target

logger = logging.getLogger(__name__)

ANTIBODY_ROLES = ("H", "L")

KNOWN_BAD: Dict[str, str] = {}


class OpenDDE40Dataset:
    """Loads the OpenDDE-40 transfer tree into ``List[Target]``."""

    name = "opendde40"

    def __init__(
        self,
        root: str | Path,
        arm: Optional[str] = None,
        only: Optional[Sequence[str]] = None,
        msa_db_dir: Optional[str | Path] = None,
        allow_known_bad: bool = True,
        task_prefix: str = "interface_antibody_antigen_opendde_transfer",
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"OpenDDE-40 root not found: {self.root}. Run "
                f"`python -m data.datasets.opendde40 --build --root {self.root} ...` first.")
        if arm not in (None, "unusable", "solvable"):
            raise ValueError(f"arm must be 'unusable', 'solvable' or None, got {arm!r}")
        self.arm = arm
        self.only = {str(x).strip() for x in only} if only else None
        self.msa_db_dir = Path(msa_db_dir) if msa_db_dir else None
        self.allow_known_bad = allow_known_bad
        self.task_prefix = task_prefix

    # ------------------------------------------------------------------ helpers
    def _rows(self) -> List[dict]:
        with open(self.root / "targets.csv") as fh:
            rows = list(csv.DictReader(fh))
        if self.arm:
            rows = [r for r in rows if r["arm"] == self.arm]
        if self.only is not None:
            rows = [r for r in rows if r["target_id"] in self.only]
        flagged = [r["target_id"] for r in rows if r["target_id"] in KNOWN_BAD]
        if not self.allow_known_bad:
            for t in flagged:
                logger.warning("dropping %s: %s", t, KNOWN_BAD[t])
            rows = [r for r in rows if r["target_id"] not in KNOWN_BAD]
        else:
            for t in flagged:
                logger.warning("%s kept with a defective reference: %s. It carries "
                               "metadata['known_bad']; score it separately.", t, KNOWN_BAD[t])
        return rows

    def _msa_for(self, target_id: str, chain_id: str) -> Optional[str]:
        """Locate a wired a3m for one chain, or ``None``."""
        if self.msa_db_dir is None:
            return None
        base = self.msa_db_dir / target_id / "msa"
        if not base.is_dir():
            return None
        hits = sorted(base.glob(f"*/*/non_pairing.a3m")) or sorted(base.glob("*/*/*.a3m"))
        seqs = self._sequences()[target_id]
        idx = [cid for cid, _ in seqs].index(chain_id)
        return str(hits[idx]) if idx < len(hits) else None

    def _sequences(self) -> Dict[str, List[List[str]]]:
        if not hasattr(self, "_seq_cache"):
            with open(self.root / "sequences.json") as fh:
                self._seq_cache = json.load(fh)
        return self._seq_cache

    # ------------------------------------------------------------------ the protocol
    def load(self) -> List[Target]:
        seqs = self._sequences()
        targets, no_msa = [], []
        for r in self._rows():
            tid = r["target_id"]
            if tid not in seqs:
                raise KeyError(f"{tid} listed in targets.csv but absent from sequences.json")
            chains = []
            for cid, seq in seqs[tid]:
                a3m = self._msa_for(tid, cid)
                if a3m is None:
                    no_msa.append(f"{tid}:{cid}")
                chains.append(Chain(chain_id=cid, sequence=seq, chain_type="protein",
                                    msa_a3m_path=a3m))
            gt = self.root / "ground_truths" / tid / f"{tid}.pdb"
            targets.append(Target(
                target_id=tid,
                chains=chains,
                ground_truth_path=str(gt) if gt.is_file() else None,
                task=self.task_prefix,
                metadata={**{k: r[k] for k in r if k != "target_id"},
                          **({"known_bad": KNOWN_BAD[tid]} if tid in KNOWN_BAD else {})},
            ))
        if no_msa and self.msa_db_dir is not None:
            logger.warning("no MSA for %d chain(s): %s. CAST's K conditionings come from "
                           "subsampling the MSA, so these would run the method with nothing to "
                           "subsample.", len(no_msa), ", ".join(no_msa[:12]))
        elif self.msa_db_dir is None:
            logger.info("msa_db_dir not set: emitting %d targets with no MSA pointers, which is "
                        "correct for the prepare -> generate_msa -> wire order.", len(targets))
        return targets


_AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
        "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
        "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def _chain_sequences(cif_path: Path) -> List[tuple]:
    """[(chain_id, sequence), ...] sorted by chain, for chains of at least 20 residues."""
    cols: List[str] = []
    rows: List[List[str]] = []
    in_loop = False
    with open(cif_path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("_atom_site."):
                cols.append(s.split(".", 1)[1]); in_loop = True; continue
            if in_loop and s.startswith(("ATOM", "HETATM")):
                rows.append(s.split()); continue
            if in_loop and cols and rows and (s == "#" or s.startswith("loop_")):
                break
    i = {c: k for k, c in enumerate(cols)}
    out: Dict[str, List[str]] = {}
    seen = set()
    for f in rows:
        if f[i["group_PDB"]] != "ATOM" or f[i["pdbx_PDB_model_num"]] != "1":
            continue
        if f[i["label_atom_id"]] != "CA":
            continue
        key = (f[i["auth_asym_id"]], f[i["auth_seq_id"]])
        if key in seen:
            continue
        seen.add(key)
        out.setdefault(key[0], []).append(_AA3.get(f[i["label_comp_id"]], "X"))
    return sorted((k, "".join(v)) for k, v in out.items() if len(v) >= 20)

def build(root: str | Path, selection_csv: str | Path, predictions_root: str | Path,
          native_dir: str | Path) -> Path:
    """Write the dataset tree from the scoring artifacts. One-time, needs the extracted predictions."""
    root = Path(root)
    (root / "ground_truths").mkdir(parents=True, exist_ok=True)
    predictions_root = Path(predictions_root)
    native_dir = Path(native_dir)

    with open(selection_csv) as fh:
        rows = list(csv.DictReader(fh))

    sequences: Dict[str, List[List[str]]] = {}
    for r in rows:
        tid = r["target_id"]
        hits = sorted(predictions_root.glob(f"output/{tid}/seed_*/predictions/*_sample_*.cif"))
        if not hits:
            raise FileNotFoundError(f"no prediction for {tid} under {predictions_root}")
        raw = _chain_sequences(hits[0])
        n_ab = 1 if ("_nb" in tid or len(raw) == 2) else 2
        roles = list(ANTIBODY_ROLES[:n_ab]) + [chr(ord("A") + i) for i in range(len(raw) - n_ab)]
        sequences[tid] = [[role, seq] for role, (_, seq) in zip(roles, raw)]

        dst = root / "ground_truths" / tid
        dst.mkdir(parents=True, exist_ok=True)
        src = native_dir / f"{tid}.pdb"
        if not src.is_file():
            raise FileNotFoundError(f"no native for {tid} in {native_dir}")
        (dst / f"{tid}.pdb").write_bytes(src.read_bytes())

    with open(root / "sequences.json", "w") as fh:
        json.dump(sequences, fh, indent=1, sort_keys=True)
    with open(root / "targets.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %d targets to %s", len(rows), root)
    return root


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true", help="materialise the dataset tree")
    p.add_argument("--root", required=True)
    p.add_argument("--selection-csv", help="the 40-target table (target_id, arm, band, ...)")
    p.add_argument("--predictions-root", help="extracted OpenDDE predictions tree")
    p.add_argument("--native-dir", help="the supplied natives, flat {tid}.pdb")
    p.add_argument("--verbose", "-v", action="count", default=0)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING)
    if a.build:
        for req in ("selection_csv", "predictions_root", "native_dir"):
            if not getattr(a, req):
                p.error(f"--build needs --{req.replace('_', '-')}")
        build(a.root, a.selection_csv, a.predictions_root, a.native_dir)
        return 0
    ds = OpenDDE40Dataset(a.root)
    ts = ds.load()
    print(f"{len(ts)} targets, {sum(t.n_chains for t in ts)} chains, "
          f"{sum(1 for t in ts if t.ground_truth_path)} with a native")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
