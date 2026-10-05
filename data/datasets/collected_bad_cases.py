"""Collected antibody-antigen "bad cases" dataset loader.

They were collected independently from public depositions.
The loader key is ``collected_bad_cases`` and the class is ``CollectedBadCasesDataset``.

Reads the on-disk layout, each root being its own constructor argument::

    {bad_cases_dir}/{target_id}/{target_id}.pdb        reference PDB (SEQRES + ATOM), required
    {predictions_root}/{target_id}/seed_{N}/...        optional predictions
    {master_csv}                                       optional master DockQ CSV

Target IDs are ``{pdb}_{ab|nb}{N}``::

    _ab  = VH-VL antibody (heavy + light paired through framework contacts)
    _nb  = VHH nanobody   (unpaired heavy chain)
    N    = 1-indexed position within the PDB entry

Produces one :class:`Target` per target subdirectory, parsing SEQRES from the reference PDB.

CURATION, as run upstream. Entries come from ANDD, a collected PDB set and SAbDab, and pass nine
stages. Entry prefilters keep X-ray and cryo-EM below 3.5 A and drop NMR, with a missing release
date treated as before 2025-06-30. Chain identification excludes ligands, ions and water, detects
variable domains with ANARCI, and rescues antibody chains misannotated as antigen by FASTA
matching; remaining protein chains are antigen candidates and an entry with none is dropped. VH and
VL pair on 25 framework contacts, and an unpaired heavy chain becomes a VHH nanobody. Antibody
validation applies AbNumber chain typing, IMGT numbering and IMGT CDR length constraints. An
interface counts on a heavy-atom contact within 5.0 A over at least 3 antigen interface residues,
and one sample is exported per contacting antigen chain. Sequence QC then removes duplicates within
each temporal split and rejects a complex unless the antibody CDR contribution is at least 0.20, the
antigen contribution at least 0.10, the interface overlap at least 0.10, and more than half the
paratope lies inside the IMGT CDRs; that last rule is the dominant filter. Representatives are
chosen by clustering on antibody CDR sequences alone, over single-antigen complexes.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from data.target import Chain, Target

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- amino-acid mapping
_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
    # Common non-standard mappings, everything else → X below.
    "MSE": "M",     # selenomethionine
    "HYP": "P", "HSD": "H", "HSE": "H", "HSP": "H",
    "CSO": "C", "CSD": "C", "OCS": "C", "SEP": "S", "TPO": "T", "PTR": "Y",
    "PCA": "E",
}


def _three_to_one(res: str) -> str:
    return _THREE_TO_ONE.get(res.upper(), "X")


# ---------------------------------------------------------------- target-ID parsing
_TID_RE = re.compile(r"^(?P<pdb>[0-9a-z]{4})_(?P<family>ab|nb)(?P<index>\d+)$")


@dataclass(frozen=True)
class TargetIdParts:
    pdb_id: str          # "8yvk"
    family: str          # "ab" | "nb"
    index: int           # 1
    raw: str             # "8yvk_ab1"

    @property
    def is_nanobody(self) -> bool:
        return self.family == "nb"

    @property
    def is_antibody(self) -> bool:
        return self.family == "ab"


def parse_target_id(tid: str) -> Optional[TargetIdParts]:
    """Return parsed parts, or None if ``tid`` isn't in the private scheme."""
    m = _TID_RE.match(tid)
    if not m:
        return None
    return TargetIdParts(pdb_id=m.group("pdb"), family=m.group("family"),
                         index=int(m.group("index")), raw=tid)


# ---------------------------------------------------------------- SEQRES parser
def parse_seqres(pdb_path: str | Path) -> Dict[str, str]:
    """Extract one-letter sequences per chain from a PDB's SEQRES records."""
    seqs: Dict[str, List[str]] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("SEQRES"):
                continue
            chain_id = line[11:12].strip()
            residues = line[19:70].split()
            if not chain_id:
                continue
            seqs.setdefault(chain_id, []).extend(residues)
    return {c: "".join(_three_to_one(r) for r in res_list) for c, res_list in seqs.items()}


# ---------------------------------------------------------------- loader
class CollectedBadCasesDataset:
    """Loads a ``bad_cases/`` tree of ``{target_id}/{target_id}.pdb`` into ``List[Target]``."""

    name = "collected_bad_cases"

    def __init__(
        self,
        bad_cases_dir: str | Path,
        predictions_root: Optional[str | Path] = None,
        master_csv: Optional[str | Path] = None,
        only: Optional[Sequence[str]] = None,
        heavy_chain_id: str = "H",
        light_chain_id: str = "L",
        task_prefix: str = "interface_antibody_antigen_bad_case",
    ):
        self.bad_cases_dir = Path(bad_cases_dir)
        if not self.bad_cases_dir.is_dir():
            raise FileNotFoundError(f"bad_cases dir not found: {self.bad_cases_dir}")
        self.predictions_root = Path(predictions_root) if predictions_root else None
        self.master_csv = Path(master_csv) if master_csv else None
        self.only = {str(x).strip() for x in only} if only else None
        self.heavy_chain_id = heavy_chain_id
        self.light_chain_id = light_chain_id
        self.task_prefix = task_prefix

    # ------------------------------------------------------------------ load
    def load(self) -> List[Target]:
        targets: List[Target] = []
        n_skipped = 0
        for entry in sorted(self.bad_cases_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            tid = entry.name
            if self.only is not None and tid not in self.only:
                continue
            parts = parse_target_id(tid)
            if parts is None:
                logger.warning("Skipping %s: doesn't match {pdb}_{ab|nb}{N}", tid)
                n_skipped += 1
                continue

            pdb = entry / f"{tid}.pdb"
            if not pdb.is_file():
                logger.warning("Skipping %s: no PDB at %s", tid, pdb)
                n_skipped += 1
                continue

            seqs = parse_seqres(pdb)
            if not seqs:
                logger.warning("Skipping %s: SEQRES empty in %s", tid, pdb)
                n_skipped += 1
                continue

            chains = self._build_chains(tid, parts, seqs)
            if not chains:
                logger.warning("Skipping %s: no usable chains after SEQRES parse", tid)
                n_skipped += 1
                continue

            meta: Dict[str, Any] = {
                "pdb_id": parts.pdb_id,
                "family": parts.family,
                "index": parts.index,
                "is_nanobody": parts.is_nanobody,
                "source_pipeline": "collected_bad_cases_v2",
            }
            if self.predictions_root is not None:
                td = self.predictions_root / tid
                if td.is_dir():
                    meta["predictions_dir"] = str(td)
            if self.master_csv is not None:
                meta["master_dockq_csv"] = str(self.master_csv)

            targets.append(Target(
                target_id=tid,
                chains=chains,
                ground_truth_path=str(pdb),
                task=self.task_prefix,
                metadata=meta,
            ))
        logger.info("CollectedBadCasesDataset: loaded %d targets (%d skipped) from %s",
                    len(targets), n_skipped, self.bad_cases_dir)
        return targets

    # ------------------------------------------------------------------ chain assembly
    def _build_chains(self, tid: str, parts: TargetIdParts, seqs: Dict[str, str]) -> List[Chain]:
        """Assemble Chain objects in canonical order: heavy, light (if ab), antigen(s)."""
        chains: List[Chain] = []

        # Heavy chain (H), always present.
        if self.heavy_chain_id in seqs:
            chains.append(Chain(chain_id=self.heavy_chain_id,
                                sequence=seqs[self.heavy_chain_id], chain_type="protein"))
        else:
            logger.warning("%s: no chain %r in SEQRES (got %s)",
                           tid, self.heavy_chain_id, sorted(seqs.keys()))

        # Light chain (L), only for VH-VL antibodies.
        if parts.is_antibody and self.light_chain_id in seqs:
            chains.append(Chain(chain_id=self.light_chain_id,
                                sequence=seqs[self.light_chain_id], chain_type="protein"))
        elif parts.is_antibody and self.light_chain_id not in seqs:
            logger.warning("%s marked as VH-VL antibody but no chain %r in SEQRES",
                           tid, self.light_chain_id)

        # Antigen chain(s): everything else.
        antigen_ids = [c for c in seqs.keys()
                       if c != self.heavy_chain_id and (not parts.is_antibody or c != self.light_chain_id)]
        for aid in antigen_ids:
            chains.append(Chain(chain_id=aid, sequence=seqs[aid], chain_type="protein"))
        if not antigen_ids:
            logger.warning("%s: no antigen chain in SEQRES (chains found: %s)",
                           tid, sorted(seqs.keys()))
        return chains

