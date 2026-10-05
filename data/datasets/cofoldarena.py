"""CoFold Arena antibody-antigen loader.

https://cofoldarena.ai is a weekly-refreshed antibody-antigen co-folding leaderboard built from
SAbDab: targets released after 2024-09-09, UniProt-annotated antigens, one antibody per antigen,
chains under 1024 tokens.

THE PANEL IS NOT PUBLISHED AS A DOWNLOAD. There is no API and no CSV, the site embeds its panel and
per-model DockQ in a JS bundle, and ``data/cofoldarena_targets.csv`` was extracted from it by
regex. The manifest is therefore supplied as data rather than re-scraped at import time, it is a
SNAPSHOT carrying its capture date, and ``verify()`` re-checks it against the structures on disk,
which is the only guard against a stale or partial capture. ``DEFAULT_MANIFEST`` points at
``data/cofoldarena_targets.csv``, which ships with the repo; pass ``manifest=`` to use another.

CHAIN IDS COME FROM THE DEPOSITED ENTRY, NOT THE ASSEMBLY. SAbDab annotates Hchain / Lchain /
agChains as auth chain ids on the deposited file, while ``-assembly1.cif`` renames chains and adds
symmetry copies, so reading assemblies loses targets to "missing chain".

Their leaderboard numbers are not directly comparable to a pool run here. They draw 5 diffusion
samples from one seed and keep the top-1 by ``0.8*ipTM + 0.2*pTM``, and their DockQ is computed
against a trimmed native they do not publish.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from data.target import Chain, Target

logger = logging.getLogger(__name__)

# One manifest row per target-antibody pair. A PDB entry with two antibodies is two rows, which is
DEFAULT_MANIFEST = "data/cofoldarena_targets.csv"
# The 50-target structure tree, `<target>/<target>.pdb` with SEQRES, shipped with the repo.
DEFAULT_TARGETS_DIR = "data/cfa50_targets"
PANEL_CUTOFF = "2024-09-09"          # their inclusion rule: released after this date


def _norm(s: str) -> str:
    return s.strip().lower().split("-")[0].split("_")[0]



def sequences_by_auth_chain(path: str | Path) -> Dict[str, str]:
    """``{auth_chain_id: one-letter sequence}`` for the amino-acid chains of a reference."""
    import numpy as np
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    cif = pdbx.CIFFile.read(str(path))
    arr = pdbx.get_structure(cif, model=1, use_author_fields=True)
    arr = arr[struc.filter_amino_acids(arr)]
    ins = getattr(arr, "ins_code", None)
    out: Dict[str, str] = {}
    for ch in np.unique(arr.chain_id):
        m = arr.chain_id == ch
        sub = arr[m]
        sub_ins = ins[m] if ins is not None else [""] * len(sub)
        seen, seq = set(), []
        for rid, code, rname in zip(sub.res_id, sub_ins, sub.res_name):
            key = (int(rid), str(code).strip())
            if key in seen:
                continue
            seen.add(key)
            seq.append(struc.info.one_letter_code(rname) or "X")
        out[str(ch)] = "".join(seq)
    return out


class CoFoldArenaDataset:
    """Load the CoFold Arena antibody-antigen panel as :class:`Target` objects."""

    name = "cofoldarena"
    task_prefix = "interface_antibody_antigen"

    def __init__(self, manifest: str | Path = DEFAULT_MANIFEST,
                 structures_dir: Optional[str | Path] = None,
                 only: Optional[Sequence[str]] = None,
                 relabel: bool = True,
                 panel_only: bool = True):
        self.manifest = Path(manifest)
        self.structures_dir = Path(structures_dir) if structures_dir else None
        self.only = {str(x).strip() for x in only} if only else None
        self.relabel = relabel
        self.panel_only = panel_only

    # ---------------------------------------------------------------- manifest
    def rows(self) -> List[Dict[str, str]]:
        with open(self.manifest) as fh:
            rows = [r for r in csv.DictReader(fh)]
        if self.panel_only:
            rows = [r for r in rows if (r.get("release") or "") > PANEL_CUTOFF]
        if self.only is not None:
            keep = self.only
            rows = [r for r in rows
                    if f'{r["pdb"]}_{r["abId"]}' in keep or r["pdb"] in keep]
        return rows

    def target_key(self, row: Mapping[str, Any]) -> str:
        return f'{row["pdb"]}_{row["abId"]}'

    # ---------------------------------------------------------------- roles
    @staticmethod
    def chain_roles(row: Dict[str, str]) -> Dict[str, str]:
        """auth chain id -> role, from the panel's own SAbDab assignment."""
        roles: Dict[str, str] = {}
        if row.get("Hchain"):
            roles[row["Hchain"]] = "H"
        if row.get("Lchain"):
            roles[row["Lchain"]] = "L"
        for c in (row.get("agChains") or "").split("|"):
            if c:
                roles[c] = "antigen"
        return roles

    def _relabel_map(self, roles: Dict[str, str]) -> Dict[str, str]:
        """Native chain id -> H / L / A,B,C..., the convention dockq_abag and the CDR maps assume."""
        out: Dict[str, str] = {}
        ag = "A"
        for cid, role in roles.items():
            if role == "H":
                out[cid] = "H"
            elif role == "L":
                out[cid] = "L"
            else:
                # antigens take A, B, C ... skipping H and L so a relabelled id is never ambiguous
                while ag in ("H", "L"):
                    ag = chr(ord(ag) + 1)
                out[cid] = ag
                ag = chr(ord(ag) + 1)
        return out

    # ---------------------------------------------------------------- load
    def load(self) -> List[Target]:
        if self.structures_dir is None:
            raise ValueError("CoFoldArenaDataset needs structures_dir: sequences and chain roles "
                             "are read from the reference, never invented")
        out: List[Target] = []
        skipped_no_struct, skipped_no_chain = [], []
        for r in self.rows():
            key = f'{r["pdb"]}_{r["abId"]}'
            cif = self.structures_dir / f'{r["pdb"]}.cif'
            if not cif.is_file():
                skipped_no_struct.append(key)
                continue
            try:
                seqs = sequences_by_auth_chain(cif)
            except Exception as exc:                                # noqa: BLE001
                logger.warning("%s: could not read sequences from %s (%s)", key, cif, exc)
                skipped_no_struct.append(key)
                continue
            roles = self.chain_roles(r)
            present = {c: s for c, s in seqs.items() if c in roles and s}
            if not present or not any(roles[c] in ("H", "L") for c in present) \
                    or not any(roles[c] == "antigen" for c in present):
                skipped_no_chain.append(key)
                continue
            mapping = self._relabel_map({c: roles[c] for c in present}) if self.relabel \
                else {c: c for c in present}
            t = Target(target_id=key, task=self.task_prefix, ground_truth_path=str(cif),
                       metadata={"pdb_id": r["pdb"], "ab_id": r["abId"],
                                 "format": r.get("format", ""), "release": r.get("release", ""),
                                 "antigen": r.get("antigen", ""), "uniprot": r.get("agUniProt", ""),
                                 "native_chains": dict(roles), "relabel_map": dict(mapping),
                                 "source": "cofoldarena", "panel_cutoff": PANEL_CUTOFF})
            for cid, seq in present.items():
                t.chains.append(Chain(chain_id=mapping[cid], sequence=seq, chain_type="protein"))
            out.append(t)
        if skipped_no_struct:
            logger.warning("cofoldarena: %d target(s) skipped, no readable structure: %s",
                           len(skipped_no_struct), skipped_no_struct[:6])
        if skipped_no_chain:
            logger.warning("cofoldarena: %d target(s) skipped, missing an antibody or antigen "
                           "chain in the reference: %s", len(skipped_no_chain), skipped_no_chain[:6])
        logger.info("CoFoldArenaDataset: loaded %d target(s) from %s", len(out), self.manifest)
        return out

    # ---------------------------------------------------------------- guard
    def verify(self) -> Dict[str, List[str]]:
        """Re-check the committed manifest against what is on disk. The only guard against a stale
                or partial capture, because the upstream panel changes weekly and is scraped, not published.
        """
        rep = {"missing_structure": [], "missing_chain_in_structure": [], "ok": []}
        for r in self.rows():
            key = f'{r["pdb"]}_{r["abId"]}'
            cif = (self.structures_dir or Path(".")) / f'{r["pdb"]}.cif'
            if not cif.is_file():
                rep["missing_structure"].append(key)
                continue
            try:
                seqs = sequences_by_auth_chain(cif)
            except Exception:                                       # noqa: BLE001
                rep["missing_structure"].append(key)
                continue
            roles = self.chain_roles(r)
            if any(c not in seqs for c in roles):
                rep["missing_chain_in_structure"].append(key)
            else:
                rep["ok"].append(key)
        return rep


# ---------------------------------------------------------------------------- subset construction
def build_50_subset(overlap_csv: str | Path, manifest: str | Path, tokens_json: str | Path,
                    our_tokens_json: str | Path, atlas_csv: str | Path,
                    scores_csv: Optional[str | Path] = None,
                    require_cfa_unsolved: bool = False, cfa_threshold: float = 0.23,
                    selector_models: Sequence[str] = ("boltz2", "chai1", "boltz1", "of3p2"),
                    n_new: int = 28, seed: int = 0) -> List[Dict[str, str]]:
    """The 50-target subset: every overlapping target kept, the rest matched to our bad cases."""
    rng = random.Random(seed)
    f = lambda v: float(v) if str(v).strip() not in ("", "None") else None
    kept = list(csv.DictReader(open(overlap_csv)))
    kept_pdb = {r["pdb_id"] for r in kept}
    atlas = {r["target"]: r for r in csv.DictReader(open(atlas_csv))}
    our_tok = json.load(open(our_tokens_json))
    cfa_tok = json.load(open(tokens_json))
    panel = list(csv.DictReader(open(manifest)))

    rest = sorted(set(atlas) - {r["our_target"] for r in kept})
    need = [{"target": t, "format": "VHH" if "_nb" in t else "Fab", "n_token": our_tok.get(t)}
            for t in rest]
    # CoFold Arena's own ranked score, best across its models, keyed by pdb id
    cfa_best: Dict[str, float] = {}
    n_sel: Dict[str, int] = {}
    if scores_csv:
        for r in csv.DictReader(l for l in open(scores_csv) if not l.startswith("#")):
            vals = [f(r.get(f"{m}_dockq")) for m in selector_models]
            vals = [v for v in vals if v is not None]
            if vals:
                k = r["pdb_id"].lower()
                cfa_best[k] = max(cfa_best.get(k, -1.0), max(vals))
                n_sel[k] = max(n_sel.get(k, 0), len(vals))

    cands = [{"key": f'{r["pdb"]}_{r["abId"]}', "pdb": r["pdb"], "abId": r["abId"],
              "format": r["format"], "release": r.get("release", ""),
              "uniprot": r.get("agUniProt", ""), "antigen": r.get("antigen", ""),
              "n_token": cfa_tok.get(f'{r["pdb"]}_{r["abId"]}'),
              "cfa_best": cfa_best.get(r["pdb"])}
             for r in panel
             if r["pdb"] not in kept_pdb and cfa_tok.get(f'{r["pdb"]}_{r["abId"]}')]
    if require_cfa_unsolved:
        before = len(cands)
        # require a real verdict: at least 2 of the selector models must have scored the target,
        # or "all models failed" could mean "no model was run"
        cands = [c for c in cands
                 if c["cfa_best"] is not None and c["cfa_best"] < cfa_threshold
                 and n_sel.get(c["pdb"], 0) >= 2]
        logger.info("cofoldarena subset: hard by %s -- %d of %d candidates remain "
                    "(every selector model < %.2f, >=2 models scored)",
                    list(selector_models), len(cands), before, cfa_threshold)

    used, used_ag, matched = set(), set(), []
    while len(matched) < n_new:
        progressed = False
        for n in sorted(need, key=lambda x: -abs((x["n_token"] or 600) - 600)):
            if len(matched) >= n_new:
                break
            pool = [c for c in cands if c["key"] not in used and c["format"] == n["format"]
                    and (c["uniprot"] or c["key"]) not in used_ag]
            if not pool:
                pool = [c for c in cands if c["key"] not in used and c["format"] == n["format"]]
            if not pool:
                continue
            want = n["n_token"] or 600
            d = lambda c: abs(c["n_token"] - want) / max(want, 1)
            best = min(d(c) for c in pool)
            ties = [c for c in pool if d(c) <= best + 0.01]  # within 1% is a tie, break it randomly
            pick = rng.choice(ties)
            used.add(pick["key"])
            used_ag.add(pick["uniprot"] or pick["key"])
            matched.append((n, pick, d(pick)))
            progressed = True
        if not progressed:                                   # pool exhausted for every format
            break

    matched.sort(key=lambda m: m[2])
    chosen = matched[:n_new]
    rows = [{"role": "kept_overlap", "target_key": "", "pdb_id": r["pdb_id"],
             "format": r["format"], "n_token": r["n_token"], "stands_in_for": r["our_target"],
             "size_mismatch_pct": "",
             "cfa_best_ranked_dockq": r.get("best_of_three", "")} for r in kept]
    rows += [{"role": "new_from_cofold", "target_key": c["key"], "pdb_id": c["pdb"],
              "format": c["format"], "n_token": c["n_token"], "stands_in_for": n["target"],
              "size_mismatch_pct": round(100 * d, 1),
              "cfa_best_ranked_dockq": "" if c.get("cfa_best") is None else round(c["cfa_best"], 4)}
             for n, c, d in chosen]
    return rows


# The 50 CFA50 targets as a single list. The second field is the CoFold Arena panel key for a
# target drawn from that panel, or None for one drawn from the collected bad-cases archive.
# Membership, ordering and provenance all derive from this list.
CFA50_TARGETS: Tuple[Tuple[str, Optional[str]], ...] = (
    ('43yy_ab1',  '43yy_sabdab2_H06A2L04R1'),
    ('8t6m_ab1',  '8t6m_sabdab2_H04KHL03HS'),
    ('8uwz_ab1',  '8uwz_sabdab2_H04JWL03H9'),
    ('8vgq_ab1',  '8vgq_sabdab2_H04C5L03CA'),
    ('8vti_ab1',  '8vti_sabdab2_H04HJL03GA'),
    ('8w5d_ab1',  '8w5d_sabdab2_H049FL03AP'),
    ('8wz5_ab1',  '8wz5_sabdab2_H04JYL03HB'),
    ('9bif_ab1',  '9bif_sabdab2_H05AYL0423'),
    ('9bjg_ab1',  '9bjg_sabdab2_H05GDL0467'),
    ('9bp1_ab1',  '9bp1_sabdab2_H04YNL03SK'),
    ('9bu6_ab1',  '9bu6_sabdab2_H05GNL046F'),
    ('9clp_ab1',  '9clp_sabdab2_H058PL040N'),
    ('9dm0_ab1',  '9dm0_sabdab2_H04ZKL03T9'),
    ('9dry_ab1',  '9dry_sabdab2_H05E7L044F'),
    ('9e7n_nb1',  '9e7n_sabdab2_H05CML0000'),
    ('9ebc_ab1',  '9ebc_sabdab2_H0615L04H9'),
    ('9fjf_nb1',  '9fjf_sabdab2_H05C1L0000'),
    ('9hbp_nb1',  '9hbp_sabdab2_H053ZL0000'),
    ('9jqt_ab1',  '9jqt_sabdab2_H04YZL03SR'),
    ('9loe_ab1',  '9loe_sabdab2_H0574L03ZF'),
    ('9m72_ab1',  '9m72_sabdab2_H0566L03YZ'),
    ('9mtf_ab1',  '9mtf_sabdab2_H05G6L0462'),
    ('9myl_ab1',  '9myl_sabdab2_H05JQL0480'),
    ('9pms_ab1',  '9pms_sabdab2_H069HL04QP'),
    ('9sfx_ab1',  '9sfx_sabdab2_H05QVL04BP'),
    ('9ua5_ab1',  '9ua5_sabdab2_H05P1L04AC'),
    ('9ugo_ab1',  '9ugo_sabdab2_H05KVL048T'),
    ('9yxv_ab1',  '9yxv_sabdab2_H0657L04M0'),
    ('8yvk_ab1',  None),
    ('9ejf_ab2',  None),
    ('9eyd_nb1',  None),
    ('9h6v_nb3',  None),
    ('9h77_nb1',  None),
    ('9hls_nb11', None),
    ('9iar_nb1',  None),
    ('9jno_ab1',  None),
    ('9kkj_ab1',  None),
    ('9mla_ab2',  None),
    ('9nsd_nb2',  None),
    ('9nsd_nb3',  None),
    ('9og1_ab1',  None),
    ('9p8x_ab1',  None),
    ('9puo_ab3',  None),
    ('9qt3_ab10', None),
    ('9rxi_nb2',  None),
    ('9uza_ab2',  None),
    ('9vnp_ab1',  None),
    ('9xqb_ab1',  None),
    ('9yc3_nb1',  None),
    ('9yc5_ab1',  None),
)

CFA50_ALL: Tuple[str, ...] = tuple(t for t, _ in CFA50_TARGETS)
CFA50_PANEL_KEYS: Dict[str, str] = {t: k for t, k in CFA50_TARGETS if k is not None}
CFA50_BAD_CASE_IDS: Tuple[str, ...] = tuple(t for t, k in CFA50_TARGETS if k is None)



_AA3 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
        "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
        "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}


def write_targets_dir(targets, dest: str | Path, overwrite: bool = False) -> Path:
    """Write one self-describing structure per target into ``<dest>/<target>/<target>.pdb``.

    The tree this produces is what ``CFA50Dataset(targets_dir=...)`` reads: one directory, keyed by
    target id, no manifest and no panel keys. Each file carries SEQRES built from the target's own
    chains, because the readers take sequences from SEQRES and a coordinates-only PDB loses every
    residue the structure does not resolve.
    """
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb
    import biotite.structure.io.pdbx as pdbx

    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for t in targets:
        out = dest / t.target_id / f"{t.target_id}.pdb"
        if out.is_file() and not overwrite:
            skipped += 1
            continue
        src = Path(t.ground_truth_path) if t.ground_truth_path else None
        if src is None or not src.is_file():
            logger.warning("%s: no source structure, skipping", t.target_id)
            continue
        # AUTHOR chain ids, because both relabel maps are keyed by them: cofoldarena builds its
        # map from sequences_by_auth_chain, and the bad-cases tree is already author-labelled.
        if src.suffix.lower() in (".cif", ".mmcif"):
            arr = pdbx.get_structure(pdbx.CIFFile.read(str(src)), model=1, use_author_fields=True)
        else:
            arr = pdb.PDBFile.read(str(src)).get_structure(model=1)
        arr = arr[struc.filter_amino_acids(arr)]
        mapping = t.metadata.get("relabel_map") or t.metadata.get("relabel") or {}
        if mapping:
            keep = arr[[c in mapping for c in arr.chain_id]]
            keep.chain_id = [mapping[c] for c in keep.chain_id]
        else:
            keep = arr
        if keep.array_length() == 0:
            logger.warning("%s: no amino-acid chains survived, skipping", t.target_id)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        f = pdb.PDBFile()
        f.set_structure(keep)
        lines = list(f.lines)
        seqres = []
        for ch in t.chains:
            res = [_AA3.get(c, "UNK") for c in ch.sequence]
            for i in range(0, len(res), 13):
                seqres.append("SEQRES %3d %s %4d  %s" % (
                    i // 13 + 1, ch.chain_id[:1], len(res), " ".join(res[i:i + 13])))
        at = next((i for i, l in enumerate(lines) if l.startswith(("ATOM", "HETATM", "MODEL"))), 0)
        f.lines = lines[:at] + seqres + lines[at:]
        f.write(str(out))
        t.metadata["source_structure"] = str(src)
        written += 1
    logger.info("targets dir: %d written, %d already present, into %s", written, skipped, dest)
    return dest

class CFA50Dataset:
    """The 50-target CAST evaluation set, built end to end from ``CFA50_TARGETS``.

    Each target is loaded from the archive its entry names: the CoFold Arena panel when the
    entry carries a panel key, the collected bad-cases archive when it carries None.
    """

    name = "cfa50"
    task_prefix = "interface_antibody_antigen"

    def __init__(self, structures_dir: Optional[str | Path] = None,
                 bad_cases_dir: Optional[str | Path] = None,
                 cfa_manifest: str | Path = DEFAULT_MANIFEST,
                 relabel: bool = True,
                 targets_dir: Optional[str | Path] = None):
        self.structures_dir = Path(structures_dir) if structures_dir else None
        self.bad_cases_dir = Path(bad_cases_dir) if bad_cases_dir else None
        self.cfa_manifest = Path(cfa_manifest)
        self.relabel = relabel
        # ONE DIRECTORY FOR ALL 50, `<dir>/<target>/<target>.pdb`, keyed by target id. Takes
        # precedence over the two-archive path, which needs a panel manifest and panel-key naming
        # for one half and a bad-cases tree for the other.
        self.targets_dir = Path(targets_dir) if targets_dir else None
        # Fall back to the shipped tree, but only when neither source archive was named, so passing
        # structures_dir/bad_cases_dir still selects the two-archive path.
        if self.targets_dir is None and self.structures_dir is None and self.bad_cases_dir is None \
                and Path(DEFAULT_TARGETS_DIR).is_dir():
            self.targets_dir = Path(DEFAULT_TARGETS_DIR)

    # ---------------------------------------------------------------- membership
    def split(self) -> Dict[str, List[str]]:
        """Source archive -> target ids, in the set's own naming."""
        return {"cofoldarena": sorted(CFA50_PANEL_KEYS),
                "collected_bad_cases": list(CFA50_BAD_CASE_IDS)}

    def legacy_ids(self) -> Dict[str, str]:
        """Target id -> the panel key its pre-rename artefacts carry."""
        return dict(CFA50_PANEL_KEYS)

    def rerun_targets(self) -> List[str]:
        """The targets that need new sampling."""
        return sorted(CFA50_PANEL_KEYS)

    # ---------------------------------------------------------------- load
    def load(self) -> List[Target]:
        want = set(CFA50_ALL)
        out: List[Target] = []

        if self.targets_dir is not None:
            from data.datasets.collected_bad_cases import (CollectedBadCasesDataset,
                                                            parse_seqres)
            out = CollectedBadCasesDataset(bad_cases_dir=self.targets_dir,
                                           only=list(CFA50_ALL)).load()
            # The reader tags every target as a bad case. The half a target came from is a property
            # of the SET, not of the directory it was read from, so restore it from CFA50_TARGETS.
            for t in out:
                if t.target_id in CFA50_PANEL_KEYS:
                    t.task = self.task_prefix
                    t.metadata["legacy_target_id"] = CFA50_PANEL_KEYS[t.target_id]
                # CHAIN ORDER IS TOKEN ORDER, and every map is a token index, so it has to be the
                # file's order and not the reader's canonical heavy-light-antigen one.
                order = {c: i for i, c in enumerate(parse_seqres(
                    self.targets_dir / t.target_id / (t.target_id + ".pdb")))}
                t.chains.sort(key=lambda ch: order.get(ch.chain_id, len(order)))
            return self._finish(out, want)

        if self.structures_dir is None:
            raise ValueError("CFA50Dataset needs structures_dir for the %d cofoldarena target(s)"
                             % len(CFA50_PANEL_KEYS))
        # panel_only=False: membership has already made the selection, and re-applying the panel's
        # release cutoff on top of it could only ever DROP a chosen target, silently.
        by_panel_key = {v: k for k, v in CFA50_PANEL_KEYS.items()}
        cfa = CoFoldArenaDataset(manifest=self.cfa_manifest, structures_dir=self.structures_dir,
                                 only=list(by_panel_key), relabel=self.relabel,
                                 panel_only=False).load()
        for t in cfa:
            t.metadata["legacy_target_id"] = t.target_id
            t.target_id = by_panel_key[t.target_id]
        out += cfa

        if self.bad_cases_dir is None:
            raise ValueError("CFA50Dataset needs bad_cases_dir for the %d collected_bad_cases "
                             "target(s)" % len(CFA50_BAD_CASE_IDS))
        from data.datasets.collected_bad_cases import CollectedBadCasesDataset
        out += CollectedBadCasesDataset(bad_cases_dir=self.bad_cases_dir,
                                        only=list(CFA50_BAD_CASE_IDS)).load()

        return self._finish(out, want)

    def _finish(self, out: List[Target], want: set) -> List[Target]:
        """Membership check and per-half tagging, shared by both source paths."""
        got = {t.target_id for t in out}
        if got != want:
            miss, extra = sorted(want - got), sorted(got - want)
            raise RuntimeError(
                "CFA50Dataset loaded %d of %d targets. Missing (%d): %s. Extra (%d): %s"
                % (len(got), len(want), len(miss), miss or "none", len(extra), extra or "none"))
        for t in out:
            t.metadata["cfa50_source"] = ("cofoldarena" if t.target_id in CFA50_PANEL_KEYS
                                          else "collected_bad_cases")
            t.metadata["cfa50_rerun"] = t.target_id in CFA50_PANEL_KEYS
        logger.info("CFA50Dataset: %d targets (%d cofoldarena, %d collected_bad_cases), %d to re-run",
                    len(out), len(CFA50_PANEL_KEYS), len(CFA50_BAD_CASE_IDS), len(self.rerun_targets()))
        return out

    # ---------------------------------------------------------------- assemble
    def assemble(self, targets: List[Target], data_root: str | Path, gt_root: str | Path,
                 wired_dirs: Sequence[str | Path] = (), gt_dirs: Sequence[str | Path] = (),
                 maps: Optional[Mapping[str, Sequence[str | Path]]] = None) -> Dict[str, Any]:
        """Materialise the rest of the dataset beside the inputs, in the same pass."""
        data_root, gt_root = Path(data_root), Path(gt_root)
        ids = [t.target_id for t in targets]
        legacy = {t.target_id: t.metadata.get("legacy_target_id") for t in targets}

        def _names(tid):
            """Current id first, then the legacy one, so a rename never silently loses a file."""
            return [n for n in (tid, legacy.get(tid)) if n]
        rerun = self.rerun_targets()
        (data_root / "inputs_wired").mkdir(parents=True, exist_ok=True)
        (data_root / "targets_50.txt").write_text("\n".join(sorted(ids)) + "\n")
        (data_root / "targets_rerun.txt").write_text("\n".join(sorted(rerun)) + "\n")

        gts = gt_root / "ground_truths"
        gts.mkdir(parents=True, exist_ok=True)
        n_gt, no_gt, n_gt_reused = 0, [], 0
        for t in targets:
            src = None
            for d in gt_dirs:
                for nm in _names(t.target_id):
                    for cand in (Path(d) / nm / f"{nm}.pdb", Path(d) / nm / f"{nm}.cif",
                                 Path(d) / f"{nm}.pdb", Path(d) / f"{nm}.cif"):
                        if cand.is_file():
                            src, n_gt_reused = cand, n_gt_reused + 1
                            break
                    if src is not None:
                        break
                if src is not None:
                    break
            if src is None and t.ground_truth_path:
                src = Path(t.ground_truth_path)
            if src is None or not src.is_file():
                no_gt.append(t.target_id)
                continue
            dst = gts / t.target_id / f"{t.target_id}{src.suffix}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            t.ground_truth_path = str(dst)
            n_gt += 1
        alias = gt_root / "ground_truth"
        if not alias.exists():
            alias.symlink_to("ground_truths")

        n_wired, no_wired, n_renamed = 0, [], 0
        for tid in ids:
            src = next((Path(d) / f"{nm}.json" for d in wired_dirs for nm in _names(tid)
                        if (Path(d) / f"{nm}.json").is_file()), None)
            if src is None:
                no_wired.append(tid)
                continue
            obj = json.load(open(src))
            body = obj[0] if isinstance(obj, list) else obj
            if body.get("name") != tid:
                body["name"] = tid
                n_renamed += 1
            with open(data_root / "inputs_wired" / f"{tid}.json", "w") as fh:
                json.dump(obj, fh, indent=1)
            n_wired += 1

        # Accept a source map keyed by either naming, and always WRITE the current id.
        by_name = {nm: tid for tid in ids for nm in _names(tid)}
        merged: Dict[str, int] = {}
        for name, sources in (maps or {}).items():
            out: Dict[str, Any] = {}
            stamps: Dict[str, Any] = {}
            for src in sources:
                obj = json.load(open(src))
                for k, v in obj.items():
                    if k.startswith("_"):
                        if k in stamps and stamps[k] != v:
                            if k in ("_source", "_builder"):
                                raise ValueError(
                                    "map %s: source maps disagree on %s (%r vs %r); a merged map "
                                    "cannot claim a provenance its halves do not share"
                                    % (name, k, stamps[k], v))
                            have = stamps[k] if isinstance(stamps[k], list) else [stamps[k]]
                            stamps[k] = have + [v] if v not in have else have
                        else:
                            stamps[k] = v
                    elif k in by_name:
                        tid = by_name[k]
                        if tid in out:
                            raise ValueError("map %s: %s is claimed by two source maps"
                                             % (name, tid))
                        out[tid] = v
            out.update(stamps)
            (data_root / "maps").mkdir(parents=True, exist_ok=True)
            json.dump(out, open(data_root / "maps" / f"{name}.json", "w"), indent=1)
            merged[name] = len(out) - len(stamps)

        rep = {"targets": len(ids), "rerun": len(rerun), "ground_truth": n_gt,
               "ground_truth_reused": n_gt_reused, "missing_ground_truth": no_gt,
               "wired": n_wired, "wired_name_rewritten": n_renamed, "missing_wired": no_wired,
               "maps": merged}
        if no_gt:
            logger.warning("assemble: %d target(s) with no reference on disk: %s", len(no_gt), no_gt)
        if no_wired:
            logger.warning("assemble: %d target(s) have no wired input in %s: %s. They will run "
                           "with no MSA unless wired first.", len(no_wired), list(wired_dirs), no_wired)
        return rep
