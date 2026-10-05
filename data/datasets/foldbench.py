"""FoldBench (BEAM-Labs/FoldBench) loader, plus installation of the Protenix results bundle.

Repo:  https://github.com/BEAM-Labs/FoldBench
Paper: Nature Communications 2025, "Benchmarking all-atom biomolecular structure prediction with
FoldBench"

Returns backend-agnostic :class:`data.target.Target` objects. Converting them to a model's input
format is the backend's job (see :meth:`models.base.ModelBackend.emit_input`).

TWO SOURCES, NOT INTERCHANGEABLE. ``targets/interface_antibody_antigen.csv`` (BEAM-Labs) carries
the CHAIN PAIRING, which chain is antibody and which antigen, and is the only place that exists;
its schema is checked against ``_IFACE_CHAIN_COLS``. The Protenix bundle at FOLDBENCH_URL is an
EVALUATION bundle of CSVs with no structures, no sequences and no ``targets/`` directory::

    foldbench/raw/{Protenix-v0_5_0,Protenix-v1}/interface_*_{ost,dockqv2}.csv
    foldbench/results/{intersection_all,intersection_af3}/interface_*.csv, monomer_*.csv

``results/intersection_all/interface_antibody_antigen.csv`` holds 320 rows, 160 antibody-antigen
interfaces over 104 ``pdb_id`` values such as ``8gho-assembly1``, scored for two models.
``intersection_all`` is the subset every benchmarked model produced output for;
``intersection_af3`` is the looser subset, 110 targets over 167 interfaces. This loader reads the
results table for its IDENTIFIER COLUMN ALONE and takes chains from the targets CSV, so the bundle
acts as a subset filter and its misnamed chain columns (``interface_Chai-1n_id_1``, from an
upstream global replace of ``chai``) never matter.

``python -m data.datasets.foldbench`` exposes ``--install`` and ``--inspect``, the latter printing
every ab-ag CSV header found in a bundle. The bundle host is firewalled on some networks, where DNS
resolves but TCP 443 never connects, so :func:`install` prints the manual command rather than
retrying.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from data.target import Chain, Target

logger = logging.getLogger(__name__)


# Column-name candidates by role. First hit wins.
_ID_COLS = ["target_id", "id", "pdb_id", "entry_id", "assembly_id", "name"]
_CHAIN_ID_COLS = ["chain_id", "chain", "chain_ids", "chain_label"]
_CHAIN_SEQ_COLS = ["sequence", "seq", "aa_sequence", "protein_sequence"]
_CHAIN_TYPE_COLS = ["chain_type", "type", "molecule_type", "entity_type"]
_GROUND_TRUTH_COLS = ["ground_truth", "native_cif", "cif_path", "reference_cif", "gt_path"]


# ---------------------------------------------------------------------- the Protenix bundle
FOLDBENCH_URL = "https://protenix.tos-cn-beijing.volces.com/benchmark/v1/foldbench.tar.gz"

_ABAG_RESULTS_REL = "results/intersection_all/interface_antibody_antigen.csv"
_ABAG_TARGETS_REL = "targets/interface_antibody_antigen.csv"


def _safe_extract(tar, dest: Path) -> None:
    """Extract refusing absolute paths, parent escapes and links, which tarfile allows by default."""
    dest = dest.resolve()
    for m in tar.getmembers():
        target = (dest / m.name).resolve()
        if not str(target).startswith(str(dest) + "/") and target != dest:
            raise RuntimeError(f"refusing tar member escaping the destination: {m.name!r}")
        if m.issym() or m.islnk():
            raise RuntimeError(f"refusing link member in the archive: {m.name!r}")
    tar.extractall(dest)


def install(dest: str | Path, url: str = FOLDBENCH_URL, force: bool = False,
            archive: Optional[str | Path] = None) -> Path:
    """Extract the FoldBench bundle into ``dest``, fetching it first unless it is already local."""
    import shutil
    import tarfile
    import urllib.request

    dest = Path(dest).expanduser()
    marker = dest / ".foldbench_installed"
    if marker.is_file() and not force:
        logger.info("FoldBench already installed at %s (pass force=True to refetch)", dest)
        return dest
    dest.mkdir(parents=True, exist_ok=True)

    if archive is not None:
        archive = Path(archive).expanduser()
        if not archive.is_file():
            raise FileNotFoundError(f"archive not found: {archive}")
        logger.info("installing from a local archive: %s", archive)
    else:
        archive = dest / "foldbench.tar.gz"
    if not archive.is_file() or force:
        tmp = archive.with_suffix(".part")
        logger.info("downloading %s", url)
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"could not fetch {url}: {exc}\n"
                "This host is firewalled on some networks: DNS resolves but TCP 443 does not "
                "connect. Fetch it somewhere with access and copy it in, then rerun:\n"
                f"    curl -fL -o {archive} {url}\n"
                f"    python -m data.datasets.foldbench --install {dest}"
            ) from exc
        tmp.replace(archive)                       # rename only once the body is complete
    logger.info("extracting %s (%.1f MB)", archive.name, archive.stat().st_size / 1e6)
    with tarfile.open(archive, "r:gz") as tar:
        _safe_extract(tar, dest)
    marker.write_text(url + "\n")

    root = _bundle_root(dest)
    try:
        fetch_targets_csv(root)
    except Exception as exc:
        # Non-fatal: the results subset is still usable, the loader just needs targets_dir passed.
        logger.warning("could not fetch the chain-pairing CSV (%s); pass targets_dir explicitly", exc)
    logger.info("FoldBench installed: %s", root)
    return root


BEAM_TARGETS_URL = ("https://raw.githubusercontent.com/BEAM-Labs/FoldBench/main/targets/"
                    "interface_antibody_antigen.csv")

# RCSB assembly mmCIF. pdb_id values are '<entry>-assembly<N>', which maps to this file name after
RCSB_ASSEMBLY_URL = "https://files.rcsb.org/download/{entry}-assembly{n}.cif"

FOLDBENCH_GT_DRIVE_ID = "17KdWDXKATaeHF6inPxhPHIRuIzeqiJxS"
FOLDBENCH_GT_MEMBER_PREFIX = "ground_truth_20250520/"


def fetch_official_ground_truth(dest: str | Path, only: Optional[Sequence[str]] = None,
                                drive_id: str = FOLDBENCH_GT_DRIVE_ID,
                                archive: Optional[str | Path] = None) -> Path:
    """Download FoldBench's own ground-truth CIFs into ``dest``, optionally only ``only`` targets."""
    import tarfile
    import urllib.parse
    import urllib.request

    dest = Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    tgz = Path(archive) if archive else None

    if tgz is None:
        tgz = dest / ".foldbench_gt.tar.gz"
        if not (tgz.is_file() and tgz.stat().st_size > 0):
            base = "https://drive.google.com/uc?export=download&id=" + drive_id
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
            with opener.open(base, timeout=120) as r:
                head = r.read(65536).decode("utf-8", "ignore")
            # A file this size returns a scan-warning page carrying the token, not the bytes.
            import re
            uuid = (re.search(r'name="uuid" value="([^"]+)"', head) or [None, ""])[1]
            conf = (re.search(r'name="confirm" value="([^"]+)"', head) or [None, "t"])[1]
            url = ("https://drive.usercontent.google.com/download?"
                   + urllib.parse.urlencode({"id": drive_id, "export": "download",
                                             "confirm": conf, "uuid": uuid}))
            logger.info("fetching FoldBench ground truth (~1 GB) -> %s", tgz)
            with opener.open(url, timeout=1800) as r, open(tgz, "wb") as fh:
                shutil.copyfileobj(r, fh)

    want = None if only is None else {f"{FOLDBENCH_GT_MEMBER_PREFIX}{t}.cif" for t in only}
    got = 0
    with tarfile.open(tgz) as tf:
        for m in tf:
            if not m.name.endswith(".cif"):
                continue
            if want is not None and m.name not in want:
                continue
            m.name = os.path.basename(m.name)          # flat layout, no release-dated directory
            tf.extract(m, dest)
            got += 1
    logger.info("official ground truth: %d cif(s) -> %s", got, dest)
    if want is not None and got != len(want):
        logger.warning("asked for %d target(s), extracted %d -- missing: %s",
                       len(want), got,
                       sorted({Path(w).stem for w in want} - {p.stem for p in dest.glob("*.cif")}))
    return dest


def fetch_targets_csv(root: str | Path, url: str = BEAM_TARGETS_URL, force: bool = False) -> Path:
    """Put the BEAM-Labs antibody-antigen task CSV at ``root/targets/`` so the bundle is complete."""
    import urllib.request
    root = _bundle_root(Path(root))
    out = root / _ABAG_TARGETS_REL
    if out.is_file() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("fetching the chain-pairing CSV: %s", url)
    tmp = out.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as r:
        tmp.write_bytes(r.read())
    tmp.replace(out)
    return out


def fetch_structures(root: str | Path, dest: str | Path, subset: str = "intersection_all",
                     task: str = "interface_antibody_antigen", force: bool = False) -> Path:
    """Download the RCSB assembly mmCIF for every identifier in a bundle subset."""
    import urllib.request
    root = _bundle_root(Path(root))
    dest = Path(dest).expanduser(); dest.mkdir(parents=True, exist_ok=True)
    ids = sorted(_subset_ids(root / "results" / subset / f"{task}.csv"))
    got = skipped = failed = 0
    for tid in ids:
        out = dest / f"{tid}.cif"
        if out.is_file() and out.stat().st_size > 0 and not force:
            skipped += 1
            continue
        entry, _, asm = tid.partition("-assembly")
        url = RCSB_ASSEMBLY_URL.format(entry=entry.upper(), n=asm or "1")
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                body = r.read()
            tmp = out.with_suffix(".part"); tmp.write_bytes(body); tmp.replace(out)
            got += 1
        except Exception as exc:
            logger.warning("%s: could not fetch %s (%s)", tid, url, exc)
            failed += 1
    logger.info("structures: %d fetched, %d already present, %d failed, into %s",
                got, skipped, failed, dest)
    if failed:
        logger.warning("%d of %d identifiers have no structure, so they will be dropped for want of "
                       "sequences", failed, len(ids))
    return dest


def _bundle_root(dest: Path) -> Path:
    """The directory holding targets/ or results/, whether or not the tar had a top-level folder."""
    dest = Path(dest)
    for cand in (dest, *(d for d in sorted(dest.iterdir()) if d.is_dir())) if dest.is_dir() else ():
        if (cand / "results").is_dir() or (cand / "targets").is_dir():
            return cand
    return dest


def find_abag_csv(root: str | Path, prefer_intersection: bool = True) -> Optional[Path]:
    """Locate the antibody-antigen CSV in an extracted bundle."""
    root = _bundle_root(Path(root))
    order = [_ABAG_RESULTS_REL, _ABAG_TARGETS_REL]
    if not prefer_intersection:
        order.reverse()
    for rel in order:
        p = root / rel
        if p.is_file():
            return p
    hits = sorted(root.rglob("interface_antibody_antigen.csv"))
    if hits:
        logger.info("interface_antibody_antigen.csv found at a non-standard path: %s", hits[0])
        return hits[0]
    return None


def _subset_ids(csv_path: Path) -> set[str]:
    """Identifiers named by a results CSV, read WITHOUT assuming its metric columns."""
    df = pd.read_csv(csv_path)
    for c in _ID_COLS:
        if c in df.columns:
            ids = {str(v).strip() for v in df[c].dropna()}
            logger.info("%s: %d rows, %d distinct %s", csv_path.name, len(df), len(ids), c)
            return ids
    raise ValueError(
        f"{csv_path} has no recognised identifier column. Looked for {_ID_COLS}. "
        f"Header is: {list(df.columns)}"
    )


# Kept as aliases for backwards compatibility with any existing scripts/docs.
FoldBenchChain = Chain
FoldBenchTarget = Target


def _first(row: pd.Series, cols: List[str]) -> Optional[str]:
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    return None


_IFACE_CHAIN_COLS = ["interface_chain_id_1", "interface_chain_id_2"]
_IFACE_TYPE_COLS = ["interface_chain_type_1", "interface_chain_type_2"]
_ROLE_COLS = {"antibody_chain": "antibody", "antigen_chain": "antigen"}

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def sequences_from_structure(path: str | Path) -> dict[str, str]:
    """``{chain_id: one-letter sequence}`` for the amino-acid chains of a reference structure."""
    import numpy as np
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    path = str(path)
    cif = pdbx.CIFFile.read(path)
    arr = pdbx.get_structure(cif, model=1, use_author_fields=False)
    arr = arr[struc.filter_amino_acids(arr)]
    out: dict[str, str] = {}
    for ch in np.unique(arr.chain_id):
        sub = arr[arr.chain_id == ch]
        # one entry per residue, in file order
        _, first_idx = np.unique(sub.res_id, return_index=True)
        names = sub.res_name[np.sort(first_idx)]
        seq = "".join(_THREE_TO_ONE.get(str(n).upper(), "X") for n in names)
        if seq:
            out[str(ch)] = seq
    return out


class FoldBenchDataset:
    """Load one or more FoldBench task CSVs into :class:`Target` objects."""

    name = "foldbench"

    def __init__(
        self,
        targets_dir: Optional[str | Path] = None,
        tasks: Optional[List[str]] = None,
        ground_truth_dir: Optional[str | Path] = None,
        root: Optional[str | Path] = None,
        subset_csv: Optional[str | Path] = None,
        install_if_missing: bool = False,
        url: str = FOLDBENCH_URL,
        relabel: bool = True,
        gt_out_dir: Optional[str | Path] = None,
    ):
        """root : the extracted Protenix bundle. Supplies ``targets_dir`` when that is not given, and,
                    unless ``subset_csv`` says otherwise, its ``results/intersection_all`` antibody-antigen
                    table as the subset filter.
                subset_csv : restrict to the identifiers in this CSV. Only its id column is read, so any
                    results table works. Pass ``""`` to load the full task with no filtering.
                install_if_missing : fetch and extract the bundle into ``root`` when it is not there yet.
                relabel : rename chains to the antibody convention H/L/A/B, as the bad-cases tree does.
                    ON BY DEFAULT because dockq_abag identifies the antibody BY CHAIN ID
                    (method/metrics/dockq.py: DEFAULT_ANTIBODY_CHAINS = ("H", "L")), so native label_asym_id
                    names would make that metric split the complex wrongly and report a number that looks
                    fine. Turn it off only to inspect the native naming.
        """
        if root is not None and install_if_missing and not _bundle_root(Path(root)).is_dir():
            install(root, url=url)
        self.root = _bundle_root(Path(root)) if root is not None else None

        if targets_dir is None:
            if self.root is None:
                raise ValueError("FoldBenchDataset needs targets_dir or root")
            targets_dir = self.root / "targets"
            if not Path(targets_dir).is_dir():
                raise FileNotFoundError(
                    f"{self.root} has no targets/ directory. The Protenix bundle is evaluation "
                    "results only (raw/ and results/), and carries no chain pairing. Pass "
                    "targets_dir pointing at the BEAM-Labs FoldBench targets/ CSVs "
                    "(https://github.com/BEAM-Labs/FoldBench); this bundle then supplies the "
                    "intersection_all subset filter.")
        self.targets_dir = Path(targets_dir)
        if not self.targets_dir.is_dir():
            raise FileNotFoundError(f"FoldBench targets dir not found: {self.targets_dir}")

        self.subset_ids: Optional[set[str]] = None
        if subset_csv is None and self.root is not None:
            subset_csv = self.root / _ABAG_RESULTS_REL
            if not Path(subset_csv).is_file():
                subset_csv = find_abag_csv(self.root) or ""
                if subset_csv and Path(subset_csv).name == Path(_ABAG_TARGETS_REL).name \
                        and _ABAG_RESULTS_REL not in str(subset_csv):
                    # only the targets CSV is present, which IS the task list, so filtering by it
                    # would be a no-op dressed up as a subset. Say so rather than imply a subset.
                    logger.info("no results/intersection_all table in the bundle, loading the full task")
                    subset_csv = ""
        self.relabel = relabel
        # When set, load() writes the relabelled references here and repoints every target at them,
        # so one prepare_dataset call yields inputs and ground truth that agree on chain ids.
        # prepare_dataset sets it by default, to {runs_root}/{dataset}/ground_truths.
        self.gt_out_dir = Path(gt_out_dir) if gt_out_dir else None
        if subset_csv:
            self.subset_ids = _subset_ids(Path(subset_csv))
            logger.info("restricting to %d identifiers from %s",
                        len(self.subset_ids), Path(subset_csv).name)
        gt = Path(ground_truth_dir) if ground_truth_dir else None
        # the released bundle extracts to a dated subdirectory (ground_truth_20250520/), so accept
        # either the bundle root or the directory holding the CIFs
        if gt is not None and gt.is_dir() and not any(gt.glob('*.cif')):
            nested = [d for d in sorted(gt.iterdir()) if d.is_dir() and any(d.glob('*.cif'))]
            if len(nested) == 1:
                logger.info('ground-truth CIFs found in nested dir %s', nested[0].name)
                gt = nested[0]
        self.ground_truth_dir = gt
        self.tasks = tasks

    # ------------------------------------------------------------------ loading
    def _iter_task_csvs(self) -> Iterable[Path]:
        if self.tasks:
            for name in self.tasks:
                p = self.targets_dir / f"{name}.csv"
                if not p.is_file():
                    logger.warning("Task CSV not found: %s", p)
                    continue
                yield p
        else:
            yield from sorted(self.targets_dir.glob("*.csv"))

    def load(self) -> List[Target]:
        targets: dict[str, Target] = {}
        for csv_path in self._iter_task_csvs():
            df = pd.read_csv(csv_path)
            task = csv_path.stem
            logger.info("Loading %d rows from %s", len(df), csv_path.name)

            for _, row in df.iterrows():
                tid = _first(row, _ID_COLS)
                if not tid:
                    logger.debug("Skipping row without id in %s: %s", csv_path.name, row.to_dict())
                    continue
                tgt = targets.setdefault(tid, Target(target_id=tid, task=task,
                                                     metadata={"raw_row": row.to_dict()}))
                chain_id = _first(row, _CHAIN_ID_COLS) or "A"
                seq = _first(row, _CHAIN_SEQ_COLS)
                chain_type = (_first(row, _CHAIN_TYPE_COLS) or "protein").lower()
                if seq:
                    tgt.chains.append(Chain(chain_id=chain_id, sequence=seq, chain_type=chain_type))
                else:
                    ctypes = tgt.metadata.setdefault("chain_types", {})
                    roles = tgt.metadata.setdefault("chain_roles", {})
                    for cid_col, ty_col in zip(_IFACE_CHAIN_COLS, _IFACE_TYPE_COLS):
                        cid = _first(row, [cid_col])
                        if cid:
                            ctypes.setdefault(cid, (_first(row, [ty_col]) or "protein").lower())
                    for col, role in _ROLE_COLS.items():
                        cid = _first(row, [col])
                        if cid:
                            roles[cid] = role
                            ctypes.setdefault(cid, "protein")
                    ab, ag = _first(row, ["antibody_chain"]), _first(row, ["antigen_chain"])
                    if ab and ag:
                        ifaces = tgt.metadata.setdefault("interfaces", [])
                        if [ab, ag] not in ifaces:
                            ifaces.append([ab, ag])

                gt = _first(row, _GROUND_TRUTH_COLS)
                if gt and not tgt.ground_truth_path:
                    tgt.ground_truth_path = gt

        # Attach ground-truth CIFs by convention
        if self.ground_truth_dir:
            for tgt in targets.values():
                if tgt.ground_truth_path:
                    continue
                for ext in (".cif", ".mmcif", ".pdb"):
                    candidate = self.ground_truth_dir / f"{tgt.target_id}{ext}"
                    if candidate.is_file():
                        tgt.ground_truth_path = str(candidate)
                        break

        n_filled = 0
        for tgt in targets.values():
            if tgt.chains or not tgt.ground_truth_path:
                continue
            try:
                seqs = sequences_from_structure(tgt.ground_truth_path)
            except Exception as exc:                      # one unreadable file must not kill the load
                logger.warning("could not read sequences from %s: %s", tgt.ground_truth_path, exc)
                continue
            ctypes = tgt.metadata.get("chain_types", {})
            roles = tgt.metadata.get("chain_roles", {})
            named = {c for c in (ctypes or roles or {}) if c in seqs}
            partners = _cognate_partners(tgt.ground_truth_path, named, roles, seqs)
            for cid, seq in partners.items():
                roles.setdefault(cid, "antibody")
                ctypes.setdefault(cid, "protein")
            present = {c: q for c, q in ((c, seqs.get(c)) for c in (set(named) | set(partners))) if q}
            # The map is built from the chains actually PRESENT, so a chain missing from the
            # structure cannot consume the H label and leave the real heavy chain as L.
            mapping = _relabel_map({c: roles[c] for c in present if c in roles}, present) \
                if (self.relabel and roles) else {}
            if mapping:
                tgt.metadata["relabel"] = mapping
                tgt.metadata["chain_roles"] = {mapping.get(c, c): r for c, r in roles.items()}
                tgt.metadata["interfaces"] = [[mapping.get(a, a), mapping.get(b, b)]
                                              for a, b in tgt.metadata.get("interfaces", [])]
            for cid, seq in present.items():
                tgt.chains.append(Chain(chain_id=mapping.get(cid, cid), sequence=seq,
                                        chain_type=ctypes.get(cid, "protein")))
            if tgt.chains:
                n_filled += 1
        if n_filled:
            logger.info("filled sequences from reference structures for %d targets", n_filled)

        out = [t for t in targets.values() if t.chains]
        # A chain the backends cannot serialise is worse than a missing one: it counts in
        # target_index.csv and vanishes from the emitted input. Refuse rather than prepare those.
        bad = {c.chain_type for t in out for c in t.chains} - {"protein", "rna", "dna", "ligand"}
        if bad:
            raise ValueError(
                f"chain_type values no backend can emit: {sorted(bad)}. chain_type is the MOLECULE "
                "type; antibody/antigen belong in metadata['chain_roles'].")
        if self.subset_ids is not None:
            kept = [t for t in out if t.target_id in self.subset_ids]
            missing = self.subset_ids - {t.target_id for t in out}
            # A subset that matches nothing means the two files key on different identifiers, which
            # would otherwise surface as "0 targets" with no cause. Fail with both id samples.
            if not kept:
                raise ValueError(
                    f"the subset filter matched none of the {len(out)} loaded targets, so the two "
                    f"CSVs key on different identifiers.\n"
                    f"  loaded ids  e.g. {sorted(t.target_id for t in out)[:3]}\n"
                    f"  subset ids  e.g. {sorted(self.subset_ids)[:3]}")
            logger.info("subset: kept %d of %d targets (%d subset ids had no target)",
                        len(kept), len(out), len(missing))
            out = kept
        if self.gt_out_dir is not None:
            write_ground_truth(out, self.gt_out_dir)
        return out


def dump_target_index(targets: List[Target], path: str | Path) -> Path:
    """Persist a CSV index of (target_id, task, n_chains, ground_truth) for later joining."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{
        "target_id": t.target_id,
        "task": t.task,
        "n_chains": t.n_chains,
        "chain_ids": ";".join(t.chain_ids()),
        "chain_types": ";".join(c.chain_type for c in t.chains),
        "chain_roles": ";".join((t.metadata.get("chain_roles") or {}).get(c.chain_id, "")
                                for c in t.chains),
        "interfaces": ";".join("+".join(pair) for pair in (t.metadata.get("interfaces") or [])),
        "ground_truth_cif": t.ground_truth_path or "",
    } for t in targets])
    df.to_csv(path, index=False)
    return path


_AB_LABELS = ("H", "L")
_AG_LABELS = ("A", "B", "C", "D")


_VH_MOTIFS = ("WVRQ", "WIRQ", "WVKQ", "WVRH", "WVQQ")
_VL_MOTIFS = ("WYQQ", "WFQQ", "WYLQ", "WLQQ", "WYRQ")
_VH_PREFIXES = ("QVQL", "EVQL", "QVKL", "QIQL", "QVTL", "EVKL", "QLQL", "QITL", "VQLV", "QMQL",
                "DVQL", "AVQL", "QEQL", "EVKV", "QVHL")
_VL_PREFIXES = ("DIQM", "DIVM", "EIVL", "AIQL", "QSVL", "QSAL", "SYEL", "QAVL", "NFML", "DIVL",
                "EIVM", "QSVV", "QLVL", "SSEL", "QPVL", "DIEL", "ETTL", "QAGL", "DIQL", "EIVQ")


def _ig_class(seq: str) -> Optional[str]:
    """``'H'`` for a heavy or VHH V domain, ``'L'`` for a light one, else ``None``."""
    s = (seq or "").upper()
    v = s[:60]
    if any(m in v for m in _VH_MOTIFS):
        return "H"
    if any(m in v for m in _VL_MOTIFS):
        return "L"
    if s[:4] in _VH_PREFIXES:
        return "H"
    if s[:4] in _VL_PREFIXES:
        return "L"
    return None


def _ig_kind(seq: str) -> Optional[str]:
    """``'antibody'`` when the sequence looks like an Ig V domain, else ``None``."""
    return "antibody" if _ig_class(seq) else None


def _cognate_partners(structure_path, named, roles, seqs) -> dict:
    """The VH/VL partner of each named ANTIBODY chain, when the CSV did not name it."""
    ab_named = [c for c in named if roles.get(c) == "antibody"]
    cand = [c for c in seqs if c not in named and _ig_kind(seqs[c])]
    if len(ab_named) != 1 or not cand:
        return {}
    try:
        import numpy as np
        import biotite.structure as struc
        import biotite.structure.io.pdbx as pdbx
        arr = pdbx.get_structure(pdbx.CIFFile.read(str(structure_path)), model=1,
                                 use_author_fields=False)
        arr = arr[struc.filter_amino_acids(arr)]
        xyz = {c: arr[arr.chain_id == c].coord for c in set(ab_named) | set(cand)}
    except Exception as exc:
        logger.warning("could not read coordinates for cognate pairing (%s): %s",
                       structure_path, exc)
        return {}
    out = {}
    for a in ab_named:
        A = xyz.get(a)
        if A is None or len(A) == 0:
            continue
        best, bd = None, 1e9
        for c in cand:
            C = xyz.get(c)
            if C is None or len(C) == 0 or c in out:
                continue
            d = float(np.sqrt(((A[:, None, :] - C[None, :, :]) ** 2).sum(-1)).min())
            if d < bd:
                best, bd = c, d
        # 5 A is a contact, and a VH/VL pair is always in contact. A chain further away is a
        # different antibody in the same assembly and must not be pulled in.
        if best is not None and bd <= 5.0 and not out:      # at most ONE partner, a Fab has two chains
            out[best] = seqs[best]
            logger.debug("%s: adding cognate partner %s of %s at %.2f A", structure_path, best, a, bd)
    return out


def _relabel_map(roles: dict, seqs: Optional[dict] = None) -> dict:
    """``{native_chain_id: H|L|A|B}`` for one target."""
    ab = sorted(c for c, r in roles.items() if r == "antibody")
    ag = sorted(c for c, r in roles.items() if r == "antigen")
    if seqs:
        def _rank(c):
            k = _ig_class(seqs.get(c, ""))
            return (0 if k == "H" else (2 if k == "L" else 1), -len(seqs.get(c, "")), c)
        ab.sort(key=_rank)
    if len(ab) > len(_AB_LABELS) or len(ag) > len(_AG_LABELS):
        raise ValueError(f"cannot relabel {len(ab)} antibody / {len(ag)} antigen chains")
    return {**{c: _AB_LABELS[i] for i, c in enumerate(ab)},
            **{c: _AG_LABELS[i] for i, c in enumerate(ag)}}


def write_ground_truth(targets, dest: str | Path, overwrite: bool = False) -> Path:
    """Write one relabelled reference structure per target, in the bad-cases layout."""
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb
    import biotite.structure.io.pdbx as pdbx

    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for t in targets:
        out = dest / t.target_id / f"{t.target_id}.pdb"
        if out.is_file() and not overwrite:
            skipped += 1
            t.ground_truth_path = str(out)          # repoint even when reusing, see below
            continue
        src = t.ground_truth_path
        if not src:
            logger.warning("%s: no source structure, skipping", t.target_id)
            continue
        mapping = t.metadata.get("relabel") or {}
        if not mapping:
            logger.warning("%s: no relabel map, skipping", t.target_id)
            continue
        arr = pdbx.get_structure(pdbx.CIFFile.read(str(src)), model=1, use_author_fields=False)
        arr = arr[struc.filter_amino_acids(arr)]
        keep = arr[[c in mapping for c in arr.chain_id]]
        if keep.array_length() == 0:
            logger.warning("%s: none of %s present in %s", t.target_id, sorted(mapping), src)
            continue
        keep.chain_id = [mapping[c] for c in keep.chain_id]
        out.parent.mkdir(parents=True, exist_ok=True)
        f = pdb.PDBFile()
        f.set_structure(keep)
        f.write(str(out))
        t.metadata["source_structure"] = src
        t.ground_truth_path = str(out)
        written += 1
    logger.info("ground truth: %d written, %d already present, into %s", written, skipped, dest)
    return dest


def reference_scores(root: str | Path, subset: str = "intersection_all",
                     task: str = "interface_antibody_antigen") -> "pd.DataFrame":
    """Published per-interface scores from the Protenix bundle, for external comparison."""
    root = _bundle_root(Path(root))
    f = root / "results" / subset / f"{task}.csv"
    if not f.is_file():
        avail = sorted(d.name for d in (root / "results").iterdir()) if (root / "results").is_dir() else []
        raise FileNotFoundError(f"{f} not found. Subsets present: {avail}")
    df = pd.read_csv(f)
    missing = [c for c in ("pdb_id", "model", "dockq_score") if c not in df.columns]
    if missing:
        raise ValueError(f"{f} is missing {missing}. Header is: {list(df.columns)}")
    return df


def _main(argv=None) -> int:
    """``python -m data.datasets.foldbench`` : install a bundle, or inspect one already extracted."""
    import argparse
    ap = argparse.ArgumentParser(description=_main.__doc__)
    ap.add_argument("--install", metavar="DIR", help="download and extract the bundle into DIR")
    ap.add_argument("--inspect", metavar="DIR", help="report the ab-ag CSVs in an extracted bundle")
    ap.add_argument("--url", default=FOLDBENCH_URL)
    ap.add_argument("--archive", metavar="TGZ",
                    help="install from a local copy of the tarball instead of downloading")
    ap.add_argument("--structures", metavar="DIR",
                    help="after installing, fetch the RCSB assembly mmCIF for the subset into DIR")
    ap.add_argument("--ground-truth", metavar="DIR",
                    help="write relabelled H/L/A reference structures into DIR, bad-cases layout. "
                         "Needs --structures (or an existing copy of them) to read from.")
    ap.add_argument("--subset", default="intersection_all",
                    help="which results subset defines the target set (default: intersection_all)")
    ap.add_argument("--force", action="store_true", help="refetch even if already installed")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not (a.install or a.inspect):
        ap.error("give --install DIR or --inspect DIR")

    root = (install(a.install, url=a.url, force=a.force, archive=a.archive) if a.install
            else _bundle_root(Path(a.inspect)))
    print(f"bundle root: {root}")
    for sub in ("targets", "results"):
        d = root / sub
        print(f"  {sub}/: {'absent' if not d.is_dir() else str(len(list(d.rglob('*.csv')))) + ' csv'}")
    if a.structures:
        fetch_structures(root, a.structures, subset=a.subset, force=a.force)
    if a.ground_truth:
        if not a.structures:
            ap.error("--ground-truth needs --structures pointing at the assembly mmCIF")
        ds = FoldBenchDataset(root=root, ground_truth_dir=a.structures,
                              tasks=["interface_antibody_antigen"])
        tgts = ds.load()
        write_ground_truth(tgts, a.ground_truth, overwrite=a.force)
        print(f"  ground truth: {len(tgts)} targets -> {a.ground_truth}")
    hits = sorted(root.rglob("interface_antibody_antigen.csv"))
    if not hits:
        print("  NO interface_antibody_antigen.csv found")
        return 1
    tgt = root / "targets"
    if not tgt.is_dir():
        print("  NOTE: no targets/ here. This bundle is evaluation results only, so it supplies the")
        print("        subset filter; chain pairing must come from the BEAM-Labs targets/ CSVs.")
    for h in hits:
        try:
            df = pd.read_csv(h)
        except Exception as exc:
            print(f"  {h.relative_to(root)}: unreadable ({exc})")
            continue
        idcol = next((c for c in _ID_COLS if c in df.columns), None)
        print(f"  {h.relative_to(root)}: {len(df)} rows")
        print(f"      columns: {list(df.columns)}")
        print(f"      id column: {idcol or 'NONE RECOGNISED -- loader would refuse this file'}"
              + (f" ({df[idcol].nunique()} distinct)" if idcol else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
