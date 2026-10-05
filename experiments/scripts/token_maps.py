"""Build the TOKEN MAPS the steering reward reads: the antibody PARATOPE (CDR) and the antigen
EPITOPE, both as indices into the model's token order.

Two-stage CDR pipeline::

  python -m experiments.scripts.token_maps cdr-map --wired-dir RUN/inputs_wired --out-json cdr_map.json
  python -m experiments.scripts.token_maps cdr     --run-dir RUN --cdr-map cdr_map.json --out-json tokenmap.json

Stage 1 needs ANARCI, imported lazily so this module imports fine without it.

The epitope map the paper reports is hybK12, ``hybrid_modes_tiles``, built in three steps::

  python -m experiments.scripts.token_maps epitope --scorer pool_modes --n-patches 3 ...
  python -m experiments.scripts.token_maps tiles   --k-want 12 ...
  python -m experiments.scripts.token_maps hybrid  --modes-json <modes> --tiles-json <tiles> ...

Pool modes first, then surface tiles ranked by mode support, to a 15-candidate budget. The tiling
depends on which pool filename sorts first, so it is reproducible only while the pool directory is
frozen: ship the built map file with a checksum.

The remaining subcommands support the arms measured against it: ``true-epitope`` (the ORACLE map
for the ground-truth ceiling), ``tiles-ensemble`` (the mode-ensemble tiling) and ``shuffle-sites``
(the A5 map-information ablation), plus ``af3red-plddt-mask``, which is a pLDDT mask for the
AF3-ReD baseline rather than an epitope map. ``jobs/tts_job_lib.py`` keeps thin ``cdr-tokenmap`` /
``epitope-tokenmap`` subcommands that delegate here.

THE TOKEN CONVENTION. A token index is the residue's position in the concatenated input-JSON
sequences, in the order the model sees the chains. ``ca_by_chain``, ``token_order`` and ``token_ca``
encode it, and every map, screen and geometry measurement must use the same one or the indices mean
different things. The paratope and epitope builders live together because the reward differences
their centroids, so they must agree on what an index means.

THAT CONVENTION IS RESIDUE SPACE, AND NOT EVERY BACKEND DIFFUSES IN IT. OpenDDE expands to a larger
structural sub-token axis before its diffusion while keeping its pair and confidence stack in
residue space, so a map written here is not directly indexable against the feature dict OpenDDE
hands its denoiser. A bounds check cannot catch it, because a residue-space index is always a valid
sub-token index. Consumers convert with ``models.protenix_hooks._map_space_ifd``, a no-op where
there is only one token space.

MULTI-SITE. ``spatial_patches`` splits a scored residue set into spatially separate surface
patches, grown greedily from the best unassigned residue out to ``patch_radius``. Slicing the top-3k
residues into three does not work, because they form one contiguous patch and the three "sites"
share a centroid.
"""
from __future__ import annotations

import argparse
import csv
import glob
import inspect
import json
import os
import sys

import numpy as np
from typing import Dict, List, Optional

from method import ANTIBODY_CHAINS

AB_CHAINS = set(ANTIBODY_CHAINS)   # the one definition, see method/__init__.py
CONTACT_R = 10.0        # Ca within this across the interface counts as a contact
TARGET_PATCH_RESIDUES = 18   # k_dir="auto": one direction patch should be ~one epitope footprint
PATCH_RADIUS = 12.0     # one epitope footprint: patches closer than this would not be distinct sites


# ------------------------------------------------------------------ shared token conventions
def _ca_by_chain_cif(path):
    """{auth_chain: (n_res,3) Ca} from an mmCIF, reading the _atom_site loop header for column order."""
    cols, out, in_loop = {}, {}, False
    for line in open(path, errors="replace"):
        t = line.strip()
        if t.startswith("_atom_site."):
            cols[t.split(".", 1)[1]] = len(cols)
            in_loop = True
            continue
        if in_loop and (t.startswith("#") or not t):
            if out:
                break
            continue
        if in_loop and (t.startswith("ATOM") or t.startswith("HETATM")):
            f = t.split()
            if len(f) < len(cols) or f[cols.get("label_atom_id", 3)] != "CA":
                continue
            cid = f[cols.get("auth_asym_id", cols.get("label_asym_id", 6))]
            out.setdefault(cid, []).append([float(f[cols["Cartn_x"]]),
                                            float(f[cols["Cartn_y"]]),
                                            float(f[cols["Cartn_z"]])])
    return {k: np.asarray(v, float) for k, v in out.items()}


_AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
        "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
        "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
        "SEC": "U", "PYL": "O"}


def ca_resid_by_chain(pdb_path):
    """{chain: [(resnum, one_letter, (x,y,z))]} from a PDB, in file order."""
    out = {}
    for line in open(pdb_path, errors="replace"):
        if line.startswith("ENDMDL"):
            break
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            out.setdefault(line[21], []).append(
                (int(line[22:26]), _AA3.get(line[17:20].strip(), "X"),
                 (float(line[30:38]), float(line[38:46]), float(line[46:54]))))
    return out


def native_token_index(res, seq):
    """Model-token index (chain-relative) for each resolved native residue, matched by SEQUENCE."""
    if not res or not seq:
        return None
    n = len(seq)
    nums = [r[0] for r in res]
    lo, hi = -max(nums) - 1, n - min(nums) + 1
    hits = [k for k in range(lo, hi + 1)
            if all(0 <= num + k < n and seq[num + k] == aa for num, aa, _ in res)]
    if len(hits) == 1:
        return [num + hits[0] for num, _a, _x in res]
    return _align_token_index([r[1] for r in res], seq)


def _align_token_index(letters, seq):
    """Global alignment of a native chain's residues onto the model sequence -> token indices."""
    m, n = len(letters), len(seq)
    if not m or not n:
        return None
    gap = -1.0
    sc = np.zeros((m + 1, n + 1))
    sc[1:, 0] = np.arange(1, m + 1) * gap
    sc[0, 1:] = 0.0                       # free leading gap in the model sequence
    ptr = np.zeros((m + 1, n + 1), dtype=np.int8)
    ptr[1:, 0] = 1
    ptr[0, 1:] = 2
    srow = np.frombuffer(seq.encode(), dtype="S1")
    for i in range(1, m + 1):
        diag = sc[i - 1, :-1] + np.where(srow == letters[i - 1].encode(), 1.0, -1.0)
        up = sc[i - 1, 1:] + gap
        best = np.maximum(diag, up)
        sc[i, 1:] = best
        ptr[i, 1:] = np.where(diag >= up, 0, 1)
        for j in range(1, n + 1):         # left moves depend on the cell to their left
            if sc[i, j - 1] + gap > sc[i, j]:
                sc[i, j] = sc[i, j - 1] + gap
                ptr[i, j] = 2
    i, j = m, int(np.argmax(sc[m, :]))    # free trailing gap in the model sequence
    out = [None] * m
    while i > 0:
        p = ptr[i, j]
        if p == 0:
            if seq[j - 1] != letters[i - 1]:
                return None
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif p == 1:
            out[i - 1] = None             # native residue absent from the model input
            i -= 1
        else:
            j -= 1
    return out


def ca_by_chain(path):
    """{chain_id: (n_res, 3) Ca coords}, for a PDB or an mmCIF."""
    if path.lower().endswith((".cif", ".mmcif")):
        out = _ca_by_chain_cif(path)
        if not out:
            raise ValueError(f"no Ca atoms parsed from {path} -- wrong format or unreadable, "
                             "NOT an empty structure")
        return out
    out = {}
    for line in open(path, errors="replace"):
        if line.startswith("ENDMDL"):
            break
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            out.setdefault(line[21], []).append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not out:
        raise ValueError(f"no Ca atoms parsed from {path} -- wrong format or unreadable, "
                         "NOT an empty structure")
    return {k: np.asarray(v, float) for k, v in out.items()}


def token_order(input_json):
    """[(chain_id, token_offset, length)] in the order the MODEL concatenates the sequences."""
    d = json.load(open(input_json))
    d = d[0] if isinstance(d, list) else d
    order, off = [], 0
    for seq in d.get("sequences", []):
        v = list(seq.values())[0]
        cid = v.get("id")
        cid = cid[0] if isinstance(cid, list) else cid
        ln = len(v.get("sequence", ""))
        order.append((cid, off, ln))
        off += ln
    return order


def _chain_sequences(input_json):
    """{chain_id: sequence} from a wired input, the same parse as :func:`token_order`."""
    d = json.load(open(input_json))
    d = d[0] if isinstance(d, list) else d
    out = {}
    for seq in d.get("sequences", []):
        v = list(seq.values())[0]
        cid = v.get("id")
        out[cid[0] if isinstance(cid, list) else cid] = v.get("sequence", "")
    return out


def token_ca(pdb_path, input_json):
    """(n_token, 3) Ca in MODEL TOKEN ORDER, or None when a chain's length disagrees."""
    X, order = ca_by_chain(pdb_path), token_order(input_json)
    n_tok = sum(ln for _c, _o, ln in order)
    out = np.full((n_tok, 3), np.nan)
    for cid, off, ln in order:
        c = X.get(cid)
        if c is None or len(c) != ln:
            return None
        out[off:off + ln] = c
    return out


def _ca_plddt_by_chain_cif(path):
    """{auth_chain: (n_res,) Ca pLDDT} from an mmCIF, reading the header for column order."""
    cols, out, in_loop = {}, {}, False
    for line in open(path, errors="replace"):
        t = line.strip()
        if t.startswith("_atom_site."):
            cols[t.split(".", 1)[1]] = len(cols)
            in_loop = True
            continue
        if in_loop and (t.startswith("#") or not t):
            if out:
                break
            continue
        if in_loop and (t.startswith("ATOM") or t.startswith("HETATM")):
            f = t.split()
            if len(f) < len(cols) or f[cols.get("label_atom_id", 3)] != "CA":
                continue
            if "B_iso_or_equiv" not in cols:
                return {}
            cid = f[cols.get("auth_asym_id", cols.get("label_asym_id", 6))]
            try:
                out.setdefault(cid, []).append(float(f[cols["B_iso_or_equiv"]]))
            except ValueError:
                return {}
    return {k: np.asarray(v, float) for k, v in out.items()}


def token_ca_plddt(cif_path, input_json):
    """(n_token,) Ca pLDDT in MODEL TOKEN ORDER, or None on a chain-length disagreement."""
    P, order = _ca_plddt_by_chain_cif(cif_path), token_order(input_json)
    if not P:
        return None
    n_tok = sum(ln for _c, _o, ln in order)
    out = np.full(n_tok, np.nan)
    for cid, off, ln in order:
        c = P.get(cid)
        if c is None or len(c) != ln:
            return None
        out[off:off + ln] = c
    return out


# ------------------------------------------------------------------ TRUE epitope (SCORING ONLY)
def build_true_epitope_tokenmap(gt_dir, out_json, wired_dir, contact_r=None, targets=None):
    """The TRUE epitope from the native complex -> antigen token indices. **SCORING ONLY.**"""
    r = float(CONTACT_R if contact_r is None else contact_r)
    names = targets or sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(wired_dir, "*.json")))
    out, rows = {}, []
    for name in names:
        gt = None
        for pat in ("%s.pdb", "%s.cif", os.path.join("%s", "%s.pdb"), os.path.join("%s", "%s.cif")):
            cand = os.path.join(gt_dir, pat % ((name, name) if pat.count("%s") == 2 else name))
            if os.path.isfile(cand):
                gt = cand
                break
        if gt is None:
            rows.append((name, "no ground-truth structure")); continue
        if gt.lower().endswith((".cif", ".mmcif")):
            rows.append((name, "native is mmCIF: the sequence-indexed reader is PDB-only")); continue
        wired = os.path.join(wired_dir, name + ".json")
        R, S = ca_resid_by_chain(gt), _chain_sequences(wired)
        ab, ag_tok, ag_xyz, shifted, dropped = [], [], [], None, {}
        for cid, off, ln in token_order(wired):
            res = R.get(cid)
            if not res:
                continue
            if cid in AB_CHAINS:
                ab.append(np.asarray([x for _n, _a, x in res], float))
                continue
            idx = native_token_index(res, S.get(cid, ""))
            if idx is None:
                shifted = "antigen chain %s could not be aligned to the model sequence" % cid
                break
            keep = [(j, x) for (j, (_n, _a, x)) in zip(idx, res) if j is not None]
            if len(keep) < len(res):
                dropped[cid] = len(res) - len(keep)
            ag_xyz.append(np.asarray([x for _j, x in keep], float))
            ag_tok += [off + j for j, _x in keep]
        if shifted:
            rows.append((name, shifted)); continue
        if not ab or not ag_tok:
            rows.append((name, "no ab/ag split")); continue
        ab = np.vstack(ab); ag_xyz = np.vstack(ag_xyz); ag_tok = np.asarray(ag_tok, int)
        from method.verifier.diversity import epitope_mask
        epi = ag_tok[epitope_mask(ab, ag_xyz, contact_r=r)]
        if epi.size == 0:
            rows.append((name, "no contacts")); continue
        out[name] = sorted(int(x) for x in epi)
        rows.append((name, "%d epitope of %d antigen residues%s"
                     % (epi.size, ag_tok.size,
                        "" if not dropped else "  [%s not in the model input]"
                        % ", ".join("%s:%d" % kv for kv in sorted(dropped.items())))))
    n_targets = write_map(out, out_json, "ground_truth", "native_contacts",
                          note=("TRUE epitope: antigen residues with Ca within %g A of any antibody "
                                "Ca in the NATIVE complex, indexed onto the model token order BY "
                                "SEQUENCE. SCORING ONLY, never read by any method." % r))
    print("true-epitope: %d targets -> %s" % (n_targets, out_json))
    # Print the ones that went wrong, not just the first five alphabetically: a target the builder
    # refused is exactly what a caller needs to see, and it was invisible behind `rows[:5]`.
    problems = [(n, m) for n, m in rows if "epitope of" not in m]
    for name, msg in (problems or rows[:5]):
        print("  %-24s %s" % (name, msg))
    if problems:
        print("  %d target(s) produced no entry" % len(problems))
    return out


# ------------------------------------------------------------------ shared builder plumbing
LEGAL_SOURCES = ("ground_truth", "predicted", "synthetic")


class EmptyMapError(ValueError):
    """write_map refused to write a map that resolved zero targets."""


def write_map(out, out_json, source, builder, note=None, extra=None):
    """Write a token map WITH its provenance. Every builder goes through here."""
    if source not in LEGAL_SOURCES:
        raise ValueError("source %r not in %r" % (source, LEGAL_SOURCES))
    if not builder:
        raise ValueError("every map must name its builder")
    n_targets = len([k for k in out if not str(k).startswith("_")])
    if n_targets == 0:
        # A map with no targets passes every existence check a job template makes and only fails
        # much later, in dataset-check or at scoring. Refuse to create the file at all.
        raise EmptyMapError(
            "refusing to write %s: the %s builder resolved 0 targets. Check that the input "
            "directories exist and hold the expected files (--wired-dir with the wired input "
            "jsons, --pool-dir / --dt3-dir with the prediction cifs for those same targets)."
            % (out_json, builder))
    out["_source"] = source
    out["_builder"] = str(builder)
    if note:
        out["_note"] = note
    for k, v in (extra or {}).items():
        out["_" + k.lstrip("_")] = v
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f)
    return n_targets


def antigen_surface(cif, wired_json, surf_pct=0.0):
    """(ag_tok, ag_xyz, centroid, radii, surf_mask) for one target's ANTIGEN, in token order."""
    ag_tok, ag_xyz = antigen_token_xyz(cif, wired_json)
    if not len(ag_tok):
        return None
    cen = ag_xyz.mean(0)
    rad = np.linalg.norm(ag_xyz - cen, axis=1)
    surf = rad >= np.percentile(rad, surf_pct) if surf_pct > 0 else np.ones(rad.shape, bool)
    return ag_tok, ag_xyz, cen, rad, surf




def _chains_from_wired(path: str):
    """``[(chain_id, sequence)]`` in the order the model sees them, from a wired input JSON."""
    obj = json.load(open(path))
    obj = obj[0] if isinstance(obj, list) else obj
    out = []
    for s in obj.get("sequences", []):
        v = list(s.values())[0]
        cid = v.get("id")
        cid = cid[0] if isinstance(cid, list) else cid
        seq = v.get("sequence", "")
        if cid is not None and seq:
            out.append((str(cid), str(seq)))
    return obj.get("name") or os.path.splitext(os.path.basename(path))[0], out


def cdr_indices(seq: str, chain_id: str, hmmer_path=None, min_len: int = 70):
    """0-based indices INTO ``seq`` of its IMGT CDR residues, or [] if ANARCI cannot number it."""
    from method.verifier.cdr import _imgt_is_cdr
    if len(seq) < min_len:                    # too short to be a variable domain
        return []
    try:
        from anarci import anarci
    except ImportError:
        raise SystemExit("ANARCI is not installed. It is required to build a CDR map, and there is "
                         "no fallback: guessing CDR positions would corrupt every method that reads "
                         "this file.")
    kw = {"scheme": "imgt", "output": False}
    if hmmer_path:
        kw["hmmerpath"] = hmmer_path
    try:
        numbered, _details, _ = anarci([(chain_id, seq)], **kw)
    except FileNotFoundError as e:
        raise SystemExit(
            f"ANARCI could not run: {e}. Install HMMER (it provides hmmscan) or pass "
            f"--hmmer-path pointing at the directory that contains it. Refusing to emit an empty "
            f"CDR map, because downstream every paratope method would silently disable itself.")
    except (AssertionError, ValueError):
        return []                              # ANARCI rejected the sequence, e.g. an unknown
                                               # residue letter. A chain that is simply not a
                                               # variable domain comes back unnumbered instead.
    except Exception as e:
        raise SystemExit(
            f"ANARCI crashed while numbering chain {chain_id}: {type(e).__name__}: {e}. This is an "
            f"environment fault, so every chain fails the same way and the map comes out empty. "
            f"The usual cause is that hmmscan is a wrapper script rather than the real binary, "
            f"which puts the wrapper's name in HMMER's output header and leaves Biopython with no "
            f"program match and every coordinate None. Point --hmmer-path at the real executable.")
    if not numbered or not numbered[0]:
        return []
    numbering, start, _end = numbered[0][0]
    idx, out = start, []
    for (pos, _ins), aa in numbering:
        if aa == "-":
            continue
        if 0 <= idx < len(seq) and _imgt_is_cdr(pos):
            out.append(idx)
        idx += 1
    return out


def build_cdr_map(wired_dir: str, out_json: str, antibody_labels=None,
          hmmer_path: Optional[str] = None, min_cdr: int = 3) -> Dict[str, Dict[str, List[int]]]:
    """Write ``{target: {chain: [cdr idx]}}`` for every wired input. Returns the map."""
    from method.verifier.cdr import ANTIBODY_LABELS
    antibody_labels = tuple(ANTIBODY_LABELS) if antibody_labels is None else antibody_labels
    ab = set(antibody_labels)
    out: Dict[str, Dict[str, List[int]]] = {}
    missing, thin = [], []
    for f in sorted(glob.glob(os.path.join(wired_dir, "*.json"))):
        name, chains = _chains_from_wired(f)
        per = {}
        for cid, seq in chains:
            if cid not in ab:
                continue
            idxs = cdr_indices(seq, cid, hmmer_path)
            if idxs:
                per[cid] = idxs
        # <3 residues cannot define a superposition, and build_cdr_tokenmap drops such targets
        # anyway. Reporting them HERE is the difference between a known gap and a silent one.
        n = sum(len(v) for v in per.values())
        if not per:
            missing.append(name)
        elif n < min_cdr:
            thin.append((name, n))
        else:
            out[name] = per
    tot = sum(sum(len(v) for v in d.values()) for d in out.values())
    n_targets = write_map(out, out_json, "predicted", "anarci_cdr",
                          extra={"min_cdr": int(min_cdr)})
    print(f"cdr-map: wrote {n_targets} targets, {tot} CDR residues -> {out_json}")
    if missing:
        print(f"  {len(missing)} with NO antibody chain ANARCI could number: {missing[:8]}")
    if thin:
        print(f"  {len(thin)} with fewer than {min_cdr} CDR residues, dropped: {thin[:8]}")
    return out


# ------------------------------------------------------------------ paratope (CDR) map
def build_cdr_tokenmap(run_dir, cdr_map, out_json, wired_dir=None):
    """Per-chain IMGT CDR map -> the PARATOPE TOKEN INDICES the cdr_rmsd / epitope rewards need."""
    with open(cdr_map) as f:
        cm = json.load(f)
    src = wired_dir or os.path.join(run_dir, "inputs_wired")
    out, missing = {}, []
    for jf in sorted(glob.glob(os.path.join(src, "*.json"))):
        obj = json.load(open(jf))
        obj = obj[0] if isinstance(obj, list) else obj
        name = obj.get("name") or os.path.splitext(os.path.basename(jf))[0]
        if name not in cm:
            missing.append(name)
            continue
        para, total = [], 0
        for cid, off, ln in token_order(jf):
            if cid in AB_CHAINS and cid in cm[name]:
                para += [off + int(i) for i in cm[name][cid] if 0 <= int(i) < ln]
            total += ln
        para = sorted({i for i in para if 0 <= i < total})
        if len(para) >= 3:                      # <3 residues cannot define a superposition
            out[name] = para
        else:
            missing.append(name)
    n_targets = write_map(out, out_json, "predicted", "imgt_cdr")
    print("cdr-tokenmap: wrote %d targets -> %s%s"
          % (n_targets, out_json,
             ("; %d without a usable map (reward falls back to ipSAE): %s" % (len(missing), missing[:5]))
             if missing else ""))
    return out


_POOL_UNITS = 6


# ------------------------------------------------------------------ multi-site patches
def spatial_patches(token_idx, scores, coords, n_patches=3, patch_radius=PATCH_RADIUS, min_size=5):
    """Split scored residues into up to ``n_patches`` SPATIALLY SEPARATE patches, best first."""
    idx = np.asarray(token_idx, int)
    sc = np.asarray(scores, float)
    xyz = np.asarray(coords, float)
    free = np.ones(len(idx), bool)
    patches = []
    while free.any() and len(patches) < int(n_patches):
        cand = np.where(free)[0]
        seed = cand[np.argmax(sc[cand])]
        near = cand[np.linalg.norm(xyz[cand] - xyz[seed], axis=1) <= float(patch_radius)]
        free[near] = False
        if len(near) >= int(min_size):
            patches.append(np.sort(idx[near]))
    return patches

EPITOPE_SCORERS = {}


def register_epitope_scorer(name):
    """Decorator registering a per-residue epitope scorer under ``name``."""
    def deco(fn):
        EPITOPE_SCORERS[name] = fn
        return fn
    return deco


@register_epitope_scorer("discotope3")
def _score_discotope3(src_dir, name, cid, chain_len, want_coords, use_flag=False):
    """DiscoTope-3 CSVs. Antibody-AGNOSTIC: it scores "antigenic surface", not where THIS antibody
    binds, which is why its centroid misses the true epitope on most of our targets."""
    hits = glob.glob(os.path.join(src_dir, f"{name}_{cid}_discotope3.csv"))
    if not hits:
        return None
    ca = _dt3_chain_ca(src_dir, name, cid) if want_coords else None
    rows = []
    with open(hits[0]) as fh:
        for r in csv.DictReader(fh):
            try:
                ridx = int(r["res_id"]) - 1                    # DiscoTope res_id is 1-based
                sc = float(r.get("calibrated_score") or r["DiscoTope-3.0_score"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= ridx < chain_len:
                continue
            if use_flag and str(r.get("epitope", "")).strip().lower() not in ("true", "1"):
                continue
            rows.append((sc, ridx, ca[ridx] if ca is not None and ridx < len(ca) else None))
    return rows


@register_epitope_scorer("pool_modes")
def _score_pool_modes(src_dir, name, cid, chain_len, want_coords, use_flag=False,
                      k=3, n_sample=None, footprint=PATCH_RADIUS):
    """The UNSTEERED co-folding pool's own binding-site MODES, scored by mode population share."""
    import glob as _glob
    from collections import defaultdict

    from method.verifier.cif_io import load_ca_coords_chains
    from method.verifier.consensus import _superpose

    def _cifs_under(root):
        """Every pool structure for ``name`` under ONE prediction root, whichever layout it uses."""
        # our pool layout first, then the flatter layout an external release uses
        c = sorted(_glob.glob(os.path.join(root, f"{name}__*", "seed_*", "predictions",
                                           "*_sample_*.cif"))) or \
            sorted(_glob.glob(os.path.join(root, name, "seed_*", "*_sample_*.cif")))
        if c:
            return c
        return sorted(_glob.glob(os.path.join(root, "**", f"{name}__*_model_*.cif"),
                                 recursive=True)) or \
            sorted(_glob.glob(os.path.join(root, "**", f"{name}_model_*.cif"), recursive=True))

    roots = [r for r in str(src_dir).split(",") if r]
    per_root = [_cifs_under(r) for r in roots]
    if len(roots) > 1:
        want = int(n_sample if n_sample is not None
                   else os.environ.get("POOL_MODES_N_SAMPLE", "150"))
        share = max(1, want // len(roots))
        cifs = []
        for c in per_root:
            cifs.extend(c[::max(1, len(c) // share)][:share] if c else [])
        already_thinned = True
    else:
        cifs = per_root[0] if per_root else []
        already_thinned = False
    _min = int(os.environ.get("POOL_MODES_MIN_CIFS", "20"))
    if len(cifs) < _min:
        # Otherwise the only signal is an empty map, whose message blames the paths.
        sys.stderr.write("pool_modes: %s has %d pool structure(s), below POOL_MODES_MIN_CIFS=%d, "
                         "so it gets no modes. Run a larger unsteered pool, or lower the "
                         "threshold and accept that modes over so few structures are noise.\n"
                         % (name, len(cifs), _min))
        return None
    if n_sample is None:
        n_sample = os.environ.get("POOL_MODES_N_SAMPLE", "150")
    k = int(os.environ.get("POOL_MODES_K", k))
    if not already_thinned:                       # the pooled branch already took an equal share
        cifs = cifs[::max(1, len(cifs) // int(n_sample))][:int(n_sample)]

    ref_ch = ref_ag = ab_m = ag_m = None
    cents, epis = [], []
    for c in cifs:
        co, ch, _pl = load_ca_coords_chains(c)
        if co is None or ch is None:
            continue
        ch = np.asarray(ch)
        if ref_ch is None:
            ref_ch = ch
            ab_m = np.array([x in AB_CHAINS for x in ch])
            if not ab_m.any():
                sizes = {c: int((ch == c).sum()) for c in dict.fromkeys(ch.tolist())}
                match = [c for c, n in sizes.items() if n == int(chain_len)]
                if len(match) != 1:
                    return None
                ag_m = ch == match[0]
                ab_m = ~ag_m
            else:
                ag_m = ~ab_m
            if not ab_m.any() or not ag_m.any():
                return None
        if len(ch) != len(ref_ch) or not np.array_equal(ch, ref_ch):
            continue
        x = np.asarray(co, float)
        ag, ab = x[ag_m], x[ab_m]
        if ref_ag is None:
            ref_ag = ag
        hit = np.linalg.norm(ag[:, None, :] - ab[None, :, :], axis=-1).min(1) < CONTACT_R
        if hit.sum() < 3:
            continue
        cents.append(_superpose(ag, ref_ag, ag)[hit].mean(0))   # a SHARED antigen frame for centroids
        epis.append(np.where(hit)[0])
    if len(cents) < 10:
        return None

    cents = np.stack(cents)
    free = np.ones(len(cents), bool)
    out, mode_id = [], 0
    while free.any() and mode_id < int(k):   # count MODES, not the residue entries in `out`
        idx = np.where(free)[0]
        sub = cents[idx]
        dd = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
        seed = idx[np.argmax((dd < float(footprint)).sum(1))]    # densest remaining = biggest mode
        near = idx[np.linalg.norm(cents[idx] - cents[seed], axis=1) <= float(footprint)]
        free[near] = False
        cnt = defaultdict(int)
        for i in near:
            for r in epis[i]:
                cnt[int(r)] += 1
        share = len(near) / len(cents)
        res = [r for r, c in cnt.items() if c >= 0.5 * len(near)]
        if len(res) >= 3:
            out += [(share, int(r), ref_ag[r] if want_coords else None, mode_id) for r in res]
            mode_id += 1
    # antigen-local -> the chain-local index the builder expects (this scorer reads only the antigen)
    ag_index = np.where(ag_m)[0]
    base = int(ag_index[0])
    return [(sc, int(ag_index[r]) - base, xyz, pid) for sc, r, xyz, pid in out
            if 0 <= int(ag_index[r]) - base < chain_len]


def _dt3_chain_ca(dt3_dir, name, cid):
    """Ca coords of the antigen structure DiscoTope itself scored, indexed by ``res_id - 1``."""
    from method.verifier.cif_io import load_ca_coords_chains
    hits = glob.glob(os.path.join(dt3_dir, f"{name}_{cid}_discotope3.cif")) or \
        glob.glob(os.path.join(dt3_dir, f"{name}_{cid}_discotope3.pdb"))
    if not hits:
        return None
    co, _ch, _pl = load_ca_coords_chains(hits[0])
    return None if co is None else np.asarray(co, float)


def build_epitope_tokenmap(run_dir, dt3_dir, out_json, wired_dir=None, top_k=20, use_flag=False,
                           n_patches=1, patch_radius=PATCH_RADIUS, scorer="pool_modes"):
    """Per-residue epitope scores -> ANTIGEN epitope TOKEN indices (for reward=epitope).

    The scorer selects the source: ``pool_modes`` (default) reads an unsteered prediction pool,
    ``discotope3`` reads DiscoTope-3 CSVs. Both come from ``--pool-dir``.
    """
    src = wired_dir or os.path.join(run_dir, "inputs_wired")
    out, missing, notes = {}, [], []
    for jf in sorted(glob.glob(os.path.join(src, "*.json"))):
        obj = json.load(open(jf))
        obj = obj[0] if isinstance(obj, list) else obj
        name = obj.get("name") or os.path.splitext(os.path.basename(jf))[0]
        rows = []                        # (score, token_idx, xyz) pooled over all antigen chains
        for cid, off, ln in token_order(jf):
            if cid in AB_CHAINS:                    # antibody chains are not the epitope
                continue
            _fn = EPITOPE_SCORERS[scorer]
            _kw = {"want_coords": int(n_patches) > 1, "use_flag": use_flag}
            if "footprint" in inspect.signature(_fn).parameters:
                _kw["footprint"] = float(patch_radius)
            scored = _fn(dt3_dir, name, cid, ln, **_kw)
            if not scored:
                continue
            for ent in scored:
                sc, ridx, xyz = ent[0], ent[1], ent[2]
                pid = ent[3] if len(ent) > 3 else None
                if int(n_patches) > 1 and xyz is None and pid is None:
                    continue                        # no coordinate and no patch id => unusable
                rows.append((sc, off + ridx, xyz, pid))
        if not rows:
            missing.append(name)
            continue
        rows.sort(key=lambda t: -t[0])
        preclustered = any(r[3] is not None for r in rows)
        if int(n_patches) <= 1:
            picked = sorted({t for _s, t, _x, _p in (rows if use_flag else rows[:int(top_k)])})
            entry = picked if len(picked) >= 3 else None
        elif preclustered:
            # the scorer already determined the patches, keep them, best-scoring patch first
            by_pid = {}
            for sc, tok, _x, pid in rows:
                by_pid.setdefault(pid, [set(), 0.0])
                by_pid[pid][0].add(int(tok))
                by_pid[pid][1] = max(by_pid[pid][1], sc)
            ordered = sorted(by_pid.values(), key=lambda v: -v[1])[:int(n_patches)]
            entry = [sorted(toks) for toks, _sc in ordered if len(toks) >= 3] or None
        else:
            # a generous candidate pool, so n footprint-sized patches can actually be carved out
            pool = rows if use_flag else rows[:int(top_k) * int(n_patches)]
            pats = spatial_patches([t for _s, t, _x, _p in pool], [s for s, _t, _x, _p in pool],
                                   np.stack([x for _s, _t, x, _p in pool]), n_patches, patch_radius)
            entry = [p.tolist() for p in pats] if pats else None
            xy = {t: x for _s, t, x, _p in rows if x is not None}
            if entry and len(entry) > 1 and all(t in xy for p in entry for t in p):
                cen = np.stack([np.stack([xy[t] for t in p]).mean(0) for p in entry])
                dd = np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=-1)
                notes.append((name, len(entry),
                              float(dd[np.triu_indices(len(cen), 1)].min()) if len(cen) > 1 else float("inf")))
        if entry:
            out[name] = entry
        else:
            missing.append(name)
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    _pred = ("%s(flag)" % scorer if use_flag else "%s(top_k=%d)" % (scorer, int(top_k))) + \
            ("" if int(n_patches) <= 1 else ",patches=%d@%.0fA" % (int(n_patches), patch_radius))
    _extra = {"predictor": _pred}
    if str(scorer) == "pool_modes":
        _extra["pool_modes_n_sample"] = int(os.environ.get("POOL_MODES_N_SAMPLE", "150"))
        _extra["pool_modes_min_cifs"] = int(os.environ.get("POOL_MODES_MIN_CIFS", "20"))
        _extra["pool_modes_draw"] = "prefix"
    n_targets = write_map(out, out_json, "predicted", str(scorer), extra=_extra)
    print("epitope-tokenmap: wrote %d targets -> %s%s"
          % (n_targets, out_json,
             ("; %d without a usable epitope: %s" % (len(missing), missing[:5])) if missing else ""))
    if notes:
        sep = np.array([n[2] for n in notes])
        n_sites = np.array([n[1] for n in notes])
        print("  patches/target: mean %.2f (min %d, max %d); patch centroid separation: "
              "min %.1f A, median %.1f A" % (n_sites.mean(), n_sites.min(), n_sites.max(),
                                             sep.min(), np.median(sep)))
        thin = [n[0] for n in notes if n[1] < int(n_patches)]
        if thin:
            print("  %d target(s) yielded FEWER than %d patches (surface too small): %s"
                  % (len(thin), int(n_patches), thin[:5]))
    return out



# ------------------------------------------------------------------ ground-truth substitution
def _ref_cif_json(target, pool_dir, wired_dir):
    """``(cif, input_json)`` for one unsteered pool structure of ``target``, or ``(None, None)``."""
    for d in sorted(glob.glob(f"{pool_dir}/{target}__*")):
        cif = sorted(glob.glob(f"{d}/**/*_sample_*.cif", recursive=True))
        if not cif:
            continue
        seed = os.path.basename(d.rstrip("/"))
        for js in (f"{wired_dir}/_seeded/{seed}.json", f"{wired_dir}/{target}.json"):
            if os.path.exists(js):
                return cif[0], js
    return None, None


# ------------------------------------------------------------------ shuffled sites (ablation A5)
def build_shuffled_sitemap(base_map, pool_dir, wired_dir, out_json, seed=0, min_shift=12.0,
                           surf_pct=40.0, tries=500, true_map=None):
    """A5: the base map's SHAPE with its INFORMATION destroyed -- candidates moved to random sites."""
    base = json.load(open(base_map))
    rng = np.random.default_rng(int(seed))
    true = json.load(open(true_map)) if true_map else None
    out, shifts, near, unmeasured, base_hit, shuf_hit = {}, [], [], [], 0, 0
    for t in [k for k in base if not str(k).startswith("_")]:
        raw = base[t]
        cand = [np.asarray(p, int) for p in raw] if (raw and isinstance(raw[0], list)) \
            else [np.asarray(raw, int)]
        cif, js = _ref_cif_json(t, pool_dir, wired_dir)
        surf = antigen_surface(cif, js, surf_pct) if cif else None
        if surf is None:
            unmeasured.append(t)
            continue
        ag_tok, ag_xyz, _cen, _rad, smask = surf
        pos = {int(k): ag_xyz[i] for i, k in enumerate(ag_tok)}
        s_tok, s_xyz = ag_tok[smask], ag_xyz[smask]
        b_cen = [np.mean([pos[int(i)] for i in p if int(i) in pos], axis=0)
                 for p in cand if any(int(i) in pos for i in p)]
        if not b_cen:
            unmeasured.append(t)
            continue
        new, t_shift = [], []
        for p in cand:
            n_i = max(1, int(len(p)))
            best, best_d = None, -1.0
            for _ in range(int(tries)):
                c = s_xyz[rng.integers(len(s_xyz))]
                d = min(float(np.linalg.norm(c - b)) for b in b_cen)
                if d > best_d:
                    # grow the patch around this centre: its n_i nearest SURFACE residues, which is
                    # the same "compact surface patch" shape the tiler produces
                    k = np.argsort(np.linalg.norm(s_xyz - c, axis=1))[:n_i]
                    best, best_d = s_tok[k], d
                if d >= float(min_shift):
                    break
            new.append(np.asarray(best, int).tolist())
            t_shift.append(best_d)
        if min(t_shift) < float(min_shift):
            near.append("%s(%.1f A)" % (t, min(t_shift)))
        shifts.extend(t_shift)
        out[t] = new
        if true is not None:
            gt = np.asarray(true[t][0] if isinstance(true[t][0], list) else true[t], int)
            g = np.mean([pos[int(i)] for i in gt if int(i) in pos], axis=0)
            base_hit += int(min(float(np.linalg.norm(b - g)) for b in b_cen) <= 6.0)
            shuf_hit += int(min(float(np.linalg.norm(
                np.mean([pos[int(i)] for i in q if int(i) in pos], axis=0) - g)) for q in new) <= 6.0)
    if unmeasured:
        raise RuntimeError(
            f"shuffle-sites: no antigen geometry for {len(unmeasured)} target(s): "
            f"{', '.join(unmeasured)}. Their candidates could not be moved, so no map was written. "
            "Check --pool-dir and --wired-dir.")
    _extra = {"derived_from": os.path.basename(base_map),
              "predictor": ("A5 CONTROL: candidate count and sizes copied from the base map, "
                            "locations redrawn uniformly on the antigen surface. Carries NO site "
                            "information and is NOT deployable."),
              "shuffle_seed": int(seed), "min_shift_a": float(min_shift),
              "median_shift_a": float(np.median(shifts)) if shifts else None}
    n = write_map(out, out_json, "synthetic", "shuffled_sites", extra=_extra)
    print("  shuffle-sites: %d targets, median shift %.1f A (min %.1f, bar %.1f)"
          % (n, np.median(shifts), np.min(shifts), min_shift))
    if near:
        print("  !! %d target(s) below the %.1f A bar, best draw kept: %s"
              % (len(near), min_shift, ", ".join(near)))
    if true is not None:
        print("  true epitope within 6 A of some candidate: base %d/%d -> shuffled %d/%d"
              % (base_hit, n, shuf_hit, n))
    return out


def antigen_token_xyz(cif_path, input_json, ab_chains=AB_CHAINS):
    """``(token_idx, xyz)`` over ANTIGEN residues only, in model token order."""
    X, order = ca_by_chain(cif_path), token_order(input_json)
    ag = [(cid, off, ln) for cid, off, ln in order if cid not in ab_chains]
    if not ag:
        return np.zeros(0, int), np.zeros((0, 3), float)

    by_id = {cid: X[cid] for cid, _o, ln in ag if cid in X and len(X[cid]) == ln}
    if len(by_id) == len(ag):
        src = by_id
    elif [len(v) for v in X.values()] == [ln for _c, _o, ln in order]:
        # dict preserves file order, and a prediction lists chains in input order
        vals = list(X.values())
        src = {cid: vals[k] for k, (cid, _o, _l) in enumerate(order)}
    else:
        return np.zeros(0, int), np.zeros((0, 3), float)

    tok, xyz = [], []
    for cid, off, ln in ag:
        for i in range(ln):
            tok.append(off + i)
            xyz.append(src[cid][i])
    return np.asarray(tok, int), np.asarray(xyz, float)


def map_patches(entry):
    """A token-map entry -> list of index lists. ONE spelling of the three shapes a map can carry."""
    if isinstance(entry, dict):
        return [list(v) for k, v in sorted(entry.items()) if not str(k).startswith("_")]
    if entry and isinstance(entry[0], (list, tuple)):
        return [list(p) for p in entry]
    return [list(entry)] if entry else []


def tile_geometry(xyz):
    """``(d, c)`` for ONE structure: pairwise distances and distance-from-centroid."""
    xyz = np.asarray(xyz, float)
    return (np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1),
            np.linalg.norm(xyz - xyz.mean(0), axis=1))


def tiles_from_geometry(tok, dbar, cbar, radius, min_size=3):
    """THE greedy covering, on whatever geometry it is handed."""
    free = np.ones(len(tok), bool)
    order = np.argsort(-cbar)
    tiles = []
    while free.any():
        seed = next(i for i in order if free[i])
        near = np.flatnonzero(free & (dbar[seed] <= float(radius)))
        free[near] = False
        if len(near) >= int(min_size):
            tiles.append(sorted(int(x) for x in tok[near]))
    return tiles


def tiles_from_geometry_at_k(tok, dbar, cbar, k_want, min_size=3, lo=4.0, hi=80.0, iters=30):
    """THE radius search: bisect until the tiling has ``k_want`` tiles. Returns ``(tiles, radius)``."""
    best, best_r = None, float(hi)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        t = tiles_from_geometry(tok, dbar, cbar, mid, min_size=min_size)
        if best is None:
            better = True
        else:
            d_new, d_old = abs(len(t) - int(k_want)), abs(len(best) - int(k_want))
            better = d_new < d_old or (d_new == d_old and len(t) < len(best))
        if better:
            best, best_r = t, mid
        if len(t) > int(k_want):
            lo = mid
        else:
            hi = mid
    return best, best_r


def surface_tiles(xyz, tok, radius, min_size=3):
    """Farthest-point tiling: every residue lands within ``radius`` of some tile seed."""
    return tiles_from_geometry(np.asarray(tok, int), *tile_geometry(xyz), radius, min_size=min_size)


def tiles_at_fixed_k(xyz, tok, k_want, min_size=3, lo=4.0, hi=80.0, iters=30):
    """Binary-search the radius that yields ``k_want`` tiles. Returns ``(tiles, radius)``."""
    return tiles_from_geometry_at_k(np.asarray(tok, int), *tile_geometry(xyz), k_want,
                                    min_size=min_size, lo=lo, hi=hi, iters=iters)


def build_surface_tilemap(pool_dir, out_json, wired_dir, radius=None, k_want=None,
                          surf_pct=0.0, min_size=3, ab_chains=AB_CHAINS):
    """Write an exhaustive surface-tiling map. Give EITHER ``radius`` (fixed radius, K varies) or
        ``k_want`` (fixed K, radius derived per target).
    """
    if (radius is None) == (k_want is None):
        raise ValueError("give exactly one of radius= (fixed radius) or k_want= (fixed K)")
    out, rows = {}, []
    for jf in sorted(glob.glob(os.path.join(wired_dir, "*.json"))):
        t = os.path.splitext(os.path.basename(jf))[0]
        cand = sorted(glob.glob(os.path.join(pool_dir, "%s__*" % t, "seed_*", "predictions", "*.cif")))
        if not cand:
            continue
        tok, xyz = antigen_token_xyz(cand[0], jf, ab_chains)
        if len(tok) < 20:
            continue
        if surf_pct > 0:                     # kept only for reproducing the older swept maps
            rad = np.linalg.norm(xyz - xyz.mean(0), axis=1)
            keep = np.flatnonzero(rad >= np.percentile(rad, float(surf_pct)))
            tok, xyz = tok[keep], xyz[keep]
        if k_want is not None:
            tiles, used_r = tiles_at_fixed_k(xyz, tok, k_want, min_size=min_size)
        else:
            tiles, used_r = surface_tiles(xyz, tok, radius, min_size=min_size), float(radius)
        if not tiles:
            continue
        out[t] = tiles
        rows.append((len(tiles), used_r, int(np.median([len(p) for p in tiles])),
                     max(len(p) for p in tiles)))
    write_map(out, out_json, "predicted", "surface_tiles",
              note=("EXHAUSTIVE surface tiling by farthest-point sampling, "
                    + ("K fixed at %d with the radius DERIVED per target" % k_want if k_want
                       else "fixed radius %.4g A" % radius)
                    + (", surface filter %g%%" % surf_pct if surf_pct > 0 else ", no surface filter")
                    + ". PREDICTED antigen geometry, no ground truth."))
    a = np.asarray(rows, float)
    print("tiles: %d targets -> %s" % (len(rows), out_json))
    if len(rows):
        print("  K: min %d median %d max %d | radius: median %.1f A | patch size median %d, worst %d"
              % (a[:, 0].min(), np.median(a[:, 0]), a[:, 0].max(), np.median(a[:, 1]),
                 np.median(a[:, 2]), a[:, 3].max()))
        print("  targets with a patch > 40 residues: %d/%d" % ((a[:, 3] > 40).sum(), len(rows)))
    return out



def ensemble_antigen_geometry(cifs, input_json, ab_chains=AB_CHAINS):
    """``(tok, dbar, cbar)`` -- pool-mean pairwise distances and centroid distances."""
    tok0 = dsum = csum = None
    used = 0
    for c in cifs:
        tok, xyz = antigen_token_xyz(c, input_json, ab_chains)
        if len(tok) < 20:
            continue
        if tok0 is None:
            tok0 = tok
            dsum = np.zeros((len(tok), len(tok)), float)
            csum = np.zeros(len(tok), float)
        elif not np.array_equal(tok, tok0):
            continue                       # different token set: not comparable, do not force it
        dsum += np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
        csum += np.linalg.norm(xyz - xyz.mean(0), axis=1)
        used += 1
    if used == 0:
        return None, None, None, 0
    return tok0, dsum / used, csum / used, used


def build_ensemble_tilemap(pool_dir, out_json, wired_dir, k_want=14, min_size=3,
                           n_sample=None, ab_chains=AB_CHAINS, targets=None,
                           fill_to=None, modes_json=None):
    """Surface tiling on the POOL-MEAN geometry rather than on one sampled conformer."""
    want = int(n_sample if n_sample is not None
               else os.environ.get("POOL_MODES_N_SAMPLE", "150"))
    nmodes = {}
    if fill_to and modes_json:
        _m = json.load(open(modes_json))
        nmodes = {k: len(map_patches(v)) for k, v in _m.items() if not str(k).startswith("_")}
    out, rows = {}, []
    for jf in sorted(glob.glob(os.path.join(wired_dir, "*.json"))):
        t = os.path.splitext(os.path.basename(jf))[0]
        if targets and t not in targets:
            continue
        cand = sorted(glob.glob(os.path.join(pool_dir, "%s__*" % t, "seed_*",
                                             "predictions", "*.cif")))
        if not cand:
            continue
        k_t = int(k_want) if not fill_to else max(1, int(fill_to) - nmodes.get(t, 0))
        # SAME DRAW AS THE MODES HALF, so the two halves of the hybrid see the same pool.
        pick = cand[::max(1, len(cand) // want)][:want]
        tok, dbar, cbar, used = ensemble_antigen_geometry(pick, jf, ab_chains)
        if tok is None:
            continue
        tiles, used_r = tiles_from_geometry_at_k(tok, dbar, cbar, k_t, min_size=min_size)
        if not tiles:
            continue
        out[t] = tiles
        rows.append((len(tiles), used_r, int(np.median([len(p) for p in tiles])),
                     max(len(p) for p in tiles), used))
    write_map(out, out_json, "predicted", "ensemble_tiles",
              note=("Farthest-point surface tiling on the POOL-MEAN geometry: mean pairwise "
                    "distance and mean centroid distance over up to %d structures, strided across "
                    "the pool. Rotation/translation invariant, so no superposition and no mean "
                    "structure. K fixed at %d with the radius derived per target. PREDICTED antigen "
                    "geometry, no ground truth." % (want, k_want)))
    a = np.asarray(rows, float)
    print("ensemble-tiles: %d targets -> %s" % (len(rows), out_json))
    if len(rows):
        print("  K: min %d median %d max %d | radius: median %.1f A | patch size median %d, worst %d"
              % (a[:, 0].min(), np.median(a[:, 0]), a[:, 0].max(), np.median(a[:, 1]),
                 np.median(a[:, 2]), a[:, 3].max()))
        print("  structures averaged per target: min %d median %d"
              % (a[:, 4].min(), np.median(a[:, 4])))
    return out


def build_af3red_plddt_masks(pool_dir, wired_dir, out_dir, targets=None,
                             min_plddt=60.0, max_structs=50):
    """AF3-ReD's pLDDT mask, one ``af3red_plddt_<target>.npy`` per target. **CPU only.**"""
    os.makedirs(out_dir, exist_ok=True)
    if not targets:
        targets = sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{wired_dir}/*.json"))
    rows, skipped = [], []
    for t in targets:
        js = os.path.join(wired_dir, f"{t}.json")
        if not os.path.exists(js):
            skipped.append((t, "no wired input")); continue
        pats = [f"{pool_dir}/{t}/**/*_sample_*.cif",        # b0b: <pool>/<target>/<seed>_<sample>/
                f"{pool_dir}/{t}__*/**/*_sample_*.cif",     # unit dirs directly under the pool
                f"{pool_dir}/**/{t}__*/**/*_sample_*.cif"]  # a RUN dir, units nested one level down
        cifs = []
        for _pat in pats:
            cifs = sorted(glob.glob(_pat, recursive=True))
            if cifs:
                break
        if not cifs:
            skipped.append((t, "no structures matched any of: " + " | ".join(pats))); continue
        vals, n_tok = [], None
        for c in cifs[:max_structs]:
            v = token_ca_plddt(c, js)
            if v is None:
                continue
            if n_tok is None:
                n_tok = len(v)
            if len(v) == n_tok:
                vals.append(v)
        if not vals:
            skipped.append((t, "no cif could be put in token correspondence")); continue
        mean = np.nanmean(np.vstack(vals), axis=0)
        keep = np.isfinite(mean) & (mean > float(min_plddt))
        if int(keep.sum()) < 3:
            skipped.append((t, "mask would keep %d residues, below the floor of 3" % int(keep.sum())))
            continue
        np.save(os.path.join(out_dir, f"af3red_plddt_{t}.npy"), keep)
        rows.append((t, n_tok, int((~keep).sum()), len(vals), float(np.median(mean))))
    if rows:
        ex = np.array([r[2] for r in rows], float); nr = np.array([r[1] for r in rows], float)
        frac = 100.0 * ex / np.maximum(nr, 1)
        print("af3red-plddt-mask: wrote %d masks -> %s" % (len(rows), out_dir))
        print("  excluded at pLDDT<=%g: median %.2f%%  mean %.2f%%  max %.2f%%  (%d target(s) above 5%%)"
              % (min_plddt, np.median(frac), frac.mean(), frac.max(), int((frac > 5).sum())))
        for t, n, k, ns, med in sorted(rows, key=lambda r: -r[2] / max(r[1], 1))[:5]:
            print("    %-14s %4d/%-4d excluded (%5.1f%%) from %d structures, median Ca pLDDT %.1f"
                  % (t, k, n, 100.0 * k / max(n, 1), ns, med))
    for t, why in skipped:
        print("  !! %s SKIPPED: %s" % (t, why))
    return len(rows), len(skipped)


def build_hybrid_tokenmap(modes_json, tiles_json, out_json, min_size=3):
    """HYBRID map: pool MODES first, then every surface tile RANKED by mode support."""
    exh = json.load(open(tiles_json))
    modes = json.load(open(modes_json))
    out, rows = {}, []
    for t in sorted(k for k in exh if not k.startswith("_")):
        cells = [set(p) for p in exh[t]]
        m = map_patches(modes.get(t, []))
        mode_tokens = set()
        for p in m:
            mode_tokens |= set(p)
        score = [len(c & mode_tokens) / max(len(c), 1) for c in cells]
        ranked = [sorted(int(x) for x in exh[t][i])
                  for i in np.argsort(-np.asarray(score), kind="stable")]
        seen, merged = set(), []
        for p in [sorted(int(x) for x in q) for q in m] + ranked:
            key = tuple(p)
            if key in seen or len(p) < int(min_size):
                continue
            seen.add(key)
            merged.append(p)
        out[t] = merged
        rows.append((len(m), len(cells), len(merged), sum(1 for s in score if s > 0)))
    _tb = (json.load(open(tiles_json)) or {}).get("_builder", "surface_tiles")
    builder = "hybrid_modes_tiles" if _tb in ("surface_tiles", "", None) \
        else "hybrid_modes_%s" % _tb
    write_map(out, out_json, "predicted", builder,
              note=("HYBRID: unsteered-pool MODES first, then EXHAUSTIVE surface cells ranked by "
                    "overlap with those modes. Coverage from the tiling, ordering from the pool. "
                    "Tiling provenance: %s. No ground truth." % _tb))
    a = np.asarray(rows, float)
    print("hybrid: %d targets -> %s" % (len(rows), out_json))
    if len(rows):
        print("  modes median %d | tiles median %d | hybrid K median %d max %d"
              % (np.median(a[:, 0]), np.median(a[:, 1]), np.median(a[:, 2]), a[:, 2].max()))
        print("  tiles with ANY mode support: median %d of %d (the rest rank last)"
              % (np.median(a[:, 3]), np.median(a[:, 1])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cdr", help="IMGT CDR map -> paratope token indices ($PROTENIX_CDR_MAP)")
    c.add_argument("--run-dir", required=True)
    c.add_argument("--cdr-map", required=True, help="{target: {chain: [CDR residue idx]}} json")
    c.add_argument("--out-json", required=True)
    c.add_argument("--wired-dir", default=None)

    e = sub.add_parser("epitope", help="per-residue epitope scores -> antigen epitope token indices")
    e.add_argument("--run-dir", required=True)
    e.add_argument("--pool-dir", "--dt3-dir", dest="src_dir", required=True,
                   help="source the scores are read from: the UNSTEERED pool predictions dir "
                        "under --scorer pool_modes, or the DiscoTope-3 CSV dir under discotope3")
    e.add_argument("--out-json", required=True)
    e.add_argument("--wired-dir", default=None)
    e.add_argument("--top-k", type=int, default=20, help="residues per patch")
    e.add_argument("--use-flag", action="store_true", help="use DiscoTope's epitope=True call instead")
    e.add_argument("--n-patches", type=int, default=1,
                   help="1 = one flat patch (historical); >1 writes a multi-SITE map")
    e.add_argument("--patch-radius", type=float, default=PATCH_RADIUS,
                   help="one epitope footprint; patches are grown to this radius from their seed")
    e.add_argument("--scorer", default="pool_modes", choices=sorted(EPITOPE_SCORERS),
                   help="which per-residue epitope scorer supplies the scores (default: the "
                        "unsteered pool's own binding modes, which hybK12 is built from)")

    sh = sub.add_parser("shuffle-sites",
                        help="ablation A5: base map's shape, candidate LOCATIONS redrawn at random")
    sh.add_argument("--base", required=True, help="map to shuffle (normally the hybk12 map the run uses)")
    sh.add_argument("--pool-dir", required=True, help="unsteered pool predictions dir (geometry source)")
    sh.add_argument("--wired-dir", required=True, help="inputs_wired dir (holds the jsons with sequences)")
    sh.add_argument("--out-json", required=True)
    sh.add_argument("--seed", type=int, default=0, help="draw seed; RECORD IT, the map is not unique")
    sh.add_argument("--min-shift", type=float, default=12.0,
                    help="reject a draw whose centroid is nearer than this to ANY base candidate "
                         "(default one epitope footprint, so an accepted draw cannot overlap)")
    sh.add_argument("--surf-pct", type=float, default=40.0,
                    help="drop the inner N%% by radius; a buried patch is not a reachable site")
    sh.add_argument("--tries", type=int, default=500)
    sh.add_argument("--true-map", default=None,
                    help="DIAGNOSTIC ONLY, never written: report true-epitope coverage before/after")

    t = sub.add_parser("tiles", help="exhaustive farthest-point surface tiling (fixed K or fixed radius)")
    t.add_argument("--pool-dir", required=True, help="unsteered pool predictions dir (geometry source)")
    t.add_argument("--wired-dir", required=True, help="inputs_wired dir (holds the jsons with sequences)")
    t.add_argument("--out-json", required=True)
    t.add_argument("--k-want", type=int, default=None,
                   help="fixed K, radius derived per target by binary search (recommended: 12)")
    t.add_argument("--radius", type=float, default=None,
                   help="fixed radius in A, K then varies with surface area. Mutually exclusive with --k-want")
    t.add_argument("--surf-pct", type=float, default=0.0,
                   help="drop the inner N%% by radius. LEAVE AT 0: sweeping it gave 43->50->51 of 52 "
                        "at 40->20->0, because groove-binding epitopes sit below the cut")
    t.add_argument("--min-size", type=int, default=3)

    et = sub.add_parser("tiles-ensemble",
                        help="surface tiling on the POOL-MEAN geometry (reproducible AND stable)")
    et.add_argument("--pool-dir", required=True, help="unsteered pool predictions dir")
    et.add_argument("--wired-dir", required=True, help="inputs_wired dir")
    et.add_argument("--out-json", required=True)
    et.add_argument("--k-want", type=int, default=14, help="fixed K, radius derived per target")
    et.add_argument("--n-sample", type=int, default=None,
                    help="structures to average, strided across the pool "
                         "(default POOL_MODES_N_SAMPLE=150, matching the modes half)")
    et.add_argument("--min-size", type=int, default=3)
    et.add_argument("--targets", nargs="*", default=None)
    et.add_argument("--fill-to", type=int, default=None,
                    help="hold the TOTAL candidate budget at this many per target: tiles requested "
                         "= fill_to - n_modes(target). Needs --modes-json")
    et.add_argument("--modes-json", default=None, help="modes map, to read n_modes per target")

    cm = sub.add_parser("cdr-map", help="STAGE 1: wired inputs -> per-chain IMGT CDR map (needs ANARCI)")
    cm.add_argument("--wired-dir", required=True, help="dir of wired input JSONs (one per target)")
    cm.add_argument("--out-json", required=True)
    cm.add_argument("--antibody-chains", default=None, help="comma-separated, default the verifier's set")
    cm.add_argument("--hmmer-path", default=None,
                    help="dir containing the REAL hmmscan executable, if not on PATH. It must be the "
                         "binary itself, not a wrapper script: HMMER writes argv[0] into its output "
                         "header and ANARCI's parser keys off that name")

    te = sub.add_parser("true-epitope",
                        help="SCORING ONLY: native-complex epitope -> antigen token indices (USES GROUND TRUTH)")
    te.add_argument("--gt-dir", required=True, help="dir of native structures, <target>.pdb/.cif or <target>/<target>.pdb")
    te.add_argument("--wired-dir", required=True, help="inputs_wired dir")
    te.add_argument("--out-json", required=True)
    te.add_argument("--contact-r", type=float, default=None, help="default: the shared CONTACT_R")
    te.add_argument("--targets", nargs="*", default=None)

    am = sub.add_parser("af3red-plddt-mask",
                        help="AF3-ReD pLDDT mask per target (CPU). Writes af3red_plddt_<target>.npy")
    am.add_argument("--pool-dir", required=True, help="UNSTEERED pool whose cifs carry pLDDT in the B-factor column")
    am.add_argument("--wired-dir", required=True, help="wired inputs, for the model token order")
    am.add_argument("--out-dir", required=True, help="the af3_red history dir the hook reads")
    am.add_argument("--targets", nargs="*", default=None, help="default: every target in --wired-dir")
    am.add_argument("--min-plddt", type=float, default=60.0, help="reference default 60, keep is strictly above")
    am.add_argument("--max-structs", type=int, default=50, help="structures averaged per target")

    h = sub.add_parser("hybrid", help="pool modes first, then surface tiles ranked by mode support")
    h.add_argument("--modes-json", required=True, help="pool-modes map (see `epitope --scorer pool_modes`)")
    h.add_argument("--tiles-json", required=True, help="surface-tiling map (see `tiles`)")
    h.add_argument("--out-json", required=True)
    h.add_argument("--min-size", type=int, default=3)

    a = ap.parse_args()
    if a.cmd == "shuffle-sites":
        build_shuffled_sitemap(a.base, a.pool_dir, a.wired_dir, a.out_json, seed=a.seed,
                               min_shift=a.min_shift, surf_pct=a.surf_pct, tries=a.tries,
                               true_map=a.true_map)
    elif a.cmd == "tiles":
        build_surface_tilemap(a.pool_dir, a.out_json, a.wired_dir, radius=a.radius, k_want=a.k_want,
                              surf_pct=a.surf_pct, min_size=a.min_size)
    elif a.cmd == "tiles-ensemble":
        build_ensemble_tilemap(a.pool_dir, a.out_json, a.wired_dir, k_want=a.k_want,
                               min_size=a.min_size, n_sample=a.n_sample, targets=a.targets,
                               fill_to=a.fill_to, modes_json=a.modes_json)
    elif a.cmd == "cdr-map":
        build_cdr_map(a.wired_dir, a.out_json,
                      antibody_labels=(a.antibody_chains.split(",") if a.antibody_chains else None),
                      hmmer_path=a.hmmer_path)
    elif a.cmd == "true-epitope":
        build_true_epitope_tokenmap(a.gt_dir, a.out_json, a.wired_dir,
                                    contact_r=a.contact_r, targets=a.targets)
    elif a.cmd == "af3red-plddt-mask":
        build_af3red_plddt_masks(a.pool_dir, a.wired_dir, a.out_dir, targets=a.targets,
                                 min_plddt=a.min_plddt, max_structs=a.max_structs)
    elif a.cmd == "hybrid":
        build_hybrid_tokenmap(a.modes_json, a.tiles_json, a.out_json, a.min_size)
    elif a.cmd == "cdr":
        build_cdr_tokenmap(a.run_dir, a.cdr_map, a.out_json, a.wired_dir)
    else:
        build_epitope_tokenmap(a.run_dir, a.src_dir, a.out_json, a.wired_dir, a.top_k, a.use_flag,
                               a.n_patches, a.patch_radius, a.scorer)


if __name__ == "__main__":
    try:
        main()
    except EmptyMapError as _e:              # only the write_map guard; every other error keeps
        sys.stderr.write("!! %s\n" % _e)     # its traceback so a real builder bug stays debuggable
        raise SystemExit(2)
