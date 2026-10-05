"""Protocol every dataset loader satisfies."""
from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from data.target import Target


@runtime_checkable
class DatasetLoader(Protocol):
    """Return a list of :class:`Target`, model-agnostic."""

    name: str

    def load(self) -> List[Target]: ...


RCSB_CIF_URL = "https://files.rcsb.org/download/{entry}.cif"


def _entry_code(target_id: str) -> str:
    """PDB entry from a target id: ``9rxi_nb2`` -> ``9rxi``, ``7txw-assembly1`` -> ``7txw``."""
    import re
    return re.split(r"[_-]", target_id, 1)[0]


def _includes_identity(oper_expression: str) -> bool:
    """Does an assembly's ``oper_expression`` include the identity operator (id 1)?"""
    import re
    expr = (oper_expression or "").strip()
    if not expr:
        return True                      # no operator listed means the coordinates as deposited
    for term in re.split(r"[,\s]+", expr.replace("(", " ").replace(")", " ")):
        term = term.strip()
        if not term:
            continue
        if term == "1":
            return True
        m = re.fullmatch(r"(\d+)-(\d+)", term)
        if m and int(m.group(1)) <= 1 <= int(m.group(2)):
            return True
    return False


def check_biological_assembly(targets, cache_dir="~/.cache/rcsb_cif", strict=True,
                              url=RCSB_CIF_URL):
    """Verify each target's chains sit in ONE biological assembly under the identity operator."""
    import logging, urllib.request
    from pathlib import Path
    import numpy as np
    from Bio.PDB import MMCIFParser
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    log = logging.getLogger(__name__)
    cache = Path(cache_dir).expanduser(); cache.mkdir(parents=True, exist_ok=True)
    parser = MMCIFParser(QUIET=True)
    out, failures = {}, []

    for t in targets:
        gt = getattr(t, "ground_truth_path", None)
        if not gt or not Path(gt).is_file():
            out[t.target_id] = (None, "no reference structure"); continue
        entry = _entry_code(t.target_id).upper()
        cif = cache / f"{entry}.cif"
        try:
            if not cif.is_file() or cif.stat().st_size == 0:
                urllib.request.urlretrieve(url.format(entry=entry), cif)
            dep = parser.get_structure("d", str(cif))[0]
            d = MMCIF2Dict(str(cif))
        except Exception as exc:
            out[t.target_id] = (None, f"deposition unavailable: {exc}"); continue

        def coords(ch):
            return np.array([a.coord for r in ch if r.id[0] == " " for a in r])
        dep_ch = {c.id: coords(c) for c in dep if len(coords(c)) > 20}

        ours = {}
        from Bio.PDB import PDBParser, MMCIFParser as _MC
        p2 = PDBParser(QUIET=True) if str(gt).lower().endswith((".pdb", ".ent")) else _MC(QUIET=True)
        ref = p2.get_structure("r", str(gt))[0]
        for c in ref:
            X = coords(c)
            if len(X) < 5:
                continue
            scored = sorted((float(np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)).min(1).max()), n)
                            for n, Y in dep_ch.items())
            hit = None
            if scored:
                best, runner = scored[0], (scored[1] if len(scored) > 1 else (1e9, None))
                if best[0] <= 5.0 and runner[0] >= max(10.0, 4 * best[0]):
                    hit = best[1]
            if hit:
                ours[c.id] = hit
        if len(ours) < 2:
            out[t.target_id] = (None, "chains not identifiable in the deposition"); continue

        lab2auth = dict(zip(d.get("_atom_site.label_asym_id", []), d.get("_atom_site.auth_asym_id", [])))
        aid = d.get("_pdbx_struct_assembly_gen.assembly_id", [])
        ops = d.get("_pdbx_struct_assembly_gen.oper_expression", [])
        lists = d.get("_pdbx_struct_assembly_gen.asym_id_list", [])
        if isinstance(aid, str): aid, ops, lists = [aid], [ops], [lists]
        need = set(ours.values())
        by = {}
        for a, op, lst in zip(aid, ops, lists):
            auth = {lab2auth.get(x, x) for x in str(lst).split(",")}
            by.setdefault(a, []).append((str(op).strip(), auth))
        ok = False
        for a, parts in by.items():
            ident = set().union(*[p[1] for p in parts if _includes_identity(p[0])]) or set()
            if need <= ident:
                ok = True; out[t.target_id] = (True, f"assembly {a}"); break
        if not ok:
            if not by:
                out[t.target_id] = (None, "no assembly annotation"); continue
            detail = (f"chains {sorted(need)} are not together in any assembly under the identity "
                      f"operator; assemblies are "
                      + "; ".join(f"{a}={sorted(set().union(*[p[1] for p in ps]))}" for a, ps in by.items()))
            out[t.target_id] = (False, detail)
            failures.append(f"{t.target_id}: {detail}")

    n_ok = sum(1 for v in out.values() if v[0] is True)
    n_sk = sum(1 for v in out.values() if v[0] is None)
    log.info("assembly check: %d pass, %d fail, %d not checked", n_ok, len(failures), n_sk)
    if failures and strict:
        raise ValueError(
            "these references pair chains from different biological assemblies, so their interface "
            "is crystal packing rather than a complex:\n  " + "\n  ".join(failures))
    for f in failures:
        log.warning("assembly check FAILED %s", f)
    return out
