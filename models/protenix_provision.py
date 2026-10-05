"""Checkpoint / model-source resolution for the Protenix backend.

Prefers a PRIVATE, site-supplied Protenix-v2 checkpoint and falls back to the PUBLIC weights when
there is none, so the same code runs on a cluster that has the private checkpoint and on one that
does not with no config edit. Only the resolved source differs, and it is logged and recorded in
the run manifest.

A private checkpoint is put on disk, by baking it into the image or placing it by hand, under
``$root`` as::

    $root/checkpoint/protenix-v2.pt
    $root/common/{*.cif,*.pkl,*.csv,*.txt}

Resolution order:
  1. private checkpoint already on disk at ``private_root_dir``  -> use it.
  2. else, if ``allow_public_fallback``                          -> public weights, which Protenix
     auto-downloads on first use.
  3. else                                                        -> raise.
"""
from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

PRIVATE_MODEL_NAME = "protenix-v2"
PRIVATE_CHECKPOINT_REL = "checkpoint/protenix-v2.pt"

PUBLIC_MODEL_NAME = "protenix-v2"

# ---------------------------------------------------------------------------------------------
# WHERE THE PUBLIC ARTEFACTS LIVE, and what to do when that host is unreachable.
#
# The package fetches everything from one origin:
#
#     https://protenix.tos-cn-beijing.volces.com/checkpoint/<model_name>.pt
#     https://protenix.tos-cn-beijing.volces.com/common/components.cif
#     https://protenix.tos-cn-beijing.volces.com/common/components.cif.rdkit_mol.pkl
#
# `download_inference_cache` always wants four cache files plus the checkpoint, under
# $PROTENIX_ROOT_DIR:
#     common/components.cif                  the PDB chemical component dictionary
#     common/components.cif.rdkit_mol.pkl    {ccd_code: rdkit mol}, read via .get(code, None)
#     common/clusters-by-entity-40.txt       training-time clustering, data_pipeline defaults it None
#     common/obsolete_release_date.csv       training-time date filter
#     checkpoint/<model_name>.pt
# Only the first and the checkpoint carry information a protein-only inference run needs. The rdkit
# cache is a dict lookup that returns None for a missing code, so an EMPTY DICT pickle is a valid
# cache rather than a stub, and a standard-residue antibody/antigen input asks it for nothing. The
# other two are presence-checked and then unused on that path, so an empty file and a header-only
# CSV satisfy them. Record that when it is done, because a target carrying a real ligand would need
# the true rdkit cache.
#
# `download_inference_cache` lays them out under $PROTENIX_ROOT_DIR as `checkpoint/<model_name>.pt`
# and `common/...`, downloads each one ONLY `if not opexists(...)`, and its own failure message says
# "Please download manually with: wget <url> -O <path>". So PRE-PLACING the files is a supported
# path rather than a workaround, which matters because that origin is firewalled on some networks.
# Two shortcuts when it is blocked:
#   * `common/components.cif` is the PDB chemical component dictionary and is better fetched from
#     its authoritative home anyway: https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
#   * `use_template: false` means the template caches (obsolete_pdbs_path, release_dates_path) are
#     never requested, and the `search_database/*` FASTAs are for MSA search only, so a run with
#     pre-built MSAs needs neither.
#
# THIRD-PARTY MIRROR of the full v2 checkpoint, for when the origin is blocked. `protenix-v2` is a
# model name the public package itself knows (configs/configs_model_type.py), so this mirrors a
# public artefact rather than any private build:
#
#     https://huggingface.co/TMF001/protenix-v2-weights  ->  protenix-v2.pt
#     sha256 8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599   (1859785497 bytes)
#     464,442,431 params, c_z=256, N_cycle=10, N_step=200. Apache-2.0
#
# TREAT IT AS UNTRUSTED CODE, NOT AS NUMBERS. `inference.py` loads with `weights_only=False`, so
# unpickling executes whatever the pickle names. Audit before any load: verify the digest above and
# scan the pickle's opcodes without executing them. A mirror also cannot be checked against what
# upstream published, only against what the publisher stated, so a run using it must record that.
# Set `checkpoint.prefer_private: false` with `public_model_name: protenix-v2` so the manifest says
# `source: public`. Do NOT place it where the private-checkpoint branch looks, because that branch
# keys off the same `protenix-v2.pt` filename and the record would then claim a provenance the file
# does not have.
# ---------------------------------------------------------------------------------------------


@dataclass
class ModelSource:
    """Outcome of :func:`resolve_model_source`."""

    source: str                       # "private" | "public"
    model_name: str                   # value for runner --model_name
    root_dir: Optional[str]           # PROTENIX_ROOT_DIR (None → package default cache)
    reason: str                       # human-readable why this source was chosen
    private_attempted: bool = False        # did we try the private source at all?
    details: Dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "model_name": self.model_name,
            "root_dir": self.root_dir,
            "reason": self.reason,
            "private_attempted": self.private_attempted,
            **self.details,
        }


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a key from either a dict or an OmegaConf node."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:  # noqa: BLE001, OmegaConf struct edge cases
            return default
    return getattr(cfg, key, default)


def _private_checkpoint_present(root: Optional[str]) -> bool:
    return bool(root) and (Path(root) / PRIVATE_CHECKPOINT_REL).is_file()


def resolve_model_source(cfg: Any) -> ModelSource:
    """Resolve which Protenix checkpoint to use, private-first with public fallback."""
    ck = _cfg_get(cfg, "checkpoint", {}) or {}

    prefer_private = bool(_cfg_get(ck, "prefer_private", True))
    allow_public = bool(_cfg_get(ck, "allow_public_fallback", True))
    private_model = _cfg_get(ck, "private_model_name", None) or _cfg_get(cfg, "model_name", PRIVATE_MODEL_NAME)
    private_root = _cfg_get(ck, "private_root_dir", None) or _cfg_get(cfg, "root_dir", None) \
        or os.environ.get("PROTENIX_ROOT_DIR")

    pub_model = _cfg_get(ck, "public_model_name", PUBLIC_MODEL_NAME)
    pub_root = _cfg_get(ck, "public_root_dir", None) or os.environ.get("PROTENIX_PUBLIC_ROOT_DIR")

    def _public(reason: str, private_attempted: bool, **details: Any) -> ModelSource:
        if not allow_public:
            raise RuntimeError(
                f"the private Protenix build checkpoint unavailable ({reason}) and "
                f"checkpoint.allow_public_fallback=false, cannot proceed."
            )
        logger.warning("Protenix checkpoint: %s → using PUBLIC weights "
                       "(model_name=%s, root=%s).", reason, pub_model, pub_root or "<pkg default>")
        return ModelSource("public", pub_model, pub_root, reason,
                           private_attempted=private_attempted, details=details)

    if not prefer_private:
        return _public("checkpoint.prefer_private=false", private_attempted=False)

    # 1. A checkpoint already on disk at private_root?
    if _private_checkpoint_present(private_root):
        same_root = bool(pub_root) and os.path.realpath(str(private_root)) == os.path.realpath(str(pub_root))
        if same_root:
            logger.warning("Protenix checkpoint: private_root_dir and public_root_dir are the SAME path "
                           "(%s), so source='private' here means 'a checkpoint was found', NOT that the private source "
                           "weights were used. Point private_root_dir at a separate cache to tell them "
                           "apart.", private_root)
        logger.info("Protenix checkpoint: using the checkpoint found at %s (provenance: on-disk, "
                    "not downloaded this run).", private_root)
        return ModelSource("private", private_model, private_root,
                           f"checkpoint found on disk at {private_root}"
                           + (" (SAME path as public_root_dir)" if same_root else ""),
                           private_attempted=True,
                           details={"provenance": "found-on-disk",
                                    "same_as_public_root": same_root})

    # NO DOWNLOAD BRANCH. A private checkpoint is put on disk, by baking it into the image or
    # placing it by hand, and step 1 above finds it.
    return _public("no checkpoint at %s" % (private_root or "<unset private_root_dir>"),
                   private_attempted=True)
