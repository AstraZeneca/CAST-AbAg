#!/usr/bin/env python3
r"""Helpers shared by the job templates in ``jobs/``, so the MSA-wiring, sharding and
completeness/checkpoint logic is identical across every template rather than inlined per template.

Run as a CLI from inside a job. REPO_ROOT must be on PYTHONPATH (the templates export it) so the
``models`` / ``method`` imports resolve::

    python jobs/tts_job_lib.py wire-msas    --run-dir $RUN_DIR --msa-db $MSA_DB
    python jobs/tts_job_lib.py wire-shard   --run-dir $RUN_DIR --msa-db $MSA_DB \
                                            --nshards 4 --shard-id 0 --out-dir $OUT/inputs_wired
    python jobs/tts_job_lib.py checkpoint   --wired-dir W --pred-dir P --todo-dir T \
                                            --n-seeds 20 --n-samples 50 [--granular]
    python jobs/tts_job_lib.py count-complete --parquet $P --exp 1000
    python jobs/tts_job_lib.py cdr-tokenmap  --run-dir $RUN_DIR --cdr-map cdr_map.json \
                                             --out-json $RUN_DIR/cdr_tokenmap.json
    python jobs/tts_job_lib.py pool-stats    --cdr-map $PROTENIX_CDR_MAP --target $TGT \
                                             --pred-dir $OUT/pred
    python jobs/tts_job_lib.py epi-dist      --cdr-map $PROTENIX_CDR_MAP --target $TGT \
                                             --true-epitope-map $TRUE_EPI --pred-dir $OUT/pred
                                             # ORACLE, uses ground truth: analysis only

It also holds the arm preflight, the health gate and the checkpoint route the templates call.

PATH / ID GRANULARITY. A run has three nested identifiers under ``pred_dir``::

    {tid}__random@d{depth}_s{SEED}/  seed_{INFER_SEED}/  predictions/  {tid}...sample_{SAMPLE}.cif
     \___target___/     \_MSA seed_/                                              \__diffusion sample_/

The MSA-subsample SEED is in the top directory NAME, a fixed inference seed directory sits under
it, and the diffusion SAMPLE index is in the file name, alongside the matching
``_pae_sample_{SAMPLE}.npz`` and summary json. So a target is matched by ``{tid}__*`` and not
``{tid}/``, and the ``__`` separator guards against prefix collisions such as 9qt3_ab1 against
9qt3_ab10.
"""
import argparse
import glob
import json
import datetime as _dt
import os
import numpy as np
import re
import shutil
import sys


# ------------------------------------------------------------------ MSA wiring
def _wire_files(files, msa_db, out_dir):
    """Wire full MSAs from msa_db onto each input json in `files`, writing to out_dir."""
    from models.protenix2 import ProtenixBackend
    from method.trunk_side import wire_msa_from_dump
    os.makedirs(out_dir, exist_ok=True)
    b = ProtenixBackend({})
    ok = miss = 0
    empty = []
    for f in files:
        tid = os.path.basename(f)[:-5]
        _, wired = wire_msa_from_dump(b, f, os.path.join(msa_db, tid), os.path.join(out_dir, tid + ".json"))
        nwired = sum(1 for v in wired.values() if v)
        if nwired == 0:
            empty.append(tid)
        if nwired < len(wired):
            miss += 1
            print(f"  {tid}: wired {nwired}/{len(wired)} chains (some unmatched)")
        else:
            ok += 1
    print(f"WIRED {ok}/{len(files)} fully ({miss} partial, {len(empty)} with NO msa)")
    assert ok + miss == len(files) and len(files) > 0, "MSA wiring produced no inputs"
    assert not empty, (
        f"{len(empty)} target(s) wired ZERO chains from {msa_db}: {empty[:6]}. The dump is looked "
        f"up at <msa_db>/<target>, so this is a wrong msa_db root or a dataset whose targets were "
        f"renamed after the dumps were built. Running on would sample with no MSA.")
    return ok, miss


def _prepared_inputs(run_dir):
    """The prepared (pre-wiring) input jsons, excluding run_ablation's -update-msa scratch files."""
    return sorted(f for f in glob.glob(os.path.join(run_dir, "inputs", "*.json")) if "-update-msa" not in f)


def wire_all_msas(run_dir, msa_db, out_dir=None):
    """Wire every prepared input (used by the 4-GPU and pool jobs, which wire the whole set)."""
    return _wire_files(_prepared_inputs(run_dir), msa_db, out_dir or os.path.join(run_dir, "inputs_wired"))


# ------------------------------------------------------------------ CDR token map (cdr_rmsd reward)
from method import ANTIBODY_CHAINS   # noqa: E402

_AB_CHAINS = ANTIBODY_CHAINS   # the one definition, see method/__init__.py


def _token_maps():
    """The builders, which live in experiments/scripts/token_maps.py (dataset prep, not job plumbing)."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from experiments.scripts import token_maps
    return token_maps


def build_cdr_tokenmap(*a, **kw):
    """Thin delegator -> experiments.scripts.token_maps.build_cdr_tokenmap."""
    return _token_maps().build_cdr_tokenmap(*a, **kw)


def _epitope_scorers():
    """Scorer names token_maps registers, so the two CLIs cannot offer different sets."""
    try:
        return _token_maps().EPITOPE_SCORERS
    except Exception:                      # noqa: BLE001 -- keep --help usable without the import
        return ("pool_modes", "discotope3")


def build_epitope_tokenmap(*a, **kw):
    """Thin delegator -> experiments.scripts.token_maps.build_epitope_tokenmap."""
    return _token_maps().build_epitope_tokenmap(*a, **kw)


# ------------------------------------------------------------------ cost-balanced sharding
_OFFLOAD_TOKENS = 950   # bin-pack cost weight only: >this counts as an offload target (~5x slower/seed).
                        # Card-agnostic on purpose (the packer cannot know each shard's card), the real
                        # per-card offload decision is models.protenix2._max_ongpu_tokens at runtime.


def _tok_cost(json_path):
    """Cost proxy = (sum of protein/NA chain sequence lengths) x (5 if it offloads else 1). A
    lightweight json read (1 token/residue) -- no heavy backend import just to bin-pack; matches the
    heuristic the job specs use, so a recovery/subset job and the shard job produce the SAME bins."""
    import json
    d = json.load(open(json_path))[0]
    ntok = sum(len(v["sequence"]) for s in d["sequences"] if isinstance(s, dict)
               for k, v in s.items() if isinstance(v, dict) and "sequence" in v)
    return ntok * (5 if ntok > _OFFLOAD_TOKENS else 1)


def cost_balanced_bins(files, nshards):
    """Greedy-LPT bin-pack of input json paths into `nshards` bins by :func:`_tok_cost` (token size x
    offload weight), so all shards finish at ~the same time. Deterministic given (files, nshards): a
    target always lands in the same bin -- so with nshards FIXED, each shard's predictions dir is
    self-consistent across resubmits."""
    bins = [[] for _ in range(nshards)]
    load = [0] * nshards
    for f in sorted(files, key=_tok_cost, reverse=True):     # heaviest first -> currently-lightest bin
        i = load.index(min(load))
        bins[i].append(f)
        load[i] += _tok_cost(f)
    return bins


def wire_shard(run_dir, msa_db, nshards, shard_id, out_dir):
    """Bin-pack all prepared inputs and wire ONLY this shard's cost-balanced bin (used by the
    sharded job so each shard wires just its own targets)."""
    bins = cost_balanced_bins(_prepared_inputs(run_dir), nshards)
    return _wire_files(bins[shard_id], msa_db, out_dir)


def cost_bins_for_targets(tids, run_dir, nshards, shard_id):
    """Cost-balanced bin for one shard over an EXPLICIT target-id list (e.g. a recovery subset or a
    full-52 rerun that wires elsewhere), rather than the whole run_dir/inputs. Same greedy-LPT +
    offload-5x weighting as cost_balanced_bins, so a recovery job and the shard job agree. Returns
    this shard's target ids (order-preserved)."""
    files = [os.path.join(run_dir, "inputs_wired", t + ".json") for t in tids]
    return [os.path.basename(f)[:-5] for f in cost_balanced_bins(files, nshards)[shard_id]]


# ------------------------------------------------------------------ merge consolidation
def consolidate_predictions(src_dirs, dest):
    """Move every `{tid}__...` prediction dir from each per-shard/per-worker src dir into ONE shared
    `dest`, so the FINAL layout matches the 4-GPU/pool jobs (dest/{tid}__.../). Skips a target already
    present in dest (never overwrites). Cost-balanced bins are disjoint, so no two srcs hold the same
    target. Returns the number moved."""
    os.makedirs(dest, exist_ok=True)
    moved = 0
    for src in src_dirs:
        for d in glob.glob(os.path.join(src, "*__*")):
            b = os.path.basename(d)
            tgt = os.path.join(dest, b)
            if not os.path.exists(tgt):
                shutil.move(d, tgt)
                moved += 1
    print(f"consolidated {moved} target dirs into {dest}")
    return moved


# ------------------------------------------------------------------ completeness / checkpoint
def _target_glob(pred_dir, tid, leaf):
    # {tid}__*/seed_*/predictions/<leaf>  -- see the PATH GRANULARITY note in the module docstring
    return glob.glob(os.path.join(pred_dir, tid + "__*", "seed_*", "predictions", leaf))


def target_counts(pred_dir, tid):
    """(#cif, #pae-npz) for a target across all its MSA-seed dirs (totals, order-independent)."""
    return len(_target_glob(pred_dir, tid, "*.cif")), len(_target_glob(pred_dir, tid, "*_pae_sample_*.npz"))


def seed_dir_complete(d, n_samples):
    """One MSA-seed dir holds a full seed: n_samples cif AND n_samples pae-npz."""
    ncif = len(glob.glob(os.path.join(d, "seed_*", "predictions", "*.cif")))
    nnpz = len(glob.glob(os.path.join(d, "seed_*", "predictions", "*_pae_sample_*.npz")))
    return ncif >= n_samples and nnpz >= n_samples


def target_complete_granular(pred_dir, tid, n_seeds, n_samples):
    """Stricter than a total count: require EACH of the target's >= n_seeds MSA-seed dirs to hold
    n_samples cif AND n_samples pae-npz. Catches an uneven seed distribution (e.g. one seed dir
    short, another duplicated) that a bare total of n_seeds*n_samples could otherwise mask."""
    seed_dirs = glob.glob(os.path.join(pred_dir, tid + "__*"))
    if len(seed_dirs) < n_seeds:
        return False
    return sum(1 for d in seed_dirs if seed_dir_complete(d, n_samples)) >= n_seeds


def checkpoint(wired_dir, pred_dir, todo_dir, n_seeds, n_samples, granular=False,
               resume_seeds=False):
    """Populate todo_dir with the inputs that still need running. A target is COMPLETE only with
        n_seeds*n_samples cif AND the matching pae-npz confidence (npz alone can be missing after an
        interrupt between the structure and confidence dumps, and scoring needs it -- counting cifs
        alone would falsely skip it and the score pass would find no confidence). Returns (n_done, n_todo).
    """
    exp = n_seeds * n_samples
    os.makedirs(todo_dir, exist_ok=True)
    n_done = n_todo = 0
    for f in sorted(glob.glob(os.path.join(wired_dir, "*.json"))):
        tid = os.path.basename(f)[:-5]
        ncif, nnpz = target_counts(pred_dir, tid)
        done = target_complete_granular(pred_dir, tid, n_seeds, n_samples) if granular \
            else (ncif >= exp and nnpz >= exp)
        if done:
            print(f"  checkpoint: {tid} COMPLETE (cif={ncif} npz={nnpz}) -> skip")
            n_done += 1
        else:
            if ncif or nnpz:
                dirs = glob.glob(os.path.join(pred_dir, tid + "__*"))
                keep = [d for d in dirs if resume_seeds and seed_dir_complete(d, n_samples)]
                for d in dirs:
                    if d not in keep:
                        shutil.rmtree(d, ignore_errors=True)
                if keep:
                    print(f"  checkpoint: {tid} PARTIAL (cif={ncif} npz={nnpz}) -> keeping "
                          f"{len(keep)} complete seed dir(s), cleaning {len(dirs) - len(keep)}")
                else:
                    print(f"  checkpoint: {tid} PARTIAL (cif={ncif} npz={nnpz}) -> clean + rerun")
            shutil.copy(f, todo_dir)
            n_todo += 1
    print(f"checkpoint: {n_done} complete, {n_todo} to run")
    return n_done, n_todo


def count_complete_targets(parquet, exp):
    """How many target_ids in the per-sample parquet reached >= exp rows (final completeness gate)."""
    if not os.path.exists(parquet):
        return 0
    import pandas as pd
    return sum(1 for _, g in pd.read_parquet(parquet).groupby("target_id") if len(g) >= exp)




def preflight(spec_overrides, n_steps=200, n_particles=8):
    """Validate a steering configuration offline. Returns a list of problems (empty = OK)."""
    import dataclasses

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from method.diffusion_side.diffusion_search import ALL_REWARDS, DiffusionSearchSpec
    from method.diffusion_side.base_search import MIN_PER_CELL, MIN_PER_SUBSAMPLE
    import method.diffusion_side.guidance as gd

    problems, fields = [], {f.name: f for f in dataclasses.fields(DiffusionSearchSpec)}

    # 1. every override must name a real field, else it is silently ignored
    kw = {}
    for k, v in spec_overrides.items():
        if k not in fields:
            problems.append("override %r is not a DiffusionSearchSpec field: it will be IGNORED" % k)
            continue
        t = fields[k].type
        try:
            kw[k] = (str(v).lower() in ("true", "1", "yes") if t in (bool, "bool")
                     else float(v) if t in (float, "float")
                     else int(v) if t in (int, "int") else str(v))
        except (TypeError, ValueError):
            problems.append("override %r=%r cannot be read as %s" % (k, v, t))
    if problems:
        return problems
    spec = DiffusionSearchSpec(**kw)

    # 2. tempering must leave room for the resampler to trigger
    if spec.lam_adaptive and spec.target_ess_frac >= spec.ess_fraction:
        problems.append(
            "target_ess_frac=%.2f >= ess_fraction=%.2f: tempering guarantees ESS above the resample "
            "trigger, so resampling can never fire and steering is a NO-OP"
            % (spec.target_ess_frac, spec.ess_fraction))

    _rs = str(getattr(spec, "resample", "ess")).strip()
    if _rs not in ("ess", "every_k", "none"):
        problems.append("resample=%r is not one of ess|every_k|none: nothing validates this field "
                        "downstream, so an unrecognised value silently disables resampling for the "
                        "whole run while the arm still looks like a filtered arm" % _rs)
    _other = []
    if getattr(spec, "boltz_perturb_enabled", False):
        _other.append("boltz_perturb")
    if spec.algorithm == "svgd":
        _other.append("svgd repulsion")
    if spec.algorithm == "af3_red":
        _other.append("af3_red history bias")
    if _rs == "none" and not spec.guidance_scale and not _other:
        problems.append("resample=none with guidance_scale=0 and no other mechanism removes BOTH "
                        "transport and selection, so this is the unsteered base model wearing a "
                        "steered arm's label, not an ablation of either. If the arm DOES have a "
                        "mechanism (boltz_perturb, svgd, af3_red), enable it and this check passes.")
    if _rs != "none" and float(getattr(spec, "ess_fraction", 0.5)) <= 0.0:
        problems.append("ess_fraction=%s disables resampling by the back door while resample=%r "
                        "still advertises a schedule; use resample=none, which is declared, and "
                        "leave ess_fraction alone because target_ess_frac tempers against it"
                        % (getattr(spec, "ess_fraction", None), _rs))

    if getattr(spec, "reward", "") == "none":
        _why = []
        if spec.guidance_scale:
            _why.append("guidance_scale=%s (the reward gradient moves particles)" % spec.guidance_scale)
        if float(getattr(spec, "lam", 0.0) or 0.0):
            _why.append("lam=%s (the FK increment reweights by the reward)" % spec.lam)
        if getattr(spec, "lam_adaptive", False):
            _why.append("lam_adaptive=true (lambda is SOLVED from the reward spread)")
        if getattr(spec, "physicality_filter", False):
            _why.append("physicality_filter=true (the mask down-weights particles)")
        if spec.algorithm == "noise_search":
            _why.append("algorithm=%s (selection IS the reward)" % spec.algorithm)
        if _why:
            problems.append("reward=none but %s: the arm declares no objective while its sampling "
                            "reads one. Either give it a real reward or turn these off."
                            % "; ".join(_why))

    # 3. each algorithm must be run as published
    if spec.algorithm == "fk" and spec.guidance_scale:
        problems.append("algorithm=fk with guidance_scale=%s: the FK reference has NO "
                        "gradient term (that is TDS)" % spec.guidance_scale)
    if spec.algorithm == "af3_red" and spec.guidance_scale:
        problems.append("algorithm=af3_red injects a FORCE and uses no reward gradient; "
                        "guidance_scale=%s would add a second steering term" % spec.guidance_scale)
    # The svgd branches here and below guard a STALE algorithm, see the note at
    # DIFFUSION_FAMILIES in method/diffusion_side/diffusion_search.py. They are unreachable for
    # every supported arm and are kept so the preflight does not pass an arm it cannot run.
    if (spec.algorithm == "svgd" and getattr(spec, "svgd_drive_only", False)
            and spec.svgd_repulsion_only):
        problems.append("algorithm=svgd with BOTH svgd_drive_only and svgd_repulsion_only: phi is "
                        "identically zero, so this arm would silently do nothing")
    if spec.algorithm == "svgd" and spec.svgd_repulsion_only and spec.guidance_scale:
        problems.append("algorithm=svgd with svgd_repulsion_only=true ignores the reward; "
                        "guidance_scale=%s has no effect (set svgd_repulsion_only=false to use it)"
                        % spec.guidance_scale)
    if spec.algorithm == "svgd" and not spec.svgd_repulsion_only and not spec.guidance_scale:
        problems.append("algorithm=svgd with the tilt ENABLED but guidance_scale=0: the reward term "
                        "is multiplied by zero, so this is silently identical to repulsion-only")
    from method.diffusion_side.diffusion_search import ALL_ALGORITHMS
    if spec.algorithm not in ALL_ALGORITHMS:
        problems.append("unknown algorithm %r (%s)" % (spec.algorithm, ", ".join(ALL_ALGORITHMS)))

    _win_label = "[%g,%g]" % (spec.resample_start_frac, spec.resample_end_frac)
    _lo, _hi = spec.resample_start_frac, spec.resample_end_frac
    try:
        _parsed = gd.parse_resample_windows(getattr(spec, "resample_windows", ""))
    except ValueError as e:
        _parsed = None
        problems.append("resample_windows: %s" % e)
    if _parsed is not None:
        _lo, _hi = _parsed
        _win_label = str(getattr(spec, "resample_windows", ""))
    ev = [i for i in range(n_steps)
          if i >= spec.schedule_start_frac * n_steps and i % max(1, spec.schedule_every) == 0]
    inwin = [i for i in ev if gd.in_resample_window(i, n_steps, _lo, _hi)]
    if spec.algorithm != "svgd" and not inwin:
        problems.append("no steering event falls inside the resample window %s "
                        "(events at %s): resampling can never fire" % (_win_label, ev[:6]))

    _fb = os.environ.get("FOLDBENCH_ROOT")
    if not _fb:
        problems.append("$FOLDBENCH_ROOT is unset: ground_truth_dir resolves to a nonexistent path "
                        "and DockQ/CA-RMSD are written as NaN WITHOUT any error")
    elif not os.path.isdir(os.path.join(_fb, "ground_truths")):
        problems.append("$FOLDBENCH_ROOT=%s has no ground_truths/ subdirectory: GT metrics will be "
                        "silently NaN" % _fb)

    _nsites, _sitelist = 1, []           # 1 site until the MAP below says otherwise
    if spec.reward in ("epitope", "site_contact"):
        _map = os.environ.get("PROTENIX_EPITOPE_MAP")
        if not _map:
            problems.append("reward=epitope needs $PROTENIX_EPITOPE_MAP; without it the run silently "
                            "falls back to cdr_rmsd (a different CV than the label reports)")
        _src = getattr(spec, "epitope_source", "predicted")
        if _src not in ("ground_truth", "predicted", "synthetic"):
            problems.append("unknown epitope_source %r (ground_truth|predicted|synthetic)" % _src)
        elif _map and os.path.exists(_map):
            try:
                with open(_map) as _f:
                    _md = json.load(_f)
                _stamp, _b = _md.get("_source"), _md.get("_builder")
                for _tk, _tv in _md.items():
                    if str(_tk).startswith("_"):
                        continue
                    if isinstance(_tv, list) and _tv and isinstance(_tv[0], list):
                        _sitelist.append(len(_tv))
                    elif isinstance(_tv, list):
                        _sitelist.append(1)
                _nsites = max(_sitelist) if _sitelist else 1
            except (OSError, ValueError):
                _stamp = _b = None
            if _stamp and _stamp != _src:
                problems.append("epitope_source=%r but %s is stamped _source=%r -- a predicted-epitope "
                                "run would be reported as an oracle one (or vice versa)"
                                % (_src, os.path.basename(_map), _stamp))
            elif _stamp is None and _src == "predicted":
                problems.append("epitope_source=predicted but %s carries no _source stamp; regenerate "
                                "it with `tts_job_lib.py epitope-tokenmap` so provenance is recorded"
                                % os.path.basename(_map))
            if _src in ("predicted", "synthetic"):
                _want = getattr(spec, "epitope_map_builder", "")
                if not _b:
                    problems.append("%s has no _builder stamp, so which construction produced this "
                                    "map cannot be recovered from the results; rebuild it with a "
                                    "current token_maps.py builder" % os.path.basename(_map))
                elif _want and _want != _b:
                    problems.append("epitope_map_builder=%r but %s is stamped _builder=%r"
                                    % (_want, os.path.basename(_map), _b))
                elif _src == "synthetic" and not _want:
                    # A synthetic map with no declared builder is the one case where "no check" is
                    # not acceptable: the two synthetics answer opposite questions.
                    problems.append("epitope_source=synthetic but the spec declares no "
                                    "epitope_map_builder, so a map that CONTAINS the answer (mode3s) "
                                    "cannot be told from one that destroys it (shuffled_sites). "
                                    "Set epitope_map_builder=%r to match %s."
                                    % (_b, os.path.basename(_map)))
    from method.diffusion_side.base_search import RMSD_SUBSETS
    _known = sorted(RMSD_SUBSETS)
    _ksub = int(getattr(spec, "subsample_pool", 1) or 1)
    if _ksub < 1:
        problems.append("subsample_pool must be >= 1 (1 = off)")
    elif _ksub > 1:
        _u = max(len(spec.sweep_list()), 1)
        _s = max(int(_nsites), 1)                     # from the map, not the spec
        _cap = int(n_particles) // (_u * _s)
        if _ksub > max(_cap, 0):
            problems.append("subsample_pool=%d exceeds the per-cell capacity: %d particles over %d "
                            "umbrellas x %d sites is %d per cell, so at most that many subsamples "
                            "can appear in every cell." % (_ksub, int(n_particles), _u, _s, _cap))
        _floor_j = MIN_PER_SUBSAMPLE * _ksub * (_u * _s)
        if int(n_particles) < _floor_j:
            _bad = [x for x in (_sitelist or [_s]) if int(n_particles) < MIN_PER_SUBSAMPLE * _ksub * _u * x]
            _tot = len(_sitelist or [_s])
            _msg = ("subsample_pool=%d needs n_samples >= %d for the WORST target (%d per latent per "
                    "cell over %d umbrellas x %d sites = %d cells); got %d = %.2f per latent per "
                    "cell there. Below 3 the cell fixates by drift rather than selecting, so the "
                    "cross-subsample comparison the joint model exists for does not happen. "
                    "Affects %d of %d targets in the map. Raise n_samples, or lower the umbrellas."
                    % (_ksub, _floor_j, MIN_PER_SUBSAMPLE, _u, _s, _u * _s, int(n_particles),
                       int(n_particles) / float(_ksub * _u * _s), len(_bad), _tot))
            # HARD only if EVERY target is under the floor; otherwise an alert, so one wide-site
            # target cannot block an arm that is correct for the rest of the set.
            problems.append(_msg if len(_bad) >= _tot else "WARN: " + _msg)
        if str(getattr(spec, "reward", "")) != "epitope":
            problems.append("subsample_pool=%d requires reward=epitope (the pure site term); "
                            "reward=%r reads more than the site, so the tilt no longer leaves the "
                            "pose-given-site law invariant under mixed conditioning."
                            % (_ksub, getattr(spec, "reward", "")))

    _resamples = (str(getattr(spec, "algorithm", "")) != "noise_search"
                  and bool(getattr(spec, "reward_affects_sampling", lambda: True)()))
    if _ksub == 1 and _resamples and str(getattr(spec, "reward", "")) in ("epitope", "site_contact"):
        _u1 = max(len(spec.sweep_list()), 1)
        _s1 = max(int(_nsites), 1)                    # from the map, not the spec
        _floor_s = MIN_PER_CELL * _u1 * _s1
        if int(n_particles) < _floor_s:
            _bad1 = [x for x in (_sitelist or [_s1]) if int(n_particles) < MIN_PER_CELL * _u1 * x]
            _tot1 = len(_sitelist or [_s1])
            _msg1 = ("separate model needs n_samples >= %d for the WORST target (%d per cell over %d "
                     "umbrellas x %d sites = %d cells); got %d = %.2f per cell there. Below 6 the "
                     "within-cell resample returns near-identity, so the umbrella/site sweep stops "
                     "selecting and the arm is steering in name only. Affects %d of %d targets in "
                     "the map. Raise n_samples, or lower the umbrellas."
                     % (_floor_s, MIN_PER_CELL, _u1, _s1, _u1 * _s1, int(n_particles),
                        int(n_particles) / float(_u1 * _s1), len(_bad1), _tot1))
            problems.append(_msg1 if len(_bad1) >= _tot1 else "WARN: " + _msg1)

    if str(getattr(spec, "reward", "")) not in ALL_REWARDS:
        problems.append("unknown reward %r (have: %s)"
                        % (getattr(spec, "reward", ""), "|".join(ALL_REWARDS)))
    from method.diffusion_side.guidance import check_route
    for _rname in ("site_route", "conf_route"):
        try:
            check_route(_rname, getattr(spec, _rname, "coord"))
        except ValueError as _e:
            problems.append(str(_e))
    # ONE predicate for "is any route latent", used by both checks below. Spelling it twice invited
    # the two copies to drift, which is how a guard ends up firing on one route and not the other.
    _any_latent = any(str(getattr(spec, _r, "coord")).lower() == "latent"
                      for _r in ("site_route", "conf_route"))
    if _any_latent and float(getattr(spec, "latent_eta", 0.0)) <= 0.0:
        problems.append("a latent guidance route is selected but latent_eta<=0, so the latent would "
                        "never move and the arm would run as an unsteered control under a steered label")
    if str(getattr(spec, "conf_route", "coord")).lower() == "latent" \
            and int(getattr(spec, "max_diffusion_chunk", 0) or 0) == 0:
        problems.append("conf_route=latent needs max_diffusion_chunk set explicitly: enabling "
                        "autograd on the denoiser retains its activation stack, and the automatic "
                        "chunk size is derived from a no-grad memory model that no longer applies.")

    for _name, _dflt in (("rmsd_subset", "paratope"), ("red_rmsd_subset", "all")):
        _sub = getattr(spec, _name, _dflt)
        if _sub not in RMSD_SUBSETS:
            problems.append("unknown %s %r (have: %s)" % (_name, _sub, "|".join(_known)))
        elif _sub == "chain" and int(getattr(spec, "rmsd_chain", -1)) < 0:
            problems.append("%s=chain needs rmsd_chain>=0; -1 is the unset sentinel and "
                            "selects no residues, so the RMSD would be taken over an empty subset"
                            % _name)

    _cmap = os.environ.get("PROTENIX_CDR_MAP", "")
    if _cmap and os.path.exists(_cmap):
        try:
            with open(_cmap) as _f:
                _cm = json.load(_f)
        except (OSError, ValueError):
            _cm = {}
        if not _cm.get("_source"):
            problems.append("%s carries no _source stamp, so an oracle-derived paratope map "
                            "cannot be told from an IMGT one; rebuild it with `token_maps.py "
                            "cdr-tokenmap`" % os.path.basename(_cmap))
        elif _cm.get("_source") != "predicted":
            problems.append("PROTENIX_CDR_MAP %s is stamped _source=%r -- a paratope map must "
                            "not come from the native complex"
                            % (os.path.basename(_cmap), _cm.get("_source")))

        u = getattr(spec, "sweep_units", "auto")
        if u not in ("auto", "rmsd_A", "centroid_A"):
            problems.append("unknown sweep_units %r (auto|rmsd_A|centroid_A)" % u)
        if u == "rmsd_A":
            problems.append("reward=epitope with sweep_units=rmsd_A: this CV is a CENTROID DISTANCE, "
                            "not a conformational RMSD -- the d* values mean a different thing")
        sw = spec.sweep_list()
        if sw and max(sw) < 5.0:
            problems.append("reward=epitope d* values %s are centroid DISTANCES in Angstrom; a max of "
                            "%.1f A is inside one binding site, so the umbrellas cannot separate sites"
                            % (sw, max(sw)))

    if spec.algorithm == "fk" and len(spec.sweep_list()) > 1:
        problems.append("algorithm=fk with umbrella_sweep=%r: FK selects rather than "
                        "transports, so MULTIPLE umbrellas cannot spread a pool. Use a single d* "
                        "(e.g. umbrella_sweep=12), which is a plain scalar reward."
                        % spec.umbrella_sweep)

    if spec.reward == "cdr_rmsd" and getattr(spec, "sweep_units", "auto") == "centroid_A":
        problems.append("reward=cdr_rmsd with sweep_units=centroid_A: this CV is a conformational "
                        "RMSD, not a centroid distance")

    if spec.algorithm == "svgd":
        n_upd = sum(1 for i in range(n_steps) if i >= spec.schedule_start_frac * n_steps)
        moved = n_upd * spec.svgd_stepsize            # AdaGrad normalises |step| to ~stepsize
        if getattr(spec, "svgd_site_kernel", False):
            from method.diffusion_side.base_search import anneal_scale
            moved = float(spec.site_step_a) * sum(
                anneal_scale(1.0 - (i / max(n_steps - 1, 1)),
                             spec.site_anneal_tau1, spec.site_anneal_tau2)
                for i in range(n_steps) if i >= spec.schedule_start_frac * n_steps)
        if moved < 0.05:
            problems.append("svgd would move ~%.4f A total (%d updates x stepsize %g): negligible "
                            "against a ~1.8 A signal, i.e. a NO-OP"
                            % (moved, n_upd, spec.svgd_stepsize))
        _rigid = (getattr(spec, "reward", "") in ("epitope", "site_contact")
                  or getattr(spec, "svgd_site_kernel", False))
        _cap = 60.0 if _rigid else 2.0
        if moved > _cap:
            problems.append("svgd would move ~%.2f A total (%d updates x stepsize %g), over the %.0f A "
                            "cap for reward=%r: large enough to tear structures apart"
                            % (moved, n_upd, spec.svgd_stepsize, _cap, getattr(spec, "reward", "")))

    if spec.algorithm == "af3_red":
        if spec.red_weight <= 0:
            problems.append("af3_red with red_weight=%g applies no bias at all" % spec.red_weight)
        if spec.red_sigma <= 0:
            problems.append("af3_red needs red_sigma > 0 (it is the Gaussian width floor)")
        if spec.red_n_smooth < 1:
            problems.append("af3_red with red_n_smooth=%d: the reference is explicit that UNSMOOTHED "
                            "repulsion does not work (no covalent constraints in the denoiser)"
                            % spec.red_n_smooth)
        if spec.guidance_scale:
            problems.append("af3_red is a repulsive FORCE, not a reward gradient: guidance_scale=%s "
                            "would add a second, different steering term" % spec.guidance_scale)

    # 5. simulate the weight/resample recursion on synthetic rewards of a realistic spread. This is
    #    the check that would have caught resamples=0 without spending a GPU-hour to discover it.
    if spec.algorithm in ("fk", "tds", "cast") and not problems:
        rng, logw, fired = np.random.default_rng(0), np.zeros(n_particles), 0
        for i in inwin:
            r_prev, r_cur = np.zeros(n_particles), rng.normal(0, 2.0, n_particles)
            lam = (gd.lambda_for_target_ess(r_prev, r_cur, spec.potential, n_particles,
                                            spec.target_ess_frac, logw_prev=logw)
                   if spec.lam_adaptive else spec.lam)
            logw = logw + gd.fk_log_increment(spec.potential, r_prev, r_cur, lam)
            e = gd.ess(gd.normalize_log_weights(logw))
            if e < spec.ess_fraction * n_particles:
                fired += 1
                logw = np.zeros(n_particles)
        if fired == 0:
            problems.append("simulated %d events and resampling NEVER fired: this arm is a no-op"
                            % len(inwin))
        elif fired == len(inwin) and spec.lam_adaptive is False and spec.lam > 2:
            problems.append("resampling fires at EVERY event with lam=%g: likely degenerate weights "
                            "(clone-the-winner)" % spec.lam)
    return problems


_HEALTH_PATTERNS = {
    # \b is not usable here: the tokens must NOT match inside INFO / inference / infer_chunk_size,
    # which a naive "nan|inf" search matches dozens of times per log.
    "naninf": r"(^|[^A-Za-z0-9_])([Nn]a[Nn]|[-+]?[Ii][Nn][Ff]([Ii][Nn][Ii][Tt][Yy])?)([^A-Za-z0-9_]|$)",
    "errors": r"Traceback|guidance skipped|reward unavailable|falling back to ipSAE",
    "guidance": r"guidance step",
    "resamples": r"RESAMPLE",
    "saturated": r"saturated=True",
}


def run_health(log_path, pred_dir, expect_cifs=None, expect_guidance=None, res_dir=None,
               require_scored=False):
    """Screen one run. Returns (line, problems): a one-line summary and a list of defects."""
    text = ""
    if os.path.exists(log_path):
        with open(log_path, errors="replace") as fh:
            text = fh.read()
    counts = {k: len(re.findall(p, text, flags=re.MULTILINE)) for k, p in _HEALTH_PATTERNS.items()}
    n_cif = len(glob.glob(os.path.join(pred_dir, "**", "*_sample_*.cif"), recursive=True))

    problems = []
    notes = []
    if expect_cifs is not None and n_cif < expect_cifs:
        problems.append("only %d/%d structures written" % (n_cif, expect_cifs))
    if counts["errors"]:
        problems.append("%d error/fallback line(s)" % counts["errors"])
    if counts["naninf"]:
        problems.append("%d non-finite value(s)" % counts["naninf"])
    # expect_guidance False means a control arm: guidance firing there would invalidate the pairing.
    if expect_guidance is True and counts["guidance"] == 0:
        problems.append("guidance never fired (silent no-op)")
    if res_dir:
        _pq = os.path.join(res_dir, "per_sample.parquet")
        if not os.path.exists(_pq):
            (problems if require_scored else notes).append(
                "no per_sample.parquet yet: nothing has scored this unit")
        else:
            try:
                import pandas as _pd
                _d = _pd.read_parquet(_pq)
                _c = "dockq_abag" if "dockq_abag" in _d.columns else ("dockq" if "dockq" in _d.columns else None)
                if _c is None:
                    (problems if require_scored else notes).append(
                        "per_sample.parquet has no dockq column yet")
                elif int(_d[_c].notna().sum()) == 0:
                    (problems if require_scored else notes).append(
                        "dockq is all-NaN over %d rows: %s" % (
                            len(_d),
                            "scoring RAN AND FAILED (check GT_INDEX and the score venv), the "
                            "structures themselves are fine" if require_scored
                            else "not scored yet, which is expected before score_unit runs"))
                else:
                    notes.append("scored: %d/%d rows carry %s" % (int(_d[_c].notna().sum()), len(_d), _c))
            except Exception as _e:                              # noqa: BLE001 - never break the gate
                problems.append("could not read per_sample.parquet (%s)" % type(_e).__name__)
    if expect_guidance is False and counts["guidance"] > 0:
        problems.append("guidance fired in a control arm")

    line = ("cifs=%d guidance=%d resamples=%d clip_saturated=%d errors=%d naninf=%d"
            % (n_cif, counts["guidance"], counts["resamples"], counts["saturated"],
               counts["errors"], counts["naninf"]))
    return line, problems, notes


def _ks_2samp(a, b):
    """Two-sample Kolmogorov-Smirnov (D, p) on 1-D samples, without scipy."""
    import numpy as np

    x = np.sort(np.asarray(a, dtype=float))
    y = np.sort(np.asarray(b, dtype=float))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        return float("nan"), float("nan")
    allv = np.concatenate([x, y])
    d = float(np.max(np.abs(np.searchsorted(x, allv, "right") / n
                            - np.searchsorted(y, allv, "right") / m)))
    en = np.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d              # Stephens' small-sample correction
    k = np.arange(1, 101)
    p = float(min(1.0, max(0.0, 2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * k ** 2 * lam ** 2)))))
    return d, p


def ks_compare(run_dir, column="dockq", ref="bon"):
    """Compare each arm's per-sample distribution against ``ref``. Returns (rows, verdicts)."""
    import glob as _glob

    import numpy as np
    import pandas as pd
    from results_io import pool_rows as _pool_rows

    rows = {}
    for pq in sorted(_glob.glob(os.path.join(run_dir, "*", "res", "per_sample.parquet"))):
        arm = os.path.basename(os.path.dirname(os.path.dirname(pq))).split("__")[-1]
        d = pd.read_parquet(pq)
        # Compare arms over DRAWS OF THE METHOD. Without this the joint arms carry K - 1 extra
        # unsteered trunk rows and are ranked against a different population than the controls.
        d = _pool_rows(d)
        if column in d:
            rows[arm] = d[column].to_numpy(dtype=float)
    verdicts = []
    if ref not in rows:
        return rows, ["reference arm %r not found (have: %s)" % (ref, ", ".join(sorted(rows)) or "none")]
    base = rows[ref]
    for arm in sorted(rows):
        v = rows[arm]
        fin = v[np.isfinite(v)]
        line = "%-9s n=%3d mean=%.4f median=%.4f max=%.4f" % (
            arm, len(fin), np.mean(fin) if len(fin) else float("nan"),
            np.median(fin) if len(fin) else float("nan"), np.max(fin) if len(fin) else float("nan"))
        if arm != ref:
            dd, pp = _ks_2samp(base, v)
            line += "   KS vs %s: D=%.3f p=%.3f  %s" % (
                ref, dd, pp, "DIFFERENT" if pp < 0.05 else "no detectable difference")
        verdicts.append(line)
    return rows, verdicts


def pool_score(run_dir, score_py=None, dockq_bin=None, jobs=6, threads=6, once=False,
               poll_s=60, code_tree=None):
    """Score finished pool units from OUTSIDE the GPU worker, so the card never waits on DockQ."""
    import subprocess
    import time
    from concurrent.futures import ThreadPoolExecutor
    score_py = score_py or os.path.join(os.environ.get("TTS_ROOT", "/data/tts"),
                                        "venv_score", "bin", "python")
    code = code_tree or os.environ.get("CODE", os.getcwd())
    done = os.path.join(run_dir, "queue", "done")

    def _unscored(arm_dir):
        pq = os.path.join(arm_dir, "res", "per_sample.parquet")
        if not os.path.isfile(pq):
            if os.path.isdir(os.path.join(arm_dir, "pred")):
                print("!! %s has predictions but NO per_sample.parquet: the verifier stage never ran "
                      "or its output was removed. --from-per-sample cannot rebuild it."
                      % os.path.basename(arm_dir), flush=True)
                return False
            return False
        try:
            import pandas as pd
            d = pd.read_parquet(pq)
        except Exception:
            return True
        return not ("dockq" in d.columns and d["dockq"].notna().all())

    tgt_index = None
    _here = os.path.abspath(run_dir)
    for _ in range(3):
        _c = os.path.join(_here, "target_index.csv")
        if os.path.isfile(_c):
            tgt_index = _c
            break
        _here = os.path.dirname(_here)
    if tgt_index is None:
        raise FileNotFoundError("no target_index.csv at or above %s" % run_dir)

    def _score(unit):
        arm = os.path.join(run_dir, unit)
        cmd = [score_py, "-m", "experiments.scripts.evaluate", "--from-per-sample",
               "--results-dir", os.path.join(arm, "res"), "--backend", "protenix2",
               "--target-index", tgt_index,
               "--metrics", "dockq", "--threads", str(threads)]
        if dockq_bin:
            cmd += ["--dockq-bin", dockq_bin]
        _pp = os.environ.get("PYTHONPATH", "")
        env = dict(os.environ, PYTHONPATH=(code + os.pathsep + _pp) if _pp else code)
        r = subprocess.run(cmd, cwd=code, env=env, capture_output=True, text=True, timeout=7200)
        tail = ""
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "").strip().split("\n")
            tail = " | " + " / ".join(x.strip() for x in err[-3:] if x.strip())
        return unit, r.returncode, tail

    hb = os.path.join(run_dir, "queue", ".pool_score_hb")

    def _beat():
        """Stamp the heartbeat. Called per UNIT, not per cycle: a cycle blocks on up to `jobs` units
        with a 7200 s subprocess timeout, so a once-per-cycle beat could go hours between writes and
        a perfectly healthy scorer would trip the worker's 900 s staleness check."""
        try:
            with open(hb + ".tmp", "w") as f:
                f.write(str(int(time.time())))
            os.replace(hb + ".tmp", hb)
        except OSError:
            pass

    # ONCE, not per beat: _beat runs per unit and queue/ is already present here (the loop below
    # lists queue/done), so re-creating the directory on every stamp was pure syscall churn.
    try:
        os.makedirs(os.path.dirname(hb), exist_ok=True)
    except OSError:
        pass

    seen = set()
    while True:
        _beat()
        units = [u for u in sorted(os.listdir(done)) if os.path.isdir(os.path.join(run_dir, u))] \
            if os.path.isdir(done) else []
        todo = [u for u in units if u not in seen and _unscored(os.path.join(run_dir, u))]
        if todo:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                for unit, rc, why in ex.map(_score, todo):
                    _beat()
                    if rc == 0:
                        seen.add(unit)
                    print("  [%s] %s%s" % ("scored" if rc == 0 else "FAILED rc=%d" % rc, unit, why),
                          flush=True)
        else:
            try:
                os.remove(os.path.join(run_dir, "NEEDS_EXTERNAL_SCORING"))
            except OSError:
                pass
            if once:
                print("  nothing unscored", flush=True)
        if once:
            return 0
        time.sleep(poll_s)



def pool_stats(tokenmap, target, pred_dir, dstars=()):
    """Pool diversity on BOTH axes for one predictions dir. Returns the report line."""
    from method.verifier.cif_io import load_ca_coords_chains
    from method.verifier.diversity import (CONTACT_R, binding_diversity,
                                           cdr_conformation_diversity, epitope_mask, kabsch_rmsd)
    from method.verifier.consensus import _superpose

    # EXACTLY the split used everywhere else. Adding "A" here would read the antigen as antibody and silently
    # make both site axes meaningless.
    ab_chains = {"H", "L"}
    para = np.asarray(json.load(open(tokenmap))[target], dtype=int)
    cifs = sorted(glob.glob(pred_dir + "/**/*_sample_*.cif", recursive=True))

    def antigen_framed(paths):
        """Antibody CA in a common antigen frame, plus the reference antigen CA."""
        ref_chains = ab_mask = ag_mask = ref_ag = None
        ab = []
        for c in paths:
            co, ch, _ = load_ca_coords_chains(c)
            if co is None or ch is None:
                continue
            ch = np.asarray(ch)
            if ref_chains is None:
                ref_chains = ch
                ab_mask = np.array([x in ab_chains for x in ref_chains])
                ag_mask = ~ab_mask
                if not ab_mask.any() or not ag_mask.any():
                    return None, None
            if len(ch) != len(ref_chains) or not np.array_equal(ch, ref_chains):
                continue
            x = np.asarray(co, float)
            if ref_ag is None:
                ref_ag = x[ag_mask]
            ab.append(_superpose(x[ag_mask], ref_ag, x[ab_mask]))
        return (np.stack(ab) if ab else None), ref_ag

    def jaccard_site_diversity(ab_pool, ref_ag):
        masks = [epitope_mask(a, ref_ag, CONTACT_R) for a in ab_pool]
        d = []
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                u = (masks[i] | masks[j]).sum()
                d.append(1.0 - (masks[i] & masks[j]).sum() / u if u else 0.0)
        return float(np.median(d)) if d else float("nan")

    co = [np.asarray(load_ca_coords_chains(c)[0], float) for c in cifs]
    co = [x for x in co if x is not None and x.ndim == 2 and x.shape[0] > int(para.max())]
    if len(co) < 2:
        return "div_conf=None n=%d" % len(co)

    a_full = np.stack(co)
    div_conf = cdr_conformation_diversity(a_full, para)

    ab_pool, ref_ag = antigen_framed(cifs)
    if ab_pool is not None and ref_ag is not None and len(ab_pool) >= 2:
        site = "div_site=%.4f div_jac=%.4f" % (binding_diversity(ab_pool, ref_ag, CONTACT_R),
                                               jaccard_site_diversity(ab_pool, ref_ag))
    else:
        # no antibody/antigen split (single-chain input, or chain ids outside ab_chains)
        site = "div_site=NA div_jac=NA"

    # Score against the anchor the objective actually optimised, when one was saved.
    anchor, anchor_src = None, "first pool member (NO anchor saved)"
    hits = glob.glob(os.path.join(pred_dir, "**", "cdr_anchor_*.npy"), recursive=True) \
        + glob.glob(os.path.join(os.path.dirname(pred_dir.rstrip("/")), "cdr_anchor_*.npy"))
    if hits:
        ref_reps = np.load(sorted(hits)[0])
        if ref_reps.ndim == 2 and ref_reps.shape[0] > int(para.max()):
            anchor, anchor_src = ref_reps[para], "STEERING ANCHOR " + os.path.basename(sorted(hits)[0])
    if anchor is None:
        anchor = a_full[:, para, :][0]

    d = np.array([kabsch_rmsd(a, anchor) for a in a_full[:, para, :]])
    out = ("div_conf=%.4f %s n=%d | rmsd_to_anchor: min=%.2f med=%.2f max=%.2f spread=%.2f  [%s]"
           % (div_conf, site, len(co), d.min(), np.median(d), d.max(), d.max() - d.min(), anchor_src))
    if dstars:
        assigned = [dstars[i % len(dstars)] for i in range(len(co))]
        pairs = " ".join("%.0f->%.2f" % (a, r) for a, r in zip(assigned, d))
        out += ("\n   d*->achieved: %s | mean|achieved-d*|=%.2f A"
                % (pairs, float(np.mean(np.abs(d - assigned)))))
    return out



def _centroid_pair_distances(cdr_json, epi_json, target, pred_dir, mode):
    """Per-structure distance over a prediction directory, for whichever pair `mode` names."""
    from method.verifier.cif_io import load_ca_coords_chains
    from method.metrics.site import site_distance_a   # GT-based: a METRIC, not a verifier signal

    pi = np.asarray(json.load(open(cdr_json))[target], int)
    ei = np.asarray(json.load(open(epi_json))[target], int)
    need = max(int(pi.max()), int(ei.max()))
    out = []
    for c in sorted(glob.glob(pred_dir + "/**/*_sample_*.cif", recursive=True)):
        x, chains, _ = load_ca_coords_chains(c)
        if x is None:
            continue
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[0] <= need:
            continue
        if mode == "contact":
            out.append(float(np.linalg.norm(x[pi].mean(0) - x[ei].mean(0))))
        elif mode == "site":
            v = site_distance_a(x, chains, ei)
            if v is None:
                continue
            out.append(v)
        else:
            raise ValueError("mode must be 'contact' or 'site', got %r" % (mode,))
    return np.asarray(out, dtype=float)


def site_distance(cdr_json, epi_json, target, pred_dir, reach_a=6.0):
    """RIGHT-SITE measure: predicted-epitope centroid to TRUE-epitope centroid, both on the antigen."""
    d = _centroid_pair_distances(cdr_json, epi_json, target, pred_dir, "site")
    if d.size == 0:
        return "site_d=None n=0"
    reach = int((d < float(reach_a)).sum())
    return ("site_d: min=%.2f med=%.2f max=%.2f mean=%.2f n=%d reach<%.1fA=%d (%.1f%%)"
            % (d.min(), np.median(d), d.max(), d.mean(), d.size,
               float(reach_a), reach, 100.0 * reach / d.size))




# MOVED to method/metrics/site.py so the scoring path can use it without importing the job layer.
# Re-exported here so every existing caller keeps working and there is only one implementation.
from method.metrics.site import native_ca_by_chain_order  # noqa: E402,F401


def epitope_distance(cdr_json, epi_json, target, pred_dir, native=None, reach_a=6.0):
    """CONTACT distance (ORACLE readout): paratope centroid to TRUE epitope centroid."""
    from method.verifier.cif_io import load_ca_coords_chains

    pi = np.asarray(json.load(open(cdr_json))[target], int)
    ei = np.asarray(json.load(open(epi_json))[target], int)
    need = max(int(pi.max()), int(ei.max()))
    d = []
    for c in sorted(glob.glob(pred_dir + "/**/*_sample_*.cif", recursive=True)):
        x, _, _ = load_ca_coords_chains(c)
        if x is None:
            continue
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[0] <= need:
            continue
        d.append(float(np.linalg.norm(x[pi].mean(0) - x[ei].mean(0))))
    if not d:
        return "epi_d=None n=0"
    d = np.asarray(d)
    nat = "" if native is None else " native=%.2f" % native
    reach = int((d < float(reach_a)).sum())
    return ("epi_d: min=%.2f med=%.2f max=%.2f mean=%.2f n=%d reach<%.1fA=%d (%.1f%%)%s"
            % (d.min(), np.median(d), d.max(), d.mean(), len(d),
               float(reach_a), reach, 100.0 * reach / len(d), nat))


# ------------------------------------------------------------------ CLI
def _structure_count(pred_dir) -> int:
    """One definition of "a finished structure", shared with results_io.structure_count."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from results_io import structure_count
        return structure_count(pred_dir)
    except Exception:                                    # noqa: BLE001
        import glob as _g
        return len(_g.glob(os.path.join(str(pred_dir), "**", "*_sample_*.cif"), recursive=True))


def compute_rederive(unit_dirs, dry_run=False):
    """Rewrite a unit's `diffusion_compute.json` from the timing records that match its structures."""
    import glob as _glob
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.base import estimate_diffusion_search_flops, flops_totals
    rc = 0
    for u in unit_dirs:
        f = os.path.join(u, "res", "diffusion_compute.json")
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            old = json.load(fh)
        pred = os.path.join(u, "pred")
        n_cif = _structure_count(pred)
        logs = sorted(_glob.glob(os.path.join(pred, "**", "diffusion_timing.jsonl"), recursive=True),
                      key=os.path.getmtime, reverse=True)
        per = []
        for jl in logs:
            rs = [json.loads(ln) for ln in open(jl) if ln.strip()]
            per.append((jl, rs, sum(int(r.get("n_structures_emitted") or 0) for r in rs)))
        if len(per) < 2 or not n_cif:
            print("[rederive] %s: nothing to do (%d stage dir(s))" % (os.path.basename(u), len(per)))
            continue
        kept, got = [], 0
        for jl, rs, n in per:
            if got >= n_cif:
                print("   dropping %s (%d rec, %d structures) -- a previous attempt"
                      % (os.path.basename(os.path.dirname(jl)), len(rs), n))
                continue
            kept += rs
            got += n
        agg = estimate_diffusion_search_flops(kept, n_cycle=10,
                                              trunk_flops=old.get("flops_trunk"),
                                              trunk_nfe=old.get("trunk_nfe"))
        new = dict(old)
        new.update(agg)
        new.update(flops_totals(new["flops_trunk"], new["flops_denoiser"],
                                new["flops_confidence"], new["flops_confidence_output"]))
        new["n_targets"] = len(kept)
        new["rederived"] = ("attempt-aware selection applied %s; the previous record summed %d "
                            "stage dir(s)" % (_dt.datetime.utcnow().strftime("%Y-%m-%d"), len(per)))
        print("[rederive] %-42s flops_total %.4e -> %.4e (%.3fx)  conf_out %s -> %s"
              % (os.path.basename(u), old["flops_total"], new["flops_total"],
                 new["flops_total"] / max(old["flops_total"], 1e-30),
                 old.get("confidence_output_nfe"), new.get("confidence_output_nfe")))
        if dry_run:
            continue
        os.replace(f, f + ".pre-rederive")
        with open(f, "w") as fh:
            json.dump(new, fh, indent=2)
        rc = rc or 0
    return rc


def compute_check(unit_dirs, quiet=False):
    """Validate each unit's `diffusion_compute.json` against the structures it actually produced."""
    import glob as _glob
    bad = tot = 0
    for u in unit_dirs:
        f = os.path.join(u, "res", "diffusion_compute.json")
        if not os.path.exists(f):
            continue
        try:
            with open(f) as fh:
                j = json.load(fh)
        except (OSError, ValueError):
            print("!! %s: diffusion_compute.json unreadable" % os.path.basename(u))
            bad += 1
            continue
        n_cif = _structure_count(os.path.join(u, "pred"))
        if not n_cif:
            continue
        tot += 1
        co = int(j.get("confidence_output_nfe") or 0)
        if co and co != n_cif:
            stages = len(_glob.glob(os.path.join(u, "pred", "_stage_*")))
            print("!! %s: confidence_output_nfe=%d but %d structures on disk (%.2fx) -- %d _stage_ "
                  "dir(s), so this record sums %s. M4 for this unit is NOT usable."
                  % (os.path.basename(u), co, n_cif, co / n_cif, stages,
                     "more than one attempt" if stages > 1 else "an inconsistent set of calls"))
            bad += 1
    if not quiet:
        print("[compute-check] %d of %d units with a compute record are inconsistent" % (bad, tot))
    return 2 if bad else 0


def oom_class(log_path, memguard_flag=None, contend_frac=0.25):
    """Classify a failed unit's OOM. Prints one WORD, then a reason line."""
    paths = [log_path] if isinstance(log_path, str) else list(log_path)
    chunks = []
    for q in paths:
        try:
            with open(q, errors="ignore") as f:
                chunks.append(f.read())
        except OSError:
            continue
    if not chunks:
        print("none")
        print("  no readable log among: %s" % ", ".join(os.path.basename(q) for q in paths))
        return 0
    log = "\n".join(chunks)
    # MEMGUARD FIRST. A host-RAM kill leaves a CUDA OOM in the log too (the child dies mid-allocation),
    # so testing the GPU pattern first would misread every RAM kill as a GPU one.
    if memguard_flag and os.path.exists(memguard_flag):
        print("ram")
        print("  the host memory guard fired; offload RAISES host RAM, so do not escalate")
        return 0
    # ORDER MATTERS THE OTHER WAY TOO: `torch.OutOfMemoryError` CONTAINS `MemoryError`, so a host test
    if "torch.OutOfMemoryError" in log or "CUDA out of memory" in log:
        by_pid = {}
        for pid, gib in re.findall(r"Process (\d+) has ([\d.]+) GiB memory in use", log):
            by_pid[pid] = max(by_pid.get(pid, 0.0), float(gib))
        named = list(by_pid.values())
        cap = re.search(r"total capacity of ([\d.]+) GiB", log)
        total = float(cap.group(1)) if cap else 0.0
        held = sum(named)
        own = re.findall(r"Of the allocated memory ([\d.]+) GiB is allocated by PyTorch", log)
        own_gib = max((float(x) for x in own), default=0.0)
        held = min(held, total) if total > 0 else held
        others_held = max(held - own_gib, 0.0)
        if named and total > 0 and others_held >= float(contend_frac) * total:
            print("gpu_contention")
            print("  %d process(es) on the card held %.1f GiB, of which %.1f GiB is ours, leaving "
                  "%.1f of %.1f GiB (%.0f%%) to neighbours -- this unit did not fit because the CARD "
                  "WAS OCCUPIED, not because it is too big. Do NOT force offload; give it a free GPU. "
                  "Check that every worker on a multi-GPU pod has its own CUDA_VISIBLE_DEVICES."
                  % (len(named), held, own_gib, others_held, total, 100.0 * others_held / total))
            return 0
        print("gpu_size")
        print("  CUDA OOM with no significant neighbour on the card (%d process(es) named, %.1f GiB "
              "held, %.1f GiB of it ours -> %.1f GiB from neighbours)"
              % (len(named), held, own_gib, others_held))
        return 0
    if re.search(r"Cannot allocate memory|Killed process|MemoryError", log):
        print("ram")
        print("  host allocation failure; offload RAISES host RAM, so do not escalate")
        return 0
    print("none")
    print("  no OOM signature in %s" % os.path.basename(log_path))
    return 0


def map_stamp(path, source=None, builder=None):
    """Assert a token map carries the provenance stamps a caller expects. 0 if it does, 2 if not."""
    try:
        with open(path) as f:
            m = json.load(f)
    except (OSError, ValueError) as e:
        print("!! %s could not be read as a token map (%s)" % (path, e))
        return 2
    bad = []
    if source is not None and m.get("_source") != source:
        bad.append("_source=%r, expected %r" % (m.get("_source"), source))
    if builder is not None and m.get("_builder") != builder:
        bad.append("_builder=%r, expected %r" % (m.get("_builder"), builder))
    n = len([k for k in m if not str(k).startswith("_")])
    if bad:
        print("!! %s is stamped %s" % (os.path.basename(path), "; ".join(bad)))
        return 2
    extra = " ".join("%s=%s" % (k, m[k]) for k in sorted(m) if str(k).startswith("_")
                     and k not in ("_source", "_builder", "_note"))
    print("[map-stamp] %s OK: _source=%s _builder=%s over %d targets%s"
          % (os.path.basename(path), m.get("_source"), m.get("_builder"), n,
             (" (" + extra + ")") if extra else ""))
    return 0


def checkpoint_route(code_tree: str = None, require: str = None) -> int:
    """Say WHICH checkpoint a run started here would load, before it spends a GPU second."""
    if code_tree:
        sys.path.insert(0, code_tree)
    try:
        from models.protenix_provision import resolve_model_source
    except Exception as exc:                                     # noqa: BLE001
        print("[route] cannot import the resolver (%s); is CODE on sys.path?" % exc)
        return 2
    private_root = os.environ.get("PROTENIX_ROOT_DIR")
    pub_root = os.environ.get("PROTENIX_PUBLIC_ROOT_DIR")
    m = resolve_model_source({"checkpoint": {"private_root_dir": private_root, "public_root_dir": pub_root}})
    prov = m.details.get("provenance", "n/a")
    print("[route] source=%s model=%s root=%s provenance=%s" %
          (m.source, m.model_name, m.root_dir or "<pkg default>", prov))
    print("[route] reason: %s" % m.reason)
    if m.source == "private" and prov == "found-on-disk":
        print("[route] NOTE: this is 'a checkpoint was found at the root', not a verified digest. "
              "Whatever sits at that path is what the run loads.")
    if m.details.get("same_as_public_root"):
        print("[route] NOTE: the private and public roots are the SAME path, so the two branches "
              "cannot be told apart from the record.")
    if require and m.source != require:
        print("[route] !! required source=%s but resolved %s" % (require, m.source))
        return 1
    return 0


# ------------------------------------------------------------------ dataset consistency gate
def dataset_check(targets, inputs_wired=None, gt_index=None, foldbench_root=None, maps=(),
                  msa_db=None):
    """Assert one dataset. Returns a list of problems, empty when every artefact covers every target."""
    probs = []
    targets = [t for t in targets if t]
    if not targets:
        return ["no targets given"]

    if inputs_wired:
        missing = [t for t in targets if not os.path.isfile(os.path.join(inputs_wired, t + ".json"))]
        if missing:
            probs.append("INPUTS_WIRED=%s has no input for %d target(s): %s"
                         % (inputs_wired, len(missing), missing[:6]))
        # The NAME INSIDE the json is what the map hooks key on, not the filename. A renamed dataset
        # whose files were moved but whose contents were not is the failure this catches.
        bad = []
        for t in targets:
            f = os.path.join(inputs_wired, t + ".json")
            if not os.path.isfile(f):
                continue
            try:
                o = json.load(open(f))
            except Exception:                                       # noqa: BLE001
                probs.append("INPUTS_WIRED/%s.json is not readable json" % t)
                continue
            o = o[0] if isinstance(o, list) else o
            if o.get("name") != t:
                bad.append("%s(name=%s)" % (t, o.get("name")))
            if not any((c.get("proteinChain") or {}).get("unpairedMsaPath")
                       for c in o.get("sequences", [])):
                probs.append("INPUTS_WIRED/%s.json carries no MSA path: it was never wired" % t)
        if bad:
            probs.append("%d wired input(s) whose internal name is not the target id, so every map "
                         "lookup will miss: %s" % (len(bad), bad[:6]))

    if gt_index:
        if not os.path.isfile(gt_index):
            probs.append("GT_INDEX does not exist: %s" % gt_index)
        else:
            import csv as _csv
            rows = {r["target_id"]: r for r in _csv.DictReader(open(gt_index))}
            missing = [t for t in targets if t not in rows]
            if missing:
                probs.append("GT_INDEX=%s has no row for %d target(s): %s"
                             % (gt_index, len(missing), missing[:6]))
            gone = [t for t in targets if t in rows and rows[t].get("ground_truth_cif")
                    and not os.path.isfile(rows[t]["ground_truth_cif"])]
            if gone:
                probs.append("GT_INDEX names a reference that does not exist for %d target(s): %s"
                             % (len(gone), gone[:6]))

    if foldbench_root:
        gts = os.path.join(foldbench_root, "ground_truths")
        if not os.path.isdir(gts):
            probs.append("FOLDBENCH_ROOT has no ground_truths/ subdirectory: %s" % gts)
        else:
            # Both layouts the readers accept: `<target>/` is FoldBench's nested form and
            # `<target>.<ext>` is what data/datasets/foldbench.py writes.
            def _has_ref(t):
                base = os.path.join(gts, t)
                return os.path.isdir(base) or any(os.path.isfile(base + e)
                                                  for e in (".cif", ".mmcif", ".pdb"))
            missing = [t for t in targets if not _has_ref(t)]
            if missing:
                probs.append("%s holds no reference for %d target(s): %s"
                             % (gts, len(missing), missing[:6]))

    for name, path in maps:
        if not os.path.isfile(path):
            probs.append("%s does not exist: %s" % (name, path))
            continue
        try:
            m = json.load(open(path))
        except Exception:                                           # noqa: BLE001
            probs.append("%s is not readable json: %s" % (name, path))
            continue
        missing = [t for t in targets if t not in m]
        if missing:
            probs.append("%s=%s has no entry for %d of %d target(s): %s. A site arm degrades to its "
                         "fallback reward under a site label. Wrong dataset's map?"
                         % (name, os.path.basename(path), len(missing), len(targets), missing[:6]))

    if msa_db:
        missing = [t for t in targets if not os.path.isdir(os.path.join(msa_db, t))]
        if missing:
            probs.append("MSA_DB=%s has no dump for %d target(s): %s. Re-wiring would produce inputs "
                         "with no MSA." % (msa_db, len(missing), missing[:6]))
    return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dataset-check",
                       help="assert every artefact covers every target, i.e. ONE dataset")
    p.add_argument("--targets", required=True, help="space-separated target ids")
    p.add_argument("--inputs-wired", default=None)
    p.add_argument("--gt-index", default=None)
    p.add_argument("--foldbench-root", default=None)
    p.add_argument("--msa-db", default=None)
    p.add_argument("--map", action="append", default=[], metavar="NAME=PATH",
                   help="a map that must cover every target; repeatable")

    p = sub.add_parser("wire-msas", help="wire MSAs onto ALL prepared inputs")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--msa-db", required=True)
    p.add_argument("--out-dir", default=None)

    p = sub.add_parser("wire-shard", help="bin-pack + wire only this shard's cost-balanced bin")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--msa-db", required=True)
    p.add_argument("--nshards", type=int, required=True)
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--out-dir", required=True)

    p = sub.add_parser("cdr-tokenmap", help="per-chain IMGT CDR map -> paratope token indices ($PROTENIX_CDR_MAP)")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--cdr-map", required=True, help="{target: {chain: [CDR residue idx]}} json")
    p.add_argument("--out-json", required=True)
    p.add_argument("--wired-dir", default=None)

    p = sub.add_parser("epitope-tokenmap", help="antigen epitope token indices "
                       "(implemented in experiments/scripts/token_maps.py)")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--pool-dir", "--dt3-dir", dest="src_dir", required=True,
                   help="unsteered pool predictions dir, or the DiscoTope-3 CSV dir")
    p.add_argument("--scorer", default="pool_modes", choices=sorted(_epitope_scorers()),
                   help="which epitope scorer supplies the scores")
    p.add_argument("--n-patches", type=int, default=1)
    p.add_argument("--patch-radius", type=float, default=12.0)
    p.add_argument("--out-json", required=True)
    p.add_argument("--wired-dir", default=None)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--use-flag", action="store_true", help="use DiscoTope's epitope=True instead of top-k")

    p = sub.add_parser("checkpoint", help="populate todo-dir with inputs still needing a run")
    p.add_argument("--wired-dir", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--todo-dir", required=True)
    p.add_argument("--n-seeds", type=int, required=True)
    p.add_argument("--n-samples", type=int, required=True)
    p.add_argument("--granular", action="store_true", help="require each seed dir complete, not just the total")
    p.add_argument("--resume-seeds", action="store_true",
                   help="keep a partial target's COMPLETE seed dirs instead of wiping them; "
                        "pair with TTS_RESUME_SEEDS=1 on the run_ablation call")

    p = sub.add_parser("count-complete", help="count target_ids with >= exp rows in a parquet")
    p.add_argument("--parquet", required=True)
    p.add_argument("--exp", type=int, required=True)

    p = sub.add_parser("cost-bins", help="print this shard's cost-balanced bin over an explicit target list")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--targets", required=True, help="space-separated target ids (their inputs_wired/{tid}.json)")
    p.add_argument("--nshards", type=int, required=True)
    p.add_argument("--shard-id", type=int, required=True)

    p = sub.add_parser("consolidate", help="mv per-shard prediction dirs into one shared dir (for merge)")
    p.add_argument("--dest", required=True)
    p.add_argument("--src-dirs", required=True, nargs="+", help="per-shard predictions dirs to drain")

    p = sub.add_parser("preflight", help="validate a steering config OFFLINE before spending GPU time")
    p.add_argument("--set", dest="sets", nargs="*", default=[], help="key=value spec overrides")
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--n-particles", type=int, default=8)

    p = sub.add_parser("health", help="screen one run for silent failure before its number is used")
    p.add_argument("--log", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--expect-cifs", type=int, default=None)
    p.add_argument("--res-dir", default=None,
                   help="the unit's res/ dir, enabling the metrics check. USE IT AFTER SCORING ONLY, "
                        "never in the in-run gate: score_unit runs after that gate passes, so "
                        "requiring a scored column there is unsatisfiable on a fresh unit")
    p.add_argument("--expect-guidance", choices=["yes", "no"], default=None,
                   help="yes = steered arm (guidance must fire), no = control (it must not)")
    p.add_argument("--require-scored", action="store_true",
                   help="with --res-dir, treat an UNSCORED unit as a failure. Off by default: the "
                        "in-run gate runs before score_unit, so demanding a scored column there is "
                        "unsatisfiable and returns finished units to the queue to be recomputed")

    p = sub.add_parser("checkpoint-route",
                       help="which checkpoint would a run load here (private-first, public fallback)")
    p.add_argument("--code-tree", default=None, help="prepend to sys.path so the resolver imports")
    p.add_argument("--require", choices=["private", "public"], default=None,
                   help="fail unless the route resolves to this source")

    p = sub.add_parser("ks-compare", help="two-sample KS across a pool run's arms (distributions, not pairs)")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--column", default="dockq")
    p.add_argument("--ref", default="bon", help="arm every other arm is tested against")

    p = sub.add_parser("pool-score", help="score finished pool units OFF the GPU pod (run with SCORE_JOBS=0)")
    p.add_argument("--run-dir", required=True, help="the pool RUN_DIR (holds queue/done and the arm dirs)")
    p.add_argument("--score-py", default=None, help="python for the scoring venv")
    p.add_argument("--dockq-bin", default=None)
    p.add_argument("--jobs", type=int, default=6, help="units scored concurrently")
    p.add_argument("--threads", type=int, default=6, help="metric threads per unit")
    p.add_argument("--once", action="store_true", help="one pass then exit (default: poll forever)")
    p.add_argument("--poll-s", type=int, default=60)
    p.add_argument("--code-tree", default=None)

    p = sub.add_parser("pool-stats", help="pool diversity on BOTH axes (conformation and binding site)")
    p.add_argument("--cdr-map", required=True, help="paratope tokenmap json ($PROTENIX_CDR_MAP)")
    p.add_argument("--target", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--dstars", default="", help="optional comma-separated d* targets to report against")

    p = sub.add_parser("epi-dist", help="ORACLE: paratope-to-TRUE-epitope centroid distance (USES GROUND TRUTH)")
    p.add_argument("--cdr-map", required=True, help="paratope tokenmap json")
    p.add_argument("--true-epitope-map", required=True, help="TRUE epitope tokenmap (ground truth)")
    p.add_argument("--target", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--native", type=float, default=None, help="native value to print for reference")
    p.add_argument("--reach-a", type=float, default=6.0,
                   help="paratope-to-true-epitope distance (A) counted as reaching the site")

    p = sub.add_parser("compute-rederive",
                       help="rewrite diffusion_compute.json from the records matching the cifs on disk")
    p.add_argument("units", nargs="+")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("compute-check",
                       help="M4 gate: diffusion_compute.json must agree with the structures on disk")
    p.add_argument("units", nargs="+", help="unit directories (<run>/<target>__<arm>)")
    p.add_argument("--quiet", action="store_true")

    p = sub.add_parser("oom-class", help="classify a failed unit's OOM: ram | gpu_contention | gpu_size | none")
    p.add_argument("--log", required=True, nargs="+",
                   help="the unit's run.log, or several per-sample ERR captures")
    p.add_argument("--memguard-flag", default=None, help="path the host memory guard touches")
    p.add_argument("--contend-frac", type=float, default=0.25,
                   help="share of the card other processes must hold to call it contention")

    p = sub.add_parser("map-stamp", help="assert a token map's _source/_builder provenance stamps")
    p.add_argument("--map", required=True)
    p.add_argument("--source", default=None, help="required value of _source (ground_truth|predicted|synthetic)")
    p.add_argument("--builder", default=None, help="required value of _builder")

    a = ap.parse_args()
    if a.cmd == "dataset-check":
        probs = dataset_check(a.targets.split(), a.inputs_wired, a.gt_index, a.foldbench_root,
                              [tuple(m.split("=", 1)) for m in a.map if "=" in m], a.msa_db)
        for x in probs:
            print("!! dataset-check: %s" % x)
        n_t = len(a.targets.split())
        print("dataset-check: %d target(s), %d problem(s)" % (n_t, len(probs)))
        return 2 if probs else 0
    if a.cmd == "wire-msas":
        wire_all_msas(a.run_dir, a.msa_db, a.out_dir)
    elif a.cmd == "wire-shard":
        wire_shard(a.run_dir, a.msa_db, a.nshards, a.shard_id, a.out_dir)
    elif a.cmd == "cdr-tokenmap":
        build_cdr_tokenmap(a.run_dir, a.cdr_map, a.out_json, a.wired_dir)
    elif a.cmd == "epitope-tokenmap":
        build_epitope_tokenmap(a.run_dir, a.src_dir, a.out_json, a.wired_dir, a.top_k, a.use_flag,
                               a.n_patches, a.patch_radius, a.scorer)
    elif a.cmd == "checkpoint":
        checkpoint(a.wired_dir, a.pred_dir, a.todo_dir, a.n_seeds, a.n_samples, a.granular,
                   a.resume_seeds)
    elif a.cmd == "count-complete":
        print(count_complete_targets(a.parquet, a.exp))
    elif a.cmd == "cost-bins":
        print(" ".join(cost_bins_for_targets(a.targets.split(), a.run_dir, a.nshards, a.shard_id)))
    elif a.cmd == "consolidate":
        consolidate_predictions(a.src_dirs, a.dest)
    elif a.cmd == "preflight":
        ov = dict(kv.split("=", 1) for kv in a.sets if "=" in kv)
        probs = preflight(ov, a.n_steps, a.n_particles)
        warns = [m for m in probs if m.startswith("WARN:")]
        hard = [m for m in probs if not m.startswith("WARN:")]
        print("[preflight] " + (("FAIL (%d)" % len(hard)) if hard else "OK")
              + (" [%d warning(s)]" % len(warns) if warns else "") + ": " + str(ov))
        for m in hard:
            print("  !! " + m)
        for m in warns:
            print("  ~~ " + m[5:].lstrip())
        sys.exit(1 if hard else 0)
    elif a.cmd == "compute-rederive":
        return compute_rederive(a.units, dry_run=a.dry_run)
    elif a.cmd == "compute-check":
        return compute_check(a.units, quiet=a.quiet)
    elif a.cmd == "checkpoint-route":
        return checkpoint_route(code_tree=a.code_tree, require=a.require)
    elif a.cmd == "oom-class":
        return oom_class(a.log, memguard_flag=a.memguard_flag, contend_frac=a.contend_frac)
    elif a.cmd == "map-stamp":
        return map_stamp(a.map, source=a.source, builder=a.builder)
    elif a.cmd == "epi-dist":
        print(epitope_distance(a.cdr_map, a.true_epitope_map, a.target, a.pred_dir, a.native,
                               reach_a=a.reach_a))
    elif a.cmd == "pool-score":
        return pool_score(a.run_dir, score_py=a.score_py, dockq_bin=a.dockq_bin, jobs=a.jobs,
                          threads=a.threads, once=a.once, poll_s=a.poll_s, code_tree=a.code_tree)
    elif a.cmd == "pool-stats":
        print(pool_stats(a.cdr_map, a.target, a.pred_dir,
                         tuple(float(x) for x in a.dstars.split(",") if x)))
    elif a.cmd == "ks-compare":
        _rows, verdicts = ks_compare(a.run_dir, a.column, a.ref)
        print("[ks-compare] %s, reference arm %r" % (a.column, a.ref))
        for line in verdicts:
            print("  " + line)
    elif a.cmd == "health":
        want = None if a.expect_guidance is None else (a.expect_guidance == "yes")
        line, problems, notes = run_health(a.log, a.pred_dir, a.expect_cifs, want,
                                           res_dir=a.res_dir, require_scored=a.require_scored)
        print("[health] " + line)
        for msg in notes:
            print("  .. " + msg)
        for msg in problems:
            print("  !! " + msg)
        sys.exit(1 if problems else 0)   # non-zero so a job step can refuse to score a bad run


if __name__ == "__main__":
    sys.exit(main() or 0)
