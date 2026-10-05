![Python version](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)

# CAST-AbAg

**CAST** (Cross-subsample Amortised Site Tilting) steers a co-folding model's diffusion sampler
toward a named antibody binding **site** rather than a pose. Training-free, no reference structure,
frozen backbone.

The problem is *site coverage*. Co-folding models often dock the antibody on the wrong epitope, and
more seeds deepen the mode the model already prefers instead of relocating it. A verifier cannot
select an epitope the sampler never proposed. CAST adds a site-measurable reward from a candidate
epitope map, carries the candidate site and the alignment-subsample index in the particle state so
K conditionings share one pool, and tilts toward the site while leaving the pose to the base model.

This repository contains the running pipeline: preprocessing, input wiring, the methods, and the job
templates to run them end to end.

## Software requirements

- Linux, NVIDIA GPU.
- Python 3.11 (backends pin `>=3.11,<3.12`), CUDA 12.6.

Clone the repository if you want a local checkout:

```bash
git clone https://github.com/.../CAST-AbAg.git
cd CAST-AbAg
```

Then install with either [Poetry](https://python-poetry.org/):

```bash
poetry install                  # core
poetry install --with guidance  # + torch, the differentiable reward path
poetry install --with cdr       # + anarci, for the CDR map
```

or pip:

```bash
pip install .
```

**`poetry install` sets up this package only.** It installs `numpy`, `pandas`, `pyarrow`,
`omegaconf`, `biopython`, `biotite` and `tqdm`. Two optional groups add the rest, `torch` in `guidance` and `anarci` in `cdr`. It deliberately
does **not** install a co-folding backend: nothing here does `import protenix`, the backends are
shelled out to in their own virtualenvs. Five things must be provisioned separately, and
`./setup_backends.sh` covers every one except the MSA database. Pick what you need, re-run it
freely, anything already on disk is skipped, and it prints the env-file lines for what it built.
It installs both backends and downloads OpenDDE's checkpoint, but leaves the Protenix weights to
you, for the reason in the **Model weights** row below. `--cdr` resolves HMMER only, since
ANARCI is already a poetry dependency.

```bash
./setup_backends.sh --protenix          # Protenix v2 venv
./setup_backends.sh --opendde           # OpenDDE venv + abag checkpoint
./setup_backends.sh --all --prefix /data/tts
./setup_backends.sh --all --dry-run     # show what it would do
```


| Component                    | How                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Pointed at by                                                                                                                                                                                                                                                                    |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Co-folding backend** | install Protenix v2 or OpenDDE in its own venv                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | `PROTENIX_COMPONENT_PATH`, or `OPENDDE_ROOT_DIR` + `OPENDDE_BIN`                                                                                                                                                                                                           |
| **Model weights**      | Protenix v2 is fetched by the`protenix` package from `protenix.tos-cn-beijing.volces.com/checkpoint/protenix-v2.pt`, or pre-placed by hand when that origin is blocked. For a PRIVATE Protenix-v2 checkpoint, put it at `$PROTENIX_ROOT_DIR/checkpoint/protenix-v2.pt`, by baking it into your image or placing it by hand, because a checkpoint already on disk wins before any fetch is tried. OpenDDE weights are `opendde_abag.pt` from [huggingface.co/aurekaresearch/OpenDDE](https://huggingface.co/aurekaresearch/OpenDDE)                                                                                                                                                                                                                            | `PROTENIX_ROOT_DIR`, `OPENDDE_ROOT_DIR`. The job template logs which checkpoint it resolved before taking a GPU. `models/protenix_provision.py` has the full artefact list, the manual-placement layout, a third-party mirror with its digest, and the caveats on using it |
| **MSA database**       | a pre-built ColabFold-style DB, or generate per target with`generate_msa`                                                                                                                                                                                                                                                                                                                                                                                                                                                               | `MSA_DB`                                                                                                                                                                                                                                                                       |
| **DockQ**              | install in a scoring venv, kept separate from the sampler's                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | `SCORE_PY`, `SCORE_DOCKQ`                                                                                                                                                                                                                                                    |
| **ANARCI + HMMER**     | only for the CDR map. ANARCI comes from`poetry install --with cdr`. HMMER is a C program and must be the real binary, not a wrapper script, which rules out the PyPI `hmmer` package's console scripts: HMMER writes `argv[0]` into its output header, a wrapper puts its own name there, Biopython then matches no program and leaves every alignment coordinate `None`, and ANARCI dies comparing them. Use a conda/apt/source HMMER, or point at the real executables the wheel bundles under `site-packages/hmmer/data/bin` | `--hmmer-path` on `token_maps cdr-map`                                                                                                                                                                                                                                       |

### Environment file

A pool run is configured by an env file passed with `--env-file`, sourced by the job template so
every export reaches the worker. The variables it must set:

| Variable                                       | Meaning                                                                                                                                                                                                                                                                                     |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CODE`, `STATS_DIR`                        | repo root, and`$CODE/jobs` for the helper scripts                                                                                                                                                                                                                                         |
| `BACKEND`, `TTS_CFG_SUFFIX`                | which backend, and the config suffix that selects its variants                                                                                                                                                                                                                              |
| `RUN_DIR`                                    | run outputs and the shared queue                                                                                                                                                                                                                                                            |
| `INPUTS_WIRED`, `GT_INDEX`, `MSA_DB`     | prepared inputs, the target index, the MSA database                                                                                                                                                                                                                                         |
| `EPI_HYBK12`, `PROTENIX_CDR_MAP`           | candidate epitope map and paratope token map                                                                                                                                                                                                                                                |
| `N_SAMPLES`, `SUBSAMPLE_POOL`, `N_SEEDS` | pool size N and conditioning count K, which must match                                                                                                                                                                                                                                      |
| `UMBRELLA_HYBK12`, `SCALE`                 | arrival radii and guidance scale                                                                                                                                                                                                                                                            |
| `TARGETS`, `ARMS_ONLY`, `SKIP_CTRL`      | what to run.`SKIP_CTRL` defaults to 1, so the matched control is opt-in with `SKIP_CTRL=0`                                                                                                                                                                                              |
| `SCORE_PY`, `SCORE_DOCKQ`, `SCORE_JOBS`  | scoring                                                                                                                                                                                                                                                                                     |
| `LAYERNORM_TYPE`                             | which Protenix layer-norm kernel to use. Every template defaults to`torch`, which is what all the measured arms ran. `tts-b0b-pool.yaml` also accepts `fast_layernorm`, the faster JIT-compiled CUDA kernel, which needs nvcc and ninja and is not bit-identical to `torch` in bf16 |

Cluster-specific paths default from `APP_ROOT` and `TTS_ROOT`, so nothing needs editing to run
elsewhere. The job templates also carry placeholders to fill in: `<YOUR-NAMESPACE>`,
`<YOUR-PVC-CLAIM>`, `<YOUR-GPU-PRODUCT-LABEL>`, `<YOUR-PROTENIX-IMAGE>`, `<YOUR-PROTENIX-VENV>`,
`<YOUR-PROTENIX-CACHE>` and `<YOUR-VENV>`.

## How to run

The full pipeline, in order, for the paper's configuration: CAST steering to the **hybK12**
epitope map. Set `PYTHONPATH` to the repo root, or install the package.

**Step 4 builds the epitope map, and is OPTIONAL if you already have one.** The map is a
small JSON that travels with a run, so it is normally built once per dataset and reused. Nothing
builds it implicitly: the job templates only ever *consume* a map, so point them at the existing
file and go straight from step 3 to step 5. Which variable depends on the path: step 5a
(`run_ablation`) reads `$PROTENIX_EPITOPE_MAP`, while step 5b (the pool) takes `EPI_HYBK12` and
exports `PROTENIX_EPITOPE_MAP` itself, per arm.

A reused map must pass two gates, both before any GPU is taken:

- its `_builder` stamp must equal the config's `epitope_map_builder`, which is `hybrid_modes_tiles`
  for `m5_cast.yaml`. A map built by `tiles-ensemble` is stamped `hybrid_modes_ensemble_tiles` and
  is refused, which is what stops two different maps being reported under one label.
- `dataset-check` requires it to cover every target in `$TARGETS`, so a map built for one dataset
  cannot be silently reused on another.

Check both without launching anything:

```bash
PYTHONPATH=$PWD python jobs/tts_job_lib.py map-stamp --map $RUN/hybk12.json \
    --source predicted --builder hybrid_modes_tiles
# exit 0 and "[map-stamp] ... OK ... over N targets", or exit 2 naming the mismatch
```

If you do need to build it, the ordering matters in one non-obvious place. **The deployable map is
derived from an unsteered pool**, so the pool inside step 4 runs before the map is built and the
steered run in step 5 cannot come first. That pool is not wasted either way, since it is also the
Best-of-N control the steered arms are compared against.

Each map is one step here but several commands: step 3 builds `cdr_map.json` then
`cdr_tokenmap.json`, and step 4 builds `modes.json` and `tiles.json` then `hybk12.json`.

<pre>
<a href="#1-prepare-the-dataset">1 prepare_dataset</a>
<a href="#2-msas-and-wire-them-into-the-inputs">2 generate_msa</a>            (own MSA DB? same command, it only wires;
      |                    own wired inputs? skip it with REUSE_WIRED=1)
      ├─ <a href="#3-paratope-map-cdr">3 cdr-map ─> cdr</a> ─> $PROTENIX_CDR_MAP ───────────────────┐
      │                                                           │
      └─ <a href="#4-epitope-map-hybk12">4 unsteered pool ─┬─ epitope(pool_modes) ─┐</a>              │
         (optional)        <a href="#4-epitope-map-hybk12">└─ tiles ───────────────┴─> hybrid</a>     │
                                                        │         │
                                            $PROTENIX_EPITOPE_MAP │
                                                        │         │
            <a href="#5-run-the-steered-arm">5 steered run</a> &lt;─────────────────────────────┴─────────┘
            │
            <a href="#6-score">6 evaluate</a> ─> <a href="#7-tabulate-compute">7 compute_table</a>
</pre>

Every block below shares this preamble.

```bash
DATASET=cfa50
RUN=runs/$DATASET
```

#### 1. Prepare the dataset

Structures to backend-native inputs plus a target index.

```bash
python -m experiments.scripts.prepare_dataset \
    --dataset $DATASET \
    --backend protenix2 --runs-root runs
# -> $RUN/inputs/*.json and $RUN/target_index.csv   (the latter is $GT_INDEX for scoring)
# cfa50 needs no --dataset-args, it reads the shipped data/cfa50_targets tree. The other
# loaders do, e.g. --dataset-args '{"bad_cases_dir": "'$DATA_ROOT'/bad_cases"}'
```

`--dataset` takes one of `cfa50`, `cofoldarena`, `foldbench`, `collected_bad_cases`, `opendde40`.

#### 2. MSAs, and wire them into the inputs

Two jobs in one command: fill the DB from the server, then attach each input's per-chain MSA from
the DB.

```bash
python -m experiments.scripts.generate_msa \
    --inputs-dir $RUN/inputs --db-dir $DATA_ROOT/msa_db \
    --wire-out $RUN/inputs_wired --backend protenix2
```

ALREADY HAVE THE MSAs? Run the same command. Generation is per target and skipped when the DB
already holds it, so with a complete DB it contacts no server and only wires. Already have WIRED
inputs? Skip this step and point the run at them with `REUSE_WIRED=1` and `INPUTS_WIRED=<dir>`,
otherwise it REWRITES every wired input the DB covers.

#### 3. Paratope map (CDR)

Stage (a) needs ANARCI, stage (b) does not.

```bash
python -m experiments.scripts.token_maps cdr-map \
    --wired-dir $RUN/inputs_wired --out-json $RUN/cdr_map.json
python -m experiments.scripts.token_maps cdr \
    --run-dir $RUN --cdr-map $RUN/cdr_map.json --out-json $RUN/cdr_tokenmap.json
export PROTENIX_CDR_MAP=$RUN/cdr_tokenmap.json
```

#### 4. Epitope map, hybK12

OPTIONAL. Already have one? Then just `export PROTENIX_EPITOPE_MAP=<existing>/hybk12.json` and skip
to [step 5](#5-run-the-steered-arm).

4a, the unsteered pool the map is derived from, which is also the Best-of-N control.

```bash
CODE=$PWD TEMPLATE=jobs/tts-b0b-pool.yaml RUN_DIR=$RUN/bon \
  bash jobs/submit_steer_pool.sh --gpus 4 --local --env-file env.sh
POOL=$RUN/bon/predictions
```

4b, epitope candidates from that pool's own binding-site modes. `--scorer` defaults to `pool_modes`,
which is what hybK12 needs; pass `--scorer discotope3` with `--pool-dir` pointing at the CSV dir for
that predictor instead.

```bash
python -m experiments.scripts.token_maps epitope \
    --n-patches 3 --run-dir $RUN --pool-dir $POOL \
    --wired-dir $RUN/inputs_wired --out-json $RUN/modes.json
```

4c, exhaustive surface tiling of the antigen, 12 tiles per target.

```bash
python -m experiments.scripts.token_maps tiles \
    --pool-dir $POOL --wired-dir $RUN/inputs_wired \
    --k-want 12 --out-json $RUN/tiles.json
```

4d, hybK12 is pool modes first, then tiles ranked by mode support, to a 15-candidate budget.

```bash
python -m experiments.scripts.token_maps hybrid \
    --modes-json $RUN/modes.json --tiles-json $RUN/tiles.json \
    --out-json $RUN/hybk12.json
export PROTENIX_EPITOPE_MAP=$RUN/hybk12.json     # EPI_HYBK12 in the job templates
```

#### 5. Run the steered arm

5a, one CAST arm directly. Paths come from the config unless overridden, so `--set` them to keep
each unit's artefacts together, exactly as the job template does.

```bash
python -m experiments.scripts.run_ablation \
    --config experiments/configs/diffusion_side/m5_cast.yaml \
    --set model.run.msa_subsample.source_inputs_dir=$RUN/inputs_wired \
          paths.inputs_dir=$OUT/inputs paths.predictions_dir=$OUT/pred \
          paths.results_dir=$OUT/res
```

5b, or a pool of workers, one per GPU. The env file exports `CODE`, `STATS_DIR`, `GT_INDEX`,
`EPI_HYBK12` and `PROTENIX_CDR_MAP`; see the [environment-file table](#environment-file) above.
With only `EPI_HYBK12` set, the queue is just the hybk12 arm: every other map arm is gated on its
own `EPI_*` variable, the oracle arms on `WANT_GT`, and the control on `SKIP_CTRL`.

```bash
CODE=$PWD TEMPLATE=jobs/tts-cast-pool.yaml RUN_DIR=$RUN/cast \
  bash jobs/submit_steer_pool.sh --gpus 4 --local --env-file env.sh
# add --dry-run to stop after preflight, before any GPU work
```

`SKIP_CTRL` defaults to 1, so NO control is queued. A ctrl unit depends only on
`(target, seed, n_samples, backend)`, so it is built once and reused; set `SKIP_CTRL=0` to run one
here when you do not already have a matching one. `WANT_GT=1` adds the oracle ceiling arms, and then
`EPI_TRUE` is required.

#### 6. Score

REQUIRED after 5a, which writes structures and leaves the parquet's `dockq` column empty. After 5b
the pool has already run this command per unit as `score_unit`, so the only thing left for it to add
is `--summary`, the rank and oracle rollup that the pool never passes. Pool-written rows carry
`--method-label inflight`, so a later pass here is distinguishable rather than duplicated.

`--from-per-sample` reads the run's `per_sample.parquet`, and without it `--predictions-dir` is
required instead and the call will find nothing.

```bash
python -m experiments.scripts.evaluate --from-per-sample \
    --target-index $RUN/target_index.csv \
    --results-dir $OUT/res --backend protenix2 \
    --metrics dockq --dockq-bin $SCORE_DOCKQ --summary
```

#### 7. Tabulate compute

Measured FLOPs and wall-clock across runs. Pass the Best-of-N pool too and `--ref-arm` to read each
steered arm as a ratio of it.

```bash
python -m experiments.scripts.compute_table $RUN/cast $RUN/bon --ref-arm b0b
```

The pool templates run steps 1, 2 and the pool in 4a themselves from `$DATASET` / `$DATASET_ARGS`, so a pool
launch needs only the two maps from steps 3 and 4 to exist and be exported. `--dry-run` stops
after the preflight, which is what checks that `$PROTENIX_EPITOPE_MAP` is present, carries a
`_builder` stamp matching `epitope_map_builder`, and that the per-cell budget is satisfiable.

## Arms

| Arm           | Config                                   | What it is                             |
| ------------- | ---------------------------------------- | -------------------------------------- |
| CAST          | `diffusion_side/m5_cast.yaml`          | this work                              |
| Best-of-N     | `baseline/b0b.yaml`                    | unsteered control                      |
| FK steering   | `diffusion_side/m1_fk.yaml`            | same reward, no gradient               |
| TDS           | `diffusion_side/m2_tds.yaml`           | separate pool per conditioning         |
| AF3-ReD       | `diffusion_side/m4_af3red.yaml`        | RMSD repulsion                         |
| Boltz-Perturb | `diffusion_side/b4_boltz_perturb.yaml` | conditioning perturbation              |
| Noise search  | `diffusion_side/b2_noise_search.yaml`  | evolutionary search over initial noise |

Three of these ship an OpenDDE variant with the `_opendde` suffix, `b0b_opendde.yaml`, `m2_tds_opendde.yaml` and `m5_cast_opendde.yaml`.

## Layout

| Path              | Contents                                                                  |
| ----------------- | ------------------------------------------------------------------------- |
| `config.py`     | YAML loader, paths resolved relative to the config file                   |
| `results_io.py` | `per_sample.parquet` / `per_target.csv` schema                        |
| `data/`         | `target.py`, `datasets/` loaders, `msa/` generation and subsampling |
| `method/`       | `diffusion_side/`, `trunk_side/`, `metrics/`, `verifier/`         |
| `models/`       | backend adapters and inference hooks                                      |
| `experiments/`  | `configs/` and pipeline `scripts/`                                    |
| `jobs/`         | job templates and submission scripts                                      |

## Key knobs

The settings usually changed, all in the `method.diffusion_search` block of an arm config or
overridable with `--set`. Defaults are CAST's reference arm.

**Budget and pool**

| Knob                        | Default | Effect                                                                                      |
| --------------------------- | ------- | ------------------------------------------------------------------------------------------- |
| `subsample_pool` (K)      | 4       | conditionings sharing one pool. K is the axis that scales; N alone does not                 |
| `n_samples_per_seed` (N)  | 720     | particles in the pool. Must satisfy`N >= 3 * K * umbrellas * sites`, or preflight refuses |
| `weight_level_correction` | true    | required at K>1; restores the cross-latent baseline the difference potential divides out    |

**The site reward**

| Knob                    | Default                | Effect                                                                                  |
| ----------------------- | ---------------------- | --------------------------------------------------------------------------------------- |
| `reward`              | `epitope`            | the site collective variable. The joint pool requires this                              |
| `umbrella_sweep`      | `9,11,13,15`         | arrival radii d*, brackets the observed native paratope-epitope range                   |
| `site_hinge`          | true                   | flat zero-reward ball inside d*. False gives a plain harmonic that keeps pulling inward |
| `site_huber_delta`    | 4.0                    | bounds the far side; 0 leaves it unbounded quadratic                                    |
| `epitope_source`      | `predicted`          | `ground_truth` swaps in the oracle map, the ablation ceiling                          |
| `epitope_map_builder` | `hybrid_modes_tiles` | pool modes first, then surface tiles ranked by mode support                             |

**The tilt**

| Knob                   | Default | Effect                                                                             |
| ---------------------- | ------- | ---------------------------------------------------------------------------------- |
| `guidance_scale`     | 75.0    | gradient strength. 0 turns the tilt off entirely, leaving the joint pool unsteered |
| `guidance_max_rms_a` | 6.0     | size-invariant displacement rail                                                   |
| `lam_adaptive`       | true    | calibrate lambda once from the observed reward spread                              |
| `lam`                | 4.0     | the cap on the calibrated lambda, not inert                                        |
| `target_ess_frac`    | 0.4     | ESS target the calibration solves for                                              |

**Resampling**

| Knob                                        | Default        | Effect                                                                 |
| ------------------------------------------- | -------------- | ---------------------------------------------------------------------- |
| `potential`                               | `difference` | score improvement rather than absolute reward                          |
| `resample` / `ess_fraction`             | `ess` / 0.5  | resample a cell when its ESS falls below half its size                 |
| `resample_windows`                        | `0.25:0.35`  | the early window resampling is confined to, which bounds pool collapse |
| `schedule_start_frac`, `schedule_every` | 0.25, 20       | when scoring events begin and how often they fire                      |

At the job level the same budget is set by `N_SAMPLES`, `SUBSAMPLE_POOL`, `N_SEEDS` (must equal
`SUBSAMPLE_POOL`), `UMBRELLA_HYBK12` and `SCALE`.

## Job submission template

`jobs/submit_steer_pool.sh` freezes a template (`jobs/tts-*-pool.yaml`) into a self-contained shell
body and starts N workers, one per GPU, all draining a shared queue under `$RUN_DIR/queue/`. A unit
is one `target x arm`. Workers are independent, so they may start at different times and the grid
still completes.

The template is not a thin wrapper. It carries the gates that decide whether a result is
trustworthy:

| Stage                      | What it does                                                                                                                                                                                        |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **checkpoint route** | resolves private-vs-public weights and logs which one,*before* a GPU is taken                                                                                                                     |
| **dataset check**    | validates every target in the index and reports problems before any GPU work                                                                                                                        |
| **preflight**        | builds the arm list and rejects invalid ones. Catches per-cell budget violations (`n_samples` too small for `umbrellas x sites x K`) and maps with no `_source`/`_builder` provenance stamp |
| **queue**            | `pending/ claimed/ done/ failed/` plus `attempts/`, `heartbeat/`, `force_offload/`. Claims are atomic `mkdir`, so workers never collide                                                   |
| **`run_unit`**     | invokes`run_ablation --config <cfg> --set ...` with the arm's overrides and per-unit paths                                                                                                        |
| **health gate**      | refuses to score a silently-failed run. Expects`N+K-1` structures for a joint pool and `N_SEEDS x N_SAMPLES` for the control, because the two arms legitimately emit different counts           |
| **memguard**         | kills the unit at`MEMGUARD_PCT` (92) of the cgroup limit rather than letting the pod be OOM-killed                                                                                                |
| **OOM classifier**   | separates`gpu_size` (escalate to confidence offload) from `gpu_contention` (requeue, do not escalate) from `ram`, and records each in `oom_events.jsonl`                                    |
| **`score_unit`**   | runs DockQ concurrently at`nice 19 ionice -c 3`, so scoring never delays the GPU                                                                                                                  |
| **`reclaim`**      | returns units whose worker died, via the heartbeat                                                                                                                                                  |
| **dead-letter**      | retires a unit after`MAX_ATTEMPTS` (2) so one bad unit cannot starve the pool                                                                                                                     |

Useful knobs: `MAX_ATTEMPTS`, `MEMGUARD_PCT`, `DEADLINE_S`/`UNIT_EST_S` (retire cleanly rather than
be killed mid-unit), `SCORE_JOBS`, `ARMS_ONLY`, `SKIP_CTRL`, `TARGETS`.

## Adding a backend

A backend is selected by name everywhere, so a new one is a registration plus wiring, not a change
to any method or experiment code.

1. **Implement `ModelBackend`** (`models/base.py`) in `models/<name>.py`:
   `emit_input`, `get_msa_sources`, `set_msa_sources`, `get_chain_sequences`, `run`,
   `iter_predictions`, `load_pae_and_coords`. `run` takes a `BackendRunSpec`
   (`seeds`, `n_cycles`, `n_samples_per_seed`, `n_diffusion_steps`, `use_msa`, `extras`).
2. **Register it** at module scope: `register_backend("<name>", <Name>Backend)`.
3. **Auto-import it** by adding `"<name>"` to the tuple in `models/__init__.py::_auto_register`.
   Import failures there are non-fatal, so a missing runtime disables that backend only.
4. **Add a peak-memory scale** to `_PEAK_SCALE_BY_BACKEND` in `models/base.py`, relative to the
   Protenix curve the model was fitted on. Measure it rather than assuming 1.0.
5. **Install sampling hooks** if the backend needs them. `models/protenix_hooks.py` holds the
   steering hook plus vendor-quirk patches, each self-gating on `cofold_pkg()` so it no-ops
   elsewhere.
6. **Add per-backend configs** as `<config>_<suffix>.yaml` beside the originals, e.g.
   `m5_cast_opendde.yaml`. The job template picks them up through
   `TTS_CFG_SUFFIX` and `cfg_for_backend`.
7. **Run it** with `BACKEND=<name> TTS_CFG_SUFFIX=_<name>` in the env file. `SCORE_BACKEND`
   defaults to `BACKEND`, since the results reader must follow the model.
8. **Check that sample chunking preserves coupling.** This is the one correctness trap. A backend
   may split the particle batch to fit memory, and that is exact for the denoiser, which has no
   cross-particle term. But under a joint pool the chunk must be taken **within one conditioning
   group**, never across groups, and every chunk's output must be scattered back into a single
   N-particle tensor before anything else reads it. CAST's coupling lives in the SMC layer, the
   reward, the weights and the resample, and that layer must see the whole pool. The reference
   implementation does this by iterating groups first and chunking inside each
   (`models/protenix_hooks.py`, the `_grp` branch with `index_copy_`). A backend that chunks the
   flat batch will silently mix conditionings and break the cross-subsample comparison the joint
   pool exists for.
9. **Verify** with a small end-to-end run: preflight passes, the health gate sees `N+K-1`
   structures, and DockQ is non-null in `per_sample.parquet`.

## Memory controls

**The memory model.** `models/base.py::peak_gib_for` is the one cross-backend entry point: it scales
the Protenix-fitted curve by `_PEAK_SCALE_BY_BACKEND` (`protenix2` 1.0, the curve it was fitted on,
`opendde` 1.03). The curve itself, `models/protenix2.py::_predicted_peak_gib`, is the larger of two
measured paths:

```
inflight = base(N_token) + chunk*slope(N_token) + max(0, n_samples - chunk)*frac*slope(N_token)
peak     = max(inflight, atom_attn_base(N_token) + chunk*atom_attn_slope(N_token))
```

The middle term is the retained-sample accumulation, which is why a large pool can OOM at a chunk
that a small one survives, and the second path is why shrinking the chunk eventually stops helping.

The decisions built on it live in `models/protenix2.py`:

- **Diffusion chunk**, `PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE=auto` -> `auto_diffusion_chunk`. It
  fits first and optimises second: `max_diffusion_chunk` asks whether anything fits on-GPU at all,
  then `optimal_diffusion_chunk` picks the chunk to actually use, which is the **smallest that
  reaches the speed knee, not the largest that fits**. Below `_MIN_ONGPU_CHUNK` it falls back to an
  offload chunk.
- **Confidence offload**, `PROTENIX_CONF_OFFLOAD_TOKENS=auto` -> `should_offload`. Free-GPU-first:
  it moves the PAE/PDE stack to host RAM only when the target cannot safely stay on-GPU, via three
  hard rails (a card-aware token ceiling, a sample ceiling, and the chunk floor) and then the chunk
  math. Offload is a fitting tool, not a speedup, and it trades GPU pressure for host pressure, so
  a pod that offloads needs a larger memory limit rather than a smaller chunk.

**Opt-in controls for large joint pools.** All three are exact, so they change memory and not
results. The first two are unset by default and do nothing until you set them; the third already
has a default of 5, which you lower.

| Variable                         | What it does                                         | When you need it                                                                                                                  |
| -------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `TTS_JOINT_CACHE_CPU=1`        | stages the joint conditioning cache through host RAM | the cache holds K trunk outputs, about 3.5 GB each, so K=16 will not fit on an 80 GB card beside the working set                  |
| `TTS_ODDE_SC_PAIR_CHUNK=16`    | shrinks OpenDDE's shape-complementarity token chunk  | the vendor default of 128 materialises`[n_sample, 128, n_atom, 3]`, about 37 GB for a 2880-particle pool on a 1171-token target |
| `PROTENIX_CONF_SAMPLE_CHUNK=2` | shrinks the confidence sub-batch                     | the confidence peak is roughly`27 + 9.5 * chunk` GiB, so the default of 5 leaves only ~5 GiB of headroom on a large pool        |

The last two fail *after* the diffusion has completed, which costs the whole unit, so set them ahead
of a large run rather than waiting for the failure.

## Licence

Apache-2.0. See [LICENSE.md](LICENSE.md).

Release notes are in [NEWS.md](NEWS.md), and [CONTRIBUTING.md](CONTRIBUTING.md) covers issues and pull requests.
