#!/usr/bin/env bash
# NOISE-SEARCH BASELINE launcher, the "is steering just search?" arm of the ICLR plan (§3.3, T5).
#
# WHY A SEPARATE LAUNCHER AND NOT AN ARM IN tts-cast-pool.yaml. noise_search is
# `diffusion_side: search`, a different FAMILY: it runs an outer ES loop over the INITIAL NOISE and
# each candidate is a full sampling run, so it does not share the per-step coupled-particle loop the
# pool templates are built around. Bolting it into the cast template would add a branch that implies
# that template can run the search family, which is exactly the kind of "config claims a capability
# the code does not have" defect this repo keeps producing. One small launcher instead.
#
# BUDGET, matched to CAST (see experiments/configs/diffusion_side/b2_noise_search.yaml):
#     UNDER COSYNE the cost is NOT popsize x generations. Each generation decodes the children
#     (2 x popsize) PLUS the permuted population (popsize), so 3 x popsize sampling runs per
#     generation on top of the initial population:
#         NFE = popsize + generations x 3 x popsize = 16 + 15 x 48 = 736 per seed
#     and n_samples_per_seed=736 keeps every one of them, so 736 generated = 736 written
#     (CAST: 723). The 1.8% excess is in the BASELINE's favour, which is the safe direction.
#     This arithmetic is specific to cosyne at 1 seed: the isotropic `es` variant costs popsize per
#     generation instead. If noise_search_algo or the seed count changes, RE-DERIVE it here AND in
#     b2_noise_search.yaml, and in jobs/tts-baseline-pool.yaml's $NS_NFE, because M4 reports this
#     number.
#
# THE REWARD IS PINNED IN THE CONFIG (reward=epitope, hybrid map). Without it the spec falls back to
# reward=ipsae -- a different objective from CAST's -- and the comparison would be meaningless while
# looking perfectly healthy. Do not override it on the command line.
#
# THE CHECKPOINT IS LEFT AT THE DEFAULT ON PURPOSE. Forcing `checkpoint.prefer_private=false` takes
# PUBLIC weights while every arm this is a baseline for logs "using private checkpoint" -- same
# architecture, different weights, so the comparison would silently measure the checkpoint as well
# as the method. A baseline must inherit the same resolution path as the arm it is compared
# against, so do not pin a checkpoint here.
#
#   TARGETS="..." RUN_DIR=${TTS_ROOT:-/data/tts}/runs/ns16 GPU=0 bash jobs/launch_noise_search.sh
#   DRY_RUN=1 ... to validate dispatch without taking a GPU.
set -uo pipefail

CODE="${CODE:-${TTS_ROOT:-/data/tts}/Protein-TTS}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
TARGETS="${TARGETS:?set TARGETS (space-separated target ids)}"
INPUTS_WIRED="${INPUTS_WIRED:-${TTS_ROOT:-/data/tts}/runs/inputs_wired}"
CFG="${CFG:-experiments/configs/diffusion_side/b2_noise_search.yaml}"
GPU="${GPU:-0}"
SEED0="${SEED0:-101}"
DRY_RUN="${DRY_RUN:-0}"

export PATH="${TTS_ROOT:-/data/tts}/protenix/.venv/bin:$PATH"
export PYTHONPATH="$CODE"
export CUDA_VISIBLE_DEVICES="$GPU"
export FOLDBENCH_ROOT="${FOLDBENCH_ROOT:-${TTS_ROOT:-/data/tts}/data/foldbench-groundTruth}"
export PROTENIX_COMPONENT_PATH="${PROTENIX_COMPONENT_PATH:-${TTS_ROOT:-/data/tts}/protenix/.venv}"
export PROTENIX_ROOT_DIR="${PROTENIX_ROOT_DIR:-${TTS_ROOT:-/data/tts}/protenix/cache_public}"
export PROTENIX_CDR_MAP="${PROTENIX_CDR_MAP:-${TTS_ROOT:-/data/tts}/runs/maps/cdr_tokenmap.json}"
# THE MAP MUST BE THE ONE THE CONFIG PINS (_builder=hybrid_modes_tiles) or preflight refuses it.
export PROTENIX_EPITOPE_MAP="${PROTENIX_EPITOPE_MAP:-${TTS_ROOT:-/data/tts}/runs/maps/hybK12_all52.json}"
export PROTENIX_CONF_OFFLOAD_TOKENS="${PROTENIX_CONF_OFFLOAD_TOKENS:-auto}"
# TOKEN-AXIS CHUNK. Quadratic in N_token and independent of sample count, so the upstream default
# of 256 kills the largest targets at ~97% completion. See the memory-controls section of the
# README.
export PROTENIX_INFER_CHUNK_SIZE="${PROTENIX_INFER_CHUNK_SIZE:-auto}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 LAYERNORM_TYPE=torch
# PIN THREADS TO THE CGROUP QUOTA. Featurisation spawns far more threads than the pod's CPU limit and
# thrashes against the CPU cap, stretching startup by minutes per unit.
# noise_search pays that startup ONCE PER GENERATION, so it is hit harder than any other arm.
_q=$(awk '{if ($1=="max") print 8; else print int($1/$2)}' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 8)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$_q}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

cd "$CODE"
mkdir -p "$RUN_DIR"
echo "===== noise_search: $(echo $TARGETS | wc -w) target(s), GPU=$GPU, threads=$OMP_NUM_THREADS, $(date -u) ====="
python - <<'PY'
import yaml, os
d=yaml.safe_load(open(os.environ.get("CFG","experiments/configs/diffusion_side/b2_noise_search.yaml")))
ds=d["method"]["diffusion_search"]; r=d["model"]["run"]
P = ds["popsize"]; G = ds["generations"]; sd = r["msa_subsample"]["n_seeds"]
algo = ds.get("noise_search_algo", "es")
# COST DEPENDS ON THE VARIANT. cosyne decodes children (2P, EvoTorch cross-over rate 2.0) PLUS the
# permuted population (P) each generation, on top of the initial population, the isotropic es decodes
# P per generation. Getting this wrong silently changes the budget by 3x.
gen = (P + G * 3 * P) if algo == "cosyne" else (P * G)
print("  variant: %s  popsize %d  generations %d" % (algo, P, G))
print("  budget : %d NFE/seed x %d seed(s) = %d evaluated, %d written  (CAST reference: 723)"
      % (gen, sd, gen * sd, r["n_samples_per_seed"] * sd))
print("  reward : %s  source=%s  builder=%s" % (ds["reward"], ds["epitope_source"], ds["epitope_map_builder"]))
assert ds["reward"] == "epitope", "reward must match the arm this is a baseline for"
assert r["n_samples_per_seed"] == gen, (
    "n_samples_per_seed (%d) must equal the evaluated count (%d) or structures are discarded" % (r["n_samples_per_seed"], gen))
PY
[ $? -eq 0 ] || { echo "!! config self-check failed"; exit 2; }

for T in $TARGETS; do
  OUT="$RUN_DIR/${T}__ns"
  if [ -s "$OUT/res/per_sample.parquet" ]; then echo "  [skip] $T already done"; continue; fi
  mkdir -p "$OUT/src"; cp "$INPUTS_WIRED/$T.json" "$OUT/src/" 2>/dev/null || { echo "  !! no input for $T"; continue; }
  echo "----- $T $(date -u +%H:%M:%S) -----"
  EXTRA=""; [ "$DRY_RUN" = "1" ] && EXTRA="--dry-run"
  t0=$(date +%s)
  python -m experiments.scripts.run_ablation --config "$CFG" $EXTRA \
    --set model.run.msa_subsample.source_inputs_dir="$OUT/src" method.label="ns_${T}" \
          model.seeds=[$SEED0] model.run.nproc_per_node=1 \
          paths.inputs_dir="$OUT/inputs" paths.predictions_dir="$OUT/pred" paths.results_dir="$OUT/res" \
    > "$OUT/run.log" 2>&1
  rc=$?
  cif=$(find "$OUT/pred" -name '*.cif' 2>/dev/null | wc -l)
  echo "  rc=$rc wall=$(( $(date +%s) - t0 ))s cifs=$cif"
  # HEALTH: the ES writes n_samples_per_seed PER SEED. Anything less means generations were lost.
  # SKIPPED ON A DRY RUN, which writes nothing by design -- a gate that fires on every dry run is a
  # gate people learn to ignore.
  if [ "$DRY_RUN" != "1" ]; then
    want=$(python -c "import yaml;d=yaml.safe_load(open('$CFG'));r=d['model']['run'];print(r['n_samples_per_seed']*r['msa_subsample']['n_seeds'])")
    [ "$cif" -lt "$want" ] && echo "  !! UNHEALTHY: $cif cifs, expected $want"
  fi
  grep -aoE "noise_search gen [0-9]+/[0-9]+: best [a-z_]+=[0-9.-]+" "$OUT/run.log" | tail -2 | sed 's/^/  /'
done
echo "===== noise_search done $(date -u) ====="
