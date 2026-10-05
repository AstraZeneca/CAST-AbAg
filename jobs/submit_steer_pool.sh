#!/usr/bin/env bash
# Submit a diffusion-side steering grid as a pool of single-GPU workers.
#
#   NGPUS=8 METHOD=tds_epi ./jobs/submit_steer_pool.sh
#   NGPUS=6 METHOD=tds_cdr TARGETS="9atr_nb2 9jno_ab1" ./jobs/submit_steer_pool.sh
#
# IT ALSO TAKES A COMMAND LINE, for the flag-gated templates (tts-cast-ablation-pool.yaml,
# tts-baseline-pool.yaml) and their thin entry points jobs/submit_{ablation,baseline}_pool.sh:
#
#   TEMPLATE=jobs/tts-cast-ablation-pool.yaml ./jobs/submit_steer_pool.sh --all --gpus 4 --local
#   jobs/submit_ablation_pool.sh --a2 --a3 --dry-run --env-file /path/env.sh --local
#
# Options this script owns: --gpus N, --targets "a b c", --run-dir DIR, --env-file FILE, --local,
# --dry-run. EVERY OTHER --flag IS FORWARDED VERBATIM in $POOL_FLAGS and is parsed by the TEMPLATE.
# That split is deliberate: a template is a shell body frozen into a job spec at submit time, so a
# parser living here could not travel with the job and the two could disagree about which arms run.
# An unknown arm flag is therefore rejected by the template, before any GPU is claimed. Passing no
# argv at all reproduces the old behaviour byte for byte, so every existing caller is unaffected.
#
# --local RUNS WORKERS IN THE CURRENT POD, one per GPU, instead of submitting jobs. That is how the
# production runs are actually launched here: a long-lived GPU pod, an --env-file that pins every
# cluster path, and one worker per card. It needs the sampler environment already set (CODE, PATH to
# the sampler venv, FOLDBENCH_ROOT, PROTENIX_COMPONENT_PATH, PROTENIX_ROOT_DIR, PROTENIX_CDR_MAP,
# INPUTS_WIRED, SCORE_PY, SCORE_DOCKQ, GT_INDEX, and whichever EPI_* maps the chosen arms read).
# the template fails loudly on anything missing rather than running with a silent default.
#
# NGPUS is the ONLY parallelism knob: it submits that many worker jobs, each holding exactly one GPU
# and draining the same shared queue of arms. Workers are independent, so they may start at different
# times, and the grid still completes -- the last worker to arrive simply takes whatever is left.
#
# WHY ONE GPU PER WORKER rather than one N-GPU job. An N-GPU job is atomic: the scheduler must free
# all N at once, so on a busy shared cluster it can sit unadmitted indefinitely while single-GPU jobs
# trickle in around it. N single-GPU workers acquire GPUs incrementally and start doing useful work
# as soon as the FIRST one is admitted.
#
# DEADLINE sizing. A worker writes its cifs only at the END of inference, so one cut off mid-unit
# loses all of it, and a deadline short enough to bite costs whole units. Set DEADLINE_S to the
# scheduler's own maximum execution time and let the WORKER decide when to stop: DEADLINE_S and
# UNIT_EST_S together make it retire cleanly rather than be killed mid-unit.
#
# SIZE NGPUS TO THE WORK. Count the units that have NOT started and submit one worker each. More
# workers than units just idle on a card or exit on a drained queue, fewer costs an extra round for
# nothing. The default below is a fallback for a caller that passes nothing, not a target, and a
# number that suited one campaign should not be carried to the next. Validate one unit before
# submitting a wave.
#
# RESUBMITTING IS SAFE AND IS THE INTENDED RECOVERY: claims are atomic and finished arms retire to
# done/, so a rerun never double-runs an arm and exits immediately if the queue is already drained.
set -uo pipefail

# ---- command line (all optional, with no argv this block sets nothing and the env-var path below
#      behaves exactly as it always has) ----
POOL_FLAGS="${POOL_FLAGS:-}"
_LOCAL=0; _ENV_FILE=""; _USAGE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus)       NGPUS="${2:?--gpus needs a number}"; shift 2 ;;
    --gpus=*)     NGPUS="${1#*=}"; shift ;;
    --targets)    TARGETS="${2:?--targets needs a list}"; shift 2 ;;
    --targets=*)  TARGETS="${1#*=}"; shift ;;
    --run-dir)    RUN_DIR="${2:?--run-dir needs a path}"; shift 2 ;;
    --run-dir=*)  RUN_DIR="${1#*=}"; shift ;;
    --env-file)   _ENV_FILE="${2:?--env-file needs a path}"; shift 2 ;;
    --env-file=*) _ENV_FILE="${1#*=}"; shift ;;
    --local)      _LOCAL=1; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    _USAGE=1; shift ;;
    # FORWARDED, NOT VALIDATED HERE -- the template owns the arm vocabulary. See the header.
    --*)          POOL_FLAGS="$POOL_FLAGS $1"; shift ;;
    *) echo "!! stray argument '$1' (options start with --)" >&2; exit 2 ;;
  esac
done
POOL_FLAGS="${POOL_FLAGS# }"
export POOL_FLAGS
[ -n "${DRY_RUN:-}" ] && export DRY_RUN
# SOURCED AFTER the argv, so a value given on the command line survives. An env file that used plain
# `export VAR=...` would still win here, which is why jobs/*-pool.yaml headers ask for the
# ${VAR:-default} form -- otherwise the run silently uses a different map than the operator asked for.
if [ -n "$_ENV_FILE" ]; then
  [ -f "$_ENV_FILE" ] || { echo "!! env file not found: $_ENV_FILE" >&2; exit 2; }
  . "$_ENV_FILE"
fi

# Size to the WORK: one worker per unstarted unit, capped at whatever GPU ceiling applies.
NGPUS="${NGPUS:-8}"
METHOD="${METHOD:-tds_epi}"
DEADLINE="${DEADLINE:-432000}"               # seconds; the cluster's own max-exec-time cap, see above
# What ONE unit costs, used ONLY by the worker's pre-claim check. Over-estimate on purpose:
# claiming too little wastes one slot, claiming too much wastes hours of GPU that writes nothing.
# 16000 s covers the slowest measured unit (2 seeds x 780 samples, 3.8 h on a 40 GB card).
UNIT_EST_S="${UNIT_EST_S:-16000}"
SUBMIT="${SUBMIT:-kubectl create -f -}"      # replace with your cluster's submit command
TEMPLATE="${TEMPLATE:-$(dirname "$0")/tts-steer-pool.yaml}"

[ -f "$TEMPLATE" ] || { echo "template not found: $TEMPLATE" >&2; exit 2; }
# VALIDATE AGAINST THE TEMPLATE, not a list kept here. A hardcoded allowlist drifts: this one still
# read "tds_epi|tds_cdr|svgd" after tds_sc and svgd_bw were added to the template, so the only way to
# launch either was to bypass this script.
#
# Read the arms of the `case "$METHOD" in` block SPECIFICALLY. Matching every `word)` in the file
# instead picked up unrelated case statements: against tts-tds-pool.yaml it reported the methods as
# "gt pred" (its per-arm epitope-map switch) and so rejected that template's own documented
# invocation, and against a template with no METHOD axis at all it invents one.
# A template that does not dispatch on $METHOD has no METHOD axis, so there is nothing to validate.
_methods=$(sed -n '/case "\$METHOD" in/,/esac/p' "$TEMPLATE" \
           | grep -oE '^[[:space:]]+[a-z_0-9|]+\)' | tr -d ' )' | tr '|' '\n' | sort -u | tr '\n' ' ')
if [ -n "$_methods" ]; then
  case " $_methods " in
    *" $METHOD "*) ;;
    *) echo "METHOD '$METHOD' is not a case in $TEMPLATE (have: $_methods)" >&2; exit 2 ;;
  esac
else
  echo "note: $(basename "$TEMPLATE") has no \$METHOD dispatch -- METHOD='$METHOD' is a label only"
fi

# Everything the template reads from the environment. Exported so a wrapper can override any of them
# without editing this script.
#
# THESE DEFAULTS ARE FOR THE LEGACY ENV-VAR TEMPLATES ONLY, AND SETTING THEM UNCONDITIONALLY WOULD BE
# A SILENT BUDGET OVERRIDE. tts-steer-pool.yaml and its siblings read n_seeds/n_samples from the
# environment and have no defaults worth the name, so these fill them in. The flag-gated templates
# are the opposite: they carry their own operating point (723 or 720 particles, K=1 or K=4) and
# derive the per-arm budget internally, so filling in N_SAMPLES=50 here would quietly run a 50-
# structure "ablation" that passed every gate and answered nothing. Skip them whenever the caller
# gave arm flags -- which is exactly the case where a template owns its own budget.
# RUN_DIR is gated for the same reason as the budget: each flag-gated template names its own root
# (./runs/abl_${METHOD}, ./runs/baseline_${METHOD}), and filling in the steer-pool name here would
# scatter results under a directory that says the wrong thing about what produced them.
[ -n "$POOL_FLAGS" ] || : "${RUN_DIR:=./runs/steer_${METHOD}}"
: "${TARGETS:=9atr_nb2}"
# EXPORTED, not just set: the --local path runs the body in this shell's environment, so a plain
# assignment would leave the template reading its own default instead of what was asked for.
export TARGETS METHOD
[ -n "${RUN_DIR:-}" ] && export RUN_DIR
if [ -z "$POOL_FLAGS" ]; then
  : "${N_SEEDS:=2}"
  : "${N_SAMPLES:=50}"
else
  # Pass through only if the caller set them; otherwise leave the template's own values alone.
  N_SEEDS="${N_SEEDS:-}"; N_SAMPLES="${N_SAMPLES:-}"
fi
# EXPORT CONDITIONALLY HERE, NEVER AS A `${VAR:+VAR=$VAR}` COMMAND PREFIX. A prefix is recognised as
# an assignment only BEFORE expansion, so a word that merely looks like one after expanding is taken
# as the COMMAND NAME, failing with "RUN_DIR=...: No such file or directory". The
# conditional export does the same job and cannot be misparsed.
[ -n "${N_SEEDS:-}" ] && export N_SEEDS
[ -n "${N_SAMPLES:-}" ] && export N_SAMPLES
export DEADLINE UNIT_EST_S

if [ "$_USAGE" = "1" ]; then
  sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

if [ -n "$POOL_FLAGS" ]; then
  echo "  POOL_FLAGS='$POOL_FLAGS'  (parsed by $(basename "$TEMPLATE"), not by this script)"
fi

# ---- --local: run the workers here, one per GPU, instead of submitting jobs ----
if [ "$_LOCAL" = "1" ]; then
  # Pull the shell body out with the YAML parser rather than sed, so an indentation change in the
  # template cannot silently truncate it.
  _BODY="${TMPDIR:-/tmp}/pool_body_$$.sh"
  python3 - "$TEMPLATE" "$_BODY" <<'PYX' || exit 2
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
body = d.get("command_script")
if not body:
    sys.exit("!! %s has no command_script" % sys.argv[1])
open(sys.argv[2], "w").write(body)
print("[body] %s -> %s (%d lines)" % (sys.argv[1], sys.argv[2], body.count("\n")))
PYX
  bash -n "$_BODY" || { echo "!! extracted body does not parse" >&2; exit 2; }
  LOG_DIR="${LOG_DIR:-./logs}"; mkdir -p "$LOG_DIR"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    # ONE worker, in the FOREGROUND. A dry run that forked would report success from here while the
    # checks it exists for were still running, which is worse than no check at all.
    echo "[dry-run] single foreground worker, no GPU work"
    WORKER_ID=0 N_WORKERS=1 CUDA_VISIBLE_DEVICES="" bash "$_BODY"
    exit $?
  fi
  # LOG NAME MUST BE UNIQUE PER POD, NOT PER WORKER_ID. WORKER_ID restarts at 0 in every pod and a
  # pool is normally drained by several pods against one RWX volume, so `${METHOD}_w0.log` would
  # collect every pod's worker 0 and their lines would interleave. That exact defect cost this
  # project two wrong diagnoses when the templates had it (a "contaminated control", then a
  # retraction). The AUTHORITY for what a unit did stays $RUN_DIR/<unit>/run.log regardless.
  _host="${HOSTNAME:-$(hostname)}"
  # GPU_IDS pins the workers to named physical cards instead of 0..NGPUS-1. On a pod whose other
  # cards are already draining another pool, the default would put a new worker on a busy GPU and
  # both would OOM, so a partially occupied pod could not be used at all. Comma separated, one per
  # worker: GPU_IDS=1 --gpus 1 runs a single worker on card 1.
  _gpu_ids="${GPU_IDS:-$(seq -s, 0 $((NGPUS - 1)))}"
  [ "$(echo "$_gpu_ids" | tr ',' '\n' | grep -c .)" = "$NGPUS" ] || {
    echo "!! GPU_IDS='$_gpu_ids' does not name exactly $NGPUS card(s)" >&2; exit 2; }
  echo "running $NGPUS worker(s) locally: METHOD=$METHOD RUN_DIR=${RUN_DIR:-<template default>}"
  for k in $(seq 0 $((NGPUS - 1))); do
    _dev=$(echo "$_gpu_ids" | cut -d, -f$((k + 1)))
    WORKER_ID="$k" N_WORKERS="$NGPUS" CUDA_VISIBLE_DEVICES="$_dev" \
      nohup bash "$_BODY" > "$LOG_DIR/${METHOD}_${_host}_w$k.log" 2>&1 &
    echo "  worker $k on GPU $_dev -> $LOG_DIR/${METHOD}_${_host}_w$k.log  (pid $!)"
    sleep 2      # stagger: the queue seed is idempotent but simultaneous mkdir storms are noisy
  done
  echo
  echo "watch progress:   ls ${RUN_DIR:-<run dir>}/queue/{pending,claimed,done,failed} | wc -l"
  exit 0
fi

echo "submitting $NGPUS single-GPU workers"
echo "  METHOD=$METHOD  TARGETS='$TARGETS'  N_SEEDS=${N_SEEDS:-<template>}  N_SAMPLES=${N_SAMPLES:-<template>}"
echo "  RUN_DIR=${RUN_DIR:-<template default>}  DEADLINE=${DEADLINE}s per worker"

for k in $(seq 0 $((NGPUS - 1))); do
  # WORKER_ID is only used for log naming and to pick the worker that writes the merged RESULT.txt.
  # It does NOT partition the work: every worker pulls from the same queue, so the grid self-balances
  # even if some workers never start. DockQ is scored per arm as it completes, by whichever worker
  # ran it, so no worker is special for scoring.
  WORKER_ID="$k" N_WORKERS="$NGPUS" DEADLINE_S="$DEADLINE" \
    "${RENDER:-cat}" "$TEMPLATE" | $SUBMIT || {
      echo "!! worker $k failed to submit" >&2; exit 1; }
  echo "  worker $k submitted"
done

echo
echo "watch progress:   ls ${RUN_DIR:-<template default>}/queue/{pending,claimed,done,failed} | wc -l"
echo "results:          ${RUN_DIR:-<template default>}/RESULT.txt   (rows: target tag div_conf div_site div_jac)"
echo "add capacity:     rerun with a larger NGPUS -- new workers join the same queue"
