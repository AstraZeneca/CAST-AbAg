#!/usr/bin/env bash
# Provision the parts `poetry install` deliberately leaves out: a co-folding backend and its
# weights, the DockQ scoring venv, and HMMER for the CDR map.
#
#   ./setup_backends.sh --protenix            # Protenix v2 venv + weights
#   ./setup_backends.sh --opendde             # OpenDDE venv + abag checkpoint
#   ./setup_backends.sh --both --scoring --cdr
#   ./setup_backends.sh --all --prefix /data/tts
#   ./setup_backends.sh --all --dry-run
#
# EVERY STEP IS SKIPPED IF ITS TARGET ALREADY EXISTS, so re-running is safe and cheap. It never
# deletes or overwrites; to rebuild a component, remove its directory first.
#
# It prints the env-file exports at the end. Nothing is written outside --prefix.
set -uo pipefail

PREFIX="${PREFIX:-$PWD/.backends}"
DO_PROTENIX=0; DO_OPENDDE=0; DO_SCORING=0; DO_CDR=0; DRY=0
PY="${PYTHON:-python3}"

while [ $# -gt 0 ]; do
  case "$1" in
    --protenix) DO_PROTENIX=1 ;;
    --opendde)  DO_OPENDDE=1 ;;
    --both)     DO_PROTENIX=1; DO_OPENDDE=1 ;;
    --scoring)  DO_SCORING=1 ;;
    --cdr)      DO_CDR=1 ;;
    --all)      DO_PROTENIX=1; DO_OPENDDE=1; DO_SCORING=1; DO_CDR=1 ;;
    --prefix)   PREFIX="${2:?--prefix needs a path}"; shift ;;
    --prefix=*) PREFIX="${1#*=}" ;;
    --dry-run)  DRY=1 ;;
    -h|--help)  awk 'NR>1 && /^#/{sub(/^# ?/,""); print; next} NR>1{exit}' "$0"; exit 0 ;;
    *) echo "!! unknown flag: $1. See --help." >&2; exit 2 ;;
  esac
  shift
done
[ $((DO_PROTENIX + DO_OPENDDE + DO_SCORING + DO_CDR)) -eq 0 ] && {
  echo "!! choose at least one of --protenix --opendde --both --scoring --cdr --all" >&2; exit 2; }

say()  { printf '  %s\n' "$*"; }
skip() { printf '  SKIP  %s\n' "$*"; }
run()  { if [ "$DRY" = 1 ]; then printf '  would run: %s\n' "$*"; else "$@"; fi; }
have() { [ -e "$1" ]; }
# A VENV WHOSE INSTALL FAILED STILL HAS bin/python, so testing for that skips a broken tree
# forever. Every skip below tests the artefact the step exists to produce instead.
has_pkg() { compgen -G "$1/lib/python*/site-packages/$2" >/dev/null 2>&1; }

PREFIX="$(mkdir -p "$PREFIX" 2>/dev/null; cd "$PREFIX" && pwd)" || { echo "!! cannot use --prefix"; exit 2; }
echo "prefix: $PREFIX"
FAILED=0

# ---------------------------------------------------------------- Protenix v2
PROTENIX_VENV="$PREFIX/protenix/.venv"
PROTENIX_CACHE="$PREFIX/protenix/cache"
if [ $DO_PROTENIX = 1 ]; then
  echo "== Protenix v2 =="
  if has_pkg "$PROTENIX_VENV" protenix; then
    skip "protenix installed: $PROTENIX_VENV"
  else
    run "$PY" -m venv "$PROTENIX_VENV" \
      && run "$PROTENIX_VENV/bin/pip" install -q --upgrade pip \
      && run "$PROTENIX_VENV/bin/pip" install -q "protenix==2.0.0" \
      || { echo "  !! protenix install failed"; FAILED=1; }
    has_pkg "$PROTENIX_VENV" protenix && say "installed: $PROTENIX_VENV"
  fi
  # WEIGHTS ARE NOT FETCHED HERE. The `protenix` package pulls them on first inference from its own
  # origin, which is the supported path. Pre-placing them is also supported and is what a network
  # that cannot reach that host needs. models/protenix_provision.py lists every artefact, the
  # layout, and a third-party mirror with the digest to check and the weights_only caveat.
  CKPT="$PROTENIX_CACHE/checkpoint/protenix-v2.pt"
  if have "$CKPT"; then
    skip "checkpoint present: $CKPT"
  else
    run mkdir -p "$PROTENIX_CACHE/checkpoint" "$PROTENIX_CACHE/common"
    say "checkpoint NOT present. Either let the first run fetch it, or place it by hand at:"
    say "    $CKPT"
    say "  origin  https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix-v2.pt"
    say "  mirror  https://huggingface.co/TMF001/protenix-v2-weights  (audit it, see"
    say "          models/protenix_provision.py for the digest and the weights_only caveat)"
  fi
fi

# ---------------------------------------------------------------- OpenDDE
OPENDDE_ROOT="$PREFIX/opendde"
OPENDDE_VENV="$OPENDDE_ROOT/.venv"
if [ $DO_OPENDDE = 1 ]; then
  echo "== OpenDDE =="
  # PINNED TO 1.1.0, as models/opendde.py requires: upstream calls this a preview release whose
  # CLI flags and checkpoints may change, and those are the attach points the hooks depend on.
  if have "$OPENDDE_VENV/bin/opendde"; then
    skip "console script exists: $OPENDDE_VENV/bin/opendde"
  else
    run "$PY" -m venv "$OPENDDE_VENV" \
      && run "$OPENDDE_VENV/bin/pip" install -q --upgrade pip \
      && run "$OPENDDE_VENV/bin/pip" install -q \
           --extra-index-url https://download.pytorch.org/whl/cu126 "opendde[gpu]==1.1.0" \
      || { echo "  !! opendde install failed"; FAILED=1; }
    have "$OPENDDE_VENV/bin/opendde" && say "installed: $OPENDDE_VENV/bin/opendde"
  fi
  # WEIGHTS COME FROM THE PROJECT'S OWN HUGGING FACE REPO, so unlike the Protenix third-party
  # mirror they are fetched automatically. Set OPENDDE_NO_FETCH=1 to place them by hand instead.
  ODDE_CKPT="$OPENDDE_ROOT/checkpoint/opendde_abag.pt"
  ODDE_URL="https://huggingface.co/aurekaresearch/OpenDDE/resolve/main/opendde_abag.pt"
  if have "$ODDE_CKPT"; then
    skip "abag checkpoint present: $ODDE_CKPT"
  elif [ -n "${OPENDDE_NO_FETCH:-}" ]; then
    run mkdir -p "$OPENDDE_ROOT/checkpoint"
    say "OPENDDE_NO_FETCH set. Place opendde_abag.pt at $ODDE_CKPT (from $ODDE_URL)"
  else
    run mkdir -p "$OPENDDE_ROOT/checkpoint"
    say "fetching opendde_abag.pt ..."
    if [ "$DRY" = 1 ]; then
      say "  would download $ODDE_URL -> $ODDE_CKPT"
    elif curl -fsSL --retry 2 -o "$ODDE_CKPT.part" "$ODDE_URL" && [ -s "$ODDE_CKPT.part" ]; then
      mv "$ODDE_CKPT.part" "$ODDE_CKPT"
      say "  downloaded: $ODDE_CKPT ($(du -h "$ODDE_CKPT" | cut -f1))"
    else
      rm -f "$ODDE_CKPT.part"
      say "  download FAILED, that host may be unreachable here. Place it by hand:"
      say "    $ODDE_CKPT   from  $ODDE_URL"
    fi
  fi
fi

# ---------------------------------------------------------------- DockQ
# A SEPARATE VENV, NOT AN EXTRA. dockq pins numpy<2 while this package runs numpy 2.x, so the two
# cannot share an environment. pyproject.toml records the same reason.
SCORE_VENV="$PREFIX/venv_score"
if [ $DO_SCORING = 1 ]; then
  echo "== DockQ =="
  if have "$SCORE_VENV/bin/DockQ"; then
    skip "DockQ present: $SCORE_VENV/bin/DockQ"
  else
    run "$PY" -m venv "$SCORE_VENV" \
      && run "$SCORE_VENV/bin/pip" install -q --upgrade pip \
      && run "$SCORE_VENV/bin/pip" install -q "dockq==2.1.3" pandas pyarrow \
      || { echo "  !! dockq install failed"; FAILED=1; }
    have "$SCORE_VENV/bin/DockQ" && say "installed: $SCORE_VENV/bin/DockQ"
  fi
fi

# ---------------------------------------------------------------- HMMER
# ANARCI IS NOT INSTALLED HERE. It is a poetry dependency, `poetry install --with cdr`. 
# What poetry cannot supply is HMMER, the C program, so that is all this step resolves.
#
# AND IT MUST BE THE REAL BINARY. The PyPI wheel's console scripts are wrappers, and HMMER writes
# argv[0] into its output header, so Biopython then matches no program, leaves every alignment
# coordinate None, and ANARCI dies comparing them. A system hmmscan is used when there is one; the
# wheel's bundled executables are the fallback, and those are real binaries.
HMMER_VENV="$PREFIX/venv_hmmer"
HMMER_BIN=""
if [ $DO_CDR = 1 ]; then
  echo "== HMMER =="
  SYS_HMMSCAN="$(command -v hmmscan 2>/dev/null || true)"
  # A wrapper starts with a shebang, a real binary with ELF's magic byte. That one byte is the test.
  if [ -n "$SYS_HMMSCAN" ] && [ "$(head -c1 "$SYS_HMMSCAN" 2>/dev/null)" != "#" ]; then
    HMMER_BIN="$(dirname "$SYS_HMMSCAN")"
    skip "system hmmscan is a real binary: $SYS_HMMSCAN"
  else
    [ -n "$SYS_HMMSCAN" ] && say "hmmscan on PATH is a wrapper script, ignoring it: $SYS_HMMSCAN"
    if has_pkg "$HMMER_VENV" hmmer; then
      skip "hmmer wheel installed: $HMMER_VENV"
    else
      run "$PY" -m venv "$HMMER_VENV" \
        && run "$HMMER_VENV/bin/pip" install -q --upgrade pip \
        && run "$HMMER_VENV/bin/pip" install -q hmmer \
        || { echo "  !! hmmer install failed"; FAILED=1; }
    fi
    if [ "$DRY" = 0 ] && has_pkg "$HMMER_VENV" hmmer; then
      HMMER_BIN="$("$HMMER_VENV/bin/python" -c 'import hmmer,os;print(os.path.join(os.path.dirname(hmmer.__file__),"data","bin"))' 2>/dev/null)"
      if [ -x "$HMMER_BIN/hmmscan" ]; then
        say "bundled hmmscan: $HMMER_BIN/hmmscan"
      else
        HMMER_BIN=""
        say "no bundled hmmscan. Install HMMER from conda, apt or source and pass its bin dir."
      fi
    fi
  fi
  say "ANARCI itself comes from: poetry install --with cdr"
fi

# ---------------------------------------------------------------- what to export
echo
echo "# env-file lines for what is now on disk"
[ $DO_PROTENIX = 1 ] && { echo "export PROTENIX_COMPONENT_PATH=$PROTENIX_VENV"; echo "export PROTENIX_ROOT_DIR=$PROTENIX_CACHE"; }
[ $DO_OPENDDE  = 1 ] && { echo "export OPENDDE_ROOT_DIR=$OPENDDE_ROOT"; echo "export OPENDDE_BIN=$OPENDDE_VENV/bin/opendde"; }
[ $DO_SCORING  = 1 ] && { echo "export SCORE_PY=$SCORE_VENV/bin/python"; echo "export SCORE_DOCKQ=$SCORE_VENV/bin/DockQ"; }
[ $DO_CDR = 1 ] && [ -n "${HMMER_BIN:-}" ] && echo "# token_maps cdr-map --hmmer-path $HMMER_BIN"
echo "# MSA_DB is yours to point at a ColabFold-style DB, or build one with generate_msa."

exit $FAILED
