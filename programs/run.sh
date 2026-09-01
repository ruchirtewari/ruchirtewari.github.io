#!/usr/bin/env bash
# run_tests.sh — isolated setup + selective runner for the book programs.
#
# Successor to run.sh: runs programs individually, by chapter, or all, inside
# a guarded virtualenv. With no arguments it prints this help.
#
# USAGE
#   ./run_tests.sh                 show this help (same as help / -h)
#   ./run_tests.sh list            list all programs in book order
#   ./run_tests.sh all             run every program
#   ./run_tests.sh 03              run all chapter-3 programs
#   ./run_tests.sh 02_01           run one program (prefix match: chapter 2, section 1)
#   ./run_tests.sh 05_02 --show    trailing args after the target pass through to python
#
# SETUP (done automatically before the first run of a session)
#   1. create ./.venv if missing
#   2. activate it
#   3. pip install -r requirements.txt  ONLY if the venv is verified active
#      (never installs into system Python)
#   Setup is skipped automatically once the venv has the deps; force it off
#   with --no-setup, or force reinstall with --setup.
#
# OPTIONS
#   --quick        pass --quick to every program (fast smoke mode)
#   --setup        force venv creation + pip install before running
#   --no-setup     skip venv setup entirely (use current interpreter)
#   -h, help       show this help
#
# Program names are CC_SS_title.py (chapter, section) so `ls` lists them in
# book order. Figures/artifacts are written under out/<program>/.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=".venv"
PYTHON="${PYTHON:-python3}"

# ---------------------------------------------------------------- helpers ---
# print only the contiguous comment header (stop at first non-comment line)
show_help() { awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "$0"; }

list_programs() { ls [0-9][0-9]_[0-9][0-9]_*.py 2>/dev/null | sort; }

venv_active() {
  local abs; abs="$(cd "$VENV" 2>/dev/null && pwd)" || return 1
  [ "${VIRTUAL_ENV:-}" = "$abs" ] && [ "$(command -v python)" = "$abs/bin/python" ]
}

deps_present() { python -c "import torch, transformers, numpy, matplotlib" 2>/dev/null; }

setup_venv() {   # $1 = force (1) or auto (0)
  local force="$1"
  if [ ! -x "$VENV/bin/python" ]; then
    echo "==> creating $VENV with $PYTHON"
    "$PYTHON" -m venv "$VENV" || { echo "ERROR: venv creation failed" >&2; exit 1; }
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  if ! venv_active; then
    echo "ERROR: venv did not activate; refusing to touch system Python." >&2
    exit 1
  fi
  if [ "$force" = "1" ] || ! deps_present; then
    echo "==> installing requirements into the venv (not system Python)"
    python -m pip install --quiet --upgrade pip
    python -m pip install -r requirements.txt
  fi
}

PYBIN="python3"   # resolved after setup: venv 'python' if active, else python3
PASS=(); FAIL=()
run_one() {
  local f="$1"; shift
  echo
  echo "======== $f ========"
  if "$PYBIN" "$f" "$@"; then PASS+=("$f"); else
    echo "FAILED: $f" >&2; FAIL+=("$f"); fi
}

# ---------------------------------------------------------------- parse ------
# Split args into: target (first non-flag), setup flags, and pass-through args.
TARGET=""; SETUP="auto"; QUICK=""; PASS_ARGS=()
for a in "$@"; do
  case "$a" in
    -h|--help|help) show_help; exit 0 ;;
    --setup)    SETUP="force" ;;
    --no-setup) SETUP="off" ;;
    --quick)    QUICK="--quick" ;;
    list|all|[0-9][0-9]|[0-9][0-9]_[0-9]*)
      if [ -z "$TARGET" ]; then TARGET="$a"; else PASS_ARGS+=("$a"); fi ;;
    *) if [ -z "$TARGET" ]; then TARGET="$a"; else PASS_ARGS+=("$a"); fi ;;
  esac
done
[ -n "$QUICK" ] && PASS_ARGS+=("$QUICK")

# no target -> help
[ -z "$TARGET" ] && { show_help; exit 0; }

# list needs no venv
if [ "$TARGET" = "list" ]; then list_programs; exit 0; fi

# ---------------------------------------------------------------- setup ------
case "$SETUP" in
  force) setup_venv 1 ;;
  auto)  setup_venv 0 ;;
  off)   [ -x "$VENV/bin/python" ] && source "$VENV/bin/activate" || true
         echo "==> --no-setup: using $(command -v python || command -v python3)" ;;
esac
# resolve interpreter: prefer venv 'python', else python3
PYBIN="$(command -v python || command -v python3)"

# ---------------------------------------------------------------- select -----
case "$TARGET" in
  all)
    for f in $(list_programs); do run_one "$f" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}; done ;;
  [0-9][0-9])
    matches=$(list_programs | grep "^${TARGET}_" || true)
    [ -z "$matches" ] && { echo "no programs for chapter $TARGET"; exit 1; }
    for f in $matches; do run_one "$f" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}; done ;;
  *)
    matches=$(list_programs | grep "^${TARGET}" || true)
    n=$(printf '%s\n' "$matches" | grep -c . || true)
    if [ -z "$matches" ]; then
      echo "no program matches '$TARGET' (try ./run_tests.sh list)"; exit 1
    elif [ "$n" -gt 1 ]; then
      echo "ambiguous prefix '$TARGET':"; printf '%s\n' "$matches"; exit 1
    fi
    run_one "$matches" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"} ;;
esac

# ---------------------------------------------------------------- summary ----
echo
echo "================ summary ================"
echo "passed: ${#PASS[@]}   failed: ${#FAIL[@]}"
if [ ${#FAIL[@]} -gt 0 ]; then
  printf 'failed: %s\n' "${FAIL[*]}" >&2; exit 1
fi
echo "ok."
