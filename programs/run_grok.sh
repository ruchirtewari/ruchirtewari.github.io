#!/usr/bin/env bash
# run_grok.sh — replicate the grokking + Fisher-spectrum experiments.
#
# Runs the book's flagship experiment end to end, logged to logs/:
#   1. 08_02  grokking: 2-layer transformer on a+b mod 97, train-frac 0.3,
#             weight decay 1.0, 100k steps. Test accuracy plateaus near
#             chance (~0.30) then jumps to ~1.0 around step 5000-5500.
#   2. 08_04  Fisher spectrum inspector: same regime, 12k steps, top-8 true
#             Fisher eigenvalues probed every 400 steps via Lanczos over
#             Fisher-vector products.
#
# USAGE
#   ./run_grok.sh              run both (08_02 then 08_04), logged
#   ./run_grok.sh 08_02        run only the grokking transition
#   ./run_grok.sh 08_04        run only the Fisher spectrum inspector
#   ./run_grok.sh --quick      smoke mode for both (minutes; too short to grok)
#
# Runtime (CPU): 08_02 full ~1-2 h; 08_04 full ~1 h. Logs stream to
# logs/grok_<prog>_<timestamp>.log (symlinked logs/grok_<prog>_latest.log),
# figures to out/<program_name>/.
#
# EXPECTED RESULTS (full, seed 0)
#   08_02: "acceptance: delayed generalization present: True" — train
#          saturates by ~step 500, test stuck at ~0.30, grok at ~5000-5500
#          (delay ~5000 steps); embedding Fourier peaks sharpen DURING the
#          plateau.
#   08_04: "acceptance: Fisher spectrum tracked through a grokking run: True"
#          — reference run: top eigenvalue climbs through the plateau
#          (3.4 -> ~4e3), spikes ~10x to ~3e4 at step ~4800 (train acc
#          briefly dips: the memorized solution destabilizing), collapses as
#          test acc snaps up; grok at step 5600 with the largest spectral
#          movement at 5200 — the spectrum LEADS behavior by ~400 steps.
#          Dynamic range across the run ~9000x.
#   Exact step numbers vary with seed/hardware; the shape (plateau ->
#   spectral crisis -> behavioral jump) is the replicable result.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

QUICK=""; TARGETS=()
for a in "$@"; do
  case "$a" in
    -h|--help|help) awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "$0"; exit 0 ;;
    --quick) QUICK="--quick" ;;
    08_02|08_04) TARGETS+=("$a") ;;
    *) echo "unknown argument: $a (use -h)"; exit 1 ;;
  esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(08_02 08_04)

# venv via the standard runner setup (creates/installs if needed)
./run_tests.sh list >/dev/null 2>&1 || true
PYBIN=".venv/bin/python"
[ -x "$PYBIN" ] || PYBIN="$(command -v python3)"

mkdir -p logs
FAIL=0
for t in ${TARGETS[@]+"${TARGETS[@]}"}; do
  prog=$(ls ${t}_*.py 2>/dev/null | head -1)
  [ -z "$prog" ] && { echo "no program for $t"; exit 1; }
  LOG="logs/grok_${t}_$(date +%Y%m%d_%H%M%S).log"
  ln -sf "$(basename "$LOG")" "logs/grok_${t}_latest.log"
  echo "==> $prog ${QUICK:+(quick)} -> $LOG"
  if "$PYBIN" -u "$prog" $QUICK 2>&1 | tee "$LOG" | grep --line-buffered -E "^step|acceptance|final|grok|saved"; then
    echo "==> OK: $prog"
  else
    echo "==> FAILED: $prog (full log: $LOG)" >&2; FAIL=1
  fi
done
exit $FAIL
