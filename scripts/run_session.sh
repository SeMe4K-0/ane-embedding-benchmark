#!/bin/bash
# One measurement session: powermetrics log, measurement grid, energy attribution.
#
#   scripts/run_session.sh SESSION [CONFIG] [SEED] [MINUTES]
#   scripts/run_session.sh s5 configs/core.yaml 1 180
#
# Environment: SKIP_PROBE=1 skips the readiness probe; COOLDOWN_START=N waits
# N seconds before starting; PY overrides the Python interpreter.
set -uo pipefail
cd "$(dirname "$0")/.."

SESSION="${1:?session name required}"
CONFIG="${2:-configs/core.yaml}"
SEED="${3:-1}"
MINUTES="${4:-180}"
PY="${PY:-.venv/bin/python}"
LOG="results/raw/power_${SESSION}.log"
# powermetrics delivers about 4.2 samples/s at -i 200 on an M4 Pro, not 5,
# because formatting and writing the log take time. The sample count bounds
# the run so that the process exits on its own.
SAMPLES=$(( MINUTES * 60 * 42 / 10 ))

echo "session ${SESSION} | config ${CONFIG} | seed ${SEED} | log up to ${MINUTES} min"

case "$(pmset -g ps | head -1)" in
    *"Battery Power"*) echo "on battery power; connect the charger and retry"; exit 1;;
esac
echo "Low Power Mode: $(pmset -g | grep -i lowpowermode | tail -1 | xargs)"
echo "free disk: $(df -h . | tail -1 | awk '{print $4}')"

# A powermetrics process left over from an earlier session distorts CPU and
# GPU results (session s2 was excluded for this reason).
if pgrep -x powermetrics >/dev/null 2>&1; then
    echo "powermetrics is already running; stop it first: sudo pkill -x powermetrics"
    exit 1
fi

if [ "${COOLDOWN_START:-0}" -gt 0 ]; then
    echo "waiting ${COOLDOWN_START} s before start"
    sleep "${COOLDOWN_START}"
fi

# Readiness probe against the reference session s1 (about 20 s).
if [ "${SKIP_PROBE:-0}" != "1" ] && [ -f results/raw/s1/bench_power.jsonl ]; then
    RATIO=$($PY -m src.probe 2>/dev/null | awk '/^RATIO/{print $2}')
    if [ -z "${RATIO}" ] || [ "${RATIO}" = "nan" ]; then
        echo "probe gave no result; continuing"
    elif [ "$(awk -v r="${RATIO}" 'BEGIN{print (r>=0.95)?1:0}')" != "1" ]; then
        echo "probe ratio ${RATIO} < 0.95: close other applications and retry (SKIP_PROBE=1 to override)"
        exit 1
    else
        echo "probe ratio to s1: ${RATIO}"
    fi
fi

sudo -n powermetrics --help >/dev/null 2>&1 || sudo -v || exit 1
mkdir -p results/raw
sudo powermetrics --samplers cpu_power,gpu_power,ane_power,thermal \
     -i 200 -n "${SAMPLES}" > "${LOG}" 2>/dev/null &
PM_PID=$!
sleep 3
kill -0 "${PM_PID}" 2>/dev/null || { echo "powermetrics did not start"; exit 1; }
trap 'sudo -n kill "${PM_PID}" 2>/dev/null || true' INT TERM

caffeinate -i $PY -m src.bench --config "${CONFIG}" --session "${SESSION}" \
    --seed "${SEED}" --baseline 60 --cooldown 30
BENCH_RC=$?

sleep 2
sudo -n kill "${PM_PID}" 2>/dev/null \
    || echo "powermetrics will exit after ${SAMPLES} samples (or: sudo pkill -x powermetrics)"

$PY -m src.power --session "${SESSION}"
$PY -m src.analyze --sessions "${SESSION}" --realistic ""
echo "session ${SESSION} finished (bench exit code ${BENCH_RC})"
