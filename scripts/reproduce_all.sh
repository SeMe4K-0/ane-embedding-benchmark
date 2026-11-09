#!/bin/bash
# Full measurement program as run for the paper. Takes about 8 hours on
# mains power; leave the machine idle. Each step can also be run on its own.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
MODELS="minilm-l6 minilm-l12 bge-small bge-base e5-base me5-small"

# 1. Core ML packages (fp16, static shapes; cached in models/).
$PY -m src.convert --models $MODELS --batches 1 32 --lengths 32 128 512
$PY -m src.convert --models minilm-l6 bge-base --batches 1 32 --lengths 128 --precision fp32

# 2. Check that the ANE path works.
$PY -m src.smoke --model minilm-l6

# 3. Planned placement.
$PY -m src.placement --models $MODELS

# 4. Measurement sessions; s1 is the reference for the readiness probe.
#    Let the machine cool down between sessions.
SKIP_PROBE=1 scripts/run_session.sh s1 configs/core.yaml 1 180
scripts/run_session.sh s3 configs/core.yaml 3 180
scripts/run_session.sh s4 configs/core.yaml 4 180
scripts/run_session.sh c1 configs/control.yaml 5 30

# 5. Corpus indexing with and without length buckets.
sudo powermetrics --samplers cpu_power,gpu_power,ane_power,thermal \
     -i 200 -n 12600 > results/raw/power_r1.log 2>/dev/null &
caffeinate -i $PY -m src.realistic --models minilm-l6 bge-base --session r1
sleep 2
$PY -m src.power --session r1

# 6. Embedding fidelity and retrieval quality on SciFact.
$PY -m src.fidelity --models $MODELS

# 7. Tables and figures.
$PY -m src.analyze --sessions s1 s3 s4 --figures
