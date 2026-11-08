#!/usr/bin/env bash
# Full pipeline on one GPU.  Usage: scripts/run_all.sh [config] [gpu]
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${1:-configs/default.yaml}
export CUDA_VISIBLE_DEVICES=${2:-3}
PY=${PY:-/home/iacopog/miniconda3/envs/leansteer/bin/python}
$PY -m leansteer.cli download -c "$CFG"
$PY -m leansteer.cli collect -c "$CFG"
$PY -m leansteer.cli probe   -c "$CFG"
$PY -m leansteer.cli steer   -c "$CFG"
$PY -m leansteer.cli report  -c "$CFG"
