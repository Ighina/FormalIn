#!/usr/bin/env bash
# DeepSeek-Prover pass over the af_ctx steered generations (whole card); waits for scripts/context_job.sh to finish steering.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/iacopog/miniconda3/envs/leansteer/bin/python
PYV=/home/iacopog/miniconda3/envs/bmuq/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=. PATH=$HOME/.elan/bin:$PATH OMP_NUM_THREADS=8
CFG=configs/context.yaml
until [ -f /scratch/users/iacopog/leansteer/runs/af_ctx/STEER_DONE ]; do sleep 300; done
$PYV -m leansteer.cli prove-gen    -c $CFG --set prove.gpu_memory_utilization=${GPU_UTIL:-0.85}
$PY  -m leansteer.cli prove-verify -c $CFG --set prove.lean_workers=32
echo "CONTEXT PROVE JOB DONE $(date)"
