#!/usr/bin/env bash
# CPU follow-ups for the context run; waits for scripts/context_job.sh to finish the probes.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/iacopog/miniconda3/envs/leansteer/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=. PATH=$HOME/.elan/bin:$PATH OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=
CFG=configs/context.yaml
until [ -f /scratch/users/iacopog/leansteer/runs/af_ctx/PROBES_DONE ]; do sleep 120; done
for ST in controls uncertainty abstain transfer baselines; do
  $PY -m leansteer.cli $ST -c $CFG
done
echo "CONTEXT CPU JOB DONE $(date)"
