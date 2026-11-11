#!/usr/bin/env bash
# Resume the af_ctx sycophancy steering on a shared-card slice; retries on OOM (the stage resumes per generated row).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/iacopog/miniconda3/envs/leansteer/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=. PATH=$HOME/.elan/bin:$PATH OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/context.yaml
S="--set steer.directions=[dm] steer.alphas=[-8,-4,4,8] steer.random_alphas=[-8,8] steer.batch_size=${STEER_BS:-3}"
for attempt in 1 2 3 4 5 6; do
  $PY -m leansteer.cli steer -c $CFG --tasks ${TASKS:-sycophancy} $S && { touch /scratch/users/iacopog/leansteer/runs/af_ctx/STEER_DONE; echo "STEER RESUME DONE $(date)"; exit 0; }
  echo "attempt $attempt failed at $(date); retrying in 5 min"; sleep 300
done
echo "STEER RESUME GAVE UP $(date)"; exit 1
