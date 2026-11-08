#!/usr/bin/env bash
# Chained jobs for the two refined external datasets. Usage: ext2_jobs.sh <prm|m2f>
set -uo pipefail
cd "$(dirname "$0")/.."
PYV=/home/iacopog/miniconda3/envs/bmuq/bin/python
export PYTHONPATH=.
if [ "$1" = prm ]; then
  $PYV scripts/judge_prm800k.py google/gemma-3-12b-it 5000 && scripts/external_job.sh prm800k_claims 0.85
else
  $PYV scripts/perturb_minif2f.py google/gemma-3-12b-it /scratch/users/iacopog/leansteer/external/minif2f_ext/statements.jsonl /scratch/users/iacopog/leansteer/external/minif2f_ext/perturbed.jsonl && scripts/external_job.sh minif2f_ext 0.85
fi
echo "EXT2 $1 DONE $(date)"
