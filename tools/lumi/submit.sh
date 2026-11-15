#!/bin/bash -l
# Generic LUMI SLURM wrapper: one job = one `experiments/train.py` invocation. Pass the train.py
# args through verbatim; set --output / --job-name / --time on the sbatch call.
#
#   sbatch --job-name=icicle_em256 --output=/scratch/project_465003095/icicle/logs/em256.log --time=12:00:00 \
#     tools/lumi/submit.sh icicle --model.emsize 256 --run.checkpoint-dir runs/em256/ckpt
#
# Runs in the lean no-torch image (JAX 0.11.2 + prebuilt CK flash kernels); the model code
# stays pure JAX. `bash -c '...' _ "$@"` preserves argument boundaries.
#SBATCH --job-name=icicle
# Use small-g for single-GCD jobs.
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --account=project_465003095
# Write Slurm logs to /scratch (%x = job name, %j = job ID). Override per submission.
#SBATCH --output=/scratch/project_465003095/icicle/logs/%x_%j.out

set -euo pipefail
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

export PYTHONUNBUFFERED=1  # live logs; block-buffered stdout hides hours of progress
export PYTHONDONTWRITEBYTECODE=1  # frozen source trees must remain byte-identical while running

source /pfs/lustrep4/projappl/project_465003095/icicle/tools/lumi/lumi_env.sh
export ICICLE_SOURCE=${ICICLE_SOURCE:-$ICICLE}

if [[ ${ICICLE_SHARED_CACHE:-1} == 1 ]]; then
  export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$ICICLE_CACHE_ROOT}
else
  export JAX_COMPILATION_CACHE_DIR=/tmp/icicle-jax-cache-${ICICLE_IMAGE_KEY}-${SLURM_JOB_ID:-local}
fi
# Reserve the BFC pool up front; growing it can fragment large training allocations.
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-true}
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date); image: $ICICLE_SIF"
echo "Source: $ICICLE_SOURCE"
echo "Compilation cache: $JAX_COMPILATION_CACHE_DIR"
echo "Args: $*"

srun --export=ALL --ntasks=1 /usr/bin/singularity exec --bind "$ICICLE_SINGULARITY_BIND" "$ICICLE_SIF" bash -c '
  set -euo pipefail
  export VENV_DIR="$ICICLE_VENV"
  source "$ICICLE_LUMI_JAX/activate.sh"
  test "$(command -v python)" = "$ICICLE_VENV/bin/python"
  source "$ICICLE_LUMI_JAX/tools/ray_visible_devices.sh"
  export PYTHONPATH="$ICICLE_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
  cd "$ICICLE_SOURCE"
  exec python -m experiments.train "$@"
' _ "$@"
