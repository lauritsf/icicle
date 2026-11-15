#!/bin/bash -l
# CPU wrapper for tests and cache-only paper renderers:
# `sbatch tools/lumi/submit_cpu.sh <script> [args]`.
#
# Run figure renderers and tests on CPU with enough host memory for JAX compilation.
#
# CPU jobs use project_465003095. Project_465002594 has no CPU submission
# allowance; using it causes AssocMaxSubmitJobLimit regardless of queue depth.
#SBATCH --job-name=icicle_cpu
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --account=project_465003095
#SBATCH --output=/scratch/project_465003095/icicle/logs/%x_%j.out

set -euo pipefail
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
export PYTHONUNBUFFERED=1
export ICICLE_BENCH_DATA=${ICICLE_BENCH_DATA:-/scratch/project_465003095/icicle/data}
export JAX_PLATFORMS=cpu
ulimit -c 0

source /pfs/lustrep4/projappl/project_465003095/icicle/tools/lumi/lumi_env.sh
export ICICLE_SOURCE=${ICICLE_SOURCE:-$ICICLE}
if [[ ${ICICLE_SHARED_CACHE:-1} == 1 ]]; then
  export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$ICICLE_CACHE_ROOT}
else
  export JAX_COMPILATION_CACHE_DIR=/tmp/icicle-jax-cache-${ICICLE_IMAGE_KEY}-${SLURM_JOB_ID:-local}
fi
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "Args: $*"
srun --export=ALL --ntasks=1 /usr/bin/singularity exec --bind "$ICICLE_SINGULARITY_BIND" "$ICICLE_SIF" bash -c '
  set -euo pipefail
  export VENV_DIR="$ICICLE_VENV"
  source "$ICICLE_LUMI_JAX/activate.sh"
  test "$(command -v python)" = "$ICICLE_VENV/bin/python"
  source "$ICICLE_LUMI_JAX/tools/ray_visible_devices.sh"
  cd "$ICICLE_SOURCE"
  export PYTHONPATH="$ICICLE_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
  exec python "$@"
' _ "$@"
