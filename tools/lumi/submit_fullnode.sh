#!/bin/bash -l
# Full-node (standard-g) LUMI job: one process with all 8 GCDs visible.
# JAX shards the batch across them; batch_size must be divisible by 8.
#
#   sbatch --output=/scratch/project_465003095/icicle/logs/fn.log tools/lumi/submit_fullnode.sh icicle --data.batch-size 64
#SBATCH --job-name=icicle_fullnode
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=56
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --account=project_465003095
# Write Slurm logs to /scratch (%x = job name, %j = job ID). Override per submission.
#SBATCH --output=/scratch/project_465003095/icicle/logs/%x_%j.out

set -euo pipefail
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
ulimit -c 0  # a ROCm GPU fault otherwise dumps a multi-GB core + gpucore files into the workdir

source /pfs/lustrep4/projappl/project_465003095/icicle/tools/lumi/lumi_env.sh
export ICICLE_SOURCE=${ICICLE_SOURCE:-$ICICLE}

if [[ ${ICICLE_SHARED_CACHE:-1} == 1 ]]; then
  export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$ICICLE_CACHE_ROOT}
else
  export JAX_COMPILATION_CACHE_DIR=/tmp/icicle-jax-cache-${ICICLE_IMAGE_KEY}-${SLURM_JOB_ID:-local}
fi
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
# Full-node jobs must preallocate: without it the allocator's per-allocation mmaps saturate
# vm.max_map_count (65530) across 8 GCDs. Fraction of the 64 GB GCD that XLA owns.
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-true}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}

echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date); image: $ICICLE_SIF"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"
echo "Compilation cache: $JAX_COMPILATION_CACHE_DIR"
# Log address-space and memory-map limits to help diagnose compilation failures
# that occur despite sufficient available RAM.
echo "limits: ulimit -v=$(ulimit -v) -m=$(ulimit -m) -u=$(ulimit -u); vm.max_map_count=$(cat /proc/sys/vm/max_map_count 2>/dev/null || echo '?')"
echo "Args: $*"

srun --export=ALL --ntasks=1 /usr/bin/singularity exec --bind "$ICICLE_SINGULARITY_BIND" "$ICICLE_SIF" bash -c '
  set -euo pipefail
  export VENV_DIR="$ICICLE_VENV"
  source "$ICICLE_LUMI_JAX/activate.sh"
  test "$(command -v python)" = "$ICICLE_VENV/bin/python"
  source "$ICICLE_LUMI_JAX/tools/ray_visible_devices.sh"
  export PYTHONPATH="$ICICLE_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
  cd "$ICICLE_SOURCE"
  python -c "import os, jax; print(\"jax devices:\", jax.device_count(), \
\"| prealloc:\", os.environ.get(\"XLA_PYTHON_CLIENT_PREALLOCATE\"), \
\"| mem_fraction:\", os.environ.get(\"XLA_PYTHON_CLIENT_MEM_FRACTION\"))"
  exec python -m experiments.train "$@"
' _ "$@"
