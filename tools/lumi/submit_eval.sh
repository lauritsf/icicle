#!/bin/bash -l
# One GPU evaluation command per job. Predictive evaluation is the default;
# ICICLE_EVAL_MODULE also selects experiments.hpo_regret or experiments.qtmeta_hpo.
# Pass the selected command's arguments through verbatim:
#
#   sbatch --job-name=eval_anchor --output=/scratch/project_465003095/icicle/logs/eval_anchor.log --time=02:00:00 \
#     tools/lumi/submit_eval.sh /scratch/.../runs/icicle --benchmark pd1_ifbo --out /scratch/.../eval.parquet
#
#SBATCH --job-name=icicle_eval
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --account=project_465003095
# Write Slurm logs to /scratch (%x = job name, %j = job ID). Override per submission.
#SBATCH --output=/scratch/project_465003095/icicle/logs/%x_%j.out

set -euo pipefail
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export ICICLE_BENCH_DATA=${ICICLE_BENCH_DATA:-/scratch/project_465003095/icicle/data}
export ICICLE_EVAL_MODULE=${ICICLE_EVAL_MODULE:-experiments.eval}
case "$ICICLE_EVAL_MODULE" in
  experiments.eval) ;;
  experiments.hpo_regret|experiments.qtmeta_hpo)
    export AMD_SERIALIZE_KERNEL=${AMD_SERIALIZE_KERNEL:-3}
    export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-true} ;;
  *) echo "Unsupported evaluation module: $ICICLE_EVAL_MODULE" >&2; exit 2 ;;
esac

source /pfs/lustrep4/projappl/project_465003095/icicle/tools/lumi/lumi_env.sh
export ICICLE_SOURCE=${ICICLE_SOURCE:-$ICICLE}

if [[ ${ICICLE_SHARED_CACHE:-1} == 1 ]]; then
  export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$ICICLE_CACHE_ROOT}
else
  export JAX_COMPILATION_CACHE_DIR=/tmp/icicle-jax-cache-${ICICLE_IMAGE_KEY}-${SLURM_JOB_ID:-local}
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "Compilation cache: $JAX_COMPILATION_CACHE_DIR"
echo "Source: $ICICLE_SOURCE; module: $ICICLE_EVAL_MODULE"
echo "Args: $*"

srun --export=ALL --ntasks=1 /usr/bin/singularity exec --bind "$ICICLE_SINGULARITY_BIND" "$ICICLE_SIF" bash -c '
  set -euo pipefail
  export VENV_DIR="$ICICLE_VENV"
  source "$ICICLE_LUMI_JAX/activate.sh"
  test "$(command -v python)" = "$ICICLE_VENV/bin/python"
  source "$ICICLE_LUMI_JAX/tools/ray_visible_devices.sh"
  export PYTHONPATH="$ICICLE_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
  cd "$ICICLE_SOURCE"
  exec python -m "$ICICLE_EVAL_MODULE" "$@"
' _ "$@"
