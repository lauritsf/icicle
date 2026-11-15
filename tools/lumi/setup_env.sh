#!/bin/bash
# Create ICICLE's writable environment over the validated ROCm 7.2.4 lean image.
set -euo pipefail

ICICLE=/pfs/lustrep4/projappl/project_465003095/icicle
LUMI_JAX_WORK=/scratch/project_465003095/larsenla_cache
LUMI_JAX_LINE=${ICICLE_JAX_LINE:-0.11.2}
case "$LUMI_JAX_LINE" in
    0.11.0)
        LUMI_JAX=/pfs/lustrep4/projappl/project_465003095/lumi-jax-8a73b1d
        LUMI_JAX_COMMIT=8a73b1d2bd186045b4083eb2f1d1f42accd440df
        DEFAULT_VENV=/pfs/lustrep4/scratch/project_465003095/icicle/jax-env-rocm724-j0110-20261013
        SIF=/pfs/lustrep4/scratch/project_465003095/larsenla_cache/images/u24r72f21m50t211-20260929_104918/lumi-jax-lean-u24r72f21m50t211_c227j0110-20261013_142858.sif
        FLAX_PIN=0.12.8
        ;;
    0.11.2)
        LUMI_JAX=/pfs/lustrep4/projappl/project_465003095/lumi-jax-v0.2.0
        LUMI_JAX_COMMIT=b26e558d6e3e3a67c415aa149b0c5b8e2cc1a847
        DEFAULT_VENV=/pfs/lustrep4/scratch/project_465003095/icicle/jax-env-lumi-jax-v0.2.0
        SIF=/pfs/lustrep4/scratch/project_465003095/larsenla_cache/images/u24r72f21m50t211-20260929_104918/lumi-jax-lean-u24r72f21m50t211_c227j0112-20261023_151217.sif
        FLAX_PIN=0.12.10
        ;;
    *)
        echo "FATAL: ICICLE_JAX_LINE must be 0.11.0 or 0.11.2" >&2
        exit 2
        ;;
esac
VENV_DIR=${ICICLE_VENV_DIR:-$DEFAULT_VENV}

if [ -e "$VENV_DIR" ] && [ "${ICICLE_REBUILD_ENV:-0}" != 1 ]; then
    echo "FATAL: $VENV_DIR already exists; the lumi-jax installer clears VENV_DIR." >&2
    echo "Set ICICLE_VENV_DIR to a fresh path, or ICICLE_REBUILD_ENV=1 to rebuild it." >&2
    exit 1
fi
test "$(git -C "$LUMI_JAX" rev-parse HEAD)" = "$LUMI_JAX_COMMIT" || {
    echo "FATAL: lumi-jax is not at validated commit $LUMI_JAX_COMMIT" >&2
    exit 1
}
test "$(readlink -f "$SIF")" = "$SIF" || {
    echo "FATAL: SIF must point directly to an immutable image file" >&2
    exit 1
}

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export LUMI_JAX_WORK LUMI_JAX_LINE VENV_DIR SIF FLAX_PIN
(cd "$LUMI_JAX" && bash install.sh)

singularity exec \
    --bind "$ICICLE:$ICICLE,$LUMI_JAX:$LUMI_JAX,/scratch/project_465003095:/scratch/project_465003095" \
    "$SIF" bash -c '
set -euo pipefail
export VENV_DIR="'"$VENV_DIR"'"
export LUMI_KERNELS_LIB=/opt/lumi-jax/lumi_kernels/lib
source "'"$LUMI_JAX"'/activate.sh"
source "'"$LUMI_JAX"'/tools/ray_visible_devices.sh"
cd "'"$ICICLE"'"
python -m pip install -c "$VENV_DIR/lumi-jax-constraints.txt" \
    -e ".[train,ray,serving,export,experiments]" "flax==$FLAX_PIN" matplotlib pytest
python -m pip freeze > "$VENV_DIR/pip-freeze.txt"
python tools/lumi/validate_lumi_env.py --write-note
'

echo "ICICLE environment created at $VENV_DIR"
echo "Immutable image: $SIF"
