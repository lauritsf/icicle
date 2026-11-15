#!/bin/bash
# Validated LUMI runtime shared by ICICLE's Slurm entry points.

ICICLE=/projappl/project_465003095/icicle
ICICLE_HOST=/pfs/lustrep4/projappl/project_465003095/icicle
ICICLE_SCRATCH=/scratch/project_465003095/icicle
ICICLE_DATA=$ICICLE_SCRATCH/data
ICICLE_JAX_LINE=${ICICLE_JAX_LINE:-0.11.2}
case "$ICICLE_JAX_LINE" in
    0.11.0)
        ICICLE_LUMI_JAX=/projappl/project_465003095/lumi-jax-8a73b1d
        ICICLE_LUMI_JAX_HOST=/pfs/lustrep4/projappl/project_465003095/lumi-jax-8a73b1d
        ICICLE_LUMI_JAX_COMMIT=8a73b1d2bd186045b4083eb2f1d1f42accd440df
        ICICLE_VENV=/pfs/lustrep4/scratch/project_465003095/icicle/jax-env-rocm724-j0110-20261013
        ICICLE_SIF=/pfs/lustrep4/scratch/project_465003095/larsenla_cache/images/u24r72f21m50t211-20260929_104918/lumi-jax-lean-u24r72f21m50t211_c227j0110-20261013_142858.sif
        ICICLE_IMAGE_KEY=rocm724-lean-c227j0110-20261013_142858
        ;;
    0.11.2)
        ICICLE_LUMI_JAX=/projappl/project_465003095/lumi-jax-v0.2.0
        ICICLE_LUMI_JAX_HOST=/pfs/lustrep4/projappl/project_465003095/lumi-jax-v0.2.0
        ICICLE_LUMI_JAX_COMMIT=b26e558d6e3e3a67c415aa149b0c5b8e2cc1a847
        ICICLE_VENV=/pfs/lustrep4/scratch/project_465003095/icicle/jax-env-lumi-jax-v0.2.0
        ICICLE_SIF=/pfs/lustrep4/scratch/project_465003095/larsenla_cache/images/u24r72f21m50t211-20260929_104918/lumi-jax-lean-u24r72f21m50t211_c227j0112-20261023_151217.sif
        ICICLE_IMAGE_KEY=lumi-jax-v0.2.0-rocm724-j0112
        ;;
    *)
        echo "FATAL: ICICLE_JAX_LINE must be 0.11.0 or 0.11.2" >&2
        return 1
        ;;
esac
ICICLE_CACHE_ROOT=/flash/project_465003095/icicle/jax-cache/$ICICLE_IMAGE_KEY
mkdir -p "$ICICLE_CACHE_ROOT"
# OOD VS Code injects this into submitted environments; JAX mistakes it for distributed setup.
unset VSCODE_PROXY_URI

test -f "$ICICLE_SIF" || { echo "FATAL: missing ICICLE image: $ICICLE_SIF" >&2; return 1; }
test -f "$ICICLE_VENV/pyvenv.cfg" && test -L "$ICICLE_VENV/bin/python" || {
    echo "FATAL: missing ICICLE environment: $ICICLE_VENV" >&2
    return 1
}
test "$(cat "$ICICLE_VENV/lumi-jax-image")" = "$ICICLE_SIF" || {
    echo "FATAL: venv/image pin mismatch" >&2
    return 1
}
test "$(git -C "$ICICLE_LUMI_JAX_HOST" rev-parse HEAD)" = "$ICICLE_LUMI_JAX_COMMIT" || {
    echo "FATAL: lumi-jax checkout moved from validated commit $ICICLE_LUMI_JAX_COMMIT" >&2
    return 1
}

# Bind the immutable runtime inputs and ICICLE's canonical storage roots explicitly.
ICICLE_SINGULARITY_BIND="$ICICLE_HOST:$ICICLE,$ICICLE_LUMI_JAX_HOST:$ICICLE_LUMI_JAX"
ICICLE_SINGULARITY_BIND+=",$ICICLE_VENV:$ICICLE_VENV,$ICICLE_DATA:$ICICLE_DATA"
ICICLE_SINGULARITY_BIND+=",$ICICLE_CACHE_ROOT:$ICICLE_CACHE_ROOT"
ICICLE_SINGULARITY_BIND+=",/scratch/project_465003095:/scratch/project_465003095"
ICICLE_SINGULARITY_BIND+=",/flash/project_465003095:/flash/project_465003095"

export ICICLE ICICLE_DATA ICICLE_JAX_LINE ICICLE_LUMI_JAX ICICLE_VENV ICICLE_SIF
export ICICLE_IMAGE_KEY ICICLE_CACHE_ROOT ICICLE_SINGULARITY_BIND
export ICICLE_FLASH_PROVIDER=${ICICLE_FLASH_PROVIDER:-lumi_kernels.attention.flash_ffi}
export SINGULARITYENV_ICICLE_FLASH_PROVIDER=$ICICLE_FLASH_PROVIDER
export LUMI_KERNELS_LIB=/opt/lumi-jax/lumi_kernels/lib
export SINGULARITYENV_LUMI_KERNELS_LIB=$LUMI_KERNELS_LIB

# HIP command buffers stall or crash this workload stochastically and change its arithmetic.
# Keep the release instrument's explicit disabled setting across every shared launcher.
case " ${XLA_FLAGS:-} " in
    *" --xla_gpu_enable_command_buffer="*) ;;
    *) export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_command_buffer=" ;;
esac

