"""Load the 12 IMDb tasks and normalization used by the ifBO reference."""

import numpy as np
import pandas as pd

from reproduction.bench._hp_spaces import TASKSET_SPACE_4P, TASKSET_SPACE_8P
from reproduction.bench.task import BenchmarkTask, hp_matrix, pivot, resolve_data_dir

TASKSET_OPTIMIZERS = ["adam4p_wide_grid_1k", "adam8p_wide_grid_1k"]

# ifBO's own 12 (src/pfns_hpo/pfns_hpo/configs/benchmark/taskset-tabular-normalized-nlp-*.yaml on
# the icml-2024 branch) -- the only taskset keys their benchmarks.md lists. All imdb.
TASKSET_IFBO_TASKS = [
    "FixedTextRNNClassification_imdb_patch128_LSTM128_avg_bs64",
    "FixedTextRNNClassification_imdb_patch128_LSTM128_bs64",
    "FixedTextRNNClassification_imdb_patch128_LSTM128_embed128_bs64",
    "FixedTextRNNClassification_imdb_patch32_GRU128_bs128",
    "FixedTextRNNClassification_imdb_patch32_GRU64_avg_bs128",
    "FixedTextRNNClassification_imdb_patch32_IRNN64_relu_avg_bs128",
    "FixedTextRNNClassification_imdb_patch32_IRNN64_relu_last_bs128",
    "FixedTextRNNClassification_imdb_patch32_LSTM128_E128_bs128",
    "FixedTextRNNClassification_imdb_patch32_LSTM128_bs128",
    "FixedTextRNNClassification_imdb_patch32_VRNN128_tanh_bs128",
    "FixedTextRNNClassification_imdb_patch32_VRNN64_relu_avg_bs128",
    "FixedTextRNNClassification_imdb_patch32_VRNN64_tanh_avg_bs128",
]


def load_taskset(task_id: str, optimizer: str, data_dir=None) -> BenchmarkTask:
    """Normalize validation loss and remove initialization.

    Clamp losses to [0, median initialization loss], then convert to higher-is-better
    responses. Remove initialization and shift the epoch grid to 1..50, following
    process_taskset_mfpbench_with_step_0_prior in ifBO's icml-2024 branch.
    """
    name = f"{task_id}-{optimizer}_10000_replica5.parquet"
    path = resolve_data_dir(data_dir) / "taskset-tabular" / name
    mat, epochs, first_rows = pivot(pd.read_parquet(path), "valid1_loss")
    mat = np.where(np.isfinite(mat), mat, np.inf)
    upper = float(np.median(mat[:, 0]))
    clamped = np.minimum(np.where(np.isfinite(mat), mat, upper), upper)
    clamped = np.maximum(clamped, 0.0)
    curves = (1.0 - clamped / max(upper, 1e-12)).astype(np.float32)
    curves, epochs = curves[:, 1:], epochs[1:] - 1
    space = TASKSET_SPACE_8P if optimizer.startswith("adam8p") else TASKSET_SPACE_4P
    hps = hp_matrix(first_rows, space)
    fidelities = (epochs / epochs.max()).astype(np.float32)
    return BenchmarkTask(f"taskset/{task_id}-{optimizer}", hps, curves, fidelities)
