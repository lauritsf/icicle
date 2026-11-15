"""LCBench tabular adapter (35 OpenML tasks; 2000 configs x 51 epochs each)."""

import numpy as np
import pandas as pd

from reproduction.bench._hp_spaces import LCBENCH_SPACE
from reproduction.bench.task import BenchmarkTask, hp_matrix, pivot, resolve_data_dir


def load_lcbench(task_id: str, data_dir=None) -> BenchmarkTask:
    """Load validation balanced accuracy at epochs 1..51, matching the ifBO reference."""
    path = resolve_data_dir(data_dir) / "lcbench-tabular" / f"{task_id}.parquet"
    mat, epochs, first_rows = pivot(pd.read_parquet(path), "val_balanced_accuracy")
    mat, epochs = mat[:, 1:], epochs[1:]
    keep = np.isfinite(mat).all(axis=1)
    curves = np.clip(mat[keep], 0.0, 1.0).astype(np.float32)
    hps = hp_matrix(first_rows[keep], LCBENCH_SPACE)
    fidelities = (epochs / epochs.max()).astype(np.float32)
    return BenchmarkTask(f"lcbench/{task_id}", hps, curves, fidelities)


def lcbench_task_ids(data_dir=None) -> list[str]:
    """All available task ids (data-driven: the parquet files present)."""
    return sorted(
        p.stem for p in (resolve_data_dir(data_dir) / "lcbench-tabular").glob("*.parquet")
    )
