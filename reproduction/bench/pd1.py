"""PD1 tabular adapter and the task selections used in our experiments."""

import numpy as np
import pandas as pd

from reproduction.bench._hp_spaces import PD1_SPACE
from reproduction.bench.task import BenchmarkTask, hp_matrix, pivot, resolve_data_dir

# Our submitted comparison uses the 28 entries listed in src/section5.2/plot.sh
# on its icml-2024 branch. Several underlying tasks appear at multiple
# subsampling levels.
# The uniref50 '-1' suffix in upstream configuration names is not a coarseness.
PD1_IFBO_TASKS = [
    ("cifar100", "wide_resnet", 2048, None),
    ("cifar100", "wide_resnet", 256, None),
    ("cifar10", "wide_resnet", 2048, None),
    ("cifar10", "wide_resnet", 256, None),
    ("fashion_mnist", "simple_cnn", 2048, None),
    ("fashion_mnist", "simple_cnn", 256, None),
    ("lm1b", "transformer", 2048, None),
    ("mnist", "simple_cnn", 2048, None),
    ("mnist", "simple_cnn", 256, None),
    ("svhn_no_extra", "wide_resnet", 1024, None),
    ("svhn_no_extra", "wide_resnet", 256, None),
    ("uniref50", "transformer", 128, None),
] + [
    (ds, model, bs, c)
    for ds, model, bs in (
        ("imagenet", "resnet", 256),
        ("imagenet", "resnet", 512),
        ("imagenet", "resnet", 1024),
        ("translate_wmt", "xformer_translate", 64),
    )
    for c in (1, 2, 5, 10)
]


def load_pd1(
    dataset: str, model: str, batch_size: int, coarseness: int | None = None, data_dir=None
) -> BenchmarkTask:
    """Curve = 1 - valid_error_rate; configs with missing or non-finite entries are dropped."""
    suffix = f"-{coarseness}" if coarseness is not None else ""
    path = (
        resolve_data_dir(data_dir)
        / "pd1-tabular"
        / f"{dataset}-{model}-{batch_size}_tabular{suffix}.parquet"
    )
    mat, epochs, first_rows = pivot(pd.read_parquet(path), "valid_error_rate")
    keep = np.isfinite(mat).all(axis=1)
    curves = np.clip(1.0 - mat[keep], 0.0, 1.0).astype(np.float32)
    hps = hp_matrix(first_rows[keep], PD1_SPACE)
    fidelities = (epochs / epochs.max()).astype(np.float32)
    name = f"pd1/{dataset}-{model}-{batch_size}"
    if coarseness is not None:
        name += f"-{coarseness}"
    return BenchmarkTask(name, hps, curves, fidelities)
