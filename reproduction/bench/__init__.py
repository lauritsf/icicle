"""Real-benchmark adapters (LCBench / PD1 / TaskSet / Meta-Album / ARLBench) -> ``BenchmarkTask``.

Backend-free (numpy + pandas only -- no jax, no torch). Data location comes from
``$ICICLE_BENCH_DATA`` or an explicit ``data_dir`` argument; see ``task.resolve_data_dir``.
"""

from reproduction.bench.arlbench import ARLBENCH_VIEWS, arlbench_stems, load_arlbench
from reproduction.bench.lcbench import lcbench_task_ids, load_lcbench
from reproduction.bench.pd1 import PD1_IFBO_TASKS, load_pd1
from reproduction.bench.qtmeta import QTMETA_SPLITS, load_qtmeta, qtmeta_datasets
from reproduction.bench.scenario import (
    BENCHMARKS,
    EvalProtocol,
    Scenario,
    benchmark_tasks,
    iter_scenarios,
    sample_scenario,
)
from reproduction.bench.task import BenchmarkTask, normalize_hp, resolve_data_dir, valid_lengths
from reproduction.bench.taskset import TASKSET_OPTIMIZERS, TASKSET_IFBO_TASKS, load_taskset

__all__ = [
    "ARLBENCH_VIEWS",
    "BENCHMARKS",
    "PD1_IFBO_TASKS",
    "QTMETA_SPLITS",
    "TASKSET_OPTIMIZERS",
    "TASKSET_IFBO_TASKS",
    "BenchmarkTask",
    "EvalProtocol",
    "Scenario",
    "arlbench_stems",
    "benchmark_tasks",
    "iter_scenarios",
    "load_arlbench",
    "lcbench_task_ids",
    "load_lcbench",
    "load_pd1",
    "load_qtmeta",
    "load_taskset",
    "normalize_hp",
    "qtmeta_datasets",
    "resolve_data_dir",
    "sample_scenario",
    "valid_lengths",
]
