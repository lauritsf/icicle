"""Readers for final-paper statistics and explicit HPO caches."""

import json
import os
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

CACHE = Path(os.environ.get("ICICLE_FIGCACHE", "/scratch/project_465003095/icicle/figcache"))


def check_fresh(benches):
    """Reject explicit caches whose selected inputs have changed."""
    meta_path = CACHE / "meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"Figure cache missing: {meta_path}; rerun aggregation with --inputs")
    meta = json.loads(meta_path.read_text())
    if meta.get("format") != "hpo-explicit":
        raise SystemExit("Expected an explicit HPO cache; rerun aggregation with --inputs")
    missing = set(benches) - set(meta["benchmarks"])
    if missing:
        raise SystemExit(f"Figure cache does not cover {sorted(missing)}")
    for source in meta["sources"]:
        path = Path(source["path"])
        if (not path.is_file() or path.stat().st_size != source["size"]
                or path.stat().st_mtime_ns != source["mtime_ns"]):
            raise SystemExit(f"Figure input changed: {path}; rerun aggregation with --inputs")
    return meta


def load_cells(bench, tag=None, iters=None, arms=None, columns=None, finals_only=False):
    """Read selected rows from an explicit HPO result cache."""
    benches = bench if isinstance(bench, (list, tuple)) else [bench]
    expr = pc.field("bench").isin(benches)
    if tag is not None:
        tags = tag if isinstance(tag, (list, tuple)) else [tag]
        expr = expr & pc.field("tag").isin(tags)
    if iters is not None:
        expr = expr & (pc.field("iters") == iters)
    if arms is not None:
        expr = expr & pc.field("arm").isin(list(arms))
    if finals_only:
        expr = expr & pc.field("is_final")
    return pq.read_table(CACHE / "cells.parquet", columns=columns, filters=expr).to_pandas()


def paper_statistics():
    """Load the statistics prepared by agg_figure_cache --paper."""
    path = CACHE / "paper-statistics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run agg_figure_cache --paper to create {path}")
    return json.loads(path.read_text())


def final_task_runs(statistics, panel_name, methods):
    """Task-card rows for fixed checkpoints, with one value per BO seed."""
    if statistics.get('schema') != 'icicle-paper-statistics-v1':
        raise ValueError('Rebuild the final-paper statistical cache')
    panel = statistics['panels'][panel_name]
    return [dict(task=task, method=method, seed=seed, bo_seed=seed,
                 split=panel_name.removeprefix('qtmeta_'), bo_conditioned_regret=value)
            for method, key in methods.items()
            for task, runs in zip(panel['tasks'], panel['models'][key]['task_runs']['endpoint'], strict=True)
            for seed, value in enumerate(runs)]
