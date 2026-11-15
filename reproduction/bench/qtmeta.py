"""Load Quick-Tune's Meta-Album learning curves (86 dataset/split tasks).

The search space contains 33 HPs, including a categorical choice among 24 timm
models. Keep each recorded curve's finite prefix; do not query missing epochs.
Two fidelity views are available:

- walltime: cumulative training plus evaluation seconds divided by the split
  budget (1, 4 or 16 hours), clipped to 1.
- epoch: (k + 1) / 50 on the shared epoch grid.

Responses are eval_top1 / 100. Encode the requested batch_size as the HP. epochs is an HP because it changes the learning-rate
schedule. JSON list positions identify epochs; args.json supplies configurations.
"""

import functools
import json

import numpy as np

from reproduction.bench._hp_spaces import QTMETA_CAT, QTMETA_NUM
from reproduction.bench.task import BenchmarkTask, resolve_data_dir

QTMETA_SPLITS = ("micro", "mini", "extended")
_T_MAX = {"micro": 3600.0, "mini": 14400.0, "extended": 57600.0}  # the paper's budgets [s]
_N_EPOCHS = 50

# One merged, name-sorted column order over numeric + categorical HPs; cat_card is aligned.
_NUM = dict(QTMETA_NUM)
_CAT = dict(QTMETA_CAT)
_COLUMNS = tuple(sorted([*_NUM, *_CAT]))
_CAT_CARD = np.array([len(_CAT[c]) if c in _CAT else 0 for c in _COLUMNS], dtype=np.int32)
_NUM_INDEX = {name: {v: i for i, v in enumerate(grid)} for name, grid in QTMETA_NUM}
_CAT_INDEX = {name: {v: i for i, v in enumerate(cats)} for name, cats in QTMETA_CAT}


@functools.lru_cache(maxsize=len(QTMETA_SPLITS))
def _split_tables(root: str, split: str) -> dict[str, dict]:
    """The four JSON tables of one split, parsed once (~40 MB) and shared by its datasets."""
    tables = {}
    for name in ("args", "eval_top1", "train_time", "eval_time"):
        with open(f"{root}/qtmeta/{split}/{name}.json") as f:
            tables[name] = json.load(f)
    return tables


def _raw_value(args: dict, column: str) -> float | str:
    """One config's raw value for a spec column -- all null/merge policies in one place."""
    if column == "data_augmentation":
        if args["trivial_augment"]:
            return "trivial"
        if args["random_augment"]:
            return "random"
        if args["auto_augment"] is not None:
            return f"auto_{args['auto_augment']}"
        return "none"
    if column in ("opt_beta1", "opt_beta2"):
        betas = args["opt_betas"] or (0.9, 0.999)  # null = the timm default
        return float(betas[0] if column == "opt_beta1" else betas[1])
    if column == "clip_grad":
        return float("inf") if args["clip_grad"] is None else float(args["clip_grad"])
    if column == "layer_decay":
        return 1.0 if args["layer_decay"] is None else float(args["layer_decay"])
    v = args[column]
    return str(v) if column in _CAT else float(v)


def _hp_row(args: dict) -> np.ndarray:
    row = np.empty(len(_COLUMNS), dtype=np.float32)
    for j, col in enumerate(_COLUMNS):
        v = _raw_value(args, col)
        if col in _CAT:
            k, card = _CAT_INDEX[col][v], len(_CAT[col])
            row[j] = (k + 0.5) / card
        else:
            grid = _NUM_INDEX[col]
            idx = grid.get(v)
            if idx is None:
                raise ValueError(f"qtmeta: {col}={v!r} not on the transcribed grid")
            row[j] = idx / (len(grid) - 1)
    return row


def qtmeta_datasets(split: str, data_dir=None) -> list[str]:
    """The split's dataset keys (e.g. ``mtlbm/micro/set1/PNU``), sorted."""
    root = str(resolve_data_dir(data_dir))
    return sorted(_split_tables(root, split)["eval_top1"])


def qtmeta_task_name(split: str, dataset: str) -> str:
    """``qtmeta/{split}/{tail}`` with the redundant ``mtlbm/{split}/`` prefix stripped."""
    tail = dataset.removeprefix(f"mtlbm/{split}/").replace("/", "-")
    return f"qtmeta/{split}/{tail}"


def load_qtmeta(
    split: str, dataset: str, fidelity: str = "walltime", data_dir=None
) -> BenchmarkTask:
    """One (split, dataset) task; see the module docstring for the conventions."""
    root = str(resolve_data_dir(data_dir))
    tables = _split_tables(root, split)
    curves_by_cid = tables["eval_top1"][dataset]
    cids = sorted(curves_by_cid, key=str)  # pivot()'s lexicographic-by-string convention
    b_max = max(len(curves_by_cid[c]) for c in cids)

    curves = np.full((len(cids), b_max), np.nan, dtype=np.float32)
    costs = np.full((len(cids), b_max), np.nan, dtype=np.float64)
    for i, cid in enumerate(cids):
        c = np.asarray(curves_by_cid[cid], dtype=np.float64)
        if c.size < 1 or not np.isfinite(c).all():
            raise ValueError(f"qtmeta: bad curve {cid}")
        curves[i, : c.size] = np.clip(c / 100.0, 0.0, 1.0)
        t = np.asarray(tables["train_time"][dataset][cid], dtype=np.float64) + np.asarray(
            tables["eval_time"][dataset][cid], dtype=np.float64
        )
        if t.size != c.size or not np.isfinite(t).all() or (t < 0).any():
            raise ValueError(f"qtmeta: bad table cost {cid}")
        costs[i, : t.size] = t

    if fidelity == "walltime":
        fid = np.full((len(cids), b_max), np.nan, dtype=np.float32)
        for i, _cid in enumerate(cids):
            t = costs[i, : np.count_nonzero(np.isfinite(costs[i]))]
            fid[i, : t.size] = np.minimum(np.cumsum(t) / _T_MAX[split], 1.0)
    elif fidelity == "epoch":
        # Recorded epochs are 0-based; (k+1)/50 keeps the first..1.0 convention of the
        # other adapters and matches the walltime view's endpoint at full length.
        fid = ((np.arange(b_max) + 1) / _N_EPOCHS).astype(np.float32)
    else:
        raise ValueError(f"unknown fidelity view {fidelity!r}")

    hps = np.stack([_hp_row(tables["args"][cid]) for cid in cids])
    return BenchmarkTask(
        qtmeta_task_name(split, dataset), hps, curves, fid, cat_card=_CAT_CARD, costs=costs
    )


def raw_scores(task: BenchmarkTask, data_dir=None) -> np.ndarray:
    """Load float64 scores in the same configuration order as the model-visible task."""
    _, split, _ = task.name.split("/", 2)
    tables = _split_tables(str(resolve_data_dir(data_dir)), split)
    dataset = next(name for name in tables["eval_top1"] if qtmeta_task_name(split, name) == task.name)
    curves = tables["eval_top1"][dataset]
    scores = np.full(task.curves.shape, np.nan, dtype=np.float64)
    for i, cid in enumerate(sorted(curves, key=str)):
        values = np.asarray(curves[cid], dtype=np.float64) / 100.0
        scores[i, :len(values)] = values
    if not np.allclose(scores, task.curves, equal_nan=True, atol=1e-7):
        raise ValueError(f"{task.name}: raw scores disagree with the loaded task")
    return scores

