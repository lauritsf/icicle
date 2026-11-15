"""Adapter tests on tiny synthetic fixtures (hand-computed expectations)."""

import json

import numpy as np
import pandas as pd
import pytest

from reproduction.bench import load_lcbench, load_pd1, load_taskset
from reproduction.bench._hp_spaces import LCBENCH_SPACE, PD1_SPACE, TASKSET_SPACE_4P


def write_table(path, ids, epochs, metric, curves, hp_cols, index_names):
    """One (config, epoch) row per curve point, HPs repeated per row -- the tabular layout."""
    records, index = [], []
    for i, cid in enumerate(ids):
        for j, epoch in enumerate(epochs):
            records.append({metric: curves[i][j], **{h: v[i] for h, v in hp_cols.items()}})
            index.append((cid, epoch))
    frame = pd.DataFrame(records, index=pd.MultiIndex.from_tuples(index, names=index_names))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def test_lcbench_drop_epochs_skip_nonfinite_and_string_order(tmp_path):
    # written in order "7", "2", "11" -- the adapter must emit lexicographic ("11", "2", "7"),
    # the order the ifBO pipeline computes in (ids cast to str + sort_index)
    ids, epochs = ["7", "2", "11"], [0, 1, 2, 3]
    curves = [
        [0.1, 0.5, 0.6, 0.7],
        [0.1, 0.4, np.nan, 0.5],  # non-finite -> dropped
        [0.2, 0.3, 0.4, 1.5],  # 1.5 -> clipped to 1
    ]
    hp_cols = {name: np.full(3, lower) for name, lower, _, _ in LCBENCH_SPACE}
    hp_cols["batch_size"] = np.array([16.0, 16.0, 512.0])  # log-scale: 0 / 0 / 1
    write_table(
        tmp_path / "lcbench-tabular" / "toy.parquet",
        ids,
        epochs,
        "val_balanced_accuracy",
        curves,
        hp_cols,
        ("id", "epoch"),
    )
    task = load_lcbench("toy", data_dir=tmp_path)
    assert task.curves.shape == (2, 3)  # Epoch 0 and config "2" dropped.
    np.testing.assert_allclose(task.curves[0], [0.3, 0.4, 1.0], atol=1e-6)  # "11", clipped
    np.testing.assert_allclose(task.curves[1], [0.5, 0.6, 0.7], atol=1e-6)  # "7"
    np.testing.assert_allclose(task.fidelities, np.array([1, 2, 3]) / 3, atol=1e-6)
    batch_col = [s[0] for s in LCBENCH_SPACE].index("batch_size")
    np.testing.assert_allclose(task.hps[:, batch_col], [1.0, 0.0], atol=1e-6)  # "11", "7"
    assert task.hps.shape == (2, len(LCBENCH_SPACE))


def test_pd1_invert_and_skip_short_config(tmp_path):
    ids, epochs = [0, 1, 2], [1, 2]
    curves = [
        [0.9, 0.4],
        [0.8, np.nan],  # missing row -> NaN after pivot -> dropped
        [0.5, 0.2],
    ]
    hp_cols = {name: np.full(3, upper) for name, _, upper, _ in PD1_SPACE}
    write_table(
        tmp_path / "pd1-tabular" / "toy-model-64_tabular.parquet",
        ids,
        epochs,
        "valid_error_rate",
        curves,
        hp_cols,
        ("id", "epoch"),
    )
    # Remove config 1's second row to create a shorter curve.
    frame = pd.read_parquet(tmp_path / "pd1-tabular" / "toy-model-64_tabular.parquet")
    frame = frame[~(frame.index.get_level_values(0) == 1) | (frame.index.get_level_values(1) == 1)]
    frame.to_parquet(tmp_path / "pd1-tabular" / "toy-model-64_tabular.parquet")

    task = load_pd1("toy", "model", 64, data_dir=tmp_path)
    assert task.curves.shape == (2, 2)
    np.testing.assert_allclose(task.curves, [[0.1, 0.6], [0.5, 0.8]], atol=1e-6)
    np.testing.assert_allclose(task.hps, np.ones((2, len(PD1_SPACE))), atol=1e-6)


@pytest.mark.parametrize("dataset,model,batch_size", [("imagenet", "resnet", 256), ("translate_wmt", "xformer_translate", 64)])
def test_pd1_ifbo_coarseness_dispatch(tmp_path, dataset, model, batch_size):
    from reproduction.bench.pd1 import PD1_IFBO_TASKS
    from reproduction.bench.scenario import benchmark_tasks

    # pd1_ifbo = their 28 instances; every coarseness variant is a separate, suffixed name.
    assert len(PD1_IFBO_TASKS) == 28
    ifbo = benchmark_tasks("pd1_ifbo", data_dir=tmp_path)
    assert len(ifbo) == 28

    stem = f"{dataset}-{model}-{batch_size}"
    assert {f"pd1/{stem}-{c}" for c in (1, 2, 5, 10)} <= {name for name, _ in ifbo}
    ids, epochs = [0, 1], [1, 2]
    curves = [[0.9, 0.4], [0.5, 0.2]]
    hp_cols = {name: np.full(2, upper) for name, _, upper, _ in PD1_SPACE}
    write_table(
        tmp_path / "pd1-tabular" / f"{stem}_tabular-1.parquet",
        ids,
        epochs,
        "valid_error_rate",
        curves,
        hp_cols,
        ("id", "epoch"),
    )
    name, loader = next(t for t in ifbo if t[0] == f"pd1/{stem}-1")
    task = loader()
    assert task.name == f"pd1/{stem}-1"
    np.testing.assert_allclose(task.curves, [[0.1, 0.6], [0.5, 0.8]], atol=1e-6)


def test_taskset_clamp_invert_hand_computed(tmp_path):
    ids, epochs = [0, 1, 2], [1, 2, 3]
    losses = [
        [4.0, 2.0, 1.0],
        [6.0, np.nan, 3.0],  # Map the crash NaN to the upper loss bound.
        [2.0, 8.0, 0.5],
    ]
    hp_cols = {name: np.full(3, lower) for name, lower, _, _ in TASKSET_SPACE_4P}
    write_table(
        tmp_path / "taskset-tabular" / "toy-adam4p_wide_grid_1k_10000_replica5.parquet",
        ids,
        epochs,
        "valid1_loss",
        losses,
        hp_cols,
        ("config_id", "epoch"),
    )
    task = load_taskset("toy", "adam4p_wide_grid_1k", data_dir=tmp_path)
    # Normalize by median initialization loss (4), then remove initialization.
    expected = 1.0 - np.array([[2, 1], [4, 3], [4, 0.5]]) / 4
    assert task.curves.shape == (3, 2)
    np.testing.assert_allclose(task.curves, expected, atol=1e-6)
    np.testing.assert_allclose(task.hps, np.zeros((3, len(TASKSET_SPACE_4P))), atol=1e-6)
    np.testing.assert_allclose(task.fidelities, np.array([1, 2]) / 2, atol=1e-6)


def _qt_base_args():
    """One config's args.json entry with every spec column on its lowest grid value."""
    return {
        "batch_size": 2,
        "bss_reg": 0.0,
        "clip_grad": 1.0,
        "cotuning_reg": 0.0,
        "cutmix": 0.0,
        "decay_epochs": 10.0,
        "decay_rate": 0.1,
        "delta_reg": 0.0,
        "drop": 0.0,
        "epochs": 50,
        "layer_decay": 0.65,
        "lr": 1e-05,
        "mixup": 0.0,
        "mixup_prob": 0.0,
        "momentum": 0.0,
        "opt_betas": [0.0, 0.99],
        "patience_epochs": 2,
        "pct_to_freeze": 0.0,
        "ra_magnitude": 8,
        "ra_num_ops": 2,
        "smoothing": 0.0,
        "sp_reg": 0.0,
        "warmup_epochs": 0,
        "warmup_lr": 0.0,
        "weight_decay": 0.0,
        "amp": False,
        "linear_probing": False,
        "stoch_norm": False,
        "model": "beit_base_patch16_384",
        "opt": "adam",
        "sched": "cosine",
        "auto_augment": None,
        "random_augment": False,
        "trivial_augment": False,
    }


def write_qtmeta(tmp_path, dataset, curves, train_times, eval_times, args_by_cid):
    root = tmp_path / "qtmeta" / "micro"
    root.mkdir(parents=True, exist_ok=True)
    tables = {
        "eval_top1": {dataset: curves},
        "train_time": {dataset: train_times},
        "eval_time": {dataset: eval_times},
        "args": args_by_cid,
    }
    for name, table in tables.items():
        (root / f"{name}.json").write_text(json.dumps(table))


def test_qtmeta_ragged_walltime_and_hp_encoding(tmp_path):
    from reproduction.bench import load_qtmeta
    from reproduction.bench.qtmeta import _COLUMNS

    # ids "10" < "2" lexicographic; lengths 3 / 1 (an epochs=1 probe) -> ragged
    a_full, a_probe = _qt_base_args(), _qt_base_args()
    # exercise every null/merge policy + one mid-grid + categorical values on config "10"
    a_full.update(
        clip_grad=None,  # -> inf, top rank
        layer_decay=None,  # -> 1.0, top rank
        opt_betas=None,  # -> timm default (0.9, 0.999): both top rank
        trivial_augment=True,  # -> data_augmentation "trivial", category 4 of 5
        lr=1e-4,  # grid index 2 of 7
        model="dla46x_c",  # category 4 of 24
        amp=True,
    )
    a_probe.update(epochs=1, auto_augment="v0")  # -> "auto_v0", category 1 of 5
    write_qtmeta(
        tmp_path,
        "mtlbm/micro/set0/TOY",
        {"10": [50.0, 75.0, 101.0], "2": [5.0]},
        {"10": [1000.0, 1500.0, 2000.0], "2": [700.0]},
        {"10": [100.0, 100.0, 100.0], "2": [20.0]},
        {"10": a_full, "2": a_probe},
    )
    task = load_qtmeta("micro", "mtlbm/micro/set0/TOY", data_dir=tmp_path)

    assert task.name == "qtmeta/micro/set0-TOY"
    np.testing.assert_allclose(task.curves[0], [0.5, 0.75, 1.0], atol=1e-6)  # "10", clipped
    assert task.curves[1][0] == np.float32(0.05) and np.isnan(task.curves[1][1:]).all()
    # walltime view: cumsum(train+eval)/3600, clipped at the 1 h budget; NaN mask == curves'
    assert task.fidelities.shape == task.curves.shape
    np.testing.assert_allclose(task.fidelities[0], [1100 / 3600, 2700 / 3600, 1.0], atol=1e-6)
    np.testing.assert_allclose(task.fidelities[1][0], 720 / 3600, atol=1e-6)
    assert np.isnan(task.fidelities[1][1:]).all()
    np.testing.assert_allclose(task.costs[0], [1100.0, 1600.0, 2100.0])
    np.testing.assert_allclose(task.costs[1][0], 720.0)
    assert np.isnan(task.costs[1][1:]).all()
    assert np.array_equal(np.isfinite(task.costs), np.isfinite(task.curves))

    cols = {name: j for j, name in enumerate(_COLUMNS)}
    row_full, row_probe = task.hps[0], task.hps[1]
    assert row_full[cols["clip_grad"]] == 1.0
    assert row_full[cols["layer_decay"]] == 1.0
    assert row_full[cols["opt_beta1"]] == 1.0 and row_full[cols["opt_beta2"]] == 1.0
    np.testing.assert_allclose(row_full[cols["lr"]], 2 / 6, atol=1e-6)
    np.testing.assert_allclose(row_full[cols["data_augmentation"]], 4.5 / 5, atol=1e-6)
    np.testing.assert_allclose(row_full[cols["model"]], 4.5 / 24, atol=1e-6)
    np.testing.assert_allclose(row_full[cols["amp"]], 1.5 / 2, atol=1e-6)
    assert row_full[cols["epochs"]] == 1.0 and row_probe[cols["epochs"]] == 0.0
    np.testing.assert_allclose(row_probe[cols["data_augmentation"]], 1.5 / 5, atol=1e-6)
    np.testing.assert_allclose(row_probe[cols["opt_beta1"]], 0.0, atol=1e-6)

    card = {name: c for name, c in zip(_COLUMNS, task.cat_card, strict=True)}
    assert card["model"] == 24 and card["opt"] == 5 and card["amp"] == 2
    assert card["lr"] == 0 and card["epochs"] == 0
    assert int((task.cat_card > 0).sum()) == 7


def test_qtmeta_epoch_view_shared_grid(tmp_path):
    from reproduction.bench import load_qtmeta

    write_qtmeta(
        tmp_path,
        "mtlbm/micro/set0/TOY",
        {"1": [10.0, 20.0]},
        {"1": [1.0, 1.0]},
        {"1": [1.0, 1.0]},
        {"1": _qt_base_args()},
    )
    task = load_qtmeta("micro", "mtlbm/micro/set0/TOY", fidelity="epoch", data_dir=tmp_path)
    assert task.fidelities.ndim == 1
    np.testing.assert_allclose(task.fidelities, [1 / 50, 2 / 50], atol=1e-6)


def write_arlbench(tmp_path, stem, perf_by_config_seed, hp_by_config, steps):
    """Long-format landscape CSV: one row per (config, seed, training_steps)."""
    records = []
    for cid, by_seed in perf_by_config_seed.items():
        for seed, perfs in by_seed.items():
            for step, p in zip(steps, perfs, strict=True):
                records.append(
                    {
                        "config_id": cid,
                        "training_steps": step,
                        "performance": p,
                        **{f"hp_config.{k}": v for k, v in hp_by_config[cid].items()},
                        "seed": seed,
                    }
                )
    path = tmp_path / "arlbench" / "landscapes" / f"{stem}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def test_arlbench_views_share_scale_truncate_nan_and_hp_encoding(tmp_path):
    from reproduction.bench import ARLBENCH_VIEWS, arlbench_stems, load_arlbench
    from reproduction.bench._hp_spaces import ARLBENCH_BOOL, ARLBENCH_NUM

    ppo_lower = {name: lower for name, lower, _, _ in ARLBENCH_NUM["ppo"]}
    hp0 = dict.fromkeys(ppo_lower, None)
    hp0.update(ppo_lower)
    hp0["normalize_advantage"] = False
    hp1 = dict(hp0)
    hp1["learning_rate"] = 0.1  # log-scale upper -> 1.0
    hp1["normalize_advantage"] = True
    perfs = {
        # seed 0 dies after 2 levels; seed 1 survives -- crash policy fills NaN with the
        # table's min observed performance (0.0) before any averaging
        0: {0: [0.0, 100.0, np.nan], 1: [0.0, 300.0, 400.0]},
        # A configuration that fails at every level becomes a constant worst-value curve.
        1: {0: [np.nan] * 3, 1: [np.nan] * 3},
    }
    write_arlbench(tmp_path, "cc_toy_ppo", perfs, {0: hp0, 1: hp1}, steps=[10, 20, 30])

    assert arlbench_stems(tmp_path) == ["cc_toy_ppo"]
    assert "mean" in ARLBENCH_VIEWS
    s0 = load_arlbench("cc_toy_ppo", "seed0", data_dir=tmp_path)
    s1 = load_arlbench("cc_toy_ppo", "seed1", data_dir=tmp_path)
    mean = load_arlbench("cc_toy_ppo", "mean", data_dir=tmp_path)

    # normalization by the full table's range (0..400), identical across views; seed0's
    # death maps to the min performance (-> 0.0), and the dead config 1 is a flat-0 curve
    assert s0.curves.shape == (2, 3) and s1.curves.shape == (2, 3)
    np.testing.assert_allclose(s0.curves[0], [0.0, 0.25, 0.0])
    np.testing.assert_allclose(s1.curves[0], [0.0, 0.75, 1.0])
    np.testing.assert_allclose(mean.curves[0], [0.0, 0.5, 0.5])  # crash-filled before the mean
    np.testing.assert_allclose(s0.curves[1], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(s0.fidelities, [1 / 3, 2 / 3, 1.0])

    cols = sorted([*ppo_lower, *ARLBENCH_BOOL["ppo"]])
    row = dict(zip(cols, s0.hps[0], strict=True))
    assert row["learning_rate"] == 0.0 and row["normalize_advantage"] == 0.25
    assert row["clip_eps"] == 0.0
    card = dict(zip(cols, s0.cat_card, strict=True))
    assert card["normalize_advantage"] == 2 and card["learning_rate"] == 0
    assert s0.name == "arlbench/seed0/cc_toy_ppo"


@pytest.mark.parametrize("benchmark", ["pd1_wmt", "arlbench_allseeds", "arlbench_allseeds_hp"])
def test_removed_benchmark_views_are_rejected(tmp_path, benchmark):
    from reproduction.bench import benchmark_tasks

    with pytest.raises(ValueError, match="unknown"):
        benchmark_tasks(benchmark, data_dir=tmp_path)
