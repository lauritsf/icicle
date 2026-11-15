import json

import pandas as pd
import pytest

from reproduction.figures import agg_figure_cache as aggregate
from reproduction.figures import figcache


def write_result(path, method="asha", step=None, checkpoint=None, runtime="test", feature_seed=0):
    pd.DataFrame(
        dict(
            runner=["hpo-regret-v2"] * 2,
            method=method,
            benchmark="lcbench_ifbo",
            checkpoint=checkpoint,
            step=step,
            acquisition=None,
            value_map="off",
            backend="xla",
            jax_version="test",
            jaxlib_version="test",
            platform="cpu",
            runtime=runtime,
            feature_seed=feature_seed,
            device="cpu",
            dtype="float32",
            chain_batch=1,
            task="lcbench/toy",
            seed=0,
            iteration=[0, 1],
            iters=2,
            f_best=[0.2, 0.4],
            regret=[0.6, 0.4],
            cost=[1.0, 2.0],
        )
    ).to_parquet(path)


def test_explicit_inputs_and_reader(tmp_path, monkeypatch):
    first, second = tmp_path / "arbitrary-name.parquet", tmp_path / "other.parquet"
    write_result(first)
    write_result(second, method="icicle", step=7, checkpoint="/model")
    out = tmp_path / "cache"
    monkeypatch.setattr(
        aggregate, "build_consts", lambda *a: {"lcbench_ifbo": {"lcbench/toy": [0.8, 0.1]}}
    )
    aggregate.main(inputs=(first, second), out=out)
    with pytest.raises(ValueError, match="--inputs and --out"):
        aggregate.main()
    monkeypatch.setattr(figcache, "CACHE", out)
    figcache.check_fresh(["lcbench_ifbo"])
    frame = figcache.load_cells("lcbench_ifbo", finals_only=True)
    assert set(frame.method) == {"asha", "icicle"}
    assert set(frame.arm) == {"asha", "icicle:/model@7"}
    assert frame.iteration.eq(1).all()
    first.unlink()
    with pytest.raises(SystemExit, match="Figure input changed"):
        figcache.check_fresh(["lcbench_ifbo"])


def test_runtime_and_steps_stay_separate(tmp_path, monkeypatch):
    paths = [tmp_path / f"{i}.parquet" for i in range(3)]
    write_result(paths[0], "icicle", 7, "/model", "old")
    write_result(paths[1], "icicle", 7, "/model", "new")
    write_result(paths[2], "icicle", 8, "/model", "new")
    monkeypatch.setattr(aggregate, "build_consts", lambda *a: {})
    aggregate.main(inputs=tuple(paths), out=tmp_path / "cache")
    frame = pd.read_parquet(tmp_path / "cache/cells.parquet")
    assert len(frame[["arm", "tag"]].drop_duplicates()) == 3
    with pytest.raises(ValueError, match="Overlapping"):
        aggregate.main(inputs=(paths[0], paths[0]), out=tmp_path / "duplicate")
    historical = tmp_path / "historical"
    historical.mkdir()
    (historical / "meta.json").write_text(json.dumps({"created": "old"}))
    with pytest.raises(ValueError, match="separate cache"):
        aggregate.main(inputs=(paths[0],), out=historical)


def test_feature_seeds_stay_separate(tmp_path, monkeypatch):
    first, second = tmp_path / "first.parquet", tmp_path / "second.parquet"
    write_result(first, method="icicle", feature_seed=0)
    write_result(second, method="icicle", feature_seed=1)
    monkeypatch.setattr(aggregate, "build_consts", lambda *a: {})
    aggregate.main(inputs=(first, second), out=tmp_path / "cache")
    frame = pd.read_parquet(tmp_path / "cache/cells.parquet")
    assert frame.tag.nunique() == 2




def test_interrupted_cache_rebuild_invalidates_completion(tmp_path, monkeypatch):
    result = tmp_path / "result.parquet"
    write_result(result)
    out = tmp_path / "cache"
    monkeypatch.setattr(aggregate, "build_consts", lambda *a: {})
    aggregate.main(inputs=(result,), out=out)
    from pathlib import Path

    original = Path.write_text

    def fail_constants(path, *args, **kwargs):
        if path == out / "consts.json":
            raise OSError("interrupted")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_constants)
    with pytest.raises(OSError, match="interrupted"):
        aggregate.main(inputs=(result,), out=out)
    assert not (out / "meta.json").exists()


def test_hierarchical_bootstrap_variance_and_pairing():
    import numpy as np

    weights = aggregate.bootstrap_weights(2, 3, draws=30000, seed=12)
    values = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    np.testing.assert_allclose(weights.sum(1), 1.0)
    samples = weights @ values.ravel()
    expected_variance = values.mean(1).var() / 2 + values.var(1).mean() / (2 * 3)
    assert samples.var() == pytest.approx(expected_variance, rel=0.04)
    assert samples.mean() == pytest.approx(values.mean(), abs=0.02)
    # A constant paired offset must remain constant in every bootstrap replicate.
    np.testing.assert_allclose(weights @ (values - (values + 0.2)).ravel(), -0.2)
    # Repeated use of a task samples its runs independently, not once per unique task.
    single = aggregate.bootstrap_weights(1, 3, draws=20000, seed=1)
    assert (single @ np.array([0.0, 1.0, 2.0])).var() == pytest.approx(2 / 9, rel=0.04)


def test_time_trajectory_exact_integral():
    curve, endpoint, area = aggregate.time_trajectory(
        [900.0, 1800.0, 1800.0], [0.625, 0.125, 0.125], 3600.0
    )
    assert curve[0] == 1.0
    assert endpoint == 0.125
    assert area == pytest.approx(0.25 + 0.625 * 0.25 + 0.125 * 0.5)
    with pytest.raises(ValueError, match="Invalid time-budget"):
        aggregate.time_trajectory([3601.0], [0.1], 3600.0)


def test_final_figure_selection_and_scales():
    from reproduction.figures import render_regret as r

    panels = {}
    for key in [
        "lcbench_ifbo",
        "pd1_ifbo",
        "taskset_ifbo",
        "qtmeta_micro",
        "qtmeta_mini",
        "qtmeta_extended",
    ]:
        end = {"qtmeta_micro": 1, "qtmeta_mini": 4, "qtmeta_extended": 16}.get(key, 999)
        panels[key] = dict(
            x=[0, end],
            models={
                k: dict(mean=[0.8, 0.1], lower=[0.7, 0.05], upper=[0.9, 0.2])
                for k in set(r.MAIN_HPO + r.MAIN_QT)
            },
        )
    for qt, keys in [(True, r.MAIN_QT), (False, r.MAIN_HPO)]:
        fig = r.draw_panels(dict(panels=panels), keys, qt=qt, main=True)
        assert all(len(ax.lines) == 4 for ax in fig.axes)
        assert all(ax.get_yscale() == ("linear" if qt else "log") for ax in fig.axes)
        assert [ax.get_xlim()[1] for ax in fig.axes] == ([1, 4, 16] if qt else [999] * 3)
        r.plt.close(fig)


def test_paper_loader_rejects_checkpoint_changes_and_duplicate_runs(tmp_path):
    import numpy as np

    model = tmp_path / "model"
    model.mkdir()
    (tmp_path / "bundled-model.json").write_text(json.dumps({"model": {"curve_identity": "none"}}))
    source = tmp_path / "time.parquet"
    write_result(source, "icicle", 79999, str(model))
    frame = pd.read_parquet(source).assign(
        benchmark="qtmeta",
        acquisition="mfpi-cost-cooled",
        value_map="adaptive",
        task="qtmeta/micro/toy",
        budget_seconds=3600.0,
    )
    frame["cost"] = [1800.0, 3600.0]
    frame["normalized_regret"] = (0.8 - frame.f_best) / 0.7
    frame = pd.concat([frame.assign(seed=seed) for seed in range(20)], ignore_index=True)
    frame.to_parquet(source)
    spec = dict(paths=[source], model_config=tmp_path / "bundled-model.json",
                settings=dict(method="icicle", checkpoint=str(model), step=79999))
    names = ["qtmeta/micro/toy"]
    array, metrics = aggregate.qt_arrays(spec, names, {names[0]: (0.1, 0.8)}, {})
    assert array.shape == (1, 20, 512)
    np.testing.assert_allclose(metrics["endpoint"], 0.4 / 0.7)
    frame.assign(step=15999).to_parquet(source)
    with pytest.raises(ValueError, match="step="):
        aggregate.qt_arrays(spec, names, {names[0]: (0.1, 0.8)}, {})
    frame.to_parquet(source)
    spec["paths"] = [source, source]
    with pytest.raises(ValueError, match="duplicate BO seed"):
        aggregate.qt_arrays(spec, names, {names[0]: (0.1, 0.8)}, {})
    spec["paths"] = [source]
    (tmp_path / "bundled-model.json").write_text(json.dumps({"model": {"curve_identity": "learned"}}))
    with pytest.raises(ValueError, match="no curve identity"):
        aggregate.qt_arrays(spec, names, {names[0]: (0.1, 0.8)}, {})


def test_native_identity_keeps_fold_models_separate():
    doc = dict(
        torch="test",
        hip="test",
        runner_sha256="runner",
        run=dict(split="micro", fold=0),
        checkpoint_hashes=dict(
            cost_sha256="cost", performance_sha256="model", training_manifest_sha256="train"
        ),
    )
    spec = {}
    aggregate.validate_native_identity(doc, spec)
    doc["run"]["fold"] = 1
    doc["checkpoint_hashes"]["performance_sha256"] = "fold1-model"
    aggregate.validate_native_identity(doc, spec)
    doc["run"]["fold"] = 0
    with pytest.raises(ValueError, match="Changed native checkpoint"):
        aggregate.validate_native_identity(doc, spec)
    doc["torch"] = "other-runtime"
    with pytest.raises(ValueError, match="Mixed native"):
        aggregate.validate_native_identity(doc, spec)


def test_explicit_split_batches_preserve_other_identity_checks(tmp_path):
    paths = [tmp_path / "micro.parquet", tmp_path / "mini.parquet"]
    spec = dict(settings=dict(method="asha"),
                chain_batch_by_split=dict(micro=128, mini=32, extended=32))
    for path, split, batch in zip(paths, ["micro", "mini"], [128, 32], strict=True):
        write_result(path)
        frame = pd.read_parquet(path).assign(
            benchmark="qtmeta", task=f"qtmeta/{split}/toy", chain_batch=batch)
        frame.to_parquet(path)
        assert len(aggregate.checked_table(path, spec, "qtmeta", {})) == 2

    frame.assign(chain_batch=64).to_parquet(paths[1])
    with pytest.raises(ValueError, match="chain_batch disagrees"):
        aggregate.checked_table(paths[1], spec, "qtmeta", {})
    frame.assign(runtime="other-runtime").to_parquet(paths[1])
    with pytest.raises(ValueError, match="Numerical implementation differs"):
        aggregate.checked_table(paths[1], spec, "qtmeta", {})
    frame.to_parquet(paths[1])
    spec["settings"]["chain_batch"] = 32
    with pytest.raises(ValueError, match="without a global chain_batch"):
        aggregate.checked_table(paths[1], spec, "qtmeta", {})


def test_split_batches_must_be_fully_declared(tmp_path):
    path = tmp_path / "time.parquet"
    write_result(path)
    spec = dict(settings=dict(method="asha"), chain_batch_by_split=dict(micro=128))
    with pytest.raises(ValueError, match="every Meta-Album split"):
        aggregate.checked_table(path, spec, "qtmeta", {})
