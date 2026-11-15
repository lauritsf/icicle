"""The example's training blocks and checkpoints preserve optimizer updates."""

import runpy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ray.tune")

ExampleModel = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "examples/raytune_freeze_thaw.py")
)["ExampleModel"]


@pytest.mark.parametrize("features", [64, 2048])
def test_training_blocks_and_resume(tmp_path, features):
    config = dict(feat_scale=1.0, l2=0.001, lr=0.03, momentum=0.8,
                  train_samples=257, updates_per_step=10, features=features)
    model = ExampleModel(config)
    one_update = ExampleModel({**config, "updates_per_step": 1})
    for _ in range(3):
        model.train_step()
        for _ in range(10):
            one_update.train_step()
    model.save(tmp_path)
    resumed = ExampleModel(config)
    resumed.load(tmp_path)
    for _ in range(6):
        assert resumed.train_step() == model.train_step()
        for _ in range(10):
            one_update.train_step()
    for other in (resumed, one_update):
        np.testing.assert_array_equal(model.w, other.w)
        np.testing.assert_array_equal(model.v, other.v)
        np.testing.assert_array_equal(model.order, other.order)
        assert model.position == other.position
        assert model.rng.bit_generator.state == other.rng.bit_generator.state


def test_progress_color_follows_best_configuration(tmp_path, monkeypatch):
    from types import SimpleNamespace

    save_plot = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples/raytune_freeze_thaw.py")
    )["save_plot"]

    results = SimpleNamespace(experiment_path=tmp_path)
    observations = [("a", 0.5), ("b", 0.4), ("b", 0.6), ("a", 0.6), ("a", 0.7)]
    plt = pytest.importorskip("matplotlib.pyplot")
    figures = []
    close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig: figures.append(fig))
    path = save_plot(results, observations, 10)
    fig = figures[-1]
    colors = [line.get_color() for line in fig.axes[0].lines]
    segments = fig.axes[1].collections[0].get_colors()
    np.testing.assert_allclose(
        segments, [colors[0], colors[0], colors[1], colors[1], colors[1], colors[0]],
    )
    assert all(line.get_linestyle() == "-" for line in fig.axes[0].lines)
    assert len(fig.axes) == 5  # Four panels and the shared time color bar.
    assert all(line.get_marker() in (None, "None", "") for line in fig.axes[0].lines)
    np.testing.assert_allclose(
        fig.axes[2].collections[0].get_colors(), plt.get_cmap("viridis")([4 / 5, 1]),
    )
    np.testing.assert_allclose(
        fig.axes[2].collections[0].get_segments(),
        [[[10, 0.5], [20, 0.6]], [[20, 0.6], [30, 0.7]]],
    )
    assert path.exists() and path.with_suffix(".png").exists()
    close(fig)
