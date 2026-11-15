"""Raw JAX predictions from learning-curve observations."""

import jax
import numpy as np
import pytest

from reproduction.bench import BenchmarkTask
from icicle_jax.hpo import make_raw_predict_fn
from tests.reproduction.jax.test_eval import MODEL_CFG, make_model


@pytest.fixture(autouse=True)
def _f32_mode():
    """Use float32 locally so other tests enabling x64 cannot change these predictions."""
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", old)


def make_task(n_cfg=6, b_max=10, seed=0):
    rng = np.random.default_rng(seed)
    return BenchmarkTask(
        name="hpo/jax-test",
        hps=rng.uniform(size=(n_cfg, 3)).astype(np.float32),
        curves=np.sort(rng.uniform(size=(n_cfg, b_max)), axis=1).astype(np.float32),
        fidelities=((np.arange(b_max) + 1) / b_max).astype(np.float32),
    )


def test_raw_predictions_respond_to_observations():
    model, criterion = make_model()
    task = make_task()
    predict = make_raw_predict_fn(model, criterion, config_dim=4, ctx_pad=16, qry_pad=6)
    query_hps = task.hps
    query_fidelities = np.ones(6, np.float32)
    context_configs = np.array([0, 0, 1])
    context_levels = np.array([0, 1, 0])
    values = task.curves[context_configs, context_levels]

    first = predict(task.hps[context_configs], task.fidelities[context_levels], values,
                    query_hps, query_fidelities)
    changed = predict(task.hps[context_configs], task.fidelities[context_levels], 1.0 - values,
                      query_hps, query_fidelities)
    assert first.shape == (6, MODEL_CFG.num_bars)
    np.testing.assert_allclose(first.sum(-1), 1.0, rtol=1e-5)
    assert not np.allclose(first, changed)

    context_configs = np.append(context_configs, [2, 3])
    context_levels = np.append(context_levels, [0, 0])
    grown = predict(task.hps[context_configs], task.fidelities[context_levels],
                    task.curves[context_configs, context_levels], query_hps, query_fidelities)
    np.testing.assert_allclose(grown.sum(-1), 1.0, rtol=1e-5)


def test_raw_predict_fn_guards():
    model, criterion = make_model()
    task = make_task()
    raw = make_raw_predict_fn(model, criterion, config_dim=4, ctx_pad=4, qry_pad=2)
    with pytest.raises(ValueError, match="queries"):
        raw(
            task.hps[:1],
            task.fidelities[:1],
            task.curves[:1, 0],
            task.hps[:3],
            task.fidelities[:3],
        )


def test_default_capacity_includes_context_and_query_configs():
    from flax import nnx
    from icicle_jax.loading import ModelConfig, build_model
    from icicle_jax.bar_distribution import BarDistribution
    from icicle_jax.hpo import make_raw_predict_fn
    import jax.numpy as jnp

    model = build_model(ModelConfig(
        config_dim=1, emsize=8, num_heads=1, hidden_dim=16,
        n_feat_layers=1, n_icl_layers=1, num_bars=8, icl_backend="xla", dtype="float32",
    ), nnx.Rngs(0))
    predict = make_raw_predict_fn(
        model, BarDistribution(jnp.linspace(0, 1, 9)), config_dim=1, ctx_pad=1, qry_pad=1,
    )
    probs = predict([[0.1]], [0.5], [0.4], [[0.9]], [1.0])
    assert probs.shape == (1, 8)
    np.testing.assert_allclose(probs.sum(), 1, atol=1e-6)
