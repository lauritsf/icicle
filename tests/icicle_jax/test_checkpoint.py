"""Checkpoint save -> restore round-trip: model + optimizer state survive, step resumes."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from icicle_jax.checkpoint import make_manager, restore_latest, save
from icicle_jax.nn.model import Surrogate


def _model():
    return Surrogate(
        emsize=16,
        num_heads=4,
        hidden_dim=32,
        n_feat_layers=1,
        n_icl_layers=2,
        num_bars=8,
        icl_backend="xla",
        rngs=nnx.Rngs(0),
    )


def _forward_args():
    b, c, h, p = 2, 4, 3, 6
    return (
        jnp.zeros((b, c, h)),  # config_hps
        jnp.array(2),  # num_params
        jnp.zeros((b, p), jnp.int32),  # point_config_idx
        jnp.linspace(0.1, 1.0, p)[None].repeat(b, 0),  # fidelities
        jnp.zeros((b, p)),  # y
        3,  # single_eval_pos
    )


def _leaves(obj):
    return [np.asarray(x) for x in jax.tree.leaves(nnx.state(obj))]


def test_save_restore_roundtrip(tmp_path):
    args = _forward_args()

    def loss(m):
        return jnp.sum(m(*args, feature_key=jax.random.key(0)) ** 2)

    # A model + optimizer taken one step off init, so its state differs from a fresh build.
    model = _model()
    opt = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
    opt.update(model, nnx.grad(loss)(model))

    ckpt = str(tmp_path / "ck")
    manager = make_manager(ckpt)
    save(manager, 10, model, opt)
    manager.wait_until_finished()

    model2 = _model()
    opt2 = nnx.Optimizer(model2, optax.adam(1e-2), wrt=nnx.Param)
    assert any(
        not np.array_equal(a, b) for a, b in zip(_leaves(model), _leaves(model2), strict=True)
    )  # sanity: they really differ before restore

    start = restore_latest(make_manager(ckpt), model2, opt2)
    assert start == 11  # resume from latest (10) + 1

    for a, b in zip(_leaves(model), _leaves(model2), strict=True):
        np.testing.assert_array_equal(a, b)  # model params restored exactly
    for a, b in zip(_leaves(opt), _leaves(opt2), strict=True):
        np.testing.assert_array_equal(a, b)  # optimizer moments + step restored exactly


def test_restore_latest_no_checkpoint_returns_zero(tmp_path):
    model = _model()
    opt = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
    assert restore_latest(make_manager(str(tmp_path / "empty")), model, opt) == 0


def test_restore_model_from_legacy_adamw_checkpoint(tmp_path):
    from icicle_jax.checkpoint import restore_model

    model = _model()
    model.decoder_linear2.bias[...] += 0.5
    optimizer = nnx.Optimizer(
        model, optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3, mask=lambda p:
            jax.tree.map(lambda x: x.ndim >= 2, p))), wrt=nnx.Param,
    )
    directory = str(tmp_path / "legacy")
    with make_manager(directory) as manager:
        save(manager, 7, model, optimizer)
        manager.wait_until_finished()
    restored = _model()
    assert restore_model(directory, restored, step=7) == 7
    for actual, expected in zip(_leaves(restored), _leaves(model), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_model_restore_distinguishes_zero_from_missing(tmp_path):
    from icicle_jax.checkpoint import restore_model

    model = _model()
    directory = str(tmp_path / "checkpoint")
    assert restore_model(directory, model) is None
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    with make_manager(directory) as manager:
        save(manager, 0, model, optimizer)
    assert restore_model(directory, model) == 0
