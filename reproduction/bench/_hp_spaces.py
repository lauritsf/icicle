"""Search-space bounds for the benchmark HPs, transcribed from the published spaces.

Bounds for LCBench (Zimmer et al. 2021), PD1 (Wang et al. 2021) and TaskSet
(Metz et al. 2020) come from mfpbench's space definitions and are checked against
ifBO's reference loader. HP columns are sorted by name to match the reference.

Quick-Tune (Arango et al. 2024) is a discrete-grid space, so its numeric HPs normalize by
grid rank (``index / (n_values - 1)``). This preserves ordering for grids mixing
0 with log-spaced values (weight_decay, warmup_lr, the regularizers). Categorical
HPs list their ordered categories; category k of cardinality c is stored at the bucket center
``(k + 0.5) / c`` (the cat-hps tokenizer decodes ``floor(u * card)``). Grids transcribed from
the meta-dataset's args.json, union over all three splits (verified identical semantics).
Null policies (clip_grad/layer_decay None -> top rank, opt_betas None -> the timm default) and
the data_augmentation merge are handled by the qtmeta adapter.
"""

HpSpec = tuple[str, float, float, bool]  # (column, lower, upper, log-scale)

LCBENCH_SPACE: tuple[HpSpec, ...] = (
    ("batch_size", 16, 512, True),
    ("learning_rate", 1e-4, 1e-1, True),
    ("max_dropout", 0.0, 1.0, False),
    ("max_units", 64, 1024, True),
    ("momentum", 0.1, 0.99, False),
    ("num_layers", 1, 5, False),
    ("weight_decay", 1e-5, 1e-1, False),
)

PD1_SPACE: tuple[HpSpec, ...] = (
    ("lr_decay_factor", 0.01, 0.99, False),
    ("lr_initial", 1e-5, 10.0, True),
    ("lr_power", 0.1, 2.0, False),
    ("opt_momentum", 1e-5, 1.0, True),
)

TASKSET_SPACE_4P: tuple[HpSpec, ...] = (
    ("beta1", 1e-4, 1.0, True),
    ("beta2", 1e-3, 1.0, True),
    ("epsilon", 1e-12, 1e3, True),
    ("learning_rate", 1e-9, 10.0, True),
)

TASKSET_SPACE_8P: tuple[HpSpec, ...] = (
    ("beta1", 1e-4, 1.0, True),
    ("beta2", 1e-3, 1.0, True),
    ("epsilon", 1e-12, 1e3, True),
    ("exponential_decay", 1e-6, 1e-3, True),
    ("l1", 1e-9, 10.0, True),
    ("l2", 1e-9, 10.0, True),
    ("learning_rate", 1e-9, 10.0, True),
    ("linear_decay", 1e-8, 1e-4, True),
)

QtGridSpec = tuple[str, tuple[float, ...]]  # (column, ascending value grid) -> rank/(n-1)
QtCatSpec = tuple[str, tuple[str, ...]]  # (column, ordered category names) -> (k+0.5)/card

QTMETA_NUM: tuple[QtGridSpec, ...] = (
    ("batch_size", (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0)),
    ("bss_reg", (0.0, 1e-4, 1e-3, 1e-2, 1e-1)),
    ("clip_grad", (1.0, 10.0, float("inf"))),  # inf = the None (no-clipping) level
    ("cotuning_reg", (0.0, 0.5, 1.0, 2.0, 4.0)),
    ("cutmix", (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)),
    ("decay_epochs", (10.0, 20.0, 90.0)),
    ("decay_rate", (0.1, 0.5)),
    ("delta_reg", (0.0, 1e-4, 1e-3, 1e-2, 1e-1)),
    ("drop", (0.0, 0.1, 0.2, 0.3, 0.4)),
    ("epochs", (1.0, 50.0)),  # the 1-epoch probe runs; changes the LR schedule -> an HP
    ("layer_decay", (0.65, 0.75, 1.0)),  # 1.0 = the None (no layer decay) level
    ("lr", (1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2)),
    ("mixup", (0.0, 0.2, 0.4, 1.0, 2.0, 4.0, 8.0)),
    ("mixup_prob", (0.0, 0.25, 0.5, 0.75, 1.0)),
    ("momentum", (0.0, 0.8, 0.9, 0.95, 0.99)),
    ("opt_beta1", (0.0, 0.9)),
    ("opt_beta2", (0.99, 0.999)),
    ("patience_epochs", (2.0, 5.0, 10.0)),
    ("pct_to_freeze", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
    ("ra_magnitude", (8.0, 9.0, 17.0)),
    ("ra_num_ops", (2.0, 3.0)),
    ("smoothing", (0.0, 0.05, 0.1)),
    ("sp_reg", (0.0, 1e-4, 1e-3, 1e-2, 1e-1)),
    ("warmup_epochs", (0.0, 5.0, 10.0)),
    ("warmup_lr", (0.0, 1e-6, 1e-5)),
    ("weight_decay", (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)),
)

QTMETA_CAT: tuple[QtCatSpec, ...] = (
    ("amp", ("False", "True")),
    ("data_augmentation", ("auto_original", "auto_v0", "none", "random", "trivial")),
    ("linear_probing", ("False", "True")),
    (
        "model",
        (
            "beit_base_patch16_384",
            "beit_large_patch16_512",
            "convnext_small_384_in22ft1k",
            "deit3_small_patch16_384_in21ft1k",
            "dla46x_c",
            "edgenext_small",
            "edgenext_x_small",
            "edgenext_xx_small",
            "mobilevit_xs",
            "mobilevit_xxs",
            "mobilevitv2_075",
            "swinv2_base_window12to24_192to384_22kft1k",
            "tf_efficientnet_b4_ns",
            "tf_efficientnet_b6_ns",
            "tf_efficientnet_b7_ns",
            "volo_d1_384",
            "volo_d3_448",
            "volo_d4_448",
            "volo_d5_448",
            "volo_d5_512",
            "xcit_nano_12_p8_384_dist",
            "xcit_small_12_p8_384_dist",
            "xcit_tiny_12_p8_384_dist",
            "xcit_tiny_24_p8_384_dist",
        ),
    ),
    ("opt", ("adam", "adamp", "adamw", "momentum", "sgd")),
    ("sched", ("cosine", "multistep", "plateau", "step")),
    ("stoch_norm", ("False", "True")),
)


# ARLBench (Becktepe et al. 2026) landscape sweeps. The sweep unions per-env-family
# subspaces (the repo's examples/configs/search_space/*.yaml differ per family), so the
# scale-type HPs (buffer_size, learning_starts, batch/minibatch grids) normalize by the
# data-envelope bounds rounded to the enclosing powers, log-scale (the families are
# log-spaced tiers); the remaining bounds are the published algorithm spaces, which contain
# the data. Booleans are 2-way categoricals at bucket centers (False -> 0.25, True -> 0.75),
# listed separately; each algorithm's HP matrix is the union of both, sorted by name.

ARLBENCH_NUM: dict[str, tuple[HpSpec, ...]] = {
    "dqn": (
        ("buffer_alpha", 0.01, 1.0, False),
        ("buffer_batch_size", 16, 256, True),
        ("buffer_beta", 0.01, 1.0, False),
        ("buffer_epsilon", 1e-7, 1e-3, True),
        ("buffer_size", 1024, 1e6, True),
        ("initial_epsilon", 0.5, 1.0, False),
        ("learning_rate", 1e-6, 0.1, True),
        ("learning_starts", 1, 32768, True),
        ("target_epsilon", 0.001, 0.2, False),
        ("target_update_interval", 1, 2000, False),
    ),
    "ppo": (
        ("clip_eps", 0.0, 0.5, False),
        ("ent_coef", 0.0, 0.5, False),
        ("gae_lambda", 0.8, 1.0, False),
        ("learning_rate", 1e-6, 0.1, True),
        ("max_grad_norm", 0.0, 1.0, False),
        ("minibatch_size", 32, 2048, True),
        ("vf_clip_eps", 0.0, 0.5, False),
        ("vf_coef", 0.0, 1.0, False),
    ),
    "sac": (
        ("buffer_alpha", 0.01, 1.0, False),
        ("buffer_batch_size", 128, 2048, True),
        ("buffer_beta", 0.01, 1.0, False),
        ("buffer_epsilon", 1e-9, 1e-4, True),
        ("buffer_size", 1024, 2e6, True),
        ("learning_rate", 1e-6, 0.1, True),
        ("learning_starts", 128, 16384, True),
        ("reward_scale", 0.1, 10.0, True),
        ("tau", 0.01, 1.0, False),
    ),
}

ARLBENCH_BOOL: dict[str, tuple[str, ...]] = {
    "dqn": ("buffer_prio_sampling", "use_target_network"),
    "ppo": ("normalize_advantage",),
    "sac": ("buffer_prio_sampling", "use_target_network"),
}
