"""Build an ONNX inference graph from a portable model export.

The graph represents one task (B = 1) and uses NumPy and ONNX for construction.
C is the number of configurations, P the number of points, H the HP count and
E the embedding width. Inputs are:

    config_hps        (C, H) f32   normalized hyperparameters
    feature_identity  (H, E) f32   feature frame computed by the caller
    num_params        () i64      number of active hyperparameters
    point_config_idx  (P,) i64    configuration row for each point
    fidelities        (P,) f32    fidelity values in (0, 1]
    y                 (P,) f32    observed values; only context entries are read
    single_eval_pos   () i64      context/query split, at least 1

Output logits have shape (P, num_bars). The graph uses primitive RMSNorm
operations and exact erf-based GELU. Provider support depends on the
installed ONNX Runtime build.
"""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from icicle.model_format import validate

OPSET = 21
NEG = np.float32(-1e9)  # masked-logit bias; large but softmax-safe in f32


class _Graph:
    """Accumulate ONNX nodes and NumPy constant initializers."""

    def __init__(self):
        self.nodes, self.inits, self.n = [], [], 0

    def emit(self, op, inputs, **attrs):
        self.n += 1
        out = f"{op.lower()}_{self.n}"
        self.nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def const(self, arr, name=None):
        self.n += 1
        name = name or f"c_{self.n}"
        self.inits.append(numpy_helper.from_array(np.asarray(arr), name))
        return name

    def i64(self, *vals):
        return self.const(np.array(vals, np.int64))

    def dim0(self, tensor):
        """Runtime size of a tensor's first axis, as a () i64."""
        return self.emit(
            "Gather", [self.emit("Shape", [tensor]), self.const(np.array(0, np.int64))]
        )


def _linear(g, x, tensors, prefix):
    w = g.const(tensors[f"{prefix}.kernel"], f"{prefix}.kernel")  # (in, out): y = x @ W + b
    b = g.const(tensors[f"{prefix}.bias"], f"{prefix}.bias")
    return g.emit("Add", [g.emit("MatMul", [x, w]), b])


def _rmsnorm(g, x, tensors, name):
    """X / sqrt(mean(x^2) + 1e-6) * scale over the last axis (flax RMSNorm semantics)."""
    ms = g.emit("ReduceMean", [g.emit("Mul", [x, x]), g.i64(-1)], keepdims=1)
    denominator = g.emit("Sqrt", [g.emit("Add", [ms, g.const(np.float32(1e-6))])])
    return g.emit(
        "Mul", [g.emit("Div", [x, denominator]), g.const(tensors[f"{name}.scale"], f"{name}.scale")]
    )


def _sincos(g, value, emsize):
    """Additive Fourier embedding of ``value (...,)`` -> ``(..., emsize)`` (even emsize)."""
    half = emsize // 2
    inv_freq = (10000.0 ** (-2.0 * np.arange(half) / emsize)).astype(np.float32)
    v = g.emit("Unsqueeze", [value, g.i64(-1)])
    ang = g.emit("Mul", [g.emit("Mul", [v, g.const(np.float32(2.0 * np.pi))]), g.const(inv_freq)])
    return g.emit("Concat", [g.emit("Sin", [ang]), g.emit("Cos", [ang])], axis=-1)


def _key_mask_bias(g, upto, length_source):
    """Additive key mask ``(1, 1, 1, S)``: 0 where index < upto, NEG elsewhere.

    ``S`` is the runtime first-axis size of ``length_source``.
    """
    one = g.const(np.array(1, np.int64))
    idx = g.emit("Range", [g.const(np.array(0, np.int64)), g.dim0(length_source), one])
    bias = g.emit("Where", [g.emit("Less", [idx, upto]), g.const(np.float32(0.0)), g.const(NEG)])
    return g.emit("Reshape", [bias, g.i64(1, 1, 1, -1)])


def _attention(g, x_q, x_kv, tensors, prefix, heads, head_dim, mask_bias, qk_norm, fused=False):
    """Multi-head attention on (N, S, E) inputs.

    mask_bias broadcasts to (N, heads, Sq, Sk): zero permits attention and NEG
    masks it. With fused=True, use ONNX Runtime's MultiHeadAttention operator;
    otherwise build score, softmax and weighted-value operations explicitly.
    Projections and per-head qk normalization are identical in both graphs.
    """
    if fused:
        split, merge = g.i64(0, 0, heads, head_dim), g.i64(0, 0, heads * head_dim)

        def proj(x, name, norm_name):
            y = _linear(g, x, tensors, name)
            if qk_norm and norm_name:
                y = g.emit("Reshape", [y, split])  # (B, S, heads, head_dim): norm per head
                y = _rmsnorm(g, y, tensors, f"{prefix}.{norm_name}")
                y = g.emit("Reshape", [y, merge])
            return y

        q = proj(x_q, f"{prefix}.q", "q_norm")
        k = proj(x_kv, f"{prefix}.k", "k_norm")
        v = _linear(g, x_kv, tensors, f"{prefix}.v")
        # Every mask here is a pure KEY mask, so it feeds MHA's key_padding_mask -- O(S_kv)
        # per row, where an attention_bias would have to materialize the (S_q, S_kv) square
        # the fused kernel exists to avoid. Derived from the additive mask: 0 bias = valid.
        valid = g.emit(
            "Cast",
            [g.emit("Equal", [mask_bias, g.const(np.float32(0.0))])],
            to=TensorProto.INT32,
        )
        b1 = g.emit("Gather", [g.emit("Shape", [x_q]), g.i64(0)])  # (1,) batch rows
        sk = g.emit("Gather", [g.emit("Shape", [x_kv]), g.i64(1)])  # (1,) kv length
        key_padding_mask = g.emit(
            "Expand",
            [g.emit("Reshape", [valid, g.i64(1, -1)]), g.emit("Concat", [b1, sk], axis=0)],
        )
        ctx = g.emit(
            "MultiHeadAttention",
            [q, k, v, "", key_padding_mask],
            domain="com.microsoft",
            num_heads=heads,
            scale=float(1.0 / np.sqrt(head_dim)),
        )
        return _linear(g, ctx, tensors, f"{prefix}.out")
    split = g.i64(0, 0, heads, head_dim)  # 0 = keep that dim
    perm = dict(perm=[0, 2, 1, 3])

    def heads_of(x, name):
        return g.emit(
            "Transpose", [g.emit("Reshape", [_linear(g, x, tensors, name), split])], **perm
        )

    q = heads_of(x_q, f"{prefix}.q")
    k = heads_of(x_kv, f"{prefix}.k")
    v = heads_of(x_kv, f"{prefix}.v")
    if qk_norm:
        q = _rmsnorm(g, q, tensors, f"{prefix}.q_norm")
        k = _rmsnorm(g, k, tensors, f"{prefix}.k_norm")
    scores = g.emit("MatMul", [q, g.emit("Transpose", [k], perm=[0, 1, 3, 2])])
    scores = g.emit("Mul", [scores, g.const(np.float32(1.0 / np.sqrt(head_dim)))])
    probs = g.emit("Softmax", [g.emit("Add", [scores, mask_bias])], axis=-1)
    ctx = g.emit("Transpose", [g.emit("MatMul", [probs, v])], **perm)
    merged = g.emit("Reshape", [ctx, g.i64(0, 0, heads * head_dim)])
    return _linear(g, merged, tensors, f"{prefix}.out")


def _block(g, x, tensors, prefix, heads, head_dim, mask_bias, qk_norm, fused=False):
    """One pre-norm block: x + attn(norm1(x)); x + ffn(norm2(x))."""
    h1 = _rmsnorm(g, x, tensors, f"{prefix}.norm1")
    x = g.emit(
        "Add",
        [
            x,
            _attention(
                g, h1, h1, tensors, f"{prefix}.attn", heads, head_dim, mask_bias, qk_norm, fused
            ),
        ],
    )
    h2 = _rmsnorm(g, x, tensors, f"{prefix}.norm2")
    ff = g.emit("Gelu", [_linear(g, h2, tensors, f"{prefix}.w_in")], approximate="none")
    return g.emit("Add", [x, _linear(g, ff, tensors, f"{prefix}.w_out")])


def build_model_graph(
    manifest: dict, tensors: dict[str, np.ndarray], *, fused: bool = False
) -> onnx.ModelProto:
    validate(manifest, tensors)
    m = manifest["model"]
    emsize, heads = m["emsize"], m["num_heads"]
    head_dim, qk_norm = emsize // heads, m["qk_norm"]
    if emsize % 4:
        raise ValueError("split y/fidelity codes need emsize divisible by 4")
    g = _Graph()

    # --- stage 1: config encoding ---
    # raw_mul_tag: the standardized raw value scales the identity tag, std(v) * u_j
    std = g.emit(
        "Mul",
        [
            g.emit("Sub", ["config_hps", g.const(np.float32(0.5))]),
            g.const(np.float32(np.sqrt(12.0))),
        ],
    )
    cells = g.emit("Mul", [g.emit("Unsqueeze", [std, g.i64(-1)]), "feature_identity"])
    h_mask = _key_mask_bias(g, "num_params", "feature_identity")  # dim 0 of (H, E) is H
    x = cells  # (C, H, E): batch = configs, sequence = HP cells
    for i in range(m["n_feat_layers"]):
        name = f"config_encoder.feat_layers.{i}"
        x = _block(g, x, tensors, name, heads, head_dim, h_mask, qk_norm, fused)
    # PMA pool: the learned seed is the single query cross-attending the valid cells
    seed = g.const(tensors["config_encoder.pool.seed"], "config_encoder.pool.seed")
    c = g.emit("Reshape", [g.dim0("config_hps"), g.i64(1)])
    q_shape = g.emit("Concat", [c, g.i64(1, emsize)], axis=0)
    q = g.emit("Expand", [seed, q_shape])  # (C, 1, E)
    pooled = _attention(
        g, q, x, tensors, "config_encoder.pool.attn", heads, head_dim, h_mask, qk_norm, fused
    )
    e_c = _rmsnorm(
        g, g.emit("Squeeze", [pooled, g.i64(1)]), tensors, "config_encoder.norm"
    )  # (C, E)

    # --- stage 2: the in-context stack over points ---
    pt = g.emit("Gather", [e_c, "point_config_idx"], axis=0)  # (P, emsize)
    one = g.const(np.array(1, np.int64))
    p_idx = g.emit("Range", [g.const(np.array(0, np.int64)), g.dim0("y"), one])
    finite_y = g.emit("Not", [g.emit("Or", [g.emit("IsNaN", ["y"]), g.emit("IsInf", ["y"])])])
    keep_y = g.emit("And", [g.emit("Less", [p_idx, "single_eval_pos"]), finite_y])
    zero = g.const(np.float32(0.0))
    observed_y = g.emit("Where", [keep_y, "y", zero])
    keep_y_col = g.emit("Unsqueeze", [keep_y, g.i64(-1)])
    # Values and fidelities occupy separate halves, each scaled to the full-width norm.
    sqrt2 = g.const(np.float32(np.sqrt(2.0)))
    y_code = g.emit("Mul", [_sincos(g, observed_y, emsize // 2), sqrt2])
    y_half = g.emit("Where", [keep_y_col, y_code, zero])
    fid_half = g.emit("Mul", [_sincos(g, "fidelities", emsize // 2), sqrt2])
    pt = g.emit("Add", [pt, g.emit("Concat", [y_half, fid_half], axis=-1)])
    x = g.emit("Unsqueeze", [pt, g.i64(0)])  # (1, P, emsize)
    p_mask = _key_mask_bias(g, "single_eval_pos", "y")
    for i in range(m["n_icl_layers"]):
        x = _block(g, x, tensors, f"icl_layers.{i}", heads, head_dim, p_mask, qk_norm, fused)
    x = _rmsnorm(g, x, tensors, "final_norm")  # model-level norm before the bar head
    hid = g.emit("Gelu", [_linear(g, x, tensors, "decoder_linear1")], approximate="none")
    logits3 = _linear(g, hid, tensors, "decoder_linear2")
    logits = g.emit("Squeeze", [logits3, g.i64(0)])
    g.nodes.append(helper.make_node("Identity", [logits], ["logits"]))

    f32, i64 = TensorProto.FLOAT, TensorProto.INT64
    graph = helper.make_graph(
        g.nodes,
        "icicle_surrogate",
        inputs=[
            helper.make_tensor_value_info("config_hps", f32, ["C", "H"]),
            helper.make_tensor_value_info("feature_identity", f32, ["H", emsize]),
            helper.make_tensor_value_info("num_params", i64, []),
            helper.make_tensor_value_info("point_config_idx", i64, ["P"]),
            helper.make_tensor_value_info("fidelities", f32, ["P"]),
            helper.make_tensor_value_info("y", f32, ["P"]),
            helper.make_tensor_value_info("single_eval_pos", i64, []),
        ],
        outputs=[helper.make_tensor_value_info("logits", f32, ["P", m["num_bars"]])],
        initializer=g.inits,
    )
    opsets = [helper.make_opsetid("", OPSET)]
    if fused:
        opsets.append(helper.make_opsetid("com.microsoft", 1))
    model = helper.make_model(
        graph, opset_imports=opsets, producer_name="icicle", ir_version=10,
    )
    model.metadata_props.append(onnx.StringStringEntryProto(key="format", value=manifest["format"]))
    onnx.checker.check_model(model)
    return model


def export_onnx(
    weights_path: str | Path, out: str | Path | None = None, *, fused: bool = False
) -> Path:
    """Build an ONNX graph from named safetensors weights and matching JSON metadata.

    Fused graphs use ONNX Runtime's attention operator and default to a
    ``_fused.onnx`` suffix. Each graph gets a matching JSON metadata file.
    """
    from safetensors.numpy import load_file

    weights_path = Path(weights_path)
    manifest = json.loads(weights_path.with_suffix(".json").read_text())
    tensors = load_file(weights_path)
    model = build_model_graph(manifest, tensors, fused=fused)
    if out is None:
        suffix = "_fused" if fused else ""
        out = weights_path.with_name(f"{weights_path.stem}{suffix}.onnx")
    out = Path(out)
    if out.suffix != ".onnx":
        raise ValueError("ONNX output must end in .onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out)
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out
