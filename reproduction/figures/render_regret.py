"""Render final-paper mean-regret curves from an explicit statistical cache.

Aggregation owns input validation and bootstrap computation. Every displayed model
is one checkpoint evaluated over 20 BO seeds; training seeds are never pooled.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

from reproduction.figures.paper_method_colors import METHOD_COLORS

SUITES = (("lcbench_ifbo", "LCBench"), ("pd1_ifbo", "PD1"), ("taskset_ifbo", "TaskSet"))
SPLITS = ("micro", "mini", "extended")
MAIN_HPO = ("icicle-s0", "ftpfn-published", "asha", "random_search")
MAIN_QT = ("icicle-s0", "quicktune-s0", "asha", "random_search")


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.4,
            "axes.linewidth": 0.55,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def style(key, main=False):
    if key.startswith("icicle-s"):
        seed = int(key[8])
        color = ["#2A6FD6", "#164595", "#679DE9"][seed]
        label = "ICICLE" if main else f"ICICLE seed {seed}"
        line = ["-", "--", ":"][seed]
        if key.endswith("-fixed"):
            label, color, line = "ICICLE fixed bounds", "#E06C00", "--"
        elif key.endswith("-off"):
            label, color, line = "ICICLE raw values", "#0F9B6C", ":"
    elif key.startswith("ftpfn128-s"):
        seed = int(key[-1])
        color = ["#008078", "#00584F", "#42A997"][seed]
        label, line = f"FT-PFN 128 seed {seed}", ["-", "--", ":"][seed]
    elif key == "ftpfn-published":
        label, color, line = ("FT-PFN" if main else "Published FT-PFN"), "#111111", "--"
    elif key.startswith("quicktune-s"):
        seed = int(key[-1])
        color = ["#D55E00", "#8B3E00", "#F09B47", "#A26842", "#EAB28C"][seed]
        label = "Quick-Tune" if main else f"Quick-Tune meta seed {seed}"
        line = ["--", "-", ":", "-.", (0, (5, 1, 1, 1, 1, 1))][seed]
    else:
        label, color, line = (
            ("ASHA" if key == "asha" else "Random"),
            METHOD_COLORS[key],
            "-." if key == "asha" else ":",
        )
    return dict(label=label, color=color, linestyle=line, linewidth=1.5)


def draw_panels(statistics, keys, *, qt=False, main=False, axes=None):
    panels = [("qtmeta_" + s, s.title()) for s in SPLITS] if qt else SUITES
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.65), sharey=qt)
        fig.subplots_adjust(left=0.085, right=0.975, bottom=0.18, top=0.88, wspace=0.26)
    else:
        fig = axes[0].figure
    for ax, (suite, title) in zip(axes, panels, strict=True):
        panel = statistics["panels"][suite]
        x = np.array(panel["x"])
        for key in keys:
            values, appearance = panel["models"][key], style(key, main)
            lower, upper, mean = [np.array(values[k]) for k in ("lower", "upper", "mean")]
            if (
                not (len(x) == len(mean) == len(lower) == len(upper))
                or not np.isfinite([lower, upper, mean]).all()
            ):
                raise ValueError("Invalid statistical cache")
            ax.fill_between(
                x,
                lower,
                upper,
                color=appearance["color"],
                alpha=0.11,
                linewidth=0,
                step="post" if qt else None,
            )
            ax.plot(x, mean, **appearance, drawstyle="steps-post" if qt else "default")
        if qt:
            ax.set_ylim(0, 1)
            ax.set_xticks(np.linspace(0, x[-1], 5))
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.set_xlabel("consumed table time [h]")
            title += f" ({int(x[-1])} h)"
        else:
            # Include all displayed CI bounds. No data clipping to make ribbons narrower.
            low = min(min(panel["models"][k]["lower"]) for k in keys)
            if low <= 0:
                raise ValueError("A log-scale confidence interval reaches zero; review axis choice")
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_ylim(10 ** (np.floor(np.log10(low) * 2) / 2), 1)
            ax.set_xlabel("BO iteration")
        ax.set_xlim(x[0], x[-1])
        ax.set_title(title, pad=3)
        ax.grid(axis="y", color="#D4D7DA", linewidth=0.42, alpha=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("mean normalized regret")
    axes[0].legend(loc="upper right" if qt or len(keys) > 4 else "lower left",
                   frameon=False, handlelength=1.7,
                   fontsize=6.2 if len(keys) > 4 else 6.8, labelspacing=0.25)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    statistics = json.loads(args.cache.read_text())
    if statistics.get("schema") != "icicle-paper-statistics-v1":
        raise ValueError("Rebuild the final-paper statistical cache")
    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    figures = [
        ("figure2_qtmeta_mean_regret_linear", MAIN_QT, True, True),
        ("figure3_ftpfn_mean_regret_log", MAIN_HPO, False, True),
        (
            "figure_ftpfn_presentations_log",
            ("icicle-s0", "icicle-s0-fixed", "icicle-s0-off", "ftpfn-published"),
            False,
            False,
        ),
    ]
    for stem, keys, qt, main_figure in figures:
        fig = draw_panels(statistics, keys, qt=qt, main=main_figure)
        for suffix in ("pdf", "png", "svg"):
            path = args.output_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=300)
            outputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        plt.close(fig)
        print(stem, flush=True)

    keys = tuple(f"{family}-s{seed}" for family in ("icicle", "ftpfn128") for seed in range(3))
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 5.2))
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.10, top=0.94, hspace=0.50, wspace=0.26)
    draw_panels(statistics, (*keys, "ftpfn-published"), axes=axes[0])
    draw_panels(statistics, (*keys, "quicktune-s0"), qt=True, axes=axes[1])
    for suffix in ("pdf", "png", "svg"):
        path = args.output_dir / f"figure_training_seeds.{suffix}"
        fig.savefig(path, dpi=300)
        outputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    print("figure_training_seeds", flush=True)
    manifest = {
        "schema": "paper-regret-render-v2",
        "cache_sha256": hashlib.sha256(args.cache.read_bytes()).hexdigest(),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "main_hpo": MAIN_HPO,
        "main_qt": MAIN_QT,
        "training_seed_curves": keys,
        "axes": {"hpo": "log", "qtmeta": "linear"},
        "uncertainty": statistics["uncertainty"],
        "outputs": outputs,
    }
    (args.output_dir / "figure_results_render_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
