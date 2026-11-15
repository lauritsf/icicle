"""Render static ARLBench: one trained model, 20 BO seeds, task/BO bootstrap CIs."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from reproduction.figures import arlbench_mean_common as core
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter

MAIN_METHODS = {"icicle-s0", "asha", "hyperband", "random_search"}
MAIN_SERIES = tuple(series for series in core.SERIES if series[0] in MAIN_METHODS)


def draw_panel(
    ax: plt.Axes, curves: dict[str, dict[str, np.ndarray]], scale: str, series: tuple
) -> None:
    """Draw one clean linear or log-y candidate without endpoint emphasis."""
    for method, label, color, linestyle, width in series:
        values = curves[method]
        ax.fill_between(
            core.ITERATIONS,
            values["lower"],
            values["upper"],
            color=color,
            alpha=0.09,
            linewidth=0,
        )
        ax.plot(
            core.ITERATIONS,
            values["mean"],
            color=color,
            linestyle=linestyle,
            linewidth=width,
            label=label,
        )
    ax.set_xlim(0, 999)
    ax.set_xlabel("BO iteration")
    ax.set_ylabel("mean normalized regret")
    if scale == "linear":
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6))
    elif scale == "log":
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1)
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    else:
        raise ValueError(f"unknown y scale {scale}")
    ax.grid(axis="y", color="#D4D7DA", linewidth=0.42, alpha=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_candidate(
    curves: dict[str, dict[str, np.ndarray]], scale: str, series: tuple
) -> plt.Figure:
    """Draw one full-width clean review candidate."""
    fig, ax = plt.subplots(figsize=(5.5, 3.25))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.17, top=0.96)
    draw_panel(ax, curves, scale, series)
    ax.legend(loc="upper right", frameon=False, handlelength=2.2)
    return fig


def main() -> None:
    """Validate the pinned cache and render the canonical log-scale figure."""
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "figures",
    )
    args = parser.parse_args()
    from reproduction.figures.figcache import paper_statistics

    statistics = paper_statistics()
    provenance = statistics["inputs"]["hpo"][core.BENCH]
    curves = statistics["panels"][core.BENCH]["models"]
    core.configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = draw_candidate(curves, "log", MAIN_SERIES)
    outputs = core.save_figure(
        fig,
        args.output_dir,
        "figure4_arlbench_mean_regret_log",
        ("pdf", "svg", "png"),
    )
    plt.close(fig)
    index_999 = core.ITERATIONS.index(999)
    manifest = {
        "schema": "arlbench-mean-figure-v2",
        "review_only": False,
        "benchmark": core.BENCH,
        "axis": {"scale": "log", "limits": [1e-3, 1.0]},
        "methods": [method for method, *_rest in MAIN_SERIES],
        "task_count": core.EXPECTED_TASKS,
        "iterations": core.ITERATIONS,
        "trajectory_range": [0, 999],
        "normalization": "per-task achievable configuration-best range, clipped to [0,1]",
        "aggregation": (
            "ICICLE training seed 0; 20 BO seeds within task, tasks equally weighted"
        ),
        "uncertainty": statistics["uncertainty"],
        "mean_normalized_regret_at_iteration_999": {
            method: float(values["mean"][index_999]) for method, values in curves.items()
        },
        "inputs": provenance,
        "renderer_sha256": core.content_sha256(Path(__file__)),
        "common_module_sha256": core.content_sha256(Path(core.__file__)),
        "outputs": outputs,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = args.output_dir / "figure4_arlbench_mean_regret_render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outputs": len(outputs), "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
