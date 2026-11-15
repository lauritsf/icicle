"""Shared styles and output helpers for final Meta-Album task cards."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reproduction.baselines.qtmeta_common import content_sha256  # noqa: E402
from reproduction.figures.paper_method_colors import METHOD_COLORS  # noqa: E402

METHODS = ("icicle", "quicktune", "asha", "random_search")
SPLITS = ("micro", "mini", "extended")
EXPECTED_TASKS = {"micro": 30, "mini": 30, "extended": 26}
METHOD_STYLE = {
    "icicle": {
        "label": "ICICLE",
        "color": METHOD_COLORS["icicle"],
        "linestyle": "-",
        "linewidth": 1.7,
        "marker": "o",
    },
    "quicktune": {
        "label": "Quick-Tune",
        "color": METHOD_COLORS["quicktune"],
        "linestyle": "--",
        "linewidth": 1.5,
        "marker": "s",
    },
    "asha": {
        "label": "ASHA",
        "color": METHOD_COLORS["asha"],
        "linestyle": "-.",
        "linewidth": 1.5,
        "marker": "^",
    },
    "random_search": {
        "label": "Random",
        "color": METHOD_COLORS["random_search"],
        "linestyle": ":",
        "linewidth": 1.25,
        "marker": "D",
    },
}
FACET_LABELS = {"micro": "micro (1 h)", "mini": "mini (4 h)", "extended": "extended (16 h)"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Meta-Album task-card validation failed: {message}")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans"],
            "font.size": 7.4,
            "axes.titlesize": 7.8,
            "axes.labelsize": 7.4,
            "axes.linewidth": 0.55,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def task_label(task: str) -> str:
    """Compact task label keeping the frozen set identity."""
    return task.split("/", 2)[2]


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> dict:
    records = {}
    for suffix, kwargs in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 600}),
    ):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, **kwargs)
        if suffix == ".svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
        records[path.name] = content_sha256(path)
    return records
