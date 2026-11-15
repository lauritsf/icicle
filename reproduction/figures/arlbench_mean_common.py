"""Shared presentation helpers for the final static ARLBench figures."""

import hashlib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reproduction.figures.paper_method_colors import METHOD_COLORS

BENCH = "arlbench_mean"
EXPECTED_TASKS = 42
ITERATIONS = list(range(1000))
SERIES = (
    ("icicle-s0", "ICICLE", METHOD_COLORS["icicle"], "-", 2.1),
    ("asha", "ASHA", METHOD_COLORS["asha"], (0, (4, 1.5)), 1.55),
    ("hyperband", "Hyperband", METHOD_COLORS["hyperband"], (0, (6, 2, 1, 2)), 1.5),
    ("random_search", "Random", METHOD_COLORS["random_search"], (0, (1, 1.5)), 1.45),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def content_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_matplotlib() -> None:
    """Apply the paper result-figure typography."""
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans"],
            "font.size": 7.4,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.4,
            "axes.linewidth": 0.55,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, suffixes: tuple[str, ...]) -> dict:
    """Save requested formats and return their content hashes."""
    outputs = {}
    for suffix in suffixes:
        path = output_dir / f"{stem}.{suffix}"
        kwargs = {"dpi": 400} if suffix == "png" else {}
        fig.savefig(path, **kwargs)
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
        outputs[path.name] = content_sha256(path)
    return outputs


