"""One shared plotting theme (CLAUDE.md section 5: no ad-hoc matplotlib styling).

Call :func:`apply_theme` once before plotting. Figures built from synthetic or demo data
must be saved with ``synthetic=True`` (or stamped via :func:`mark_synthetic`) so a
progress-report reader never mistakes a pipeline demonstration for a measured result
(CLAUDE.md section 8).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

#: Single accent colour, reused everywhere.
ACCENT = "#2563eb"
#: Semantic colours for CheXpert label states.
LABEL_COLORS = {
    "positive": "#16a34a",
    "negative": "#94a3b8",
    "uncertain": "#f59e0b",
    "blank": "#cbd5e1",
}
#: Fixed colours per split so every figure in the report agrees.
SPLIT_COLORS = {"train": "#2563eb", "val": "#f59e0b", "test": "#16a34a"}


def apply_theme() -> None:
    """Set repo-wide matplotlib rcParams. Idempotent; safe to call per script/notebook."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 150,
            "figure.dpi": 110,
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
        }
    )


def mark_synthetic(fig: plt.Figure) -> None:
    """Stamp a figure as synthetic/demo. Use on every figure not built from real data."""
    fig.text(
        0.99,
        0.01,
        "SYNTHETIC DATA - pipeline demonstration, not a result",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#b91c1c",
        alpha=0.85,
        style="italic",
    )


def save_figure(fig: plt.Figure, path: str | Path, *, synthetic: bool = False) -> Path:
    """Save ``fig`` to ``path`` (creating parents), optionally stamping it synthetic, then
    close it. Returns the path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if synthetic:
        mark_synthetic(fig)
    fig.savefig(path)
    plt.close(fig)
    return path
