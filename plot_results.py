"""Plot utilities for assignment results."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def create_plots(rows: list[dict[str, str]], output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plots_dir / "relative_error.png",
        plots_dir / "iterations.png",
        plots_dir / "time.png",
    ]
    _plot_metric(rows, "relative_error", "Errore relativo", paths[0], log_y=True)
    _plot_metric(rows, "iterations", "Iterazioni", paths[1], log_y=False)
    _plot_metric(rows, "elapsed_seconds", "Tempo [s]", paths[2], log_y=True)
    return paths


def _plot_metric(
    rows: list[dict[str, str]],
    metric: str,
    ylabel: str,
    path: Path,
    log_y: bool,
) -> None:
    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["matrix"], row["method"])].append(row)

    matrices = sorted({row["matrix"] for row in rows})
    fig, axes = plt.subplots(len(matrices), 1, figsize=(9, 3.2 * len(matrices)), squeeze=False)

    for ax, matrix_name in zip(axes[:, 0], matrices):
        for (current_matrix, method), values in grouped.items():
            if current_matrix != matrix_name:
                continue
            values = sorted(values, key=lambda item: float(item["tol"]))
            xs = [float(item["tol"]) for item in values]
            ys = [float(item[metric]) for item in values]
            ax.plot(xs, ys, marker="o", label=method)

        ax.set_title(matrix_name)
        ax.set_xlabel("Tolleranza")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.invert_xaxis()
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
