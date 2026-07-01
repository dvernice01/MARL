#!/usr/bin/env python3
"""
plot_csv.py — Elegant trend plotter for W&B-exported CSV files.

It expects a W&B *chart-panel* export: many rows (one per logged step), an
x-axis column (e.g. "Step") and one or more metric columns. Columns ending in
__MIN / __MAX / __STEP are ignored. Smoothing follows W&B's bias-corrected
exponential moving average: raw curves are drawn faded, smoothed curves on top.

It does NOT accept a W&B *runs-table* export (one row per run): that file has no
time axis and the script will refuse it with an explanatory message.

Examples:
    python plot_csv.py run_for_thesis/lin_mae_vc.csv
    python plot_csv.py run_for_thesis/fin_dist_first.csv \
        --columns "solar-hill-41 - Info / Episode_Info/final_distance_to_goal" \
        --title "Position Tracking Error" --xlabel "Training steps" \
        --ylabel "Tracking Error" --ema 0.9 --dpi 300
"""

import argparse
import warnings
from pathlib import Path

# Silence the pandas-3.0 pyarrow DeprecationWarning before importing pandas.
warnings.filterwarnings("ignore", message=".*[Pp]yarrow.*")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot trend curves from a W&B chart-panel CSV export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("csv", type=Path, help="Input CSV file (W&B chart-panel export).")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output PNG path. Default: <csv stem>.png next to the CSV.")
    p.add_argument("-c", "--columns", type=str, nargs="+", default=None,
                   help="Metric columns to plot. Space-tolerant exact match, so "
                        "'Info/Episode_Info/lin_vel_mae' matches the column "
                        "'Info / Episode_Info/lin_vel_mae'. Default: all metric columns.")
    p.add_argument("--title", type=str, default=None, help="Plot title.")
    p.add_argument("--xlabel", type=str, default="Step", help="X-axis label.")
    p.add_argument("--ylabel", type=str, default="Value", help="Y-axis label.")
    p.add_argument("--x-col", type=str, default=None,
                   help="X-axis column. Default: 'Step' if present, else first column.")
    p.add_argument("--ema", type=float, default=0.9,
                   help="EMA smoothing factor in [0, 1). Set to 0 to disable smoothing.")
    p.add_argument("--no-raw", action="store_true",
                   help="Hide raw curves, show only smoothed lines.")
    p.add_argument("--figsize", type=float, nargs=2, default=(8.0, 5.0),
                   metavar=("W", "H"), help="Figure size in inches.")
    p.add_argument("--dpi", type=int, default=200, help="Output DPI.")
    p.add_argument("--logy", action="store_true", help="Use a logarithmic y-axis.")
    return p.parse_args()


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """W&B-style exponential moving average with bias correction.

    s_t = alpha * s_{t-1} + (1 - alpha) * x_t
    w_t = alpha * w_{t-1} + (1 - alpha)
    out = s_t / w_t
    NaNs are preserved and do not advance the state.
    """
    if alpha <= 0.0:
        return values.astype(np.float64, copy=True)
    out = np.empty_like(values, dtype=np.float64)
    last = 0.0
    weight = 0.0
    for i, v in enumerate(values):
        if np.isnan(v):
            out[i] = np.nan
            continue
        last = alpha * last + (1.0 - alpha) * v
        weight = alpha * weight + (1.0 - alpha)
        out[i] = last / weight if weight > 0.0 else v
    return out


def normalize(name: str) -> str:
    """Drop all spaces so 'Info / Episode_Info/x' == 'Info/Episode_Info/x'."""
    return name.replace(" ", "")


def pick_x_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise SystemExit(
                f"Column '{requested}' not found. Available: {list(df.columns)}"
            )
        return requested
    for candidate in ("Step", "step", "global_step", "Epoch", "epoch"):
        if candidate in df.columns:
            return candidate
    return df.columns[0]


def all_metric_columns(df: pd.DataFrame, x_col: str) -> list[str]:
    """Every numeric column except the x-axis and W&B __MIN/__MAX/__STEP aux columns."""
    drop_suffixes = ("__MIN", "__MAX", "__STEP")
    cols: list[str] = []
    for c in df.columns:
        if c == x_col:
            continue
        if any(c.endswith(s) for s in drop_suffixes):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def resolve_requested_columns(df: pd.DataFrame, requested: list[str],
                              x_col: str) -> list[str]:
    """Map user-given names to real column names via space-tolerant exact match."""
    lookup = {normalize(c): c for c in df.columns}
    resolved: list[str] = []
    for q in requested:
        key = normalize(q)
        if key not in lookup:
            available = [c for c in all_metric_columns(df, x_col)]
            raise SystemExit(
                f"Requested column '{q}' not found.\n"
                f"Available metric columns:\n  " + "\n  ".join(available)
            )
        col = lookup[key]
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise SystemExit(f"Column '{col}' is not numeric and cannot be plotted.")
        resolved.append(col)
    return resolved


def shorten_label(name: str) -> str:
    """W&B columns look like '<run-name> - <metric>'. Keep the run-name part."""
    if " - " in name:
        return name.split(" - ", 1)[0]
    return name


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 1.8,
    })


def main() -> None:
    args = parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"Input CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    if len(df) < 2:
        raise SystemExit(
            "This CSV has only one data row, so there is no trend to plot.\n"
            "It is a W&B *runs-table* export (one row per run, no time axis).\n"
            "Export the *run history* instead: in the W&B workspace open the "
            "chart panel for your metric, use its top-right menu and download "
            "the panel data as CSV. That file has many rows (one per step)."
        )

    x_col = pick_x_column(df, args.x_col)

    if args.columns is not None:
        metric_cols = resolve_requested_columns(df, args.columns, x_col)
    else:
        metric_cols = all_metric_columns(df, x_col)

    if not metric_cols:
        raise SystemExit("No numeric metric columns found to plot.")

    apply_style()
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    cmap = plt.get_cmap("tab10")

    x = df[x_col].to_numpy()

    for i, col in enumerate(metric_cols):
        y = df[col].to_numpy(dtype=np.float64)
        color = cmap(i % 10)
        label = shorten_label(col)
        if args.ema > 0.0:
            y_smooth = ema(y, args.ema)
            if not args.no_raw:
                ax.plot(x, y, color=color, alpha=0.25, linewidth=1.0)
            ax.plot(x, y_smooth, color=color, label=label)
        else:
            ax.plot(x, y, color=color, label=label)

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    if args.title is not None:
        ax.set_title(args.title)
    if args.logy:
        ax.set_yscale("log")

    if len(metric_cols) > 1:
        # Legend outside the axes on the right: never overlaps the curves and
        # never breaks the layout, regardless of how many series are plotted.
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    out_path = args.output or args.csv.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
