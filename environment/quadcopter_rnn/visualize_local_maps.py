"""
Visualize local combined maps saved by QuadcopterRnnEnv.
Each .npy file has shape (2, n, n, n):
  channel 0 = local binary occupancy
  channel 1 = local SVS (Shannon entropy of visit distribution)

Usage:
  python visualize_local_maps.py --path outputs/local_maps/env0_step100.npy
  python visualize_local_maps.py --dir  outputs/local_maps/ --env 0
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_map(path: str) -> np.ndarray:
    data = np.load(path)
    assert data.shape[0] == 2, f"Expected (2, n, n, n), got {data.shape}"
    return data  # (2, n, n, n)


def plot_horizontal_slices(data: np.ndarray, title_prefix: str, save_path: str | None = None):
    """
    For each channel, show horizontal (z-axis) slices in a grid.
    Each row = one channel, each column = one z slice.
    """
    n = data.shape[1]
    channel_names = ["Occupancy", "SVS (Shannon entropy)"]
    channel_cmaps = ["Reds", "viridis"]

    # Pick evenly spaced z slices (max 8)
    z_indices = np.linspace(0, n - 1, min(8, n), dtype=int)
    ncols = len(z_indices)
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 5))
    fig.suptitle(title_prefix, fontsize=13)

    for row, (ch_name, cmap) in enumerate(zip(channel_names, channel_cmaps)):
        vmin, vmax = data[row].min(), data[row].max()
        for col, zi in enumerate(z_indices):
            ax = axes[row, col]
            im = ax.imshow(
                data[row, zi],          # (n_y, n_x) horizontal slice at z=zi
                origin="lower",
                cmap=cmap,
                vmin=vmin, vmax=vmax,
                interpolation="nearest"
            )
            ax.set_title(f"z={zi}", fontsize=8)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(ch_name, fontsize=9)

        fig.colorbar(im, ax=axes[row, :], shrink=0.6, pad=0.02)

    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved slices → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_3d_voxels(data: np.ndarray, title_prefix: str, save_path: str | None = None):
    """
    3D scatter plot for both channels side by side.
    Occupancy: binary — show occupied voxels.
    SVS: show voxels with entropy > threshold, colored by value.
    """
    occ = data[0]   # (n, n, n)
    svs = data[1]   # (n, n, n)

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(title_prefix, fontsize=13)

    # ── Occupancy ─────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(121, projection="3d")
    iz, iy, ix = np.where(occ > 0.5)
    ax1.scatter(ix, iy, iz, s=2, c="red", alpha=0.4, marker="s")
    ax1.set_box_aspect([16, 16, 8]) 
    ax1.set_title("Occupancy (occupied voxels)")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")

    # ── SVS ───────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(122, projection="3d")
    threshold = svs.max() * 0.05   # show top 95% of entropy mass
    iz, iy, ix = np.where(svs > threshold)
    vals = svs[iz, iy, ix]
    sc = ax2.scatter(ix, iy, iz, s=2, c=vals, cmap="viridis", alpha=0.5, marker="s")
    ax2.set_box_aspect([16, 16, 8]) 
    fig.colorbar(sc, ax=ax2, shrink=0.5, label="entropy")
    ax2.set_title("SVS — visit entropy")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")

    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved 3D plot → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_center_cross_sections(data: np.ndarray, title_prefix: str, save_path: str | None = None):
    """
    Show XY, XZ, YZ cross-sections through the cube center for both channels.
    """
    nz, ny, nx = data.shape[1], data.shape[2], data.shape[3]
    channel_names = ["Occupancy", "SVS"]
    channel_cmaps = ["Reds", "viridis"]
    plane_names   = ["XY (top-down)", "XZ (front)", "YZ (side)"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"{title_prefix} — center cross-sections", fontsize=13)

    for row, (ch_name, cmap) in enumerate(zip(channel_names, channel_cmaps)):
        ch = data[row]
        slices = [ch[nz//2, :, :], ch[:, ny//2, :], ch[:, :, nx//2]]
        for col, (sl, plane) in enumerate(zip(slices, plane_names)):
            ax = axes[row, col]
            im = ax.imshow(sl, origin="lower", cmap=cmap, interpolation="nearest")
            ax.set_title(f"{ch_name} | {plane}", fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, shrink=0.8)

    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved cross-sections → {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_file(path: str, output_dir: str | None = None):
    data = load_map(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    title = stem.replace("_", " ")

    def out(suffix):
        if output_dir is None:
            return None
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"{stem}_{suffix}.png")

    print(f"\n[{stem}]  shape={data.shape}  occ_max={data[0].max():.3f}  svs_max={data[1].max():.4f}")
    plot_center_cross_sections(data, title, out("cross"))
    plot_horizontal_slices(data, title, out("slices"))
    plot_3d_voxels(data, title, out("3d"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--path", type=str, help="Path to a single .npy file")
    group.add_argument("--dir",  type=str, help="Directory of .npy files to scan")
    parser.add_argument("--env",    type=int, default=None, help="Filter by env id when using --dir")
    parser.add_argument("--step",   type=int, default=None, help="Filter by step when using --dir")
    parser.add_argument("--save",   type=str, default=None, help="Directory to save PNGs (omit to show interactively)")
    args = parser.parse_args()

    if args.path:
        visualize_file(args.path, args.save)

    elif args.dir:
        files = sorted(f for f in os.listdir(args.dir) if f.endswith(".npy"))

        if args.env is not None:
            files = [f for f in files if f.startswith(f"env{args.env}_")]
        if args.step is not None:
            files = [f for f in files if f"step{args.step}" in f]

        if not files:
            print("No matching .npy files found.")
            return

        for fname in files:
            visualize_file(os.path.join(args.dir, fname), args.save)


if __name__ == "__main__":
    main()