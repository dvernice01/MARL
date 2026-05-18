"""
Filter local map dataset for 3D autoencoder training.

Reads .npy files (shape: 2, 8, 16, 16) from one or more input directories,
applies quality and diversity filters, cleans SVS/OCC overlap, and copies
passing samples to an output directory.

Usage:
  python filter_dataset.py --input outputs/local_maps --output outputs/dataset_filtered
  python filter_dataset.py --input run1/local_maps run2/local_maps --output outputs/dataset_filtered
  python filter_dataset.py --input outputs/local_maps --output outputs/dataset_filtered --min-svs 0.005 --min-occ 0.01
"""

import argparse
import os
import shutil
import numpy as np
from pathlib import Path


def compute_stats(data: np.ndarray) -> dict:
    occ = data[0]
    svs = data[1]
    return {
        "occ_fill": (occ > 0.5).mean(),
        "svs_fill": (svs > 0).mean(),
        "occ_max": occ.max(),
        "svs_max": svs.max(),
        "svs_nonzero_mean": svs[svs > 0].mean() if (svs > 0).any() else 0.0,
    }


def clean_overlap(data: np.ndarray) -> np.ndarray:
    """Zero out SVS where occupancy is present."""
    data = data.copy()
    data[1][data[0] > 0.5] = 0.0
    return data


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0 if (norm_a < 1e-12 and norm_b < 1e-12) else 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def parse_filename(fname: str):
    """Extract (env_id, step) from filenames like env0_step1000.npy."""
    stem = Path(fname).stem
    parts = stem.split("_")
    env_id = int(parts[0].replace("env", ""))
    step = int(parts[1].replace("step", ""))
    return env_id, step


def main():
    parser = argparse.ArgumentParser(description="Filter local map dataset")
    parser.add_argument("--input", nargs="+", required=True, help="Input directories with .npy files")
    parser.add_argument("--output", required=True, help="Output directory for filtered samples")
    parser.add_argument("--min-svs", type=float, default=0.005,
                        help="Min fraction of non-zero SVS voxels (default: 0.005 = 0.5%%)")
    parser.add_argument("--min-occ", type=float, default=0.01,
                        help="Min fraction of occupied voxels (default: 0.01 = 1%%)")
    parser.add_argument("--max-occ", type=float, default=0.5,
                        help="Max fraction of occupied voxels (default: 0.5 = 50%%)")
    parser.add_argument("--max-similarity", type=float, default=0.98,
                        help="Max cosine similarity to previous kept sample from same env (default: 0.98)")
    parser.add_argument("--clean-overlap", action="store_true", default=True,
                        help="Zero out SVS where occupancy is present (default: True)")
    parser.add_argument("--no-clean-overlap", action="store_false", dest="clean_overlap")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without copying files")
    args = parser.parse_args()

    # Collect all .npy files from all input dirs
    all_files = []
    for input_dir in args.input:
        if not os.path.isdir(input_dir):
            print(f"WARNING: {input_dir} is not a directory, skipping")
            continue
        for fname in sorted(os.listdir(input_dir)):
            if fname.endswith(".npy"):
                all_files.append((input_dir, fname))

    print(f"Found {len(all_files)} .npy files across {len(args.input)} input dir(s)")
    if not all_files:
        return

    # Pass 1: quality filter
    rejected_svs = 0
    rejected_occ_low = 0
    rejected_occ_high = 0
    quality_passed = []

    for input_dir, fname in all_files:
        path = os.path.join(input_dir, fname)
        data = np.load(path)

        if data.shape != (2, 8, 16, 16):
            print(f"  SKIP {fname}: unexpected shape {data.shape}")
            continue

        if args.clean_overlap:
            data = clean_overlap(data)

        stats = compute_stats(data)

        if stats["svs_fill"] < args.min_svs:
            rejected_svs += 1
            continue
        if stats["occ_fill"] < args.min_occ:
            rejected_occ_low += 1
            continue
        if stats["occ_fill"] > args.max_occ:
            rejected_occ_high += 1
            continue

        env_id, step = parse_filename(fname)
        quality_passed.append({
            "input_dir": input_dir,
            "fname": fname,
            "data": data,
            "env_id": env_id,
            "step": step,
            "stats": stats,
        })

    print(f"\n── Quality filter ──")
    print(f"  Rejected (SVS too sparse < {args.min_svs:.1%}):  {rejected_svs}")
    print(f"  Rejected (OCC too empty  < {args.min_occ:.1%}):  {rejected_occ_low}")
    print(f"  Rejected (OCC too full   > {args.max_occ:.1%}):  {rejected_occ_high}")
    print(f"  Passed quality: {len(quality_passed)} / {len(all_files)}")

    # Pass 2: diversity filter (per-env, remove near-duplicates)
    quality_passed.sort(key=lambda x: (x["input_dir"], x["env_id"], x["step"]))

    kept = []
    rejected_similar = 0
    last_kept_per_env = {}  # (input_dir, env_id) -> data array

    for sample in quality_passed:
        key = (sample["input_dir"], sample["env_id"])
        if key in last_kept_per_env:
            sim = cosine_similarity(sample["data"], last_kept_per_env[key])
            if sim > args.max_similarity:
                rejected_similar += 1
                continue

        last_kept_per_env[key] = sample["data"]
        kept.append(sample)

    print(f"\n── Diversity filter ──")
    print(f"  Rejected (too similar, cosine > {args.max_similarity}):  {rejected_similar}")
    print(f"  Final kept: {len(kept)} / {len(all_files)}")

    # Stats summary of kept samples
    if kept:
        occ_fills = [s["stats"]["occ_fill"] for s in kept]
        svs_fills = [s["stats"]["svs_fill"] for s in kept]
        print(f"\n── Kept samples stats ──")
        print(f"  OCC fill: min={min(occ_fills):.4f}  median={np.median(occ_fills):.4f}  max={max(occ_fills):.4f}")
        print(f"  SVS fill: min={min(svs_fills):.4f}  median={np.median(svs_fills):.4f}  max={max(svs_fills):.4f}")

    if args.dry_run:
        print("\n[dry-run] No files copied.")
        return

    # Save filtered samples (skip duplicates already in output)
    os.makedirs(args.output, exist_ok=True)

    import hashlib
    existing_hashes = set()
    existing_files = [f for f in os.listdir(args.output) if f.endswith(".npy")]
    for f in existing_files:
        d = np.load(os.path.join(args.output, f))
        existing_hashes.add(hashlib.md5(d.tobytes()).hexdigest())

    start_idx = len(existing_files)
    written = 0
    skipped_dup = 0
    for sample in kept:
        h = hashlib.md5(sample["data"].tobytes()).hexdigest()
        if h in existing_hashes:
            skipped_dup += 1
            continue
        out_path = os.path.join(args.output, f"sample_{start_idx + written:06d}.npy")
        np.save(out_path, sample["data"])
        existing_hashes.add(h)
        written += 1

    print(f"\nAppended {written} new samples to {args.output} "
          f"(skipped {skipped_dup} duplicates already present)")
    if written > 0:
        print(f"  Indices {start_idx}–{start_idx + written - 1}")
    print(f"  Total samples in output: {len(existing_files) + written}")


if __name__ == "__main__":
    main()
