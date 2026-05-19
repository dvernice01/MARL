import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from autoencoder3D import VAE3D
import wandb
import yaml

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)


# ── DATASET ──────────────────────────────────────────────────────────────────

class LocalMap3DDataset(Dataset):
    """Loads (2, 8, 16, 16) .npy files: channel 0 = occupancy, channel 1 = SVS."""

    def __init__(self, file_list, svs_max=None):
        self.file_list = file_list
        self.svs_max = svs_max

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx]).astype(np.float32)  # (2, 8, 16, 16)

        occ = np.clip(data[0], 0.0, 1.0)
        svs = np.clip(data[1], 0.0, None)

        if self.svs_max is not None and self.svs_max > 0:
            svs = svs / self.svs_max
        svs = np.clip(svs, 0.0, 1.0)

        combined = np.stack([occ, svs], axis=0)  # (2, 8, 16, 16)
        return torch.from_numpy(combined).float()


def make_loaders(data_dir, val_ratio=0.1, test_ratio=0.1,
                 batch_size=32, num_workers=2):
    """Train/val/test split. SVS normalization stat is computed on TRAIN ONLY
    so no val/test information leaks into the input scaling."""
    all_files = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")
    ])
    if len(all_files) == 0:
        raise RuntimeError(f"No .npy files found in {data_dir}")

    n_total = len(all_files)
    indices = list(range(n_total))
    random.Random(42).shuffle(indices)

    n_test = max(1, int(n_total * test_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val - n_test

    train_files = [all_files[i] for i in indices[:n_train]]
    val_files = [all_files[i] for i in indices[n_train:n_train + n_val]]
    test_files = [all_files[i] for i in indices[n_train + n_val:]]

    # SVS max from TRAIN ONLY (no leakage)
    svs_max = 1e-8
    for path in train_files:
        svs_max = max(svs_max, float(np.load(path)[1].max()))
    print(f"  SVS max (train only): {svs_max:.6f}")
    print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test}")

    def mk(files, shuffle):
        return DataLoader(
            LocalMap3DDataset(files, svs_max=svs_max),
            batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker,
        )

    return mk(train_files, True), mk(val_files, False), mk(test_files, False), svs_max


# ── LOSS (autoencoder, no KL) ────────────────────────────────────────────────

def ae_loss(recon, target, occ_weight=1.0, svs_weight=1.0,
            occ_pos_weight=12.0, svs_nonzero_weight=50.0):
    """recon = raw decoder output (model built with with_logits=True → no final
    sigmoid). channel 0 = occupancy (binary), channel 1 = SVS ([0,1]).

    The dataset is extremely imbalanced (~7% occupied, <1% SVS-nonzero), so a
    plain BCE/MSE collapses to predicting zeros. Occupancy uses a stable,
    pos-weighted BCE-with-logits; SVS uses an MSE that upweights the rare
    nonzero voxels.
    """
    occ_logits = recon[:, 0:1]
    occ_target = target[:, 0:1]
    pw = torch.tensor(occ_pos_weight, device=recon.device)
    occ_loss = F.binary_cross_entropy_with_logits(occ_logits, occ_target, pos_weight=pw)

    svs_pred = torch.sigmoid(recon[:, 1:2])          # logits → [0,1]
    svs_target = target[:, 1:2]
    w = 1.0 + svs_nonzero_weight * (svs_target > 0).float()
    svs_loss = (w * (svs_pred - svs_target) ** 2).sum() / w.sum()

    total = occ_weight * occ_loss + svs_weight * svs_loss
    return total, occ_loss, svs_loss


@torch.no_grad()
def recon_metrics(recon, target, occ_thresh=0.5):
    """Threshold-based metrics — loss alone is misleading on imbalanced data
    (an all-zeros model has near-zero loss but IoU≈0)."""
    occ_p = torch.sigmoid(recon[:, 0:1]) > occ_thresh
    occ_t = target[:, 0:1] > 0.5
    inter = (occ_p & occ_t).sum().float()
    union = (occ_p | occ_t).sum().float().clamp(min=1)
    tp = inter
    fp = (occ_p & ~occ_t).sum().float()
    fn = (~occ_p & occ_t).sum().float()

    svs_p = torch.sigmoid(recon[:, 1:2])
    svs_t = target[:, 1:2]
    m = svs_t > 0
    svs_mse_nz = ((svs_p - svs_t) ** 2)[m].mean().item() if m.any() else 0.0

    return {
        "occ_iou": (inter / union).item(),
        "occ_f1": (2 * tp / (2 * tp + fp + fn).clamp(min=1)).item(),
        "svs_mse_nonzero": svs_mse_nz,
    }


# ── TEST (frozen split, evaluated once at the end) ───────────────────────────

@torch.no_grad()
def run_test(model, test_loader, device, occ_weight=1.0, svs_weight=1.0):
    model.eval()
    keys = ["loss", "occ", "svs", "occ_iou", "occ_f1", "svs_mse_nonzero"]
    tot = {k: 0.0 for k in keys}
    for batch in test_loader:
        batch = batch.to(device)
        recon, *_ = model(batch)
        loss, occ_l, svs_l = ae_loss(recon, batch, occ_weight, svs_weight)
        m = recon_metrics(recon, batch)
        tot["loss"] += loss.item()
        tot["occ"] += occ_l.item()
        tot["svs"] += svs_l.item()
        for k in ("occ_iou", "occ_f1", "svs_mse_nonzero"):
            tot[k] += m[k]
    nb = len(test_loader)
    return {f"test/{k}": v / nb for k, v in tot.items()}


# ── LATENCY ──────────────────────────────────────────────────────────────────

def measure_latency(model, device, n_warmup=10, n_runs=100):
    model.eval()
    dummy = torch.randn(1, 2, 8, 16, 16, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        timings = []
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            for _ in range(n_runs):
                start_ev.record()
                _ = model(dummy)
                end_ev.record()
                torch.cuda.synchronize()
                timings.append(start_ev.elapsed_time(end_ev))
    else:
        timings = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model(dummy)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000)

    timings = np.array(timings)
    return {
        "latency/mean_ms": float(np.mean(timings)),
        "latency/std_ms": float(np.std(timings)),
        "latency/min_ms": float(np.min(timings)),
        "latency/max_ms": float(np.max(timings)),
        "latency/p95_ms": float(np.percentile(timings, 95)),
    }


# ── VISUALIZATION ────────────────────────────────────────────────────────────

def visualize_3d(model, dataset, device, epoch, save_dir="debug_epochs"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    n_samples = min(3, len(dataset))
    indices = random.sample(range(len(dataset)), n_samples)

    with torch.no_grad():
        for col, idx in enumerate(indices):
            sample = dataset[idx]  # (2, 8, 16, 16)
            recon, *_ = model(sample.unsqueeze(0).to(device))
            # model outputs logits (with_logits=True) → squash for display
            recon = torch.sigmoid(recon).squeeze(0).cpu().numpy()
            target = sample.numpy()

            fig = plt.figure(figsize=(16, 8))
            fig.suptitle(f"Epoch {epoch} | Sample {idx}", fontsize=13)

            for row, (data, label) in enumerate([
                (target, "Target"),
                (recon, "Recon"),
            ]):
                occ = data[0]
                svs = data[1]

                ax = fig.add_subplot(2, 3, row * 3 + 1, projection="3d")
                iz, iy, ix = np.where(occ > 0.5)
                ax.scatter(ix, iy, iz, s=20, c="red", alpha=0.4, marker="s")
                ax.set_xlim(0, 15); ax.set_ylim(0, 15); ax.set_zlim(0, 7)
                ax.set_title(f"{label} OCC ({len(iz)} voxels)")
                ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

                ax2 = fig.add_subplot(2, 3, row * 3 + 2, projection="3d")
                threshold = max(svs.max() * 0.05, 1e-6)
                iz2, iy2, ix2 = np.where(svs > threshold)
                vals = svs[iz2, iy2, ix2] if len(iz2) > 0 else []
                if len(iz2) > 0:
                    sc = ax2.scatter(ix2, iy2, iz2, s=20, c=vals,
                                     cmap="viridis", alpha=0.5, marker="s")
                    fig.colorbar(sc, ax=ax2, shrink=0.5)
                ax2.set_xlim(0, 15); ax2.set_ylim(0, 15); ax2.set_zlim(0, 7)
                ax2.set_title(f"{label} SVS")
                ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")

                ax3 = fig.add_subplot(2, 3, row * 3 + 3)
                nz = occ.shape[0]
                slice_z = nz // 2
                combined_slice = np.concatenate([occ[slice_z], svs[slice_z]], axis=1)
                ax3.imshow(combined_slice, origin="lower", cmap="viridis")
                ax3.set_title(f"{label} z={slice_z} slice (OCC|SVS)")
                ax3.axis("off")

            plt.tight_layout()
            path = os.path.join(save_dir, f"epoch_{epoch}_sample_{col}.png")
            plt.savefig(path, dpi=120)
            plt.close()


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # ── run directory ─────────────────────────────────────────────────────
    sweep_name = wandb.run.sweep_id or "manual_run"
    run_name = wandb.run.name or wandb.run.id
    run_dir = os.path.join("runs", sweep_name, run_name)
    os.makedirs(run_dir, exist_ok=True)

    cfg = wandb.config
    print(f"Run dir: {run_dir}")
    print(f"Config: {dict(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, svs_max = make_loaders(
        data_dir="/workspace/environment/vae_container/Vae/dataset_3d_collection",
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size=cfg.batch_size,
        num_workers=2,
    )
    val_data = val_loader.dataset

    # ── model (deterministic autoencoder) ─────────────────────────────────
    #   inference_mode=True → z = mean, no sampling (plain AE, no KL).
    #   with_logits=True    → decoder returns raw logits (stable BCE).
    model = VAE3D(
        input_dim=2,
        latent_dim=cfg.latent_dim,
        with_logits=True,
        inference_mode=True,
        num_conv_layers=cfg.num_conv_layers,
        use_residual=cfg.use_residual,
        residual_every=cfg.residual_every,
        num_deconv_layers=cfg.num_deconv_layers,
        use_skip=cfg.use_skip,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    epochs = cfg.epochs
    occ_weight = cfg.occ_weight
    svs_weight = cfg.svs_weight
    occ_pos_weight = cfg.get("occ_pos_weight", 12.0)
    svs_nonzero_weight = cfg.get("svs_nonzero_weight", 50.0)

    # ── tensorboard ───────────────────────────────────────────────────────
    writer = SummaryWriter(os.path.join(run_dir, "tensorboard"))

    vis_dir_train = os.path.join(run_dir, "visualizations_train")
    visualize_3d(model, val_data, device, epoch=0, save_dir=vis_dir_train)

    best_val_recon = float("inf")
    ckpt_path = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):

        # ── train ─────────────────────────────────────────────────────────
        model.train()
        t_loss = t_occ = t_svs = 0.0

        for batch in train_loader:
            batch = batch.to(device)  # (B, 2, 8, 16, 16)
            optimizer.zero_grad()
            recon, *_ = model(batch)
            loss, occ_l, svs_l = ae_loss(
                recon, batch, occ_weight, svs_weight,
                occ_pos_weight, svs_nonzero_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item()
            t_occ += occ_l.item()
            t_svs += svs_l.item()

        n_train = len(train_loader)
        t_loss /= n_train
        t_occ /= n_train
        t_svs /= n_train

        # ── validate ──────────────────────────────────────────────────────
        model.eval()
        v_loss = v_occ = v_svs = 0.0
        v_iou = v_f1 = v_svs_nz = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, *_ = model(batch)
                loss, occ_l, svs_l = ae_loss(
                    recon, batch, occ_weight, svs_weight,
                    occ_pos_weight, svs_nonzero_weight,
                )
                m = recon_metrics(recon, batch)
                v_loss += loss.item()
                v_occ += occ_l.item()
                v_svs += svs_l.item()
                v_iou += m["occ_iou"]
                v_f1 += m["occ_f1"]
                v_svs_nz += m["svs_mse_nonzero"]

        n_val = len(val_loader)
        v_loss /= n_val
        v_occ /= n_val
        v_svs /= n_val
        v_iou /= n_val
        v_f1 /= n_val
        v_svs_nz /= n_val

        scheduler.step(v_loss)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train={t_loss:.4f} (occ={t_occ:.4f} svs={t_svs:.4f}) | "
              f"val={v_loss:.4f} | occ_IoU={v_iou:.3f} F1={v_f1:.3f}")

        # ── tensorboard ───────────────────────────────────────────────────
        writer.add_scalars("loss", {"train": t_loss, "val": v_loss}, epoch)
        writer.add_scalars("occ_loss", {"train": t_occ, "val": v_occ}, epoch)
        writer.add_scalars("svs_loss", {"train": t_svs, "val": v_svs}, epoch)
        writer.add_scalar("val/occ_iou", v_iou, epoch)
        writer.add_scalar("val/occ_f1", v_f1, epoch)
        writer.add_scalar("val/svs_mse_nonzero", v_svs_nz, epoch)

        # ── wandb ─────────────────────────────────────────────────────────
        wandb.log({
            "epoch": epoch,
            "train/loss": t_loss,
            "train/occ_loss": t_occ,
            "train/svs_loss": t_svs,
            "val/loss": v_loss,
            "val/occ_loss": v_occ,
            "val/svs_loss": v_svs,
            "val/occ_iou": v_iou,
            "val/occ_f1": v_f1,
            "val/svs_mse_nonzero": v_svs_nz,
        })

        # ── visualization ─────────────────────────────────────────────────
        if epoch % 10 == 0 or epoch == epochs:
            visualize_3d(model, val_data, device, epoch, save_dir=vis_dir_train)

        # ── checkpoint (best val reconstruction) ──────────────────────────
        if v_loss < best_val_recon:
            best_val_recon = v_loss
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"ae3d_best_{timestamp}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved best model (val_loss={v_loss:.4f}) -> {ckpt_path}")
            wandb.run.summary["best_val_recon"] = best_val_recon
            wandb.run.summary["best_checkpoint"] = ckpt_path

    # ── latency ───────────────────────────────────────────────────────────
    latency_metrics = measure_latency(model, device)
    print("Latency metrics:")
    for k, v in latency_metrics.items():
        print(f"  {k}: {v:.3f} ms")
    wandb.log(latency_metrics)
    for k, v in latency_metrics.items():
        wandb.run.summary[k] = v

    # ── TEST on frozen split (load best checkpoint, evaluate once) ─────────
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    test_metrics = run_test(model, test_loader, device, occ_weight, svs_weight)
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    wandb.log(test_metrics)
    for k, v in test_metrics.items():
        wandb.run.summary[k] = v

    # ── final visualization ───────────────────────────────────────────────
    vis_dir_final = os.path.join(run_dir, "visualizations_final")
    for i in range(5):
        visualize_3d(model, test_loader.dataset, device,
                     epoch=f"final_{i}", save_dir=vis_dir_final)

    writer.flush()
    writer.close()


# ── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open("config_3d.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    sweep_id = os.environ.get("WANDB_SWEEP_ID", "manual_run")

    run = wandb.init(
        project="quadcopter_vae3d_sweep",
        config=config,
        group=sweep_id,
    )

    main()
    wandb.finish()
