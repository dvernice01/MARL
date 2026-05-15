import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from vae_residual_batch import VAE
import wandb
import yaml
import random

MAX_DEPTH = 10.0                                                                                                                                                                  
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


# ── DATASETS ──────────────────────────────────────────────────────────────────

class IsaacLabDepthDataset(Dataset):
    def __init__(self, data_dir, augment=False, target_size=(270, 480), max_samples=None):
        self.depth_dir     = os.path.join(data_dir, "depth")
        self.collision_dir = os.path.join(data_dir, "collision")
        self.augment       = augment
        self.target_size   = target_size

        depth_files     = set(f for f in os.listdir(self.depth_dir)     if f.endswith('.npy'))
        collision_files = set(f for f in os.listdir(self.collision_dir) if f.endswith('.npy'))
        self.files = sorted(depth_files & collision_files)

        if max_samples is not None:
            self.files = random.sample(self.files, min(max_samples, len(self.files)))

        print(f"  [IsaacLab] depth: {self.depth_dir}")
        print(f"  [IsaacLab] collision: {self.collision_dir}")
        print(f"  [IsaacLab] paired samples: {len(self.files)}")

        if len(self.files) == 0:
            raise RuntimeError("No paired .npy files found in IsaacLab dataset.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        depth = np.load(os.path.join(self.depth_dir, fname)).astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = np.clip(depth, 0.0, MAX_DEPTH) / MAX_DEPTH

        coll = np.load(os.path.join(self.collision_dir, fname)).astype(np.float32)
        coll = np.nan_to_num(coll, nan=0.0, posinf=0.0, neginf=0.0)
        if coll.ndim == 3:
            coll = coll[:, :, 0]
        coll = np.clip(coll, 0.0, 1.0)

        valid_mask = (coll > 0).astype(np.float32)

        depth_t = torch.from_numpy(depth).float().unsqueeze(0)
        coll_t  = torch.from_numpy(coll).float().unsqueeze(0)
        mask_t  = torch.from_numpy(valid_mask).float().unsqueeze(0)

        if depth_t.shape[-2:] != self.target_size:
            depth_t = F.interpolate(depth_t.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
            coll_t  = F.interpolate(coll_t.unsqueeze(0),  size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
            mask_t  = F.interpolate(mask_t.unsqueeze(0),  size=self.target_size, mode='nearest').squeeze(0)

        if self.augment and torch.rand(1).item() > 0.5:
            depth_t = torch.flip(depth_t, dims=[2])
            coll_t  = torch.flip(coll_t,  dims=[2])
            mask_t  = torch.flip(mask_t,  dims=[2])

        if torch.isnan(depth_t).any() or torch.isnan(coll_t).any():
            depth_t = torch.nan_to_num(depth_t, nan=0.0)
            coll_t  = torch.nan_to_num(coll_t,  nan=0.0)
            mask_t  = torch.zeros_like(mask_t)

        return depth_t, coll_t, mask_t


class WarehouseDepthDataset(Dataset):
    def __init__(self, depth_dir, collision_dir, augment=False, target_size=(270, 480)):
        self.depth_dir     = depth_dir
        self.collision_dir = collision_dir
        self.augment       = augment
        self.target_size   = target_size
        self.files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
        print(f"  [Warehouse] Found {len(self.files)} samples in {depth_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        depth = np.load(os.path.join(self.depth_dir, fname)).astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = np.clip(depth, 0, MAX_DEPTH) / MAX_DEPTH

        coll = np.load(os.path.join(self.collision_dir, fname)).astype(np.float32)
        coll = np.nan_to_num(coll, nan=0.0, posinf=0.0, neginf=0.0)
        if coll.ndim == 3:
            coll = coll[:, :, 0]

        valid_mask = (coll > 0).astype(np.float32)
        coll = np.clip(coll, 0.0, 1.0)

        depth_t = torch.from_numpy(depth).float().unsqueeze(0)
        coll_t  = torch.from_numpy(coll).float().unsqueeze(0)
        mask_t  = torch.from_numpy(valid_mask).float().unsqueeze(0)

        depth_t = F.interpolate(depth_t.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        coll_t  = F.interpolate(coll_t.unsqueeze(0),  size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        mask_t  = F.interpolate(mask_t.unsqueeze(0),  size=self.target_size, mode='nearest').squeeze(0)

        if self.augment and torch.rand(1) > 0.5:
            depth_t = torch.flip(depth_t, dims=[2])
            coll_t  = torch.flip(coll_t,  dims=[2])
            mask_t  = torch.flip(mask_t,  dims=[2])

        if torch.isnan(depth_t).any() or torch.isnan(coll_t).any():
            depth_t = torch.nan_to_num(depth_t, nan=0.0)
            coll_t  = torch.nan_to_num(coll_t,  nan=0.0)
            mask_t  = torch.zeros_like(mask_t)

        return depth_t, coll_t, mask_t


# ── LOADERS ───────────────────────────────────────────────────────────────────

def make_isaaclab_loaders(data_dir, val_ratio=0.1, batch_size=32,
                          num_workers=2, target_size=(270, 480)):
    full_dataset = IsaacLabDepthDataset(data_dir=data_dir, augment=False, target_size=target_size)
    n_total = len(full_dataset)
    n_val   = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_set.dataset.augment = True
    print(f"  Train: {n_train} | Val: {n_val}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker)
    return train_loader, val_loader


def make_warehouse_loader(base_dir, batch_size=32, num_workers=2, target_size=(270, 480)):
    dataset = WarehouseDepthDataset(
        depth_dir     = os.path.join(base_dir, 'raw/test/depth'),
        collision_dir = os.path.join(base_dir, 'collision/test'),
        augment       = False,
        target_size   = target_size,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker)


# ── LOSS ──────────────────────────────────────────────────────────────────────

def build_beta_schedule(warmup_end=100, beta_max=10.0, total_epochs=200):
    schedule = np.zeros(total_epochs)
    schedule[:warmup_end] = np.linspace(0.0, beta_max, warmup_end)
    schedule[warmup_end:] = beta_max
    return schedule


def dce_loss(recon, target, valid_mask, mean, logvar, beta=10.0, use_free_bits=False):
    squared_error = (recon - target) ** 2
    masked_error  = squared_error * valid_mask
    n_valid       = valid_mask.sum().clamp(min=1)
    recon_loss    = masked_error.sum() / n_valid

    kl_per_dim = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
    if use_free_bits:
        kl_per_dim = torch.clamp(kl_per_dim, min=0.5)
    kl_loss    = kl_per_dim.mean()

    total = recon_loss + beta * kl_loss

    if torch.isnan(total):
        zero = torch.tensor(0.0, requires_grad=True, device=recon.device)
        return zero, zero, zero

    return total, recon_loss, kl_loss


# ── LATENCY MEASUREMENT ───────────────────────────────────────────────────────

def measure_latency(model, device, input_size=(1, 1, 270, 480),
                    n_warmup=10, n_runs=100, input_tensor=None):                                                                                                     

    """
    Measures forward pass latency in milliseconds.
    Returns mean and std over n_runs.
    Uses CUDA events for GPU timing (much more accurate than time.time()).
    """
    model.eval()                                                
    dummy = input_tensor if input_tensor is not None else torch.randn(input_size, device=device)

    # warmup — first runs are always slower due to CUDA JIT
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    if device.type == 'cuda':
        # GPU timing — use CUDA events for accuracy
        torch.cuda.synchronize()
        timings = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for _ in range(n_runs):
                start_event.record()
                _ = model(dummy)
                end_event.record()
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event))  # ms
    else:
        # CPU timing
        timings = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model(dummy)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000)  # ms

    timings = np.array(timings)
    return {
        "latency/mean_ms":   float(np.mean(timings)),
        "latency/std_ms":    float(np.std(timings)),
        "latency/min_ms":    float(np.min(timings)),
        "latency/max_ms":    float(np.max(timings)),
        "latency/p95_ms":    float(np.percentile(timings, 95)),
    }


# ── VISUALIZATION ─────────────────────────────────────────────────────────────

def visualize(model, dataset, device, epoch, save_dir='debug_epochs'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    row_titles = ['Depth input', 'Collision target', 'Reconstruction']
    indices = random.sample(range(len(dataset)), 3)

    with torch.no_grad():
        for col in range(3):
            idx = indices[col]
            depth_t, coll_t, mask_t = dataset[idx]
            recon, *_ = model(coll_t.unsqueeze(0).to(device)) # PER ADESSO LO STO TESTANDO SU DEPTH
            #recon, *_ = model(coll_t.unsqueeze(0).to(device))
            for row, arr in enumerate([
                depth_t.squeeze().numpy(),
                coll_t.squeeze().numpy(),
                recon.squeeze().cpu().numpy(),
            ]):
                ax = axes[row, col]
                ax.imshow(arr, cmap='plasma', vmin=0, vmax=1)
                ax.axis('off')
                if col == 0:
                    ax.set_ylabel(row_titles[row])

    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    path = os.path.join(save_dir, f'epoch_{epoch}.png')
    plt.savefig(path, dpi=120)
    plt.close()

    # log image to wandb
    #wandb.log({"viz/reconstruction": wandb.Image(path)}, commit=False)


# ── TEST LOOP ─────────────────────────────────────────────────────────────────

def run_test(model, test_loader, device, beta, use_free_bits=False):
    """
    Evaluates the model on the Warehouse test set.
    Returns a dict of test metrics.
    """
    model.eval()
    test_loss       = 0.0
    test_recon_loss = 0.0
    test_kl_loss    = 0.0

    with torch.no_grad():
        for inputs, labels, masks in test_loader:
            inputs  = inputs.to(device)
            labels  = labels.to(device)
            masks   = masks.to(device)

            recon, mean, logvar, _ = model(labels)
            loss, recon_l, kl_l = dce_loss(recon, labels, masks, mean, logvar, beta=beta, use_free_bits=use_free_bits)

            test_loss       += loss.item()
            test_recon_loss += recon_l.item()
            test_kl_loss    += kl_l.item()

    n = len(test_loader)
    return {
        "test/loss":       test_loss       / n,
        "test/recon_loss": test_recon_loss / n,
        "test/kl_loss":    test_kl_loss    / n,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    # ── wandb run name = sweep_name/run_id ────────────────────────────────────
    sweep_name = wandb.run.sweep_id or "manual_run"
    run_name   = wandb.run.name    or wandb.run.id
    run_dir    = os.path.join("runs", sweep_name, run_name)
    os.makedirs(run_dir, exist_ok=True)

    cfg = wandb.config
    print(f"Run dir: {run_dir}")
    print(f"Config: {dict(cfg)}")

    # ── device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    target_res = (270, 480)

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = make_isaaclab_loaders(
        data_dir    = "isaaclab_dataset_right_size",
        val_ratio   = 0.1,
        batch_size  = 8,
        num_workers = 2,
        target_size = target_res,
    )
    test_loader = make_warehouse_loader(
        base_dir    = "warehouse_detection_dataset",
        batch_size  = 8,
        num_workers = 2,
        target_size = target_res,
    )
    val_data = val_loader.dataset

    # ── model ─────────────────────────────────────────────────────────────────
    model = VAE(
        input_dim          = 1,
        latent_dim         = cfg.latent_dim,
        with_logits        = False,
        inference_mode     = False,
        num_conv_layers    = cfg.num_conv_layers,
        use_residual       = cfg.use_residual,
        num_deconv_layers  = cfg.num_deconv_layers,
        residual_every     = cfg.residual_every,
        use_skip           = cfg.use_skip,
        #decoder_num_dense = cfg.decoder_num_dense,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=True
    )

    epochs        = 200
    beta_schedule = build_beta_schedule(
        warmup_end = 100,
        beta_max   = cfg.beta_max,
        total_epochs = epochs,
    )

    # ── tensorboard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(os.path.join(run_dir, "tensorboard"))

    # ── initial visualization ─────────────────────────────────────────────────
    vis_dir_test = os.path.join(run_dir, "visualizations_train")
    visualize(model, val_data, device, epoch=0, save_dir=vis_dir_test)

    best_val_recon = float('inf')
    use_free_bits = (cfg.beta_max <= 1)
    timestamp     = datetime.now().strftime('%Y%m%d_%H%M%S')
    #scaler = torch.amp.GradScaler('cuda')
    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        beta = float(beta_schedule[epoch - 1])
        #beta = 0.0
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = train_recon = train_kl = 0.0
        accum_steps = 4
        optimizer.zero_grad()
        for i, (inputs, labels, masks) in enumerate(train_loader):
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
            recon, mean, logvar, _ = model(labels)
            loss, recon_l, kl_l = dce_loss(recon, labels, masks, mean, logvar, beta=beta, use_free_bits=use_free_bits)
            loss = loss / accum_steps
            loss.backward()
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            train_loss  += loss.item() * accum_steps
            train_recon += recon_l.item()
            train_kl    += kl_l.item()

        n_train      = len(train_loader)
        train_loss  /= n_train
        train_recon /= n_train
        train_kl    /= n_train

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = val_recon = val_kl = 0.0

        with torch.no_grad():
            for inputs, labels, masks in val_loader:
                inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
                recon, mean, logvar, _ = model(labels)
                loss, recon_l, kl_l = dce_loss(recon, labels, masks, mean, logvar, beta=beta, use_free_bits=use_free_bits)
                val_loss  += loss.item()
                val_recon += recon_l.item()
                val_kl    += kl_l.item()

        n_val      = len(val_loader)
        val_loss  /= n_val
        val_recon /= n_val
        val_kl    /= n_val

        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train={train_loss:.4f} | val={val_loss:.4f} | beta={beta:.3f}")

        # ── tensorboard ───────────────────────────────────────────────────────
        writer.add_scalars('loss',       {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('recon_loss', {'train': train_recon, 'val': val_recon}, epoch)
        writer.add_scalars('kl_loss',    {'train': train_kl, 'val': val_kl}, epoch)
        writer.add_scalar('beta', beta, epoch)

        # ── wandb log — clean nested structure ────────────────────────────────
        wandb.log({
            "epoch": epoch,
            "beta":  beta,
            "train/loss":       train_loss,
            "train/recon_loss": train_recon,
            "train/kl_loss":    train_kl,
            "val/loss":         val_loss,
            "val/recon_loss":   val_recon,
            "val/kl_loss":      val_kl,
        })

        # ── periodic visualization ────────────────────────────────────────────
        if epoch % 10 == 0 or epoch == epochs:
            visualize(model, val_data, device, epoch, save_dir=vis_dir_test)

        # ── checkpoint ────────────────────────────────────────────────────────
        if val_recon < best_val_recon:
            ckpt_dir  = os.path.join(run_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"vae_best_{timestamp}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved best model (val_recon={val_recon:.4f}) → {ckpt_path}")
            wandb.run.summary["best_val_recon"] = best_val_recon
            wandb.run.summary["best_checkpoint"] = ckpt_path

    # ── TEST on Warehouse dataset ──────────────────────────────────────────────
    print("\n── Running test on Warehouse dataset ──")

    # load best checkpoint for testing
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    #test_metrics = run_test(model, test_loader, device, beta=float(beta_schedule[-1]))
    test_metrics = run_test(model, test_loader, device, beta= beta, use_free_bits=use_free_bits)

    vis_dir_test = os.path.join(run_dir, "visualizations_test")
    for i in range(5):                                                                                                                                                   
        visualize(model, test_loader.dataset, device, epoch=f"test_{i}", save_dir=vis_dir_test)

    rand_idx = random.randint(0, len(test_loader.dataset) - 1)                                                                                                           
    sample_depth, _, _ = test_loader.dataset[rand_idx]
    real_input = sample_depth.unsqueeze(0).to(device)                                                                                           
                                                                        
    latency_metrics = measure_latency(                                                                                                                                   
        model,                        
        device,                                                                                                                                                          
        input_tensor = real_input,
        n_warmup     = 20,        
        n_runs       = 100,
    )  

    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("Latency metrics:")
    for k, v in latency_metrics.items():
        print(f"  {k}: {v:.3f} ms")

    # log test + latency to wandb
    wandb.log({**test_metrics, **latency_metrics})

    # save to run summary so it's visible in sweep table
    for k, v in {**test_metrics, **latency_metrics}.items():
        wandb.run.summary[k] = v

    writer.flush()
    writer.close()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # ✅ get sweep_id from environment variable — set automatically by wandb agent
    sweep_id = os.environ.get("WANDB_SWEEP_ID", "manual_run")

    run = wandb.init(
        project = "quadcoptervae_sweep",
        config  = config,
        group   = sweep_id,   # ✅ passed at init time, not after
    )

    main()
    wandb.finish()