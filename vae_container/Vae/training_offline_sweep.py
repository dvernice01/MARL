# train.py — loads precomputed collision images, no on-the-fly computation
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from VAE import VAE
import wandb
import yaml

MAX_DEPTH = 10.0

# config variable with hyperparameter values
config = {"beta":0.001}




# ── DATASET: loads precomputed pairs ──────────────────────────────────────────
class WarehouseDepthDataset(Dataset):
    def __init__(self, depth_dir, collision_dir, augment=False,
                 target_size=(270, 480)):
        self.depth_dir     = depth_dir
        self.collision_dir = collision_dir
        self.augment       = augment
        self.target_size   = target_size
        self.files = sorted([f for f in os.listdir(depth_dir)
                             if f.endswith('.npy')])
        print(f"  Found {len(self.files)} samples in {depth_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        # load and sanitize depth
        depth = np.load(os.path.join(self.depth_dir, fname)).astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = np.clip(depth, 0, MAX_DEPTH) / MAX_DEPTH

        # load precomputed collision image
        coll = np.load(os.path.join(self.collision_dir, fname)).astype(np.float32)
        coll = np.nan_to_num(coll, nan=0.0, posinf=0.0, neginf=0.0)
        if coll.ndim == 3:
            coll = coll[:, :, 0]

        # valid mask: collision image was set to 0 for invalid pixels
        valid_mask = (coll > 0).astype(np.float32)
        coll       = np.clip(coll, 0.0, 1.0)

        # to tensors (1, H, W)
        depth_t = torch.from_numpy(depth).float().unsqueeze(0)
        coll_t  = torch.from_numpy(coll).float().unsqueeze(0)
        mask_t  = torch.from_numpy(valid_mask).float().unsqueeze(0)

        # resize to VAE input resolution (270, 480)
        depth_t = F.interpolate(depth_t.unsqueeze(0), size=self.target_size,
                                mode='bilinear', align_corners=False).squeeze(0)
        coll_t  = F.interpolate(coll_t.unsqueeze(0),  size=self.target_size,
                                mode='bilinear', align_corners=False).squeeze(0)
        mask_t  = F.interpolate(mask_t.unsqueeze(0),  size=self.target_size,
                                mode='nearest').squeeze(0)

        # augmentation
        if self.augment and torch.rand(1) > 0.5:
            depth_t = torch.flip(depth_t, dims=[2])
            coll_t  = torch.flip(coll_t,  dims=[2])
            mask_t  = torch.flip(mask_t,  dims=[2])

        # final NaN guard — catches any residual issues
        if torch.isnan(depth_t).any() or torch.isnan(coll_t).any():
            depth_t = torch.nan_to_num(depth_t, nan=0.0)
            coll_t  = torch.nan_to_num(coll_t,  nan=0.0)
            mask_t  = torch.zeros_like(mask_t)   # invalidate whole sample

        return depth_t, coll_t, mask_t

# ── LOSS ──────────────────────────────────────────────────────────────────────
def get_beta(epoch, warmup=50, beta_max=3.0):
    """KL annealing: ramp beta from 0 to beta_max over warmup epochs."""
    return beta_max * min(1.0, epoch / warmup)

def dce_loss(recon, target, valid_mask, mean, logvar, beta=3.0):
    squared_error = (recon - target) ** 2
    masked_error  = squared_error * valid_mask
    n_valid       = valid_mask.sum().clamp(min=1)
    recon_loss    = masked_error.sum() / n_valid
    kl_loss       = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    kl_loss       = kl_loss / mean.shape[0]
    total         = recon_loss + beta * kl_loss
    # catch NaN loss — skip this batch if it happens
    if torch.isnan(total):
        return torch.tensor(0.0, requires_grad=True, device=recon.device)
    return total, recon_loss, beta*kl_loss


# ── VISUALIZATION ─────────────────────────────────────────────────────────────
def visualize(model, dataset, device, epoch, save_dir='debug_epochs'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    row_titles = ['Depth input', 'Collision target', 'Reconstruction']
    with torch.no_grad():
        for col in range(3):
            idx = col * (len(dataset) // 3)
            depth_t, coll_t, mask_t = dataset[idx]
            recon, *_ = model(depth_t.unsqueeze(0).to(device))
            for row, arr in enumerate([
                depth_t.squeeze().numpy(),
                coll_t.squeeze().numpy(),
                recon.squeeze().cpu().numpy()
            ]):
                ax = axes[row, col]
                ax.imshow(arr, cmap='plasma', vmin=0, vmax=1)
                ax.axis('off')
                if col == 0:
                    ax.set_ylabel(row_titles[row])
    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch:03d}.png'), dpi=120)
    plt.close()

def main():
    
    # ── DATALOADERS ───────────────────────────────────────────────────────────────
    base       = 'warehouse_detection_dataset'
    target_res = (270, 480) # must match VAE architecture

    train_data = WarehouseDepthDataset(
        depth_dir     = os.path.join(base, 'raw/train/depth'),
        collision_dir = os.path.join(base, 'collision/train'),
        augment=True, target_size=target_res)
    val_data = WarehouseDepthDataset(
        depth_dir     = os.path.join(base, 'raw/val/depth'),
        collision_dir = os.path.join(base, 'collision/val'),
        augment=False, target_size=target_res)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False, num_workers=2)

    # ── MODEL ─────────────────────────────────────────────────────────────────────
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model     = VAE(input_dim=1, latent_dim=64, with_logits=False,
                    inference_mode=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=True)


    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # ── TRAINING LOOP ─────────────────────────────────────────────────────────────
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer     = SummaryWriter(f'runs/dce_{timestamp}')
    best_vloss = float('inf')

    visualize(model, val_data, device, epoch=0, save_dir='debug_epochs')

    with wandb.init(config=config) as run:
        for epoch in np.arange(1, run.config['epochs']):

            model.train()
            train_loss = 0.0
            train_recon_loss = 0.0
            train_kl_loss = 0.0
            for inputs, labels, masks in train_loader:
                inputs, labels, masks = (inputs.to(device),
                                        labels.to(device),
                                        masks.to(device))
                optimizer.zero_grad()
                recon, mean, logvar, z = model(inputs)
                beta = run.config['beta']
                loss, recon_loss, kl_loss = dce_loss(recon, labels, masks, mean, logvar, beta=beta)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                train_recon_loss += recon_loss
                train_kl_loss += kl_loss
            train_loss /= len(train_loader)
            train_recon_loss /= len(train_loader)
            train_kl_loss /= len(train_loader)

            # validate
            model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            with torch.no_grad():
                for vinputs, vlabels, vmasks in val_loader:
                    vinputs, vlabels, vmasks = (vinputs.to(device),
                                                vlabels.to(device),
                                                vmasks.to(device))
                    vrecon, vmean, vlogvar, _ = model(vinputs)
                    new_loss, new_recon, new_kl = dce_loss(vrecon, vlabels, vmasks,
                                        vmean, vlogvar, beta=beta)
                    val_loss += new_loss  
                    val_recon_loss += new_recon
                    val_kl_loss += new_kl      
                val_loss /= len(val_loader)
                val_recon_loss /= len(val_loader)
                val_kl_loss /= len(val_loader)

            print(f'  train: {train_loss:.6f}  val: {val_loss:.6f}')
            scheduler.step(val_loss)

            writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalar('Beta', run.config['beta'], epoch)
            writer.flush()

            if epoch % 10 == 0 or epoch == run.config['epochs'] - 1:
                visualize(model, val_data, device, epoch)

            if val_loss < best_vloss:
                best_vloss = val_loss
                os.makedirs('checkpoint', exist_ok=True)
                torch.save(model.state_dict(),
                        f'checkpoint/vae_best_{timestamp}.pt')
                print(f'  saved best model (val={val_loss:.6f})')

            run.log(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "val_recon_loss": val_recon_loss,
                    "val_kl_loss": val_kl_loss,
                    "train_recon_loss": train_recon_loss,
                    "train_kl_loss": train_kl_loss,
                }
            )

main()

