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

MAX_DEPTH = 10.0
beta = 0.0

# # ── LOSS ──────────────────────────────────────────────────────────────────────
# def get_beta(epoch, warmup=50, beta_max=3.0):
#     """KL annealing: ramp beta from 0 to beta_max over warmup epochs."""
#     return beta_max * min(1.0, epoch / warmup)

def dce_loss(recon, target, valid_mask, mean, logvar, beta=0.0):
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

base       = 'warehouse_detection_dataset'
target_res = (270, 480)   # must match VAE encoder architecture
checkpoint = 'checkpoint/vae_best_20260420_090841.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

test_data = WarehouseDepthDataset(
    depth_dir     = os.path.join(base, 'raw/test/depth'),
    collision_dir = os.path.join(base, 'collision/test'),
    augment=False, target_size=target_res)

test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)

model = VAE(input_dim=1, latent_dim=64, with_logits=False,
            inference_mode=True).to(device)   # inference_mode=True → use mean, no sampling
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()
print(f"Loaded weights from: {checkpoint}")

# TensorBoard writer — results go to runs/test_results/
writer = SummaryWriter('runs/test_results')


# ── TEST LOOP ─────────────────────────────────────────────────────────────────
test_loss       = 0.0
test_recon_loss = 0.0
test_kl_loss    = 0.0

# collect some samples for visualization
vis_depths  = []
vis_colls   = []
vis_recons  = []
vis_masks   = []
MAX_VIS     = 16   # save 16 samples for visualization grids

print("\nRunning test evaluation...")
with torch.no_grad():
    for batch_idx, (depth_t, coll_t, mask_t) in enumerate(test_loader):

        # move each tensor separately — NOT batch.to(device)
        depth_t = depth_t.to(device)
        coll_t  = coll_t.to(device)
        mask_t  = mask_t.to(device)

        recon, mean, logvar, z = model(depth_t)

        loss, rec, kl = dce_loss(recon, coll_t, mask_t, mean, logvar)

        test_loss       += loss.item()
        test_recon_loss += rec.item()
        test_kl_loss    += kl.item()

        # collect samples for visualization (from first few batches only)
        if len(vis_depths) < MAX_VIS:
            n = min(MAX_VIS - len(vis_depths), depth_t.shape[0])
            vis_depths.extend(depth_t[:n].cpu())
            vis_colls.extend(coll_t[:n].cpu())
            vis_recons.extend(recon[:n].cpu())
            vis_masks.extend(mask_t[:n].cpu())

        # log per-batch loss to TensorBoard
        writer.add_scalar('Test/batch_loss',       loss.item(), batch_idx)
        writer.add_scalar('Test/batch_recon_loss', rec.item(),  batch_idx)
        writer.add_scalar('Test/batch_kl_loss',    kl.item(),   batch_idx)

# normalize by number of batches
n_batches       = len(test_loader)
test_loss       /= n_batches
test_recon_loss /= n_batches
test_kl_loss    /= n_batches

print(f"\n{'='*40}")
print(f"Test loss       : {test_loss:.6f}")
print(f"  Reconstruction: {test_recon_loss:.6f}")
print(f"  KL (scaled)   : {test_kl_loss:.6f}")
print(f"{'='*40}")

# log final scalars
writer.add_scalar('Test/final_loss',       test_loss)
writer.add_scalar('Test/final_recon_loss', test_recon_loss)
writer.add_scalar('Test/final_kl_loss',    test_kl_loss)


# ── TENSORBOARD IMAGE GRIDS ───────────────────────────────────────────────────
# stack collected samples into grids of shape (N, 1, H, W)
def make_grid_tb(tensor_list, nrow=4):
    """Stack a list of (1,H,W) tensors into a TensorBoard image grid."""
    from torchvision.utils import make_grid
    stacked = torch.stack(tensor_list)          # (N, 1, H, W)
    grid    = make_grid(stacked, nrow=nrow,
                        normalize=True, value_range=(0, 1),
                        pad_value=0.5)          # (3, H_grid, W_grid)
    return grid

grid_depth = make_grid_tb(vis_depths)
grid_coll  = make_grid_tb(vis_colls)
grid_recon = make_grid_tb(vis_recons)
grid_mask  = make_grid_tb(vis_masks)

# add to TensorBoard — visible under the "Images" tab
writer.add_image('Test/1_depth_input',         grid_depth, global_step=0)
writer.add_image('Test/2_collision_target',     grid_coll,  global_step=0)
writer.add_image('Test/3_collision_recon',      grid_recon, global_step=0)
writer.add_image('Test/4_valid_mask',           grid_mask,  global_step=0)

# also add a side-by-side comparison for the first 4 samples
for i in range(min(4, len(vis_depths))):
    comparison = torch.cat([
        vis_depths[i],    # (1, H, W)
        vis_colls[i],
        vis_recons[i],
        vis_masks[i]
    ], dim=2)             # concat along width → (1, H, 4W)
    writer.add_image(f'Test/sample_{i:02d}_depth|target|recon|mask',
                     comparison, global_step=0)

writer.flush()
writer.close()

print(f"\nTensorBoard results saved to: runs/test_results/")
print("Launch with:")
print("  tensorboard --logdir runs/test_results --port 6006")
print("Then open: http://localhost:6006")
print("\nDone.")
# # ── DATALOADERS ───────────────────────────────────────────────────────────────
# base       = 'warehouse_detection_dataset'
# target_res = (270, 480) # must match VAE architecture

# test_data = WarehouseDepthDataset(
#     depth_dir     = os.path.join(base, 'raw/test/depth'),
#     collision_dir = os.path.join(base, 'collision/test'),
#     augment=False, target_size=target_res)

# test_loader   = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)


# # ── MODEL ─────────────────────────────────────────────────────────────────────
# device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
# model     = VAE(input_dim=1, latent_dim=64, with_logits=False,
#                 inference_mode=False).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=15, verbose=True)

# # load the best checkpoint and run on test set — only do this once
# model.load_state_dict(torch.load('checkpoint/vae_best_20260416_110853.pt'))
# model.eval()
# test_loss = 0.0
# with torch.no_grad():
#     for batch in test_loader:
#         batch = batch.to(device)
#         recon, mean, logvar, z = model(batch)
#         loss, rec, kl += dce_loss(recon, batch, mean, logvar).item()

#         test_loss += loss.item()
#         test_recon_loss += rec.item()
#         test_kl_loss += kl.item()
#     test_loss /= len(test_loader)
#     test_recon_loss /= len(test_loader)
#     test_kl_loss /= len(test_loader)
# print(f"\nFinal test loss: {test_loss:.2f}")
# print("Done. Use checkpoints/vae_best.pt with VAEImageEncoder.")