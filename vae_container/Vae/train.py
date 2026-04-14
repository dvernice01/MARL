import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import kagglehub
import matplotlib.pyplot as plt
from VAE import VAE
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from CollisionMap import depth_to_collision_image
import cv2
from scipy.ndimage import grey_dilation
import trimesh as tm 
from tqdm import tqdm

def visualize_dce_debug(model, dataset, device, num_samples=3, save_path='debug_visualization.png'):
    """
    Visualizes for each sample:
      Row 1: Original raw depth map (input x)
      Row 2: Collision image (target xcoll)
      Row 3: Reconstructed collision image (output of VAE after forward pass)
      Row 4: Valid pixel mask
    """
    model.eval()

    fig, axes = plt.subplots(nrows=4, ncols=num_samples, figsize=(5 * num_samples, 16))

    row_titles = [
        'Raw depth input (x)',
        'Collision image target (x_coll)',
        'Reconstructed collision (x_coll_recon)',
        'Valid pixel mask'
    ]

    with torch.no_grad():
        for col in range(num_samples):
            # pick a random sample from the dataset
            idx = torch.randint(len(dataset), (1,)).item()
            depth_input, collision, valid_mask = dataset[idx]

            # add batch dimension for model: (1, 1, H, W)
            depth_input_batch = depth_input.unsqueeze(0).to(device)

            # forward pass through the VAE
            recon, mean, logvar, z = model(depth_input_batch)

            # move everything to numpy for plotting — squeeze channel dim
            depth_np  = depth_input.squeeze(0).cpu().numpy()          # (H, W)
            coll_np   = collision.squeeze(0).cpu().numpy()             # (H, W)
            recon_np  = recon.squeeze(0).squeeze(0).cpu().numpy()      # (H, W)
            mask_np   = valid_mask.squeeze(0).cpu().numpy()            # (H, W)

            arrays = [depth_np, coll_np, recon_np, mask_np]
            cmaps  = ['plasma', 'plasma', 'plasma', 'gray']

            for row in range(4):
                ax = axes[row, col]
                im = ax.imshow(arrays[row], cmap=cmaps[row], vmin=0, vmax=1)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # column title only on top row
                if row == 0:
                    ax.set_title(f'Sample {idx}', fontsize=11, fontweight='bold')

                # row label only on leftmost column
                if col == 0:
                    ax.set_ylabel(row_titles[row], fontsize=10)

                ax.axis('off')

    plt.suptitle('DCE Debug Visualization\n(before/during/after training)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved debug visualization to: {save_path}")


def visualize_training_progress(model, dataset, device, epoch, save_dir='debug_epochs'):
    """
    Call this at the end of each epoch to track how reconstruction improves.
    Saves one file per epoch so you can compare them side by side.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'epoch_{epoch:03d}.png')
    visualize_dce_debug(model, dataset, device, num_samples=3, save_path=save_path)

# ── EXACT AUTHOR PARAMETERS ───────────────────────────────────────────────────
CX = 240.0
CY = 135.0
FX = 252.91646
FY = 252.91646
MAX_DEPTH      = 10.0
MIN_DEPTH      = 0.2
ROBOT_EDGE_LEN = 0.2    # cube side = 2r, so r = 0.1m
OFFSET_DIST    = 0.1    # robot radius for D_offset


# ── EXACT AUTHOR: meshgrid creation ───────────────────────────────────────────
def create_meshgrid(height, width, cx, cy, fx, fy):
    """Exact copy from author's code."""
    x = np.arange(0, height, dtype=np.float32)
    y = np.arange(0, width,  dtype=np.float32)
    x, y = np.meshgrid(y, x)
    z = np.ones((height, width))
    x = (x - cx) / fx
    y = (y - cy) / fy
    return np.stack([x, y, z], axis=0)   # shape (3, H, W)


# ── EXACT AUTHOR: depth → pointcloud + D_offset ───────────────────────────────
def depth_to_pointcloud(depth_img_no_resize, meshgrid, offset_dist=OFFSET_DIST):
    """
    Exact copy from author's code.
    Returns:
        point_cloud   : (3, H, W) 3D coordinates
        z_offset      : D_offset image (depth moved closer by offset_dist)
    """
    depth_img = cv2.resize(
        depth_img_no_resize, 
        dsize=(480, 270),  # OpenCV wants (W, H)
        interpolation=cv2.INTER_LINEAR # This is the same as mode='bilinear'
    )

    x = meshgrid[0] * depth_img
    y = meshgrid[1] * depth_img
    z = meshgrid[2] * depth_img

    z_pcl = z.copy()
    z_pcl[z_pcl < 1.0] = MAX_DEPTH   # clamp near points for mesh creation

    range_img = np.sqrt(x**2 + y**2 + z**2)
    # equation 5: subtract offset_dist in range space, convert back to depth
    z_offset  = (1 - offset_dist / range_img) * z

    point_cloud = np.stack([x, y, z_pcl], axis=0)
    return point_cloud, z_offset


# ── EXACT AUTHOR: edge detection with neighbor search ─────────────────────────
def detect_edges(depth_img_uint8, depth_img_float, H=270, W=480,
                 threshold1=30, threshold2=50):
    """
    Exact copy of author's EdgeDetector.process_image().
    For each edge pixel with zero/invalid depth, searches neighbors
    to find the nearest valid depth value.
    Returns edges as array of (row, col) pairs.
    """
    edge_image = cv2.Canny(depth_img_uint8, threshold1, threshold2)
    edges_rc   = np.where(edge_image > 0)
    edges      = np.array(list(zip(edges_rc[0], edges_rc[1])))

    if len(edges) == 0:
        return edges, edge_image

    for i in range(edges.shape[0]):
        edge = edges[i]

        # 4 neighbors + 8 neighbors (2 pixels away)
        neighbor_list = [
            (max(edge[0]-1, 0),   edge[1]),
            (min(edge[0]+1, H-1), edge[1]),
            (edge[0], max(0,   edge[1]-1)),
            (edge[0], min(W-1, edge[1]+1)),
            (max(edge[0]-2, 0),   edge[1]),
            (min(edge[0]+2, H-1), edge[1]),
            (edge[0], max(0,   edge[1]-2)),
            (edge[0], min(W-1, edge[1]+2)),
        ]

        min_depth = depth_img_float[edge[0], edge[1]]

        # if this edge pixel has invalid depth, find nearest valid neighbor
        if min_depth <= 0.0:
            for j in neighbor_list:
                if depth_img_float[j[0], j[1]] > 0.0:
                    min_depth = depth_img_float[j[0], j[1]]
                    edge = j
                    break

        # snap to the closer of the neighbors (gets the nearer surface at discontinuity)
        min_neighbor = (0, 0)
        for j in neighbor_list:
            if depth_img_float[j[0], j[1]] < min_depth and \
               depth_img_float[j[0], j[1]] > 0.1:
                min_depth    = depth_img_float[j[0], j[1]]
                min_neighbor = j

        if min_depth < depth_img_float[edge[0], edge[1]] and \
           depth_img_float[edge[0], edge[1]] > 0.1:
            edges[i] = min_neighbor

    return edges, edge_image


# ── APPROXIMATION: replace Warp ray casting with dilation ────────────────────
def build_D_M_approx(edges, point_cloud, depth_img_float,
                     robot_radius, H, W, fx):
    """
    Approximates the author's Warp mesh rendering + ray casting.
    Author places cube of side ROBOT_EDGE_LEN at each edge point,
    then renders a depth image of all cubes.

    Approximation: for each edge pixel at depth z, paint a circle
    of radius (robot_radius * fx / z) pixels in D_M with value z.
    Uses every 5th edge pixel exactly as the author does.
    """
    D_M = np.full((H, W), fill_value=MAX_DEPTH, dtype=np.float32)

    # author samples every 5th edge — exact replication
    edges_sampled = edges[::5]

    for edge in edges_sampled:
        v, u = int(edge[0]), int(edge[1])
        z    = depth_img_float[v, u]

        if z < MIN_DEPTH:
            continue

        # project robot physical size to pixels at this depth
        r_px = max(1, int(robot_radius * fx / z))
        r_px = min(r_px, 40)  # safety clamp

        # paint circle — vectorized
        v_min = max(0,   v - r_px)
        v_max = min(H,   v + r_px + 1)
        u_min = max(0,   u - r_px)
        u_max = min(W,   u + r_px + 1)

        # Check for empty windows before calling mgrid
        if v_max <= v_min or u_max <= u_min:
            continue

        vv, uu = np.mgrid[v_min:v_max, u_min:u_max]
        inside  = (vv - v)**2 + (uu - u)**2 <= r_px**2
        D_M[v_min:v_max, u_min:u_max][inside] = np.minimum(
            D_M[v_min:v_max, u_min:u_max][inside], z
        )

    return D_M


# ── EXACT AUTHOR: ImageProcessor normalization ────────────────────────────────
def process_image_like_author(image, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH,
                               scaling_factor=0.1):
    """
    Exact copy of author's ImageProcessor.process_image().
    scaling_factor=0.1 = divide by 10.0 = normalize to [0,1] for max_depth=10m
    """
    image = image.copy()
    image[image < min_depth]  = -1.0       # mark too-close/invalid as -1
    image[image > max_depth]  = max_depth
    image = image * scaling_factor          # → [0, 1] for valid pixels
    image[image < 0.0] = 0.0               # clip -1 sentinel to 0 for display
    image[image > 1.0] = 1.0
    return image


# ── MAIN FUNCTION: full pipeline ──────────────────────────────────────────────
def depth_to_collision_image(depth_raw: np.ndarray,
                              robot_radius: float = ROBOT_EDGE_LEN / 2,
                              H: int = 270, W: int = 480) -> np.ndarray:
    """
    Full pipeline following the author's collision_image_generator.py.
    Input : raw depth image in meters, shape (H, W)
    Output: collision image normalized to [0,1], invalid pixels = 0
    """
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]

    depth_raw = depth_raw.astype(np.float32)

    # ── step 1: build meshgrid (done once, reuse across images) ──────────────
    meshgrid = create_meshgrid(H, W, CX, CY, FX, FY)

    # ── step 2: point cloud + D_offset (exact author equation 5) ─────────────
    point_cloud, z_offset = depth_to_pointcloud(depth_raw, meshgrid,
                                                 offset_dist=OFFSET_DIST)

    # normalize D_offset exactly as author does
    normalized_offset = process_image_like_author(z_offset)

    # ── step 3: edge detection with neighbor search (exact author) ────────────
    # author normalizes with scaling=0.1, then multiplies by 255 for Canny
    depth_norm_uint8 = (np.clip(depth_raw, 0, MAX_DEPTH) / MAX_DEPTH * 255).astype(np.uint8)
    edges, edge_image = detect_edges(depth_norm_uint8, depth_raw, H, W,
                                     threshold1=30, threshold2=50)

    # if too few edges, return only D_offset (no mesh inflation possible)
    if len(edges) < 10:
        return normalized_offset

    # ── step 4: D_M — approximate mesh rendering ──────────────────────────────
    D_M_raw       = build_D_M_approx(edges, point_cloud, depth_raw,
                                      robot_radius, H, W, FX)
    normalized_D_M = process_image_like_author(D_M_raw)

    # ── step 5: pixel-wise minimum (exact author equation 6) ─────────────────
    collision = np.minimum(normalized_offset, normalized_D_M)

    return collision
# ── PREPROCESSING: convert and save the full dataset ─────────────────────────

# create meshgrid ONCE outside the loop — expensive to recompute per image
MESHGRID = create_meshgrid(270, 480, CX, CY, FX, FY)


# ── 2. DATASET ─────────────────────────────────────────────────────────────────

def _prepare_sample(depth_raw: np.ndarray,
                    target_size=(120, 212)) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        depth_input    : normalized raw depth tensor (1, H, W)  — VAE input x
        collision_img  : collision image tensor      (1, H, W)  — VAE target xcoll
        valid_mask     : boolean mask tensor         (1, H, W)  — True = valid pixel
    """
    # ensure (H, W)
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]

    # --- input: just normalize raw depth, keep invalids as 0 ---
    invalid = (depth_raw <= 0) | (~np.isfinite(depth_raw))
    depth_input = np.clip(depth_raw, 0, 10.0) / 10.0
    depth_input[invalid] = 0.0
    depth_input_t = torch.from_numpy(depth_input).float().unsqueeze(0)

    # --- target: collision image with invalid mask ---
    collision = depth_to_collision_image(depth_raw)
    valid_mask = (collision >= 0).astype(np.float32)      # 1=valid, 0=invalid
    #collision = np.clip(collision, 0, 1)                   # remove -1 sentinel
    collision_t  = torch.from_numpy(collision).float().unsqueeze(0)
    valid_mask_t = torch.from_numpy(valid_mask).float().unsqueeze(0)

    # --- resize all three to VAE input resolution ---
    depth_input_t = F.interpolate(depth_input_t.unsqueeze(0),
                                  size=target_size, mode='bilinear',
                                  align_corners=False).squeeze(0)
    # collision_t   = F.interpolate(collision_t.unsqueeze(0),
    #                               size=target_size, mode='bilinear',
    #                               align_corners=False).squeeze(0)
    valid_mask_t  = F.interpolate(valid_mask_t.unsqueeze(0),
                                  size=target_size, mode='nearest').squeeze(0)

    return depth_input_t, collision_t, valid_mask_t

# From pytorch documentation

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_loader):
        # Unpack the 3 items from your WarehouseDepthDataset
        inputs, labels, masks = data
        
        # Move to GPU
        inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)

        optimizer.zero_grad()

        # 1. Forward pass - Unpack the VAE tuple
        recon, mean, logvar, z = model(inputs)

        # 2. Compute the specialized DCE loss
        loss = dce_loss(recon, labels, masks, mean, logvar)
        
        # 3. Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Reporting every 10 batches (1000 is too high for smaller datasets)
        if i % 10 == 9:
            last_loss = running_loss / 10
            print(f'  batch {i + 1} loss: {last_loss:.6f}')
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

class WarehouseDepthDataset(Dataset):
    def __init__(self, depth_dir, augment=False, target_size=(120, 212)):
        self.depth_dir   = depth_dir
        self.augment     = augment
        self.target_size = target_size
        self.files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
        print(f"  Found {len(self.files)} depth maps in {depth_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        depth_raw = np.load(os.path.join(self.depth_dir, self.files[idx])).astype(np.float32)
        depth_input, collision, valid_mask = _prepare_sample(depth_raw, self.target_size)

        # data augmentation: random horizontal flip (train only)

        """" Data augmentation is a machine learning technique that artificially increases the size 
        and diversity of training datasets by creating modified copies of existing data.
        It reduces overfitting, improves model generalizability, and addresses class imbalances 
        by applying transformations such as rotation, flipping, zooming, or color adjustments to input data. """
        
        if self.augment and torch.rand(1) > 0.5:
            depth_input  = torch.flip(depth_input,  dims=[2])
            collision    = torch.flip(collision,     dims=[2])
            valid_mask   = torch.flip(valid_mask,    dims=[2])

        return depth_input, collision, valid_mask

target_res = (270, 480)
train_data = WarehouseDepthDataset(os.path.join('warehouse_detection_dataset/raw/train/depth/'), augment = False, target_size = target_res)
val_data   = WarehouseDepthDataset(os.path.join('warehouse_detection_dataset/raw/val/depth/'),augment = False, target_size = target_res)
test_data  = WarehouseDepthDataset(os.path.join('warehouse_detection_dataset/raw/test/depth/'), augment = False, target_size = target_res)


#train_data = WarehouseDepthDataset(os.path.join('warehouse_detection_dataset/raw/train/depth/'), augment = True, target_size = target_res)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False, num_workers=2)
# ── 4. MODEL ───────────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# with_logits=False: decoder output goes through sigmoid → values in [0,1]
# This is correct because xcoll is a continuous normalized image, not logits
model = VAE(input_dim=1, latent_dim=64, with_logits=False, inference_mode=False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ── 5. LOSS FUNCTION — exactly as described in the paper ──────────────────────

beta_norm = 3.0   # from paper Section IV-C

def dce_loss(recon: torch.Tensor,
             target: torch.Tensor,
             valid_mask: torch.Tensor,
             mean: torch.Tensor,
             logvar: torch.Tensor,
             beta: float = beta_norm) -> torch.Tensor:
    """
    Paper equation (1): L = Lrecon + beta_norm * LKL

    Lrecon = MSE(xcoll, xcoll_recon)   — only over VALID pixels
    LKL    = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    valid_mask: 1 where pixel is valid, 0 where it is an invalid depth pixel.
    The paper explicitly states invalid pixels are removed from the loss.
    """
    # --- reconstruction loss: MSE over valid pixels only ---
    squared_error = (recon - target) ** 2          # (B, 1, H, W)
    masked_error  = squared_error * valid_mask     # zero out invalid pixels
    n_valid       = valid_mask.sum().clamp(min=1)  # avoid division by zero
    recon_loss    = masked_error.sum() / n_valid   # mean over valid pixels

    # --- KL divergence loss ---
    # paper formula: -0.5 * sum_j(1 + log(sigma^2_j) - mu^2_j - sigma^2_j)
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    # normalize by batch size for stability
    kl_loss = kl_loss / mean.shape[0]

    return recon_loss + beta * kl_loss

visualize_dce_debug(model, train_data, device,
                    num_samples=3,
                    save_path='debug_epoch_000_pretrain.png')

# ──────────────────────── 6. TRAINING  ──────────────────────
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0
EPOCHS = 100
best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels, vmasks = vdata
                vinputs, vlabels, vmasks = vinputs.to(device), vlabels.to(device), vmasks.to(device)
                
                # Unpack here too
                v_recon, v_mean, v_logvar, v_z = model(vinputs)
                
                # Use all arguments for loss
                vloss = dce_loss(v_recon, vlabels, vmasks, v_mean, v_logvar)
                running_vloss += vloss.item()

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()


    # ── save debug visualization every 10 epochs ──────────────────────────────
    if epoch_number % 10 == 0 or epoch_number == EPOCHS - 1:
        visualize_training_progress(model, val_data, device, epoch=epoch_number)
        print(f"  [debug] saved epoch {epoch_number} visualization")

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        os.makedirs('checkpoint', exist_ok=True)
        model_path = 'checkpoint/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1