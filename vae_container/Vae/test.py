import numpy as np
import cv2
import trimesh as tm   # pip install trimesh
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
# ── EXACT AUTHOR PARAMETERS ───────────────────────────────────────────────────
CX = 240.0
CY = 135.0
FX = 252.91646
FY = 252.91646
MAX_DEPTH      = 10.0
MIN_DEPTH      = 0.2
ROBOT_EDGE_LEN = 0.4    # cube side = 2r, so r = 0.2m
OFFSET_DIST    = 0.2    # robot radius for D_offset


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

def preprocess_split(depth_dir, output_dir, split_name):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
    print(f"\n[{split_name}] {len(files)} files")

    for fname in tqdm(files, desc=split_name):
        out_path = os.path.join(output_dir, fname)
        if os.path.exists(out_path):
            continue

        depth_raw = np.load(os.path.join(depth_dir, fname)).astype(np.float32)
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]

        collision = depth_to_collision_image(depth_raw)
        np.save(out_path, collision)


# ── VISUALIZATION: compare depth vs collision ─────────────────────────────────

def visualize_conversion(depth_dir: str, collision_dir: str,
                         num_samples: int = 4,
                         save_path: str = 'collision_conversion_check.png'):
    """
    Shows side-by-side: raw depth | collision image | difference | edge mask
    for num_samples random files.
    """
    files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
    indices = np.linspace(0, len(files) - 1, num_samples, dtype=int)

    fig, axes = plt.subplots(4, num_samples, figsize=(5 * num_samples, 16))

    row_labels = [
        'Raw depth (normalized)',
        'Collision image',
        'Difference (coll - depth)',
        'Edge mask (Canny)'
    ]

    for col, idx in enumerate(indices):
        fname     = files[idx]
        depth_raw = np.load(os.path.join(depth_dir, fname)).astype(np.float32)
        coll_raw  = np.load(os.path.join(collision_dir, fname)).astype(np.float32)

        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]

        # normalize depth for display
        depth_norm = np.clip(depth_raw, 0, MAX_DEPTH) / MAX_DEPTH

        # recover valid mask from collision (-1 = invalid)
        valid_mask = (coll_raw >= 0).astype(np.float32)
        coll_display = np.where(coll_raw >= 0, coll_raw, 0.0)
        depth_norm_resize = cv2.resize(
            depth_norm, 
            dsize=(480, 270),  # OpenCV wants (W, H)
            interpolation=cv2.INTER_LINEAR # This is the same as mode='bilinear'
        )

        # difference — should show STRUCTURE at edges, not uniform color
        diff = coll_display - depth_norm_resize

        # edge mask for visualization
        depth_uint8 = (np.clip(depth_raw, 0, MAX_DEPTH) / MAX_DEPTH * 255).astype(np.uint8)
        edges = cv2.Canny(depth_uint8, 10, 30)

        images = [depth_norm, coll_display, diff, edges / 255.0]
        cmaps  = ['plasma',   'plasma',     'RdBu', 'gray']
        vmins  = [0,           0,           -0.3,   0]
        vmaxs  = [1,           1,            0.3,   1]

        for row in range(4):
            ax = axes[row, col]
            im = ax.imshow(images[row], cmap=cmaps[row],
                           vmin=vmins[row], vmax=vmaxs[row])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if row == 0:
                ax.set_title(f'Sample: {fname}', fontsize=9, fontweight='bold')
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=10)

            ax.axis('off')

    plt.suptitle(
        'Depth → Collision Conversion Check\n'
        'Row 3 (difference) must show STRUCTURE at edges — not uniform color',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved visualization to: {save_path}")
    print("Check row 3: if it is a uniform flat color, the conversion is not working correctly.")


# ── MAIN: run conversion for all splits ──────────────────────────────────────

if __name__ == '__main__':

    base_path   = 'warehouse_detection_dataset/raw'
    output_base = 'warehouse_detection_dataset/collision'

    splits = ['train', 'val', 'test']

    for split in splits:
        depth_dir  = os.path.join(base_path,   split, 'depth')
        output_dir = os.path.join(output_base, split, 'collision')
        preprocess_split(depth_dir, output_dir, split_name=split)

    # ── visualize one split to verify quality ─────────────────────────────────
    print("\nGenerating verification visualization for train split...")
    visualize_conversion(
        depth_dir     = os.path.join(base_path,   'train', 'depth'),
        collision_dir = os.path.join(output_base, 'train', 'collision'),
        num_samples   = 4,
        save_path     = 'collision_conversion_check.png'
    )

    print("\n✓ Preprocessing complete.")
    print(f"  Collision images saved to: {output_base}/")
    print("  Your dataset folder is now:")
    print("    warehouse_detection_dataset/")
    print("      raw/train/depth/        ← original depth (.npy)")
    print("      collision/train/collision/ ← converted collision (.npy)")
    print("      raw/val/depth/")
    print("      collision/val/collision/")
    print("      raw/test/depth/")
    print("      collision/test/collision/")