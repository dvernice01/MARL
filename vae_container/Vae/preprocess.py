# preprocess.py — run this ONCE before training
import os
import numpy as np
import cv2
from tqdm import tqdm

# ── PARAMETERS ────────────────────────────────────────────────────────────────
CX = 240.0;  CY = 135.0
FX = 252.91646;  FY = 252.91646
MAX_DEPTH = 10.0;  MIN_DEPTH = 0.2
ROBOT_EDGE_LEN = 0.2   # Crazyflie: cube side = 2r, r = 0.1m
OFFSET_DIST    = 0.1
H, W = 270, 480

def sanitize_depth(depth):
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth < 0] = 0.0
    depth[depth > MAX_DEPTH] = MAX_DEPTH
    return depth

def create_meshgrid(H, W, cx, cy, fx, fy):
    x = np.arange(0, H, dtype=np.float32)
    y = np.arange(0, W, dtype=np.float32)
    x, y = np.meshgrid(y, x)
    z = np.ones((H, W))
    x = (x - cx) / fx
    y = (y - cy) / fy
    return np.stack([x, y, z], axis=0)

MESHGRID = create_meshgrid(H, W, CX, CY, FX, FY)

def process_image_like_author(image):
    image = image.copy()
    image[image < MIN_DEPTH] = -1.0
    image[image > MAX_DEPTH] = MAX_DEPTH
    image = image * 0.1          # scale to [0, 1]
    image[image < 0.0] = 0.0
    image[image > 1.0] = 1.0
    return image

def detect_edges(depth_uint8, depth_float):
    edge_image = cv2.Canny(depth_uint8, 30, 50)
    edges_rc   = np.where(edge_image > 0)
    edges      = np.array(list(zip(edges_rc[0], edges_rc[1])))
    if len(edges) == 0:
        return edges, edge_image
    for i in range(len(edges)):
        edge = edges[i]
        neighbors = [
            (max(edge[0]-1,0), edge[1]), (min(edge[0]+1,H-1), edge[1]),
            (edge[0], max(0,edge[1]-1)), (edge[0], min(W-1,edge[1]+1)),
            (max(edge[0]-2,0), edge[1]), (min(edge[0]+2,H-1), edge[1]),
            (edge[0], max(0,edge[1]-2)), (edge[0], min(W-1,edge[1]+2)),
        ]
        min_d = depth_float[edge[0], edge[1]]
        if min_d <= 0.0:
            for j in neighbors:
                if depth_float[j[0], j[1]] > 0.0:
                    min_d = depth_float[j[0], j[1]]
                    edge  = j
                    break
        for j in neighbors:
            if 0.1 < depth_float[j[0], j[1]] < min_d:
                min_d = depth_float[j[0], j[1]]
                edges[i] = j
        if min_d < depth_float[edge[0], edge[1]] and depth_float[edge[0], edge[1]] > 0.1:
            edges[i] = (0,0)
    return edges, edge_image

def build_D_M_from_cubes(edges, point_cloud, depth_float,
                          edge_length=ROBOT_EDGE_LEN):
    """
    Replicates the author's create_cube_mesh + Warp ray casting without Warp.

    The author places a cube of side `edge_length` at each edge point in 3D,
    then renders a depth image of all cubes from the camera.

    We replicate this by:
      1. Taking every 5th edge pixel (same as author: edges[::5])
      2. Getting its 3D position from the point_cloud
      3. Computing how large that cube appears in the image
         (cube side in pixels = edge_length * fx / Z)
      4. Painting a square of that size in D_M with depth value Z
         (this is what the ray caster would return for rays hitting the cube)

    A cube projects as a square on the image plane (not a circle),
    so we fill a square patch — no mgrid/circle needed.
    """
    D_M = np.full((H, W), MAX_DEPTH, dtype=np.float32)

    # sample every 5th edge — exact replication of author's edges[::5]
    sampled_edges = edges[::5]

    for i in range(len(sampled_edges)):
        # pixel coordinates of this edge
        v = int(sampled_edges[i, 0])   # row
        u = int(sampled_edges[i, 1])   # col

        # 3D position of this edge pixel from the point cloud
        # point_cloud shape: (3, H, W) → [x, y, z] at pixel (v, u)
        X = point_cloud[0, v, u]
        Y = point_cloud[1, v, u]
        Z = point_cloud[2, v, u]   # this is the depth z of the edge point

        # skip invalid points
        if Z < MIN_DEPTH or not np.isfinite(Z):
            continue

        # the cube has side edge_length in meters
        # its half-side projected onto the image plane at depth Z:
        #   half_side_px = (edge_length / 2) * fx / Z
        half_px_u = int((edge_length / 2.0) * FX / Z)
        half_px_v = int((edge_length / 2.0) * FY / Z)

        # clamp to reasonable size
        half_px_u = max(1, min(half_px_u, 60))
        half_px_v = max(1, min(half_px_v, 60))

        # image bounds of the projected cube face
        v0 = max(0, v - half_px_v)
        v1 = min(H, v + half_px_v + 1)
        u0 = max(0, u - half_px_u)
        u1 = min(W, u + half_px_u + 1)

        if v1 <= v0 or u1 <= u0:
            continue

        # paint the square patch with depth Z
        # np.minimum keeps the closest cube if two overlap
        D_M[v0:v1, u0:u1] = np.minimum(D_M[v0:v1, u0:u1], Z)

    return D_M

def depth_to_collision_image(depth_raw: np.ndarray) -> np.ndarray:
    depth = sanitize_depth(depth_raw)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.shape != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        depth = sanitize_depth(depth)

    # D_offset — equation 5
    x = MESHGRID[0] * depth
    y = MESHGRID[1] * depth
    z = MESHGRID[2] * depth
    range_img   = np.sqrt(x**2 + y**2 + z**2)
    range_img   = np.nan_to_num(range_img, nan=MAX_DEPTH)
    z_offset    = np.where(range_img > 0, (1 - OFFSET_DIST / range_img) * z, 0.0)
    norm_offset = process_image_like_author(z_offset)

    # edge detection
    depth_uint8 = (depth / MAX_DEPTH * 255).astype(np.uint8)
    edges, _    = detect_edges(depth_uint8, depth)
    if len(edges) < 10:
        return norm_offset

    # build point cloud for cube placement
    point_cloud = np.stack([x, y, z], axis=0)   # (3, H, W)

    # D_M — cube mesh approximation
    D_M_raw    = build_D_M_from_cubes(edges, point_cloud, depth,
                                       edge_length=ROBOT_EDGE_LEN)
    norm_D_M   = process_image_like_author(D_M_raw)

    # equation 6
    collision  = np.minimum(norm_offset, norm_D_M)
    collision  = np.nan_to_num(collision, nan=0.0)
    return collision

def preprocess_split(depth_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
    print(f"Processing {len(files)} files: {depth_dir} → {out_dir}")
    nan_count = 0
    for fname in tqdm(files):
        out_path = os.path.join(out_dir, fname)
        if os.path.exists(out_path):
            continue
        raw = np.load(os.path.join(depth_dir, fname))
        col = depth_to_collision_image(raw)
        # final check before saving
        if np.isnan(col).any():
            col = np.nan_to_num(col, nan=0.0)
            nan_count += 1
        np.save(out_path, col.astype(np.float32))
    if nan_count > 0:
        print(f"  WARNING: {nan_count} files had NaN after conversion — replaced with 0")

base = 'warehouse_detection_dataset'
for split in ['train', 'val', 'test']:
    preprocess_split(
        depth_dir = os.path.join(base, 'raw',       split, 'depth'),
        out_dir   = os.path.join(base, 'collision',  split)
    )
print("Done.")