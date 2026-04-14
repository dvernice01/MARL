import numpy as np
import cv2
from scipy.ndimage import grey_dilation


# better approximation: divide image into zones and use local edge depth
def compute_adaptive_dilation_mask(depth_clean, edges, robot_radius, fx,
                                   n_zones=4):
    """
    Divides the image into n_zones x n_zones regions.
    Computes a separate dilation size for each region based on
    the median edge depth in that region.
    Returns a dilation size map (H, W).
    """
    H, W = depth_clean.shape
    dilation_map = np.ones((H, W), dtype=np.int32) * 5  # default

    zone_h = H // n_zones
    zone_w = W // n_zones

    for row in range(n_zones):
        for col in range(n_zones):
            # region boundaries
            r0, r1 = row * zone_h, (row + 1) * zone_h
            c0, c1 = col * zone_w, (col + 1) * zone_w

            # edge pixels in this zone
            zone_edges = edges[r0:r1, c0:c1]
            zone_depth = depth_clean[r0:r1, c0:c1]
            local_edge_depths = zone_depth[zone_edges > 0]

            if len(local_edge_depths) > 0:
                valid = local_edge_depths[local_edge_depths > 0.1]
                if len(valid) > 0:
                    local_median = np.median(valid)
                    dil_px = int(robot_radius * fx / local_median)
                    dil_px = max(2, min(dil_px, 40))
                    dilation_map[r0:r1, c0:c1] = dil_px

    return dilation_map

def depth_to_range(depth: np.ndarray, fx: float, fy: float,
                   cx: float, cy: float) -> np.ndarray:
    """
    Convert depth image (z along camera axis) to range image
    (Euclidean distance to each point).
    This is the R() function in equation (5) of the paper.
    
    depth[v, u] = z distance (along optical axis)
    range[v, u] = sqrt(x² + y² + z²) = true distance to point
    """
    H, W = depth.shape
    """
    For a 4×3 image, uu looks like:        vv looks like:
    [[0, 1, 2, 3],                         [[0, 0, 0, 0],
    [0, 1, 2, 3],                          [1, 1, 1, 1],
    [0, 1, 2, 3]]                          [2, 2, 2, 2]]
 """
    u_coords = np.arange(W, dtype=np.float32)
    v_coords = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u_coords, v_coords)
    # is computer vision approch. Is a rapport between focal length, pixel distance and cartesian coordinates.
    # 3D coordinates of each pixel
    x = (cx - uu) / fx * depth
    y = (cy - vv) / fy * depth

    # Euclidean distance
    range_img = np.sqrt(x**2 + y**2 + depth**2)
    return range_img


def range_to_depth(range_img: np.ndarray, fx: float, fy: float,
                   cx: float, cy: float) -> np.ndarray:
    """
    Convert range image back to depth image.
    This is the R⁻¹() function in equation (5) of the paper.
    """
    H, W = range_img.shape
    u_coords = np.arange(W, dtype=np.float32)
    v_coords = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u_coords, v_coords)

    """
    range = sqrt(x² + y² + z²)
        = sqrt(((cx-u)/fx * z)² + ((cy-v)/fy * z)² + z²)
        = z * sqrt(((cx-u)/fx)² + ((cy-v)/fy)² + 1)
        z è tirato fuori
    """
    # direction cosine along z for each pixel
    denom = np.sqrt(((cx - uu) / fx)**2 + ((cy - vv) / fy)**2 + 1.0)
    depth = range_img / denom
    return depth


def depth_to_collision_image(depth: np.ndarray,
                              robot_radius: float = 0.05,
                              max_depth: float = 10.0,
                              fx: float = 462.0,
                              fy: float = 462.0,
                              cx: float = None,
                              cy: float = None) -> np.ndarray:
    """
    Full implementation of Section 3.2 of the paper.

    Steps:
      1. Edge detection on depth image
      2. Project edge pixels to 3D
      3. Approximate mesh inflation with dilation (replaces Warp rendering)
      4. Compute D_offset for interior pixels (equation 5)
      5. Pixel-wise minimum (equation 6)

    Parameters:
        depth       : raw depth image (H, W), values in meters
        robot_radius: half the robot size in meters (r in the paper)
        max_depth   : maximum valid depth in meters
        fx, fy      : camera focal lengths in pixels
        cx, cy      : camera optical center (defaults to image center)

    Returns:
        collision   : collision image in [0,1], invalid pixels = -1
    """
    depth = depth.astype(np.float32)
    H, W = depth.shape

    if cx is None:
        cx = W / 2.0
    if cy is None:
        cy = H / 2.0

    # --- identify invalid pixels ---
    invalid_mask = (~np.isfinite(depth)) | (depth <= 0)
    depth_clean = depth.copy()
    depth_clean[invalid_mask] = 0.0

    # ── STEP 1: Edge detection ────────────────────────────────────────────────
    # normalize depth to uint8 for Canny
    depth_vis = np.clip(depth_clean, 0, max_depth)
    depth_uint8 = (depth_vis / max_depth * 255).astype(np.uint8) #cv2.Canny requires uint8 input

    # Canny edge detection — finds the obstacle boundaries
    edges = cv2.Canny(depth_uint8, threshold1=10, threshold2=30) # thresholds can be tuned; lower values → more edges, higher values → fewer edges
    # edges is a binary image: 255 at edge pixels, 0 elsewhere

    # ── STEP 2 & 3: Project edges to 3D and approximate mesh inflation ────────
    # The paper uses NVIDIA Warp to render robot-sized meshes at each edge point.
    # Here we approximate this with adaptive dilation:
    # at distance d, robot_radius corresponds to (robot_radius * fx / d) pixels.
    # We dilate the edge mask by this amount to simulate the mesh inflation.

    edge_mask = (edges > 0).astype(np.float32)

    # compute per-pixel dilation size based on actual depth value
    # pixels farther away → smaller dilation (robot appears smaller in image)
    # pixels closer → larger dilation
    depth_safe = np.where(depth_clean > 0.1, depth_clean, max_depth)
    dilation_size_map = (robot_radius * fx / depth_safe).astype(np.int32) # is the correspective pixel size of the robot at each depth
    dilation_size_map = np.clip(dilation_size_map, 1, 50)

    # since per-pixel dilation is expensive, use the median dilation size
    # weighted by edge pixels (dilate where edges actually are)
    edge_depths = depth_clean[edges > 0]
    if len(edge_depths) > 0:
        median_edge_depth = np.median(edge_depths[edge_depths > 0.1]) # compute_adaptive_dilation_mask to be used because median is less precise
        if median_edge_depth > 0.1:
            dilation_px = int(robot_radius * fx / median_edge_depth)
        else:
            dilation_px = 5
    else:
        dilation_px = 5

    dilation_px = max(3, min(dilation_px, 40)) # dilatatiom between 3 and 40 pixels

    # dilate the edge mask to simulate robot-sized mesh rendering (D_M)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilation_px * 2 + 1, dilation_px * 2 + 1)
    )
    dilated_edges = cv2.dilate(edge_mask, kernel)  # shape (H, W), values 0 or 1

    # for dilated edge pixels: collision distance = depth of the original edge
    # (the drone's body would hit the edge surface even from the neighboring ray)
    # broadcast the edge depth values into the dilated region
    # use grey_dilation to propagate actual depth values (not just binary)
    D_M = grey_dilation(
        np.where(edges > 0, depth_clean, 0.0),
        size=(dilation_px * 2 + 1, dilation_px * 2 + 1)
    )
    # where there's no edge influence, set to max_depth (no constraint)
    D_M = np.where(dilated_edges > 0, D_M, max_depth)

    # ── STEP 4: Offset depth image for interior pixels (equation 5) ───────────
    # D_offset = R⁻¹(R(D) - r)
    # Convert depth → range, subtract robot radius, convert back → depth
    range_img = depth_to_range(depth_clean, fx, fy, cx, cy)
    range_offset = np.clip(range_img - robot_radius, 0, None)
    D_offset = range_to_depth(range_offset, fx, fy, cx, cy)

    # ── STEP 5: Pixel-wise minimum (equation 6) ───────────────────────────────
    # D_coll = min(D_M, D_offset)
    D_coll = np.minimum(D_M, D_offset)

    # ── Normalize to [0, 1] and mark invalids ─────────────────────────────────
    D_coll = np.clip(D_coll, 0, max_depth)
    collision_normalized = D_coll / max_depth

    # mark original invalid pixels as -1 (masked out of loss)
    collision_normalized[invalid_mask] = -1.0

    return collision_normalized

# target_res = (270, 480)
# train_data = WarehouseDepthDataset(os.path.join('warehouse_detection_dataset/raw/train/depth/'), augment = True, target_size = target_res)
# train_loader = DataLoader(train_data, batch_size=32, shuffle=True,  num_workers=2)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

