import torch

def hits_to_occupancy_map(ray_hits_w, grid_size=0.05, map_dims=(200, 200)):
    """
    ray_hits_w: (N, B, 3) tensor from RayCaster
    grid_size: meters per cell
    """
    # Take XY positions only, flatten batch
    xy = ray_hits_w[0, :, :2]  # (B, 2)
    
    # Filter out invalid hits (inf/nan)
    valid = torch.isfinite(xy).all(dim=-1)
    xy = xy[valid]
    
    # Convert to grid indices
    cx, cy = map_dims[0] // 2, map_dims[1] // 2
    ix = (xy[:, 0] / grid_size + cx).long()
    iy = (xy[:, 1] / grid_size + cy).long()
    
    # Clip to map bounds
    mask = (ix >= 0) & (ix < map_dims[0]) & (iy >= 0) & (iy < map_dims[1])
    ix, iy = ix[mask], iy[mask]
    
    # Fill occupancy grid
    occ_map = torch.zeros(map_dims, dtype=torch.float32)
    occ_map[ix, iy] = 1.0
    
    return occ_map


def depth_to_occupancy(depth_img, K, robot_pose, 
                        z_min=0.1, z_max=2.0, grid_res=0.05):
    """
    depth_img: (H, W) tensor
    K: camera intrinsics (3x3)
    robot_pose: (4x4) world transform
    z_min/z_max: height band to keep (filters floor and ceiling)
    """
    H, W = depth_img.shape
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    
    # Pixel grid
    u = torch.arange(W).float()
    v = torch.arange(H).float()
    uu, vv = torch.meshgrid(u, v, indexing='xy')
    
    d = depth_img.squeeze()
    
    # Back-project to camera frame
    X = (uu - cx) * d / fx
    Y = (vv - cy) * d / fy
    Z = d
    
    pts_cam = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=-1)  # (H,W,4)
    pts_cam = pts_cam.reshape(-1, 4)
    
    # Transform to world frame
    pts_world = (robot_pose @ pts_cam.T).T  # (N, 4)
    
    # Filter by height band (remove floor + ceiling)
    z_world = pts_world[:, 2]
    mask = (z_world > z_min) & (z_world < z_max)
    pts_world = pts_world[mask]
    
    # Project to 2D occupancy
    # ... same grid binning as above


class BayesianOccupancyGrid:
    def __init__(self, size=200, res=0.05):
        # Log-odds: 0 = unknown, >0 = occupied, <0 = free
        self.log_odds = torch.zeros(size, size)
        self.l_occ  =  0.85   # log-odds update for occupied
        self.l_free = -0.40   # log-odds update for free
        self.l_min  = -5.0
        self.l_max  =  5.0
    
    def update(self, ray_hits_xy, free_cells_xy):
        # Update occupied cells
        for ix, iy in ray_hits_xy:
            self.log_odds[ix, iy] = torch.clamp(
                self.log_odds[ix, iy] + self.l_occ, 
                self.l_min, self.l_max
            )
        # Update free cells along ray
        for ix, iy in free_cells_xy:
            self.log_odds[ix, iy] = torch.clamp(
                self.log_odds[ix, iy] + self.l_free,
                self.l_min, self.l_max
            )
    
    def get_map(self):
        # Convert to probability [0, 1]
        return torch.sigmoid(self.log_odds)