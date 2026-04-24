import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect depth dataset from warehouse")
parser.add_argument("--n_positions", type=int, default=1000, help="Number of camera positions")
parser.add_argument("--save_path",   type=str, default="data/warehouse_depth.h5")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
import h5py
import os
import random
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# ── Warehouse bounds ───────────────────────────────────────────────────────────
# Adjust these to match your full_warehouse scene dimensions
X_MIN, X_MAX =  -8.0,  8.0   # meters
Y_MIN, Y_MAX =  -8.0,  8.0   # meters
Z_MIN, Z_MAX =   0.5,  2.5   # camera height range

# ── Camera looks horizontally with random yaw ──────────────────────────────────
PITCH_DEG_RANGE = (-15.0, 15.0)   # slight up/down tilt
YAW_DEG_RANGE   = (0.0,   360.0)  # full rotation


def random_camera_pose():
    """Sample a random position and orientation for the camera."""
    pos = torch.tensor([
        random.uniform(X_MIN, X_MAX),
        random.uniform(Y_MIN, Y_MAX),
        random.uniform(Z_MIN, Z_MAX),
    ])

    pitch_rad = torch.tensor(random.uniform(*PITCH_DEG_RANGE)) * (torch.pi / 180.0)
    yaw_rad   = torch.tensor(random.uniform(*YAW_DEG_RANGE))   * (torch.pi / 180.0)
    roll_rad  = torch.tensor(0.0)

    quat = quat_from_euler_xyz(roll_rad, pitch_rad, yaw_rad)  # (w, x, y, z)
    return pos.unsqueeze(0), quat.unsqueeze(0)                 # (1, 3), (1, 4)


def main():
    # ── Simulation setup ───────────────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(dt=1/60)
    sim     = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[10.0, 10.0, 5.0], target=[0.0, 0.0, 0.0])

    # ── Load warehouse USD ─────────────────────────────────────────────────────
    # Replace with the actual path to your full_warehouse USD file
    warehouse_cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd")
    warehouse_cfg.func("/World/Warehouse", warehouse_cfg)

    # ── Camera sensor ──────────────────────────────────────────────────────────
    camera_cfg = CameraCfg(
        prim_path="/World/DataCamera",
        update_period=0,            # update every step
        height=270,
        width=480,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
    )
    camera = Camera(camera_cfg)

    # ── Play simulation ────────────────────────────────────────────────────────
    sim.reset()
    print("[Collector] Simulation started.")

    # ── Collect ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args_cli.save_path), exist_ok=True)

    collected_depths  = []
    collected_poses   = []
    n_invalid         = 0

    for i in range(args_cli.n_positions):
        # 1. Sample random pose and move camera there
        pos, quat = random_camera_pose()
        camera.set_world_poses(pos, quat, convention="ros")

        # 2. Step simulation a few times so sensor buffers update
        for _ in range(5):
            sim.step()
            camera.update(sim.current_time)

        # 3. Grab depth image
        depth = camera.data.output["distance_to_image_plane"]  # (1, H, W, 1)
        depth = depth.squeeze(0).squeeze(-1)                   # (H, W)

        # 4. Validate — skip if mostly invalid (inf/nan pixels)
        valid_ratio = torch.isfinite(depth).float().mean().item()
        if valid_ratio < 0.5:
            n_invalid += 1
            print(f"  [{i+1}/{args_cli.n_positions}] Skipped — only {valid_ratio:.1%} valid pixels")
            continue

        # 5. Clip and normalize to [0, 1] using clipping range
        depth_clipped = torch.clamp(depth, min=0.1, max=20.0)
        depth_norm    = (depth_clipped - 0.1) / (20.0 - 0.1)

        collected_depths.append(depth_norm.cpu().numpy().astype(np.float32))
        collected_poses.append(torch.cat([pos.squeeze(0), quat.squeeze(0)]).cpu().numpy())

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{args_cli.n_positions}] Collected {len(collected_depths)} frames  "
                  f"(skipped {n_invalid})")

    # ── Save ───────────────────────────────────────────────────────────────────
    print(f"\n[Collector] Saving {len(collected_depths)} frames to {args_cli.save_path} ...")
    with h5py.File(args_cli.save_path, "w") as f:
        f.create_dataset("depth", data=np.stack(collected_depths), compression="gzip")
        f.create_dataset("poses", data=np.stack(collected_poses))
        f.attrs["n_frames"]    = len(collected_depths)
        f.attrs["n_skipped"]   = n_invalid
        f.attrs["image_shape"] = [270, 480]
        f.attrs["depth_min"]   = 0.1
        f.attrs["depth_max"]   = 20.0

    print(f"[Collector] Done. Dataset saved to {args_cli.save_path}")
    simulation_app.close()


if __name__ == "__main__":
    main()