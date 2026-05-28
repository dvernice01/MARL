"""
Sanity check for body-frame yaw rotation in quadcopter_rnn's local OCC map.

Places the drone at a fixed warehouse-local position near the east wall,
then for yaw in {0, pi/2, pi, 3pi/2}:
  1. Writes pose to sim and steps once so the buffers refresh.
  2. Calls raw_env._build_local_combined_map(0).
  3. Saves a 3D scatter of the OCC channel (channel 0).

Expected (drone at warehouse-local (7, 0, 2), eastern wall at x ~= 8):
  yaw=0        : cells at body +x   (high ix,           iy near centre)
  yaw=pi/2     : cells at body -y   (iy near 0,         ix near centre)
  yaw=pi       : cells at body -x   (low ix,            iy near centre)
  yaw=3pi/2    : cells at body +y   (high iy,           ix near centre)

Usage (from environment/quadcopter_rnn/):
  python scripts/skrl/verify_yaw_rotation.py --task=Template-Quadcopter-Rnn-Direct-v0 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Template-Quadcopter-Rnn-Direct-v0")
parser.add_argument("--wh_x", type=float, default=7.0,
                    help="Warehouse-local x of the drone (east wall at x ~= 8).")
parser.add_argument("--wh_y", type=float, default=0.0)
parser.add_argument("--wh_z", type=float, default=2.0)
parser.add_argument("--save_dir", type=str, default="outputs/yaw_check")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import math
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — needed for 3d projection

from isaaclab.envs import DirectRLEnvCfg, DirectMARLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import quadcopter_rnn.tasks  # noqa: F401 — registers the gym task

def summarise(label, occ):
    iz, iy, ix = np.where(occ > 0.5)
    if len(iz) == 0:
        print(f"  {label}: NO occupied cells")
        return
    print(f"  {label}: n={len(iz)}  ix mean={ix.mean():.1f} (range {ix.min()}-{ix.max()})  "
          f"iy mean={iy.mean():.1f} (range {iy.min()}-{iy.max()})  "
          f"iz mean={iz.mean():.1f}")

@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         experiment_cfg: dict):
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1000.0

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    device = raw_env.device

    # First reset puts the drone somewhere random; we overwrite below.
    env.reset()

    os.makedirs(args_cli.save_dir, exist_ok=True)

    # Warehouse-local target + env_origin = world position.
    env_origin = raw_env._terrain.env_origins[0]
    world_pos = torch.tensor(
        [args_cli.wh_x + env_origin[0].item(),
         args_cli.wh_y + env_origin[1].item(),
         args_cli.wh_z],
        device=device,
    )
    env_ids = torch.tensor([0], device=device)
    physics_dt = raw_env.sim.cfg.dt
    print(f"[YawCheck] Drone target world pos = {world_pos.tolist()} "
          f"(warehouse-local ({args_cli.wh_x}, {args_cli.wh_y}, {args_cli.wh_z}))")

    def measure(yaw_rad: float, label: str) -> int:
        # Quat (w, x, y, z) for a yaw rotation about world z.
        qw = math.cos(yaw_rad / 2.0)
        qz = math.sin(yaw_rad / 2.0)
        quat = torch.tensor([[qw, 0.0, 0.0, qz]], device=device)
        pose = torch.cat([world_pos.unsqueeze(0), quat], dim=-1)   # (1, 7)
        vel = torch.zeros(1, 6, device=device)

        raw_env._robot.write_root_pose_to_sim(pose, env_ids)
        raw_env._robot.write_root_velocity_to_sim(vel, env_ids)

        # Push to sim, step once, refresh buffers — so root_pos_w / root_quat_w
        # reflect the values we just wrote.
        raw_env.scene.write_data_to_sim()
        raw_env.sim.step(render=False)
        raw_env.scene.update(dt=physics_dt)

        combined = raw_env._build_local_combined_map(0)            # (2, NZ, NY, NX)
        occ = combined[0].cpu().numpy()
        n_occ = int((occ > 0.5).sum())
        print(f"[YawCheck] yaw={yaw_rad:+.3f} rad ({label}): {n_occ} occupied cells")

        iz, iy, ix = np.where(occ > 0.5)
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        if len(iz) > 0:
            ax.scatter(ix, iy, iz, s=40, c="red", alpha=0.55, marker="s",
                       label=f"OCC ({n_occ} cells)")
        ax.scatter([raw_env.local_hx], [raw_env.local_hy], [raw_env.local_hz],
                   s=180, c="blue", marker="o", label="drone")
        ax.quiver(raw_env.local_hx, raw_env.local_hy, raw_env.local_hz,
                  3.5, 0, 0, color="green", arrow_length_ratio=0.25,
                  label="body +x (forward)")
        ax.set_xlim(0, raw_env.local_nx - 1)
        ax.set_ylim(0, raw_env.local_ny - 1)
        ax.set_zlim(0, raw_env.local_nz - 1)
        ax.set_xlabel("body x (forward, high ix)")
        ax.set_ylabel("body y (left,    high iy)")
        ax.set_zlabel("body z (up,      high iz)")
        ax.set_title(f"Local OCC — yaw = {yaw_rad:+.3f} rad ({label})")
        ax.legend(loc="upper right")
        save_path = os.path.join(args_cli.save_dir, f"occ_{label}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[YawCheck] Saved -> {save_path}")
        return occ

    occ0   = measure(0.0,                  "yaw_0")
    occ90  = measure(math.pi / 2.0,        "yaw_pi_over_2")
    occ180 = measure(math.pi,              "yaw_pi")
    occ270 = measure(3.0 * math.pi / 2.0,  "yaw_3pi_over_2")

    print("\n[YawCheck] Centroids (drone is at ix=iy=8, iz=4):")
    summarise("yaw_0       ", occ0)
    summarise("yaw_pi/2    ", occ90)
    summarise("yaw_pi      ", occ180)
    summarise("yaw_3pi/2   ", occ270)

    print()
    if (occ0.sum() + occ90.sum() + occ180.sum() + occ270.sum()) == 0:
        print("[YawCheck] WARNING: zero occupied voxels at all four yaws.")
        print("           The drone is in an empty region. Move closer to a")
        print("           wall by adjusting --wh_x / --wh_y.")
    else:
        print("[YawCheck] Expected pattern (drone at (7, 0, 2), east wall at x ~= 8):")
        print("  yaw_0:           cells at body +x   (HIGH ix, iy near centre)")
        print("  yaw_pi_over_2:   cells at body -y   (iy near 0, ix near centre)")
        print("  yaw_pi:          cells at body -x   (LOW ix,  iy near centre)")
        print("  yaw_3pi_over_2:  cells at body +y   (HIGH iy, ix near centre)")
        print()
        print("  If yaw_pi_over_2 shows cells at HIGH iy instead of LOW iy, swap")
        print("  yaw_quat and quat_inv(yaw_quat) in _build_local_occ_map AND")
        print("  _build_local_svs_map (both must be consistent).")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
