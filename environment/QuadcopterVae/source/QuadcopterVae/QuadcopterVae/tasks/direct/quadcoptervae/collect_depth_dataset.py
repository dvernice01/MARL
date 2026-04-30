import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect depth dataset from Isaac Lab")
parser.add_argument("--num_envs",    type=int, default=16)
parser.add_argument("--num_samples", type=int, default=10000)
parser.add_argument("--output_dir",  type=str, default="/workspace/vae_container/Vae/isaaclab_dataset_right_size")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
import os
import cv2
from tqdm import tqdm

import isaacsim.core.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera
from isaaclab.terrains import TerrainImporter
from pxr import UsdGeom, Gf
from QuadcopterVae.tasks.direct.quadcoptervae.quadcoptervae_env_cfg import QuadcoptervaeEnvCfg
from QuadcopterVae.tasks.direct.quadcoptervae.quadcoptervae_env import CollisionImage
from pxr import UsdGeom, Gf
import omni.usd
from pxr import UsdGeom, Gf, Usd


class DepthCollectionEnv(DirectRLEnv):
    """
    Minimal env that spawns only the camera (no robot).
    Each reset teleports the camera prim to a random position/orientation.
    """
    cfg: QuadcoptervaeEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # No robot, no thrust/moment tensors needed

    # ------------------------------------------------------------------
    # Scene setup — camera + terrain only, no articulation
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self.camera = Camera(self.cfg.camera)
        self.scene.sensors["camera"] = self.camera

        self.scene.clone_environments(copy_from_source=False)

        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Physics step — no robot to actuate
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        pass  # camera is static each step; it is moved only on reset

    def _apply_action(self):
        pass

    # ------------------------------------------------------------------
    # Observations / rewards / dones — minimal stubs
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Time-out after one "episode" = one camera pose; no death condition
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died     = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return died, time_out

    # ------------------------------------------------------------------
    # Reset — teleport camera prims to random positions
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == 0:
            env_ids = torch.arange(self.num_envs, device=self.device)

        super()._reset_idx(env_ids)

        n = len(env_ids)

        """
        Buonds full_warehouse:
        X: -28.00 → 8.00  (width=36.00m)
        Y: -41.40 → 33.42  (depth=74.82m)
        Z: -0.01 → 9.30  (height=9.31m)
        """

        pos_x = torch.zeros(n, device=self.device).uniform_(-27.0, 7.0) + self._terrain.env_origins[env_ids, 0]
        pos_y = torch.zeros(n, device=self.device).uniform_(-40.0, 32.0) + self._terrain.env_origins[env_ids, 1]
        pos_z = torch.zeros(n, device=self.device).uniform_( 1.0, 8.0) + self._terrain.env_origins[env_ids, 2]

        positions = torch.stack([pos_x, pos_y, pos_z], dim=-1)

        yaw   = torch.zeros(n, device=self.device).uniform_(-3.14159, 3.14159)
        pitch = torch.zeros(n, device=self.device).uniform_(-0.4, -0.2)
        roll  = torch.zeros(n, device=self.device)

        # Base quaternion: rotate -90 deg around X to point camera forward
        base_qx = torch.full((n,), -0.7071, device=self.device)
        base_qw = torch.full((n,),  0.7071, device=self.device)
        base_qy = torch.zeros(n, device=self.device)
        base_qz = torch.zeros(n, device=self.device)

        # Compose with yaw rotation
        cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
        qw_yaw = cy
        qx_yaw = torch.zeros(n, device=self.device)
        qy_yaw = torch.zeros(n, device=self.device)
        qz_yaw = sy

        # Multiply: q_final = q_yaw * q_base
        qw = qw_yaw * base_qw - qx_yaw * base_qx - qy_yaw * base_qy - qz_yaw * base_qz
        qx = qw_yaw * base_qx + qx_yaw * base_qw + qy_yaw * base_qz - qz_yaw * base_qy
        qy = qw_yaw * base_qy - qx_yaw * base_qz + qy_yaw * base_qw + qz_yaw * base_qx
        qz = qw_yaw * base_qz + qx_yaw * base_qy - qy_yaw * base_qx + qz_yaw * base_qw

        orientations = torch.stack([qw, qx, qy, qz], dim=-1)

        self.camera.set_world_poses(positions, orientations, env_ids)


def collect_dataset():
    os.makedirs(os.path.join(args_cli.output_dir, "depth"),     exist_ok=True)
    os.makedirs(os.path.join(args_cli.output_dir, "collision"), exist_ok=True)

    cfg = QuadcoptervaeEnvCfg()
    cfg.scene.num_envs          = args_cli.num_envs
    cfg.scene.replicate_physics = False
    cfg.episode_length_s        = 0.5   # short episodes → camera moves often

    env = DepthCollectionEnv(cfg)
    collision_processor = CollisionImage()

    samples_collected = 0
    sample_idx        = 0

    print(f"Collecting {args_cli.num_samples} samples...")
    obs, _ = env.reset()

    with tqdm(total=args_cli.num_samples) as pbar:
        while samples_collected < args_cli.num_samples:
            obs, _ = env.reset() 
            # No-op actions — camera doesn't move between resets
            actions = torch.zeros(args_cli.num_envs, 4, device=env.device)
            obs, reward, terminated, truncated, info = env.step(actions)

            depth    = env.camera.data.output["distance_to_camera"]
            depth_np = depth.squeeze(-1).cpu().numpy()  # (N, H, W)

            for i in range(args_cli.num_envs):
                if samples_collected >= args_cli.num_samples:
                    break

                d = depth_np[i]
                d = np.where(np.isfinite(d), d, 10.0)  # replace inf with max range
                valid_pixels = np.sum((d > 0.1) & (d < 9.9))
                if valid_pixels < (d.size * 0.05):
                    continue

                collision = collision_processor.depth_to_collision_image(d)

                np.save(f"{args_cli.output_dir}/depth/{sample_idx:06d}.npy",     d.astype(np.float32))
                np.save(f"{args_cli.output_dir}/collision/{sample_idx:06d}.npy", collision.astype(np.float32))

                cv2.imwrite(
                    f"{args_cli.output_dir}/depth/{sample_idx:06d}.png",
                    (np.clip(d / 10.0, 0, 1) * 255).astype(np.uint8)
                )
                cv2.imwrite(
                    f"{args_cli.output_dir}/collision/{sample_idx:06d}.png",
                    (np.clip(collision, 0, 1) * 255).astype(np.uint8)
                )

                sample_idx        += 1
                samples_collected += 1
                pbar.update(1)

    env.close()
    print(f"Done. {samples_collected} samples saved to {args_cli.output_dir}")


if __name__ == "__main__":
    collect_dataset()
    simulation_app.close()