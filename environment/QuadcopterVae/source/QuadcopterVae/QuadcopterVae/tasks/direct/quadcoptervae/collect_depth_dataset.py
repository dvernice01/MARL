# # scripts/collect_depth_dataset.py

# import argparse
# from isaaclab.app import AppLauncher

# parser = argparse.ArgumentParser(description="Collect depth dataset from Isaac Lab")
# parser.add_argument("--num_envs",    type=int, default=16)
# parser.add_argument("--num_samples", type=int, default=10000)
# parser.add_argument("--output_dir",  type=str, default="/workspace/vae_container/Vae/isaaclab_dataset")
# AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()

# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# import torch
# import numpy as np
# import os
# import cv2
# from tqdm import tqdm

# import isaaclab.sim as sim_utils
# from isaaclab.assets import Articulation
# from isaaclab.envs import DirectRLEnv
# from isaaclab.sensors import Camera
# from isaaclab.terrains import TerrainImporter
# from QuadcopterVae.tasks.direct.quadcoptervae.quadcoptervae_env_cfg import QuadcoptervaeEnvCfg
# from QuadcopterVae.tasks.direct.quadcoptervae.quadcoptervae_env import CollisionImage


# class DepthCollectionEnv(DirectRLEnv):
#     """
#     Minimal env that only spawns the robot + camera.
#     No VAE, no policy network, no reward shaping.
#     """
#     cfg: QuadcoptervaeEnvCfg

#     def __init__(self, cfg, render_mode=None, **kwargs):
#         super().__init__(cfg, render_mode, **kwargs)

#         self._actions = torch.zeros(
#             self.num_envs, 4, device=self.device
#         )
#         self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
#         self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

#         self._body_id       = self._robot.find_bodies("body")[0]
#         self._robot_mass    = self._robot.root_physx_view.get_masses()[0].sum()
#         self._gravity_mag   = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
#         self._robot_weight  = (self._robot_mass * self._gravity_mag).item()

#     def _setup_scene(self):
#         self._robot = Articulation(self.cfg.robot)
#         self.scene.articulations["robot"] = self._robot

#         self.camera = Camera(self.cfg.camera)
#         self.scene.sensors["camera"] = self.camera

#         self.scene.clone_environments(copy_from_source=False)

#         self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
#         self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
#         self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

#         light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
#         light_cfg.func("/World/Light", light_cfg)

#     def _pre_physics_step(self, actions: torch.Tensor):
#         # random hover-like thrust so drone stays airborne
#         self._actions = actions.clamp(-1.0, 1.0)
#         self._thrust[:, 0, 2] = (
#             self.cfg.thrust_to_weight * self._robot_weight
#             * (self._actions[:, 0] + 1.0) / 2.0
#         )
#         self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

#     def _apply_action(self):
#         self._robot.permanent_wrench_composer.set_forces_and_torques(
#             body_ids=self._body_id,
#             forces=self._thrust,
#             torques=self._moment,
#         )

#     def _get_observations(self) -> dict:
#         # minimal — just return a dummy obs so the env loop works
#         return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}

#     def _get_rewards(self) -> torch.Tensor:
#         return torch.zeros(self.num_envs, device=self.device)

#     def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
#         pos = self._robot.data.root_pos_w

#         died = (
#             (pos[:, 2] < 0.3)    |   # too low — about to hit floor
#             (pos[:, 2] > 9.5)    |   # too high
#             (torch.abs(pos[:, 0] - self._terrain.env_origins[:, 0]) > 12.0) |  # too far x
#             (torch.abs(pos[:, 1] - self._terrain.env_origins[:, 1]) > 8.0)     # too far y
#         )

#         lin_vel = self._robot.data.root_lin_vel_b
#         vel_exploded = torch.norm(lin_vel, dim=-1) > 20.0

#         time_out = self.episode_length_buf >= self.max_episode_length - 1

#         return died | vel_exploded, time_out

#     def _reset_idx(self, env_ids):
#         if env_ids is None or len(env_ids) == self.num_envs:
#             env_ids = self._robot._ALL_INDICES

#         super()._reset_idx(env_ids)

#         n = len(env_ids)
#         default_root_state = self._robot.data.default_root_state[env_ids].clone()
#         default_root_state[:, 7:] = 0.0
#         default_root_state[:, 0] = torch.zeros(n, device=self.device).uniform_(-8.0, 0.0)
#         default_root_state[:, 1] = torch.zeros(n, device=self.device).uniform_(-4.0, 4.0)
#         default_root_state[:, 2] = torch.zeros(n, device=self.device).uniform_(1.5, 5.0)  # min 1.5m — safe margin above floor

#         default_root_state[:, 3] = 1.0
#         default_root_state[:, 4] = 0.0
#         default_root_state[:, 5] = 0.0
#         default_root_state[:, 6] = 0.0
#         default_root_state[:, :3] += self._terrain.env_origins[env_ids]

#         self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
#         self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)


# def collect_dataset():
#     os.makedirs(os.path.join(args_cli.output_dir, "depth"),     exist_ok=True)
#     os.makedirs(os.path.join(args_cli.output_dir, "collision"), exist_ok=True)

#     cfg = QuadcoptervaeEnvCfg()
#     cfg.scene.num_envs          = args_cli.num_envs
#     cfg.scene.replicate_physics = False
#     cfg.episode_length_s        = 10.0   # short episodes → more diverse positions

#     env = DepthCollectionEnv(cfg)
#     collision_processor = CollisionImage()

#     samples_collected = 0
#     sample_idx        = 0

#     print(f"Collecting {args_cli.num_samples} samples...")

#     obs, _ = env.reset()

#     with tqdm(total=args_cli.num_samples) as pbar:
#         while samples_collected < args_cli.num_samples:

#             actions = torch.zeros(args_cli.num_envs, 4, device=env.device)
#             actions[:, 0] = 0.5   # constant hover thrust — don't randomize this
#             actions[:, 1:] = torch.zeros(
#                 args_cli.num_envs, 3, device=env.device
#             ).uniform_(-0.05, 0.05)   # very small moments for gentle movement

#             obs, reward, terminated, truncated, info = env.step(actions)

#             # get raw depth (N, H, W, 1)
#             depth = env.camera.data.output["distance_to_camera"]
#             depth_np = depth.squeeze(-1).cpu().numpy()   # (N, H, W)

#             for i in range(args_cli.num_envs):
#                 if samples_collected >= args_cli.num_samples:
#                     break

#                 d = depth_np[i]

#                 # skip invalid frames (drone just reset or depth all zeros)
#                 valid_pixels = np.sum((d > 0.1) & (d < 9.9))
#                 if valid_pixels < (d.size * 0.05):   # need at least 5% valid pixels
#                     continue

#                 # compute collision image
#                 collision = collision_processor.depth_to_collision_image(d)

#                 # save .npy for training
#                 np.save(f"{args_cli.output_dir}/depth/{sample_idx:06d}.npy",     d.astype(np.float32))
#                 np.save(f"{args_cli.output_dir}/collision/{sample_idx:06d}.npy", collision.astype(np.float32))

#                 # save .png for visual inspection
#                 cv2.imwrite(
#                     f"{args_cli.output_dir}/depth/{sample_idx:06d}.png",
#                     (np.clip(d / 10.0, 0, 1) * 255).astype(np.uint8)
#                 )
#                 cv2.imwrite(
#                     f"{args_cli.output_dir}/collision/{sample_idx:06d}.png",
#                     (np.clip(collision, 0, 1) * 255).astype(np.uint8)
#                 )

#                 sample_idx        += 1
#                 samples_collected += 1
#                 pbar.update(1)

#     env.close()
#     print(f"Done. {samples_collected} samples saved to {args_cli.output_dir}")


# if __name__ == "__main__":
#     collect_dataset()
#     simulation_app.close()

# scripts/collect_depth_dataset.py

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect depth dataset from Isaac Lab")
parser.add_argument("--num_envs",    type=int, default=16)
parser.add_argument("--num_samples", type=int, default=10000)
parser.add_argument("--output_dir",  type=str, default="/workspace/vae_container/Vae/isaaclab_dataset")
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

        pos_x = torch.zeros(n, device=self.device).uniform_(-8.0, 0.0) + self._terrain.env_origins[env_ids, 0]
        pos_y = torch.zeros(n, device=self.device).uniform_(-4.0, 4.0) + self._terrain.env_origins[env_ids, 1]
        pos_z = torch.full((n,), 4.5, device=self.device) + self._terrain.env_origins[env_ids, 2]

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