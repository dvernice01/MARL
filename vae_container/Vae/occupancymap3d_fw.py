import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="/workspace/environment/quadcopter_rnn/warehouse_3d_denser")
parser.add_argument("--cell_size",  type=float, default=0.05)
parser.add_argument("--scan_height_min", type=float, default=0.5)
parser.add_argument("--scan_height_max", type=float, default=9.0)
parser.add_argument("--scan_height_step", type=float, default=0.05)
parser.add_argument("--scan_spacing", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import numpy as np
import torch
import omni.usd
from pxr import Gf, UsdGeom, Usd

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.sensors import RayCaster, RayCasterCfg, MultiMeshRayCaster, MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab_mimic.locomanipulation_sdg.occupancy_map_utils import (
    OccupancyMap,
    OccupancyMapDataValue,
)
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import warp as wp
from isaaclab.utils.warp import convert_to_warp_mesh


def build_combined_warp_mesh(warehouse_prim_path: str, device: str = "cuda"):
    """
    Merge all mesh prims in the warehouse into a single warp mesh
    so the RayCaster can raycast against the entire scene at once.
    """

    stage = omni.usd.get_context().get_stage()
    
    root  = stage.GetPrimAtPath(warehouse_prim_path)

    all_points  = []
    all_indices = []
    vertex_offset = 0

    print("Merging mesh prims...")
    count = 0

    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "Mesh":
            continue

        mesh_prim = UsdGeom.Mesh(prim)

        points_attr   = mesh_prim.GetPointsAttr().Get()
        indices_attr  = mesh_prim.GetFaceVertexIndicesAttr().Get()

        if points_attr is None or indices_attr is None:
            continue

        points  = np.asarray(points_attr)
        indices = np.asarray(indices_attr)

        # transform vertices to world space
        transform_matrix = np.array(
            omni.usd.get_world_transform_matrix(mesh_prim)
        ).T
        points = np.matmul(points, transform_matrix[:3, :3].T)
        points += transform_matrix[:3, 3]

        all_points.append(points)
        all_indices.append(indices + vertex_offset)
        vertex_offset += len(points)
        count += 1

    print(f"Merged {count} mesh prims → "
          f"{vertex_offset} vertices, "
          f"{sum(len(i) for i in all_indices)} indices")

    all_points  = np.concatenate(all_points,  axis=0)
    all_indices = np.concatenate(all_indices, axis=0)

    wp_mesh = convert_to_warp_mesh(all_points, all_indices, device=device)
    return wp_mesh

def get_warehouse_bounds(warehouse_prim_path: str): 
    stage      = omni.usd.get_context().get_stage()
    prim       = stage.GetPrimAtPath(warehouse_prim_path)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    bbox       = bbox_cache.ComputeWorldBound(prim)
    r          = bbox.GetRange()
    mn, mx     = r.GetMin(), r.GetMax()
    print(f"Warehouse bounds:")
    print(f"  X: {mn[0]:.2f} → {mx[0]:.2f}  ({mx[0]-mn[0]:.1f}m)")
    print(f"  Y: {mn[1]:.2f} → {mx[1]:.2f}  ({mx[1]-mn[1]:.1f}m)")
    print(f"  Z: {mn[2]:.2f} → {mx[2]:.2f}  ({mx[2]-mn[2]:.1f}m)")
    return float(mn[0]), float(mx[0]), float(mn[1]), float(mx[1]), float(mn[2]), float(mx[2])


def move_sensor(xformable, sx, sy, sz):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(float(sx), float(sy), float(sz)))
            return


def build_3d_voxel_map(ray_caster, sim, xformable,
                        x_min, x_max, y_min, y_max, z_min, z_max,
                        cell, scan_spacing,
                        scan_height_min, scan_height_max, scan_height_step):
    """
    Build a true 3D binary voxel grid.
    - OCCUPIED: ray hit this voxel
    - FREE:     sensor was here (line of sight confirmed empty)
    - UNKNOWN:  never observed
    """
    n_x = int((x_max - x_min) / cell)
    n_y = int((y_max - y_min) / cell)
    n_z = int((z_max - z_min) / cell)
    print(f"  Voxel grid: {n_x} × {n_y} × {n_z}  ({n_x*n_y*n_z} voxels)")

    # uint8: 0=unknown, 1=free, 2=occupied
    grid = np.zeros((n_z, n_y, n_x), dtype=np.uint8)

    # scan positions — horizontal grid at multiple heights
    scan_x = np.arange(x_min + scan_spacing / 2, x_max, scan_spacing)
    scan_y = np.arange(y_min + scan_spacing / 2, y_max, scan_spacing)
    scan_z = np.arange(scan_height_min, scan_height_max, scan_height_step)

    total   = len(scan_x) * len(scan_y) * len(scan_z)
    current = 0

    for sz in scan_z:
        for sx in scan_x:
            for sy in scan_y:
                current += 1
                if current % 200 == 0:
                    print(f"    {current}/{total} scan positions")

                move_sensor(xformable, sx, sy, sz)

                for _ in range(3):
                    sim.step()
                    ray_caster.update(sim.current_time)

                # ── Mark sensor voxel as FREE ──────────────────────────────────
                sx_i = int((sx - x_min) / cell)
                sy_i = int((sy - y_min) / cell)
                sz_i = int((sz - z_min) / cell)
                if 0 <= sx_i < n_x and 0 <= sy_i < n_y and 0 <= sz_i < n_z:
                    if grid[sz_i, sy_i, sx_i] != 2:   # don't overwrite occupied
                        grid[sz_i, sy_i, sx_i] = 1

                # ── Get ray hits ───────────────────────────────────────────────
                ray_hits = ray_caster.data.ray_hits_w[0]          # (B, 3)
                valid    = torch.isfinite(ray_hits).all(dim=-1)
                hits     = ray_hits[valid].cpu().numpy()           # (M, 3)

                if hits.shape[0] == 0:
                    continue

                # ── Filter floor and ceiling ───────────────────────────────────
                # keep only hits that are NOT on the ground plane
                # and NOT on the ceiling (more than cell above scan height)
                floor_thresh   = z_min + cell              # anything below = floor
                ceiling_thresh = z_max - cell              # anything above = ceiling

                wall_mask = (hits[:, 2] > floor_thresh) & (hits[:, 2] < ceiling_thresh)
                hits      = hits[wall_mask]

                if hits.shape[0] == 0:
                    continue

                # ── Vectorized voxel indexing ──────────────────────────────────
                ix = ((hits[:, 0] - x_min) / cell).astype(int)
                iy = ((hits[:, 1] - y_min) / cell).astype(int)
                iz = ((hits[:, 2] - z_min) / cell).astype(int)

                in_bounds = (
                    (ix >= 0) & (ix < n_x) &
                    (iy >= 0) & (iy < n_y) &
                    (iz >= 0) & (iz < n_z)
                )
                ix = ix[in_bounds]
                iy = iy[in_bounds]
                iz = iz[in_bounds]

                # mark as OCCUPIED (value=2) — overwrites free
                grid[iz, iy, ix] = 2

    return grid, (n_z, n_y, n_x)


def save_results(grid, x_min, x_max, y_min, y_max, z_min, z_max, cell, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # ── Save raw 3D voxel grid ─────────────────────────────────────────────────
    npy_path = os.path.join(output_dir, "occupancy_3d.npy")
    np.save(npy_path, grid)

    # ── Save metadata ──────────────────────────────────────────────────────────
    meta = {
        "cell_size" : cell,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max,
        "shape": list(grid.shape),   # (n_z, n_y, n_x)
        "values": {"UNKNOWN": 0, "FREE": 1, "OCCUPIED": 2},
    }
    np.save(os.path.join(output_dir, "occupancy_3d_meta.npy"), meta)

    # ── Save horizontal slices as OccupancyMap (ROS format) for visualization ──
    n_z      = grid.shape[0]
    z_coords = np.linspace(z_min, z_max, n_z)
    origin   = (float(x_min), float(y_min), 0.0)

    slices_dir = os.path.join(output_dir, "slices")
    os.makedirs(slices_dir, exist_ok=True)

    for iz in range(n_z):
        occ_slice  = grid[iz] == 2   # OCCUPIED
        free_slice = grid[iz] == 1   # FREE

        omap = OccupancyMap.from_masks(
            freespace_mask=free_slice,
            occupied_mask=occ_slice,
            resolution=cell,
            origin=origin,
        )
        omap.save_ros(os.path.join(slices_dir, f"z_{z_coords[iz]:.2f}m"))

    print(f"\nSaved:")
    print(f"  3D grid  : {npy_path}")
    print(f"  Slices   : {slices_dir}/")
    print(f"  Unknown  : {(grid == 0).sum()} voxels")
    print(f"  Free     : {(grid == 1).sum()} voxels")
    print(f"  Occupied : {(grid == 2).sum()} voxels")


def main():
    # ── Simulation ─────────────────────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(dt=1/60, render_interval=1)
    sim     = SimulationContext(sim_cfg)

    # ── Load warehouse ─────────────────────────────────────────────────────────
    warehouse_cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd")
    warehouse_cfg.func("/World/Warehouse", warehouse_cfg)

    # ── Spawn sensor prim ──────────────────────────────────────────────────────
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, "/World/ScanSensor")
    sim_utils.standardize_xform_ops(xform.GetPrim())
    xformable = UsdGeom.Xformable(stage.GetPrimAtPath("/World/ScanSensor"))
    mesh_paths = []

    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith("/World/Warehouse"):
            if prim.IsA(UsdGeom.Mesh):
                mesh_paths.append(prim.GetPath().pathString)

    print(f"Found {len(mesh_paths)} meshes")

    # print("Building combined warehouse mesh (this takes ~1 min for 3473 prims)...")
    # combined_mesh = build_combined_warp_mesh("/World/Warehouse", device="cuda")

    # # inject into RayCaster class cache so it skips _initialize_warp_meshes
    # RayCaster.meshes["/World/Warehouse"] = combined_mesh
    # print("Combined mesh injected into RayCaster cache.")


    ray_caster_cfg = MultiMeshRayCasterCfg(
        prim_path="/World/ScanSensor",
        update_period=0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        mesh_prim_paths=mesh_paths,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=(4.0, 4.0),
        ),
        debug_vis=False,
    )
    ray_caster = MultiMeshRayCaster(ray_caster_cfg)

    sim.reset()
    for _ in range(20):
        sim.step()

    # ── Auto-detect bounds ─────────────────────────────────────────────────────
    x_min, x_max, y_min, y_max, z_min_w, z_max_w = get_warehouse_bounds("/World/Warehouse")

    # ── Build 3D voxel map ─────────────────────────────────────────────────────
    print(f"\nBuilding 3D voxel map...")
    grid, shape = build_3d_voxel_map(
        ray_caster   = ray_caster,
        sim          = sim,
        xformable    = xformable,
        x_min        = x_min,
        x_max        = x_max,
        y_min        = y_min,
        y_max        = y_max,
        z_min        = max(0.0, z_min_w),
        z_max        = z_max_w,
        cell         = args_cli.cell_size,
        scan_spacing = args_cli.scan_spacing,
        scan_height_min  = args_cli.scan_height_min,
        scan_height_max  = args_cli.scan_height_max,
        scan_height_step = args_cli.scan_height_step,
    )

    print(f"\n3D voxel map complete:")
    print(f"  Shape    : {shape}  (Z, Y, X)")
    print(f"  Unknown  : {(grid == 0).sum()}")
    print(f"  Free     : {(grid == 1).sum()}")
    print(f"  Occupied : {(grid == 2).sum()}")

    # ── Save everything ────────────────────────────────────────────────────────
    save_results(
        grid, x_min, x_max, y_min, y_max,
        max(0.0, z_min_w), z_max_w,
        args_cli.cell_size, args_cli.output_dir
    )


if __name__ == "__main__":
    main()
    simulation_app.close()