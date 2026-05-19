"""
deviation.py
Computes point-to-mesh (or point-to-cloud) distances between the registered
reconstruction and the BIM model. Outputs a colored point cloud and a CSV report.
Usage:
    python deviation.py --source data/pointclouds/registered.ply \
                        --bim data/bim/model.obj \
                        --output data/pointclouds/deviation.ply \
                        --threshold 0.05
"""

import argparse
import numpy as np
import open3d as o3d
import pandas as pd


def colormap(distances: np.ndarray, max_dist: float) -> np.ndarray:
    """Maps distances to a green→yellow→red colormap."""
    t = np.clip(distances / max_dist, 0, 1)
    colors = np.zeros((len(t), 3))
    colors[:, 0] = t                    # red increases with distance
    colors[:, 1] = 1.0 - t             # green decreases with distance
    return colors


def compute_deviation_cloud_to_mesh(source_pcd, bim_mesh_path: str):
    mesh = o3d.io.read_triangle_mesh(bim_mesh_path)
    if not mesh.has_triangles():
        raise ValueError(f"No mesh found in {bim_mesh_path}")
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    pts = np.asarray(source_pcd.points).astype(np.float32)
    query = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
    distances = scene.compute_point_cloud_distance(query).numpy()
    return distances


def compute_deviation_cloud_to_cloud(source_pcd, target_pcd):
    dists = source_pcd.compute_point_cloud_distance(target_pcd)
    return np.asarray(dists)


def main():
    parser = argparse.ArgumentParser(description="Compute deviations between reconstruction and BIM.")
    parser.add_argument("--source", required=True, help="Registered reconstruction (.ply)")
    parser.add_argument("--bim", required=True, help="BIM mesh (.obj/.ply) or BIM cloud (.ply)")
    parser.add_argument("--output", required=True, help="Output deviation cloud (.ply)")
    parser.add_argument("--threshold", type=float, default=0.05, help="Deviation alert threshold in meters")
    parser.add_argument("--max-color-dist", type=float, default=0.2, help="Distance mapped to full red")
    args = parser.parse_args()

    print("Loading source cloud...")
    source = o3d.io.read_point_cloud(args.source)

    ext = args.bim.lower().split(".")[-1]
    if ext in ("obj", "stl"):
        print("Computing cloud-to-mesh distances...")
        distances = compute_deviation_cloud_to_mesh(source, args.bim)
    else:
        print("Computing cloud-to-cloud distances (no mesh supplied)...")
        target = o3d.io.read_point_cloud(args.bim)
        distances = compute_deviation_cloud_to_cloud(source, target)

    print(f"Distance stats — mean: {distances.mean():.4f}m  max: {distances.max():.4f}m  "
          f">{args.threshold}m: {(distances > args.threshold).sum():,} pts "
          f"({100*(distances > args.threshold).mean():.1f}%)")

    colors = colormap(distances, args.max_color_dist)
    source.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(args.output, source)
    print(f"Deviation cloud saved to {args.output}")

    pts = np.asarray(source.points)
    df = pd.DataFrame({
        "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
        "deviation_m": distances,
        "alert": distances > args.threshold,
    })
    csv_path = args.output.replace(".ply", "_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"Report saved to {csv_path}")


if __name__ == "__main__":
    main()
