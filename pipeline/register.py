"""
register.py
Aligns a reconstructed point cloud (from Meshroom/COLMAP) to the BIM point cloud.
Uses FPFH + RANSAC for coarse alignment, then ICP for fine registration.
Usage:
    python register.py --source data/pointclouds/reconstructed.ply \
                       --target data/bim/bim_cloud.ply \
                       --output data/pointclouds/registered.ply
"""

import argparse
import numpy as np
import open3d as o3d


VOXEL_SIZE = 0.05  # meters — adjust based on your scene scale


def preprocess(pcd: o3d.geometry.PointCloud, voxel_size: float):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
    )
    return pcd_down, fpfh


def coarse_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size: float):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 500),
    )
    return result


def fine_registration(src, tgt, init_transform, voxel_size: float):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=distance_threshold,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Register reconstructed cloud to BIM.")
    parser.add_argument("--source", required=True, help="Reconstructed point cloud (.ply)")
    parser.add_argument("--target", required=True, help="BIM point cloud (.ply)")
    parser.add_argument("--output", required=True, help="Output registered cloud (.ply)")
    parser.add_argument("--voxel", type=float, default=VOXEL_SIZE, help="Voxel size in meters")
    args = parser.parse_args()

    print("Loading point clouds...")
    source = o3d.io.read_point_cloud(args.source)
    target = o3d.io.read_point_cloud(args.target)
    print(f"  Source: {len(source.points):,} pts | Target: {len(target.points):,} pts")

    print("Preprocessing (downsample + FPFH features)...")
    src_down, src_fpfh = preprocess(source, args.voxel)
    tgt_down, tgt_fpfh = preprocess(target, args.voxel)

    print("Coarse registration (RANSAC)...")
    coarse = coarse_registration(src_down, tgt_down, src_fpfh, tgt_fpfh, args.voxel)
    print(f"  Coarse fitness: {coarse.fitness:.4f}  RMSE: {coarse.inlier_rmse:.4f}")

    print("Fine registration (ICP)...")
    fine = fine_registration(source, target, coarse.transformation, args.voxel)
    print(f"  Fine fitness:   {fine.fitness:.4f}  RMSE: {fine.inlier_rmse:.4f}")

    source.transform(fine.transformation)
    o3d.io.write_point_cloud(args.output, source)
    print(f"Registered cloud saved to {args.output}")

    np.save(args.output.replace(".ply", "_transform.npy"), fine.transformation)
    print("Transformation matrix saved alongside output.")


if __name__ == "__main__":
    main()
