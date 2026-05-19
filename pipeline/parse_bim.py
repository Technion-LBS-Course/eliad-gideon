"""
parse_bim.py
Converts an IFC or OBJ/PLY BIM model into an Open3D point cloud (.ply).
Usage:
    python parse_bim.py --input data/bim/model.ifc --output data/bim/bim_cloud.ply --samples 100000
"""

import argparse
import numpy as np
import open3d as o3d


def ifc_to_pointcloud(ifc_path: str, n_samples: int) -> o3d.geometry.PointCloud:
    try:
        import ifcopenshell
        import ifcopenshell.geom
    except ImportError:
        raise ImportError("ifcopenshell is not installed. Run: pip install ifcopenshell")

    ifc_file = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    vertices_all = []
    for product in ifc_file.by_type("IfcProduct"):
        if not product.Representation:
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            verts = np.array(shape.geometry.verts).reshape(-1, 3)
            vertices_all.append(verts)
        except Exception:
            continue

    if not vertices_all:
        raise ValueError("No geometry found in IFC file.")

    all_verts = np.vstack(vertices_all)

    # Sample uniformly from all vertices
    idx = np.random.choice(len(all_verts), size=min(n_samples, len(all_verts)), replace=False)
    sampled = all_verts[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sampled)
    return pcd


def mesh_to_pointcloud(mesh_path: str, n_samples: int) -> o3d.geometry.PointCloud:
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_triangles():
        raise ValueError(f"No mesh geometry found in {mesh_path}")
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=n_samples)
    return pcd


def main():
    parser = argparse.ArgumentParser(description="Convert BIM model to point cloud.")
    parser.add_argument("--input", required=True, help="Path to IFC, OBJ, or PLY file")
    parser.add_argument("--output", required=True, help="Output PLY path")
    parser.add_argument("--samples", type=int, default=100_000, help="Number of points to sample")
    args = parser.parse_args()

    ext = args.input.lower().split(".")[-1]

    if ext == "ifc":
        print(f"Parsing IFC: {args.input}")
        pcd = ifc_to_pointcloud(args.input, args.samples)
    elif ext in ("obj", "ply", "stl"):
        print(f"Sampling mesh: {args.input}")
        pcd = mesh_to_pointcloud(args.input, args.samples)
    else:
        raise ValueError(f"Unsupported format: .{ext}. Use .ifc, .obj, .ply, or .stl")

    print(f"Point cloud has {len(pcd.points):,} points.")
    o3d.io.write_point_cloud(args.output, pcd)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
