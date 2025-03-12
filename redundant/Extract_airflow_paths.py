import open3d as o3d
import numpy as np

# Load the CAD model (assuming exported as .stl)
stl_file = "HVAC_Building.stl"
mesh = o3d.io.read_triangle_mesh(stl_file)

# Convert to point cloud
pcd = mesh.sample_points_uniformly(number_of_points=10000)

# Visualize point cloud
o3d.visualization.draw_geometries([pcd])

# Extract airflow paths (basic approach: detect empty spaces)
points = np.asarray(pcd.points)
empty_space = points[np.where(points[:, 2] > np.mean(points[:, 2]))]  # Higher than avg Z-axis

print(f"Total Points: {len(points)}, Empty Space Points (Potential Airflow Paths): {len(empty_space)}")
