import open3d as o3d
import trimesh
import numpy as np
import os
from matplotlib.path import Path





def crop_mesh(input_dir, output_dir, axis_min=-1.0, axis_max = 1.0, min_x=-1.0, min_y = -1.0, max_x= 1.0, max_y = 1.0, use_path = False):


    mesh = trimesh.load(input_dir)


    # 打印BBox范围
    print(f"BBox范围: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")

    # 2. 使用 Bounding Box 来筛选点云
    # 获取点云的 x 和 y 坐标
    vertices = np.array(mesh.vertices)
    vertices_2d = vertices[:, :2]
    z_mask = ((vertices[:, 2] >= axis_min) & (vertices[:, 2] <= axis_max))

    # 3. 判断哪些点位于 BBox 内
    bbox = (vertices_2d[:, 0] >= min_x) & (vertices_2d[:, 0] <= max_x) & \
           (vertices_2d[:, 1] >= min_y) & (vertices_2d[:, 1] <= max_y)
    mask = z_mask & bbox



    # 获取三角形面片的所有顶点是否都在多边形内
    faces_in_polygon = np.all(mask[mesh.faces], axis=1)

    # 提取在多边形内的面
    mesh_in_polygon = mesh.submesh([faces_in_polygon], append=True)





    mesh_in_polygon.export(output_dir)
    print("save mesh_in_polygon.ply")



