import open3d as o3d
import trimesh
import numpy as np
import os
from matplotlib.path import Path





def crop_mesh(input_dir, output_dir, axis_min=-1.0, axis_max = 1.0, min_x=-1.0, min_y = -1.0, max_x= 1.0, max_y = 1.0, use_path = False):
    # 读取PLY格式的三维网

    mesh = trimesh.load(input_dir)
    #axis_min = -1.0
    #axis_max = 1.0

    if use_path:
        # 多边形包围盒坐标 (只取X和Y平面上的坐标)
        bounding_box_coords = np.array([
            [-16.852591729920281, 3.895802325141847],
            [-11.135200946341929, 3.754112054454025],
            [-11.013322229723768, 6.7348741029415331],
            [-5.7507196201927995, 6.7204542966325933],
            [-5.5108492653671446, 7.0281604243033149],
            [-1.3832735517493147, 6.9838978884158802],
            [-1.3624588738083565, 6.344285436399443],
            [3.4256693903696176, 6.1704954229717224],
            [3.2398419644147696, -5.3797694304872516],
            [0.023723443283972578, -5.2557590962303875],
            [-0.012890327551935421, -4.3687529238006126],
            [-1.566217016302704, -4.3002901912381786],
            [-1.566217016302704, -4.8247949720232715],
            [-1.566217016302704, -4.972253165234668],
            [-5.8999332753117404, -4.8965178347077272],
            [-5.9379263320352997, -4.090513356848156],
            [-9.7234391455149272, -4.0704510176357216],
            [-9.8622454616633863, -4.8148891920371319],
            [-16.942496091870094, -4.5925733956393389],
            [-16.809332308898405, 3.803264785524489]
        ])
        polygon_path = Path(bounding_box_coords)
        # 获取网格顶点的二维投影（X和Y坐标）
        vertices_2d = mesh.vertices[:, :2]  # 只取X和Y

        # 判断哪些顶点在多边形内
        vertices_in_polygon = polygon_path.contains_points(vertices_2d)
        # z_mask = ((mesh.vertices[:, 2] >= axis_min) & (mesh.vertices[:, 2] <= axis_max))
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        z_mask = ((vertices[:, 2] >= axis_min) & (vertices[:, 2] <= axis_max))
        mask = z_mask & vertices_in_polygon



    else:
        #min_x, min_y = -1.0, -1.0
        #min_x, min_y = -0.6, -1.0
        #max_x, max_y = 1.0, 1.0

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




    # 保存分割后的三维网格为新的PLY文件
    #mesh_in_polygon.export('exp/demo/test/d_crop_mesh.ply')
    #mesh_in_polygon.export('exp/real_storage/start/point_cloud/iteration_30000/point_cloud1.ply')
    mesh_in_polygon.export(output_dir)
    print("分割后的三维网格已保存到 mesh_in_polygon.ply")

if __name__ == "__main__":
    input_dir = '/media/wd/软件/GOF_vanilla/exp/100109/test/ours_40000/tsdf/d1_USB_tsdf.ply'
    output_dir = '/media/wd/软件/GOF_vanilla/exp/100109/test/ours_40000/tsdf/test.ply'
    crop_mesh(input_dir, output_dir, min_y = -2.0)

