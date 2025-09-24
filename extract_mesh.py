import torch
from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import open3d as o3d
import open3d.core as o3c
import math
from crop_mesh import crop_mesh

from cd_eval import compute_recon_error

id_mapping = {"103706":"blade", "102255": "foldchair", "10905":"fridge", "10211":"laptop", "101917":"oven"
              , "11100":"scissor", "103111":"stapler", "45135":"storage", "100109":"USB", "103776":"washer"}

def color_filter(mesh, render_path, black_threshold = 0.1):
    import trimesh
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=np.asarray(mesh.vertex_colors)
    )
    vertex_colors = mesh.visual.vertex_colors[:, :3]  # 去除可能的alpha通道

    # 定义黑色阈值（对应0-255范围）
    black_threshold *= 255  # 相当于Open3D中的0.01阈值
    is_black = np.all(vertex_colors <= black_threshold, axis=1)

    # 创建非黑色顶点掩码
    non_black_mask = ~is_black

    # 检查每个面片的所有顶点是否都是非黑色
    valid_faces_mask = np.all(non_black_mask[mesh.faces], axis=1)

    # 直接通过面片掩码创建子网格
    filtered_mesh = mesh.submesh(
        [valid_faces_mask],  # 使用布尔掩码直接筛选
        append=True,  # 合并所有有效面片
        repair=False  # 关闭自动修复以提升性能
    )

    # 清理未使用的顶点（trimesh特有优化方法）
    filtered_mesh.remove_unreferenced_vertices()
    filtered_mesh_path = f"{render_path}/filtered_mesh.ply"
    # 导出结果（自动保持顶点颜色）
    filtered_mesh.export(filtered_mesh_path)
    return filtered_mesh_path


def tsdf_fusion(model_path, name, iteration, views, test_views, gaussians, pipeline, background, kernel_size, only_extract_mesh=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "tsdf")

    makedirs(render_path, exist_ok=True)
    #o3d_device = o3d.core.Device("CUDA:0")
    o3d_device = o3d.core.Device("cpu:0")
    
    #voxel_size = 0.002
    voxel_size = 0.004
    alpha_thres=0.5
    max_depth = 5.0

    vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=voxel_size,
            block_resolution=16,
            block_count=50000,
            device=o3d_device)
    
    with torch.no_grad():
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            
            rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]
            
            depth = rendering[6:7, :, :]
            alpha = rendering[7:8, :, :]
            rgb = rendering[:3, :, :]
            
            if view.gt_alpha_mask is not None:
                depth[(view.gt_alpha_mask < 0.5)] = 0
            
            depth[(alpha < alpha_thres)] = 0


            depth[depth > max_depth] = 0

            W = view.image_width
            H = view.image_height
            ndc2pix = torch.tensor([
                [W / 2, 0, 0, (W-1) / 2],
                [0, H / 2, 0, (H-1) / 2],
                [0, 0, 0, 1]]).float().cuda().T
            intrins =  (view.projection_matrix @ ndc2pix)[:3,:3].T
            intrinsic=o3d.camera.PinholeCameraIntrinsic(
                width=W,
                height=H,
                cx = intrins[0,2].item(),
                cy = intrins[1,2].item(), 
                fx = intrins[0,0].item(), 
                fy = intrins[1,1].item()
            )
            
            extrinsic = np.asarray((view.world_view_transform.T).cpu().numpy())
            
            o3d_color = o3d.t.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy(), order="C"))
            o3d_depth = o3d.t.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C"))
            o3d_color = o3d_color.to(o3d_device)
            o3d_depth = o3d_depth.to(o3d_device)

            intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64)#.to(o3d_device)
            extrinsic = o3d.core.Tensor(extrinsic, o3d.core.Dtype.Float64)#.to(o3d_device)
            
            frustum_block_coords = vbg.compute_unique_block_coordinates(
                o3d_depth, intrinsic, extrinsic, 1.0, 6.0)

            vbg.integrate(frustum_block_coords, o3d_depth, o3d_color, intrinsic,
                          intrinsic, extrinsic, 1.0, 6.0)
            
        mesh = vbg.extract_triangle_mesh().to_legacy()
        filter_mesh_path = color_filter(mesh, render_path, black_threshold= 0.01)  # storage 0.2, foldchair = 0.06, oven = 0.2
        crop_mesh(filter_mesh_path, filter_mesh_path, axis_min=-1, axis_max=1, min_x=-1, min_y=-1, max_y=1, max_x=1) # max_y=0.5 laptop min_y = 1.5 oven, min_y=-2:blade
        # write mesh
        o3d.io.write_triangle_mesh(f"{render_path}/tsdf.ply", mesh)
        #o3d.io.write_triangle_mesh(f"{render_path}/tsdf.obj", mesh)

            
def extract_mesh(dataset : ModelParams, iteration : int, pipeline : PipelineParams, only_extract_mesh=False):
    with torch.no_grad():
        dataset.load_time_camera = True
        dataset.init_num = False
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_root=True)
        #deform = DeformModel(dataset.is_blender, dataset.is_6dof, use_2dgs=dataset.use_2dgs)
        #deform.load_weights(dataset.model_path)
        train_cameras = scene.getTrainCameras()
    
        gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size
        
        cams = train_cameras
        test_cams = scene.getTestCameras()
        tsdf_fusion(dataset.model_path, "test", iteration, cams, test_cams, gaussians, pipeline, background, kernel_size, only_extract_mesh=only_extract_mesh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--only_extract_mesh", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    args.train_ours = True
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    extract_mesh(model.extract(args), args.iteration, pipeline.extract(args), only_extract_mesh=args.only_extract_mesh)