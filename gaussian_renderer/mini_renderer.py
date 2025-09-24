import torch
import math

from ag_diff_gaussian_rasterization import GaussianRasterizationSettings as ag_GaussianRasterizationSettings
from ag_diff_gaussian_rasterization import GaussianRasterizer as ag_GaussianRasterizer
from scene.gaussian_model import GaussianModel

import numpy as np
import seaborn as sns

def quaternion_to_rotation_matrix(q, get_R4 = False):
    """
    将四元数转换为 4x4 旋转矩阵
    :param q: 四元数 (w, x, y, z) -> Tensor of shape (N, 4)
    :return: 4x4 旋转矩阵 Tensor of shape (N, 4, 4)
    """
    # 提取四元数分量
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # 计算旋转矩阵的分量 (3x3)
    R = torch.stack([
        1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)
    ], dim=-1).reshape(-1, 3, 3)




    # 扩展为 4x4 旋转矩阵
    R_4 = torch.cat([
        torch.cat([R, torch.zeros(R.shape[0], 3, 1).cuda()], dim=-1),  # 添加第三列（0, 0, 0）
        torch.tensor([0.0, 0.0, 0.0, 1.0]).reshape(1, 4).repeat(R.shape[0], 1).unsqueeze(1).cuda()  # 最后一行 (0, 0, 0, 1)
    ], dim=-2)
    if get_R4:
        return R_4
    else:
        return R


def update_rotation_matrix(current_rotation_matrix, delta_quaternion):
    """
    使用四元数增量更新当前的旋转矩阵
    :param current_rotation_matrix: 当前旋转矩阵 (3, 3)
    :param delta_quaternion: 四元数增量 (w, x, y, z) -> Tensor of shape (4,)
    :return: 更新后的旋转矩阵 (3, 3)
    """
    # 将四元数增量转换为旋转矩阵
    delta_rotation_matrix = quaternion_to_rotation_matrix(delta_quaternion)

    # 更新旋转矩阵
    updated_rotation_matrix = torch.mm(delta_rotation_matrix, current_rotation_matrix)
    return updated_rotation_matrix

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def ag_render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz=None, d_rot=None,
           scaling_modifier=1.0, random_bg_color=False, scale_const=None, mask=None, vis_mask=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    bg = bg_color if not random_bg_color else torch.rand_like(bg_color)

    raster_settings = ag_GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = ag_GaussianRasterizer(raster_settings=raster_settings)

    xyz = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    sh_features = pc.get_features

    means3D = xyz + d_xyz if d_xyz is not None else xyz
    means2D = screenspace_points
    if scale_const is not None:
        opacity = torch.ones_like(pc.get_opacity)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(d_rot, scaling_modifier)
    else:
        rotations = quaternion_multiply(d_rot, rotations) if d_rot is not None else rotations

    if mask != None:
        shs = None
        pallete = torch.from_numpy(np.array(sns.color_palette("hls", mask.max() + 1))).float().to(pc.get_xyz.device)
        colors_precomp = pallete[mask]
    else:
        shs = sh_features
        colors_precomp = None

    if scale_const is not None:
        scales = scale_const * torch.ones_like(scales)

    # Rasterize visible Gaussians to image.
    if vis_mask is not None:
        means3D = means3D[vis_mask]
        means2D = means2D[vis_mask]
        shs = shs[vis_mask] if shs is not None else None
        colors_precomp = colors_precomp[vis_mask] if colors_precomp is not None else None
        opacity = opacity[vis_mask]
        scales = scales[vis_mask]
        rotations = rotations[vis_mask]
        cov3D_precomp = cov3D_precomp[vis_mask] if cov3D_precomp is not None else None

    rendered_image, radii, depth, alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            "alpha": alpha,
            "bg_color": bg}


