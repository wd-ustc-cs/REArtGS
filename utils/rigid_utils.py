import torch
from scene import GaussianModel
import torch.nn.functional as F
from torch import nn
import numpy as np
from plyfile import PlyData
from pytorch3d.loss import chamfer_distance
from utils.pointnet2_utils import farthest_point_sample, index_points
from scipy.optimize import linear_sum_assignment
import open3d as o3d
from sklearn.cluster import SpectralClustering
import os

def match_pcd(pc0, pc1, N=5000):
    """
    Input:
        pc0, pc1: tensor [1, N0, 3], [1, N1, 3]
        N: downsample number
    Return:
        idx_s, idx_e: [N], [N]
    """
    # Downsample with farthest point sampling
    num_fps = min(pc0.shape[1], pc1.shape[1], N)
    s_idx = farthest_point_sample(pc0, num_fps)
    pc_s = index_points(pc0, s_idx)
    e_idx = farthest_point_sample(pc1, num_fps)
    pc_e = index_points(pc1, e_idx)

    # Matching
    with torch.no_grad():
        cost = torch.cdist(pc_s, pc_e).cpu().numpy()
    idx_s, idx_e = linear_sum_assignment(cost.squeeze())
    idx_s, idx_e = s_idx[0].cpu().numpy()[idx_s], e_idx[0].cpu().numpy()[idx_e]
    return idx_s, idx_e


def skew(w: torch.Tensor) -> torch.Tensor:
    """Build a skew matrix ("cross product matrix") for vector w.

    Modern Robotics Eqn 3.30.

    Args:
      w: (N, 3) A 3-vector

    Returns:
      W: (N, 3, 3) A skew matrix such that W @ v == w x v
    """
    zeros = torch.zeros(w.shape[0], device=w.device)
    w_skew_list = [zeros, -w[:, 2], w[:, 1],
                   w[:, 2], zeros, -w[:, 0],
                   -w[:, 1], w[:, 0], zeros]
    w_skew = torch.stack(w_skew_list, dim=-1).reshape(-1, 3, 3)
    return w_skew


def rp_to_se3(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Rotation and translation to homogeneous transform.

    Args:
      R: (3, 3) An orthonormal rotation matrix.
      p: (3,) A 3-vector representing an offset.

    Returns:
      X: (4, 4) The homogeneous transformation matrix described by rotating by R
        and translating by p.
    """
    bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=R.device).repeat(R.shape[0], 1, 1)
    transform = torch.cat([torch.cat([R, p], dim=-1), bottom_row], dim=1)

    return transform

def exp_so3_44(w, theta):
    """
    计算 SE(3) 中的旋转部分（4x4 矩阵）。
    输入:
        w: [N, 3] 旋转轴
        theta: [N, 1] 旋转角度
    输出:
        d_xyz: [N, 4, 4] 旋转齐次矩阵
    """

    W = skew(w)
    R = exp_so3(w, theta)
    p = torch.tensor([[0.0, 0.0, 0.0]], device=R.device).unsqueeze(dim=-1).repeat(R.shape[0], 1, 1)
    bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=R.device).repeat(R.shape[0], 1, 1)
    transform = torch.cat([torch.cat([R, p], dim=-1), bottom_row], dim=1)
    # batch_size = w.shape[0]
    #
    # # 计算反对称矩阵 [w]
    # w_hat = torch.zeros((batch_size, 3, 3), device=w.device)
    # w_hat[:, 0, 1] = -w[:, 2]
    # w_hat[:, 0, 2] = w[:, 1]
    # w_hat[:, 1, 0] = w[:, 2]
    # w_hat[:, 1, 2] = -w[:, 0]
    # w_hat[:, 2, 0] = -w[:, 1]
    # w_hat[:, 2, 1] = w[:, 0]
    #
    # # 单位矩阵
    # I = torch.eye(3, device=w.device).unsqueeze(0).repeat(batch_size, 1, 1)
    #
    # # Rodrigues公式计算旋转矩阵 R
    # R = I + torch.sin(theta).unsqueeze(-1) * w_hat + \
    #     (1 - torch.cos(theta).unsqueeze(-1)) * torch.bmm(w_hat, w_hat)
    #
    # # 构造 4x4 齐次变换矩阵
    # T = torch.eye(4, device=w.device).unsqueeze(0).repeat(batch_size, 1, 1)
    # T[:, :3, :3] = R  # 填入旋转部分

    return transform

def exp_so3(w: torch.Tensor, theta: float) -> torch.Tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3.

    Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.

    Args:
      w: (3,) An axis of rotation.
      theta: An angle of rotation.

    Returns:
      R: (3, 3) An orthonormal rotation matrix representing a rotation of
        magnitude theta about axis w.
    """
    W = skew(w)
    identity = torch.eye(3).unsqueeze(0).repeat(W.shape[0], 1, 1).to(W.device)
    W_sqr = torch.bmm(W, W)  # batch matrix multiplication
    R = identity + torch.sin(theta.unsqueeze(-1)) * W + (1.0 - torch.cos(theta.unsqueeze(-1))) * W_sqr
    return R


def exp_se3(S: torch.Tensor, theta: float) -> torch.Tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3.

    Modern Robotics Eqn 3.88.

    Args:
      S: (6,) A screw axis of motion.
      theta: Magnitude of motion.

    Returns:
      a_X_b: (4, 4) The homogeneous transformation matrix attained by integrating
        motion of magnitude theta about S for one second.
    """
    w, v = torch.split(S, 3, dim=-1)
    W = skew(w)
    R = exp_so3(w, theta)

    identity = torch.eye(3).unsqueeze(0).repeat(W.shape[0], 1, 1).to(W.device)
    W_sqr = torch.bmm(W, W)
    theta = theta.view(-1, 1, 1)

    p = torch.bmm((theta * identity + (1.0 - torch.cos(theta)) * W + (theta - torch.sin(theta)) * W_sqr),
                  v.unsqueeze(-1))
    return rp_to_se3(R, p)


def to_homogenous(v: torch.Tensor) -> torch.Tensor:
    """Converts a vector to a homogeneous coordinate vector by appending a 1.

    Args:
        v: A tensor representing a vector or batch of vectors.

    Returns:
        A tensor with an additional dimension set to 1.
    """
    return torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)


def from_homogenous(v: torch.Tensor) -> torch.Tensor:
    """Converts a homogeneous coordinate vector to a standard vector by dividing by the last element.

    Args:
        v: A tensor representing a homogeneous coordinate vector or batch of homogeneous coordinate vectors.

    Returns:
        A tensor with the last dimension removed.
    """
    return v[..., :3] / v[..., -1:]

def R_from_axis_angle(k: torch.tensor, theta: torch.tensor):
    if torch.norm(k) == 0.:
        return torch.eye(3)
    k = F.normalize(k, p=2., dim=0)
    kx, ky, kz = k[0], k[1], k[2]
    cos, sin = torch.cos(theta), torch.sin(theta)
    R = torch.zeros((3, 3)).to(k)
    R[0, 0] = cos + (kx**2) * (1 - cos)
    R[0, 1] = kx * ky * (1 - cos) - kz * sin
    R[0, 2] = kx * kz * (1 - cos) + ky * sin
    R[1, 0] = kx * ky * (1 - cos) + kz * sin
    R[1, 1] = cos + (ky**2) * (1 - cos)
    R[1, 2] = ky * kz * (1 - cos) - kx * sin
    R[2, 0] = kx * kz * (1 - cos) - ky * sin
    R[2, 1] = ky * kz * (1 - cos) + kx * sin
    R[2, 2] = cos + (kz**2) * (1 - cos)
    return R
    # batch_size = k.size(0)
    # if batch_size == 0:
    #     return torch.eye(3).expand(batch_size, 3, 3)
    # k = F.normalize(k, p=2., dim=1)
    # R = exp_so3(k, theta)
    return  R

def rigid_transform(gaussians: GaussianModel, state):
    '''
    Perform the rigid transformation: R @ (x - c) + c

    Transform the positions from canonical state=0.5 to state=0 or state=1
    '''
    if gaussians.use_canonical:
        scaling = (gaussians.canonical - state) / gaussians.canonical
    else:
        scaling = state
    positions = gaussians.get_xyz
    positions = positions - gaussians.axis_o
    if scaling == 1.:
        R = R_from_quaternions(gaussians.quaternions)
        #positions = torch.matmul(R, positions.T).T
    elif scaling == -1.:
        inv_sc = torch.tensor([1., -1., -1., -1]).to(gaussians.quaternions)
        inv_q = inv_sc * gaussians.quaternions
        R = R_from_quaternions(inv_q)
        #positions = torch.matmul(R, positions.T).T
    else:
        axis, angle = quaternion_to_axis_angle(gaussians.quaternions)  # the angle means from t=0 to t=0.5
        tgt_angle = scaling * angle
        R = R_from_axis_angle(axis, tgt_angle)
        #positions = torch.matmul(R, positions.T).T
    rotation_matrix = torch.eye(3).unsqueeze(dim=0).repeat(gaussians.get_xyz.shape[0],1,1).cuda()
    rotation_matrix[gaussians.dynamic_part_mask] = R
    #rotation_matrix = R
    positions = torch.matmul(rotation_matrix, positions.unsqueeze(dim=-1)).squeeze(dim=-1)
    positions = positions + gaussians.axis_o

    return positions

def R_from_quaternions(quaternions: torch.tensor):
    quaternions = F.normalize(quaternions, p=2., dim=0)

    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3)).to(quaternions)

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    axis = quaternions[..., 1:] / sin_half_angles_over_angles
    axis = F.normalize(axis, p=2., dim=0)
    return axis, angles

def revolute_judge_dynamic(rotation_angle, gaussians: GaussianModel, dynamic_threshold):
    new_dynamic_mask = (rotation_angle.squeeze(dim=1) > dynamic_threshold).cpu().numpy()
    #static2dynamic_mask = (rotation_angle.squeeze(dim=1) > dynamic_threshold).cpu().numpy()
    # new_dynamic_mask = static2dynamic_mask | gaussians.dynamic_part_mask
    dynamic_point_num = new_dynamic_mask.sum()
    # if dynamic_point_num > round(0.75*gaussians.get_xyz.shape[0]) or dynamic_point_num < round(0.25*gaussians.get_xyz.shape[0]) :
    #     new_dynamic_mask = gaussians.dynamic_part_mask
    #     dynamic_point_num = -1
    return new_dynamic_mask, dynamic_point_num

def revolute_judge_static(rotation_angle, gaussians: GaussianModel, static_threshold):
    dynamic2static_mask = (rotation_angle.squeeze(dim=1) < static_threshold).cpu().numpy()
    new_static_mask = dynamic2static_mask | (~gaussians.dynamic_part_mask)
    remove_dynamic_point_num = new_static_mask.sum() - (~gaussians.dynamic_part_mask).sum()
    if remove_dynamic_point_num > round(0.75 * gaussians.dynamic_part_mask.sum()):
        new_static_mask = ~gaussians.dynamic_part_mask
        remove_dynamic_point_num = -1

    return ~new_static_mask, remove_dynamic_point_num

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)

def match_gaussians(xyz0, xyz1, path, cano_gs, num_slots, visualize=False):
    print("Init canonical Gaussians by matching.")
    # load single state gaussians
    xyzs, opacities, features_dcs, features_extras, scales, rots, feats = [], [], [], [], [], [], []
    for state in (0 , 1):
        # if state == 0:
        #     #ply_path = os.path.join(path, "s1", "point_cloud", "iteration_30000", "point_cloud.ply")
        #     ply_path = os.path.join(path, "s1", "point_cloud", "iteration_10000", "point_cloud_0.ply")
        # else:
        #     ply_path = os.path.join(path, "s1", "point_cloud", "iteration_10000", "point_cloud_1.ply")
        # plydata = PlyData.read(ply_path)
        #
        # xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
        #                 np.asarray(plydata.elements[0]["y"]),
        #                 np.asarray(plydata.elements[0]["z"])), axis=1)
        # # point_cloud = o3d.geometry.PointCloud()
        # # point_cloud.points = o3d.utility.Vector3dVector(xyz)
        # # o3d.visualization.draw_geometries([point_cloud])
        #
        # xyzs.append(xyz)
        # opacities.append(np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis])
        #
        # features_dc = np.zeros((xyz.shape[0], 3, 1))
        # features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        # features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        # features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        # features_dcs.append(features_dc)
        #
        # extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        # for idx, attr_name in enumerate(extra_f_names):
        #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # features_extra = features_extra.reshape((features_extra.shape[0], 3, (cano_gs.max_sh_degree + 1) ** 2 - 1))
        # features_extras.append(features_extra)
        #
        # scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        # scale = np.zeros((xyz.shape[0], len(scale_names)))
        # for idx, attr_name in enumerate(scale_names):
        #     scale[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # scales.append(scale)
        #
        # rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        # rot = np.zeros((xyz.shape[0], len(rot_names)))
        # for idx, attr_name in enumerate(rot_names):
        #     rot[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # rots.append(rot)

        # fea_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("fea")]
        # feat = np.zeros((xyz.shape[0], cano_gs.fea_dim))
        # for idx, attr_name in enumerate(fea_names):
        #     feat[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # feats.append(feat)
        if state == 0:
            xyzs.append(xyz0)
        else:
            xyzs.append(xyz1)
    pc0, pc1 = torch.tensor(xyzs[0])[None].cuda(), torch.tensor(xyzs[1])[None].cuda()
    idx = match_pcd(pc0, pc1) # idx: [idx_start, idx_end]

    cd, _ = chamfer_distance(pc0, pc1, batch_reduction=None, point_reduction=None) # cd: [cd_start2end, cd_end2start]
    #cd, _ = chamfer_distance(pc0, pc1, batch_reduction=None)
    larger_motion_state = 0 if cd[0].mean().item() > cd[1].mean().item() else 1
    print("Larger motion state: ", larger_motion_state)

    threshould = [cano_gs.dynamic_threshold_ratio * cd[0].max().item(), cano_gs.dynamic_threshold_ratio * cd[1].max().item()]
    mask_static = [(cd[i].squeeze() < threshould[i]).cpu().numpy() for i in range(2)]
    mask_dynamic = [~mask_static[i] for i in range(2)]

    s = larger_motion_state
    # xyz = np.concatenate([xyzs[s][mask_static[s]], (xyzs[0][idx[0]] + xyzs[1][idx[1]]) * 0.5])
    # opacities = np.concatenate([opacities[s][mask_static[s]], (opacities[0][idx[0]] + opacities[1][idx[1]]) * 0.5])
    # features_dcs = np.concatenate([features_dcs[s][mask_static[s]], (features_dcs[0][idx[0]] + features_dcs[1][idx[1]]) * 0.5])
    # features_extras = np.concatenate([features_extras[s][mask_static[s]], (features_extras[0][idx[0]] + features_extras[1][idx[1]]) * 0.5])
    # scales = np.concatenate([scales[s][mask_static[s]], (scales[0][idx[0]] + scales[1][idx[1]]) * 0.5])
    # rots = np.concatenate([rots[s][mask_static[s]], (rots[0][idx[0]] + rots[1][idx[1]]) * 0.5])
    #feats = np.concatenate([feats[s][mask_static[s]], (feats[0][idx[0]] + feats[1][idx[1]]) * 0.5])

    # cano_gs._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    # cano_gs._features_dc = nn.Parameter(torch.tensor(features_dcs, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    # cano_gs._features_rest = nn.Parameter(torch.tensor(features_extras, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    # cano_gs._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    # cano_gs._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    # cano_gs._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
    # if cano_gs.fea_dim > 0:
    #     cano_gs.feature = nn.Parameter(torch.tensor(feats, dtype=torch.float, device="cuda").requires_grad_(True))

    # cano_gs.max_radii2D = torch.zeros((cano_gs.get_xyz.shape[0]), device="cuda")
    # cano_gs.active_sh_degree = cano_gs.max_sh_degree
    # cano_gs.vanilla_save_ply(os.path.join(path, "point_cloud_cano.ply"))

    if num_slots > 3 or 'real' in path: # larger threshold for complex or real wolrd multi-part objects
        ratio = 0.05
        threshould = [ratio * cd[0].max().item(), ratio * cd[1].max().item()]
        mask_static = [(cd[i].squeeze() < threshould[i]).cpu().numpy() for i in range(2)]
        mask_dynamic = [~mask_static[i] for i in range(2)]
    np.save(os.path.join(path,  'xyz_static.npy'), xyzs[s][mask_static[s]])
    np.save(os.path.join(path, 'xyz_dynamic.npy'), xyzs[s][mask_dynamic[s]])
    np.save(os.path.join(path, 'xyz_static_0.npy'), xyzs[0][mask_static[0]])
    np.save(os.path.join(path, 'xyz_dynamic_0.npy'), xyzs[0][mask_dynamic[0]])
    np.save(os.path.join(path, 'xyz_static_1.npy'), xyzs[1][mask_static[1]])
    np.save(os.path.join(path, 'xyz_dynamic_1.npy'), xyzs[1][mask_dynamic[1]])
    # np.save(path.replace('point_cloud.ply', 'xyz_static.npy'), xyzs[s][mask_static[s]])
    # np.save(path.replace('point_cloud.ply', 'xyz_dynamic.npy'), xyzs[s][mask_dynamic[s]])
    # np.save(path.replace('point_cloud.ply', 'xyz_static_0.npy'), xyzs[0][mask_static[0]])
    # np.save(path.replace('point_cloud.ply', 'xyz_dynamic_0.npy'), xyzs[0][mask_dynamic[0]])
    # np.save(path.replace('point_cloud.ply', 'xyz_static_1.npy'), xyzs[1][mask_static[1]])
    # np.save(path.replace('point_cloud.ply', 'xyz_dynamic_1.npy'), xyzs[1][mask_dynamic[1]])
    if visualize:
        import seaborn as sns
        pallete = np.array(sns.color_palette("hls", 2))
        point_cloud = o3d.geometry.PointCloud()
        x_s = xyzs[s][mask_static[s]]
        x_matched = (xyzs[0][idx[0]] + xyzs[1][idx[1]]) * 0.5
        x = np.concatenate([x_s, x_matched])
        color = np.concatenate([pallete[0:1].repeat(x_s.shape[0], 0), pallete[1:2].repeat(x_matched.shape[0], 0)])
        point_cloud.points = o3d.utility.Vector3dVector(x)
        point_cloud.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([point_cloud])
    return larger_motion_state


def cal_cluster_centers(path, num_slots, visualize=False):
    xyz_static = np.load(os.path.join(path, 'xyz_static.npy'))
    xyz_dynamic = np.load(os.path.join(path, 'xyz_dynamic.npy'))
    print("Finding centers by Spectral Clustering")
    if num_slots > 2:
        cluster = SpectralClustering(num_slots - 1, assign_labels='discretize', random_state=0)
        labels = cluster.fit_predict(xyz_dynamic)
        center_dynamic = np.array([xyz_dynamic[labels == i].mean(0) for i in range(num_slots - 1)])
        labels = np.concatenate([np.zeros(xyz_static.shape[0]), labels + 1])
        center = np.concatenate([xyz_static.mean(0, keepdims=True), center_dynamic])
    else:
        labels = np.concatenate([np.zeros(xyz_static.shape[0]), np.ones(xyz_dynamic.shape[0])])
        center = np.concatenate([xyz_static.mean(0, keepdims=True), xyz_dynamic.mean(0, keepdims=True)])
    x = np.concatenate([xyz_static, xyz_dynamic])
    labels = np.asarray(labels, np.int32)
    dist = (x - center[labels]) # [N, 3]
    mask = np.zeros([dist.shape[0], num_slots])
    mask[np.arange(dist.shape[0]), labels] = 1
    dist_max = (np.linalg.norm(dist, axis=-1)[:, None] * mask).max(0)[:, None] / 2 # [K, 1]
    center_info = np.concatenate([center, dist_max], -1)
    path = os.path.join(path, 'center_info.npy')
    np.save(path, center_info)

    if visualize:
        import seaborn as sns
        pallete = np.array(sns.color_palette("hls", num_slots))
        point_cloud = o3d.geometry.PointCloud()
        c = (center[None] + np.random.randn(1000, 1, 3) * 0.05).reshape(-1, 3)
        x1 = np.concatenate([x, c], 0)

        #color = np.concatenate([pallete[labels], pallete[None].repeat(1000, 0).reshape(-1, 3)], 0)
        color = np.concatenate([np.array([[1,0,0]]).repeat(x.shape[0],axis=0),  np.array([[0,1,0]]).repeat(c.shape[0],axis=0) ], 0)
        point_cloud.points = o3d.utility.Vector3dVector(x1)
        point_cloud.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([point_cloud])