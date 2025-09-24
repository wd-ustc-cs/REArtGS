
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_ply, load_obj
from pytorch3d.structures import Meshes
import torch
import open3d as o3d
import numpy as np
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import json
from pytorch_lightning import seed_everything
from itertools import permutations
from piq import LPIPS
from piq import ssim as ssim_func
import math
# Code is build on Artgs (https://github.com/YuLiu-LY/ArtGS). Thanks for their great work!

lpips = LPIPS()

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    return 20 * torch.log10(1.0 / torch.sqrt(mse(img1, img2)))


def load_mesh(path):
    if path.endswith('.ply'):
        verts, faces = load_ply(path)
    elif path.endswith('.obj'):
        obj = load_obj(path)
        verts = obj[0]
        faces = obj[1].verts_idx
    return verts, faces


def combine_pred_mesh(paths, exp_path):
    recon_mesh = o3d.geometry.TriangleMesh()
    for path in paths:
        mesh = o3d.io.read_triangle_mesh(path)
        recon_mesh += mesh
    o3d.io.write_triangle_mesh(exp_path, recon_mesh)


def compute_chamfer(recon_pts, gt_pts):
	with torch.no_grad():
		recon_pts = recon_pts.cuda()
		gt_pts = gt_pts.cuda()
		dist,_ = chamfer_distance(recon_pts, gt_pts, batch_reduction=None, single_directional=False)
		dist = dist.item()
	return dist


def compute_recon_error(recon_path, gt_path, n_samples=10000, vis=False):
    verts, faces = load_mesh(recon_path)
    recon_mesh = Meshes(verts=[verts], faces=[faces])
    verts, faces = load_mesh(gt_path)
    gt_mesh = Meshes(verts=[verts], faces=[faces])

    gt_pts = sample_points_from_meshes(gt_mesh, num_samples=n_samples)
    recon_pts = sample_points_from_meshes(recon_mesh, num_samples=n_samples)


    if vis:
        pts = gt_pts.clone().detach().squeeze().numpy()
        gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        o3d.io.write_point_cloud("gt_points.ply", gt_pcd)
        pts = recon_pts.clone().detach().squeeze().numpy()
        recon_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        o3d.io.write_point_cloud("recon_points.ply", recon_pcd)

    return compute_chamfer(recon_pts, gt_pts)


def eval_CD(pred_s_ply, pred_d_ply_list, pred_w_ply, gt_s_ply, gt_d_ply_list, gt_w_ply):
    # combine the part meshes as a whole
    # combine_pred_mesh([pred_s_ply, pred_d_ply], pred_w_ply)
    # compute synmetric distance
    chamfer_dist_d_list = [compute_recon_error(pred_d_ply, gt_d_ply, n_samples=10000, vis=False) * 1000
                           for pred_d_ply, gt_d_ply in zip(pred_d_ply_list, gt_d_ply_list)]
    chamfer_dist_s = compute_recon_error(pred_s_ply, gt_s_ply, n_samples=10000, vis=False) * 1000
    chamfer_dist_w = compute_recon_error(pred_w_ply, gt_w_ply, n_samples=10000, vis=False) * 1000

    return chamfer_dist_s, chamfer_dist_d_list, chamfer_dist_w


def eval_CD_2states(gt_path, pred_path, id_d_list, n_trials=1):
    num_d_joints = len(id_d_list)
    dyn_names = [''] if num_d_joints == 1 else [f'_{i}' for i in range(num_d_joints)]
    states = ['start', 'end']
    cd_s_2s, cd_w_2s = {'start':0., 'end':0.}, {'start':0., 'end':0.}
    cd_d_2s = {'start':[0.] * num_d_joints, 'end':[0.] * num_d_joints}
    for state in states:
        for seed in range(n_trials):
            seed_everything(seed)
            pred_w_ply = f'{pred_path}/{state}_-1.ply'
            pred_s_ply = f'{pred_path}/{state}_0.ply'
            pred_d_ply_list = [f'{pred_path}/{state}_{id}.ply' for id in id_d_list]
            gt_w_ply = f'{gt_path}/{state}/{state}_rotate.ply'
            gt_s_ply = f'{gt_path}/{state}/{state}_static_rotate.ply'
            gt_d_ply_list = [f'{gt_path}/{state}/{state}_dynamic{name}_rotate.ply' for name in dyn_names]

            cd_s_2s[state] += compute_recon_error(pred_s_ply, gt_s_ply, n_samples=10000, vis=False) * 1000
            cd_w_2s[state] += compute_recon_error(pred_w_ply, gt_w_ply, n_samples=10000, vis=False) * 1000
            for i in range(num_d_joints):
                cd_d_2s[state][i] += compute_recon_error(pred_d_ply_list[i], gt_d_ply_list[i], n_samples=10000, vis=False) * 1000
        cd_s_2s[state] /= n_trials
        print(f'CD_static_{state} {cd_s_2s[state]:.4f}', end=', ')
        for i in range(num_d_joints):
            cd_d_2s[state][i] /= n_trials
            print(f'CD_dynamic_{state}_{i} {cd_d_2s[state][i]:.4f}', end=', ')
        cd_w_2s[state] /= n_trials
        print(f'CD_whole_{state} {cd_w_2s[state]:.4f}')
    return cd_s_2s, cd_d_2s, cd_w_2s


def interpret_transforms(base_R, base_t, R, t, joint_type='revolute'):
    """
    base_R, base_t, R, t are all from canonical to world
    rewrite the transformation = global transformation (base_R, base_t) {R' part + t'} --> s.t. R' and t' happens in canonical space
    R', t':
    - revolute: R'p + t' = R'(p - a) + a, R' --> axis-theta representation; axis goes through a = (I - R')^{-1}t'
    - prismatic: R' = I, t' = l * axis_direction
    """
    R = np.matmul(base_R.T, R)
    t = np.matmul(base_R.T, (t - base_t).reshape(3, 1)).reshape(-1)

    if joint_type == 'revolute':
        rotvec = Rotation.from_matrix(R).as_rotvec()
        theta = np.linalg.norm(rotvec, axis=-1)
        axis_direction = rotvec / max(theta, (theta < 1e-8))
        try:
            axis_position = np.matmul(np.linalg.inv(np.eye(3) - R), t.reshape(3, 1)).reshape(-1)
        except:   # TO DO find the best solution
            axis_position = np.zeros(3)
        axis_position += axis_direction * np.dot(axis_direction, -axis_position)
        joint_info = {'axis_position': axis_position,
                      'axis_direction': axis_direction,
                      'theta': np.rad2deg(theta),
                      'rotation': R, 'translation': t}

    elif joint_type == 'prismatic':
        theta = np.linalg.norm(t)
        axis_direction = t / max(theta, (theta < 1e-8))
        joint_info = {'axis_direction': axis_direction, 'axis_position': np.zeros(3), 'theta': theta,
                      'rotation': R, 'translation': t}

    return joint_info, R, t


def line_distance(a_o, a_d, b_o, b_d):
    normal = np.cross(a_d, b_d)
    normal_length = np.linalg.norm(normal)
    if normal_length < 1e-6:   # parallel
        return np.linalg.norm(np.cross(b_o - a_o, a_d))
    else:
        return np.abs(np.dot(normal, a_o - b_o)) / normal_length


def eval_axis_and_state(axis_a, axis_b, joint_type='r', reverse=False):
    a_d, b_d = axis_a['axis_direction'], axis_b['axis_direction']

    angle = np.rad2deg(np.arccos(np.dot(a_d, b_d) / np.linalg.norm(a_d) / np.linalg.norm(b_d)))
    angle = min(angle, 180 - angle)

    if joint_type == 'r':
        a_o, b_o = axis_a['axis_position'], axis_b['axis_position']
        distance = line_distance(a_o, a_d, b_o, b_d)

        a_r, b_r = axis_a['rotation'], axis_b['rotation']
        if reverse:
            a_r = a_r.T

        r_diff = np.matmul(a_r, b_r.T)
        state = np.rad2deg(np.arccos(np.clip((np.trace(r_diff) - 1.0) * 0.5, a_min=-1, a_max=1)))
    elif joint_type == 'p':
        distance = 0
        a_t, b_t = axis_a['translation'], axis_b['translation']
        if reverse:
            a_t = -a_t

        state = np.linalg.norm(a_t - b_t)
    else:
        raise ValueError(f'Unknown joint type {joint_type}')

    return angle, distance * 10, state

def eval_axis_and_state_all(pred_joint_list, pred_joint_types, gt_info_list, print_perm=False):
    num_d_joints = len(pred_joint_list)
    gt_joint_types = ['r' if gt_info['type'] == 'revolute' else 'p' for gt_info in gt_info_list]
    if num_d_joints <= 1:
        pred_joint = pred_joint_list[0]
        gt_joint = {key: value for key, value in gt_info_list[0].items()}
        angle, distance, theta_diff = eval_axis_and_state(pred_joint, gt_joint, gt_joint_types[0])
        return [(angle, distance, theta_diff)], [0]
    else:
        perms = permutations(range(num_d_joints))
        ps = []
        results = [[] for _ in range (math.factorial(num_d_joints))]
        results_score = [0. for _ in range (math.factorial(num_d_joints))]
        gt_types = ''.join(gt_joint_types)
        for p, perm in enumerate(perms):
            ps.append(perm)
            pred_types = ''.join([pred_joint_types[idx] for idx in perm])
            if pred_types != gt_types:
                results_score[p] = 1e6
                continue
            pred_ls = [pred_joint_list[idx] for idx in perm]
            for i in range(num_d_joints):
                pred_joint = pred_ls[i]
                gt_joint = {key: value for key, value in gt_info_list[i].items()}
                angle, distance, theta_diff = eval_axis_and_state(pred_joint, gt_joint, gt_joint_types[i])
                results[p].append((angle, distance, theta_diff))
                results_score[p] += angle + distance + theta_diff
        real_idx = np.argmin(results_score)
        real_perm = ps[real_idx]
        real_result = results[real_idx]
        if print_perm:
            print(f'Permutation: {real_perm}')
        return real_result, real_perm


def geodesic_distance(pred_R, gt_R):
    '''
    q is the output from the network (rotation from t=0.5 to t=1)
    gt_R is the GT rotation from t=0 to t=1
    '''
    pred_R, gt_R = pred_R.cpu(), gt_R.cpu()
    R_diff = torch.matmul(pred_R, gt_R.T)
    cos_angle = torch.clip((torch.trace(R_diff) - 1.0) * 0.5, min=-1., max=1.)
    angle = torch.rad2deg(torch.arccos(cos_angle))
    return angle


def axis_metrics(motion, gt):
    # pred axis
    pred_axis_d = motion['axis_d'].cpu().squeeze(0)
    pred_axis_o = motion['axis_o'].cpu().squeeze(0)
    # gt axis
    gt_axis_d = gt['axis_d']
    gt_axis_o = gt['axis_o']
    # angular difference between two vectors
    cos_theta = torch.dot(pred_axis_d, gt_axis_d) / (torch.norm(pred_axis_d) * torch.norm(gt_axis_d))
    ang_err = torch.rad2deg(torch.acos(torch.abs(cos_theta)))
    # positonal difference between two axis lines
    w = gt_axis_o - pred_axis_o
    cross = torch.cross(pred_axis_d, gt_axis_d)
    if (cross == torch.zeros(3)).sum().item() == 3:
        pos_err = torch.tensor(0)
    else:
        pos_err = torch.abs(torch.sum(w * cross)) / torch.norm(cross)
    return ang_err, pos_err


def translational_error(motion, gt):
    dist_half = motion['dist'].cpu()
    dist = dist_half * 2.
    gt_dist = gt['dist']

    axis_d = F.normalize(motion['axis_d'].cpu().squeeze(0), p=2, dim=0)
    gt_axis_d = F.normalize(gt['axis_d'].cpu(), p=2, dim=0)

    err = torch.sqrt(((dist * axis_d - gt_dist * gt_axis_d) ** 2).sum())
    return err


def read_gt(gt_path):
    with open(gt_path, 'r') as f:
        info = json.load(f)

    all_trans_info = info['trans_info']
    if isinstance(all_trans_info, dict):
        all_trans_info = [all_trans_info]
    ret_list = []
    for trans_info in all_trans_info:
        axis = trans_info['axis']
        axis_o, axis_d = np.array(axis['o']), np.array(axis['d'])
        axis_type = trans_info['type']
        l, r = trans_info[axis_type]['l'], trans_info[axis_type]['r']

        if axis_type == 'rotate':
            rotvec = axis_d * np.deg2rad(r - l)
            rot = Rotation.from_rotvec(rotvec).as_matrix()
            trans = np.matmul(np.eye(3) - rot, axis_o.reshape(3, 1)).reshape(-1)
            joint_type = 'revolute'
        else:
            rot = np.eye(3)
            trans = (r - l) * axis_d
            joint_type = 'prismatic'
        ret_list.append({'axis_position': axis_o, 'axis_direction': axis_d, 'theta': r - l, 'joint_type': axis_type, 'rotation': rot, 'translation': trans,
                         'type': joint_type})
    return ret_list