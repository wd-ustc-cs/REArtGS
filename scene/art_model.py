# Code is build on Artgs (https://github.com/YuLiu-LY/ArtGS). Thanks for their great work!

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scene.gaussian_model import GaussianModel
from utils.dual_quaternion import *
import tinycudann as tcnn
from utils.general_utils import get_expon_lr_func



def searchForMaxIteration(folder):
    if not os.path.exists(folder):
        return None
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if "_" in fname and 'best' not in fname]
    return max(saved_iters) if saved_iters != [] else None

class ArtModel:
    def __init__(self, args, joint_type = 'r'):
        self.deform = ArtGS(args, joint_type).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    @property
    def reg_loss(self):
        return self.deform.reg_loss

    def step(self, gaussians, is_training=True):
        return self.deform(gaussians, is_training=is_training)

    def train_setting(self, training_args):
        l = [
            {'params': group['params'],
             'lr': training_args.position_lr_init * self.spatial_lr_scale * training_args.deform_lr_scale,
             "name": group['name']}
            for group in self.deform.trainable_parameters()
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale * training_args.deform_lr_scale,
            lr_final=training_args.position_lr_final * training_args.deform_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration, is_best=False):
        if is_best:
            out_weights_path = os.path.join(model_path, "deform/iteration_best")
            os.makedirs(out_weights_path, exist_ok=True)
            torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))
            with open(os.path.join(out_weights_path, "iter.txt"), 'w') as f:
                f.write(f"iteration: {iteration}")
        else:
            out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
            os.makedirs(out_weights_path, exist_ok=True)
            torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        if os.path.exists(weights_path):
            self.deform.load_state_dict(torch.load(weights_path))
            return True
        else:
            return False

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform" or param_group["name"] == "mlp":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == 'slot':
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr

    def update(self, iteration):
        self.deform.update(iteration)


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0

class ProgressiveBandHashGrid(nn.Module):
    def __init__(self, in_channels, start_level=6, n_levels=12, start_step=1000, update_steps=1000, dtype=torch.float32):
        super().__init__()

        encoding_config = {
            "otype": "Grid",
            "type": "Hash",
            "n_levels": n_levels,  # 16 for complex motions
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 2.0,
            "interpolation": "Linear",
            "start_level": start_level,
            "start_step": start_step,
            "update_steps": update_steps,
        }

        self.n_input_dims = in_channels
        with torch.cuda.device(get_rank()):
            self.encoding = tcnn.Encoding(in_channels, encoding_config, dtype=dtype)
        self.n_output_dims = self.encoding.n_output_dims
        self.n_level = encoding_config["n_levels"]
        self.n_features_per_level = encoding_config["n_features_per_level"]
        self.start_level, self.start_step, self.update_steps = (
            encoding_config["start_level"],
            encoding_config["start_step"],
            encoding_config["update_steps"],
        )
        self.current_level = self.start_level
        self.mask = torch.zeros(
            self.n_level * self.n_features_per_level,
            dtype=torch.float32,
            device=get_rank(),
        )
        self.mask[: self.current_level * self.n_features_per_level] = 1.0

    def forward(self, x):
        enc = self.encoding(x)
        enc = enc * self.mask + enc.detach() * (1 - self.mask)
        return enc

    def update_step(self, global_step):
        current_level = min(
            self.start_level
            + max(global_step - self.start_step, 0) // self.update_steps,
            self.n_level,
        )
        if current_level > self.current_level:
            print(f"Update current level of HashGrid to {current_level}")
            self.current_level = current_level
            self.mask[: self.current_level * self.n_features_per_level] = 1.0




def gumbel_softmax(logits, tau=1., hard=True, dim=-1, is_training=False):
    # modified from torch.nn.functional.gumbel_softmax
    if is_training:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        logits = (logits + gumbels)
    logits = logits / tau
    y_soft = logits.softmax(dim)
    index = y_soft.max(dim, keepdim=True)[1]
    if hard:
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        # Straight through estimator.
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y, index.squeeze(dim)


class CenterBasedSeg(nn.Module):
    def __init__(self, num_slots, slot_size, scale_factor=1.0, shift_weight=0.5):
        super().__init__()
        self.num_slots = num_slots

        self.grid = ProgressiveBandHashGrid(3, start_level=6, n_levels=12, start_step=0, update_steps=500)
        dim = num_slots * 4 + self.grid.n_output_dims + 3
        self.mlp = nn.Sequential(
            nn.Linear(dim, slot_size),
            nn.ReLU(),
            nn.Linear(slot_size, num_slots * 2),
        )
        self.center = nn.Parameter(torch.randn(num_slots, 3) * 0.01)
        self.logscale = nn.Parameter(torch.randn(num_slots, 3) * 0.01)
        self.rot = nn.Parameter(torch.Tensor([[1, 0, 0, 0]]).repeat(self.num_slots, 1))

        self.scale_factor = scale_factor
        self.shift_weight = shift_weight

    def train_setting(self, training_args):
        self.spatial_lr_scale = 5
        l = [
            {'params': group['params'],
             'lr': training_args.position_lr_init * self.spatial_lr_scale * training_args.deform_lr_scale,
             "name": group['name']}
             for group in self.trainable_parameters()
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.seg_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale * training_args.deform_lr_scale,
                                                       lr_final=training_args.position_lr_final * training_args.deform_lr_scale,
                                                       lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.deform_lr_max_steps)

    def trainable_parameters(self):
        params = [
            #{'params': self.joints, 'name': 'mlp'},
            {'params': list(self.parameters()), 'name': 'seg'},
            ]
        return params

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            # if param_group["name"] == "deform" or param_group["name"] == "mlp":
            #     lr = self.deform_scheduler_args(iteration)
            #     param_group['lr'] = lr
            if param_group["name"] == 'seg':
                lr = self.seg_scheduler_args(iteration)
                param_group['lr'] = lr

    def update(self, iteration, *args, **kwargs):
        self.tau = self.cosine_anneal(iteration, self.tau_decay_steps, 0, 1.0, 0.1)
        self.grid.update_step(global_step=iteration)


    def cosine_anneal(self, step, final_step, start_step=0, start_value=1.0, final_value=0.1):
        if start_value <= final_value or start_step >= final_step:
            return final_value

        if step < start_step:
            value = start_value
        elif step >= final_step:
            value = final_value
        else:
            a = 0.5 * (start_value - final_value)
            b = 0.5 * (start_value + final_value)
            progress = (step - start_step) / (final_step - start_step)
            value = a * math.cos(math.pi * progress) + b
        return value

    def forward(self, x, tau, is_training=False):
        '''
            x: position of canonical gaussians [N, 3]
        '''
        rel_pos = self.cal_relative_pos(x)  # [N, K, 3]
        dist = (rel_pos ** 2).sum(-1)  # [N, K]

        x_rel = torch.cat([rel_pos, torch.norm(rel_pos, p=2, dim=-1, keepdim=True)], dim=-1)  # [N, K, 4]
        info = torch.cat([x_rel.reshape(x.shape[0], -1), self.grid(x), x], -1)
        delta = self.mlp(info)  # [N, K * 2]
        logscale, shift = torch.split(delta, delta.shape[-1] // 2, dim=-1)  # [N, K]

        dist = dist * (self.shift_weight * logscale).exp()
        logits = -dist + shift * self.shift_weight

        slots = None
        hard = (tau - 0.1) < 1e-3
        mask, _ = gumbel_softmax(logits, tau=tau / (self.num_slots - 1), hard=hard, dim=1, is_training=is_training)
        return slots, mask

    def init_from_file(self, path):
        center_info = torch.from_numpy(np.load(path)).float().to(self.rot.device)  # [K, 4], center and radius
        self.center = nn.Parameter(center_info[:, :3])
        self.logscale = nn.Parameter(torch.log(center_info[:, 3:4].repeat(1, 3)))
        return center_info[:, :3], center_info[:, 3:4]

    def cal_relative_pos(self, x):
        center = self.center[None]
        rot = self.get_rot[None]
        scale = self.get_scale[None] * self.scale_factor
        return quaternion_apply(rot, (x[:, None] - center)) / scale  # [N, K, 3]

    @property
    def get_scale(self):
        return torch.exp(self.logscale)

    @property
    def get_rot(self):
        return F.normalize(self.rot, p=2, dim=-1)


class ArtGS(nn.Module):
    def __init__(self, args, joint_type = 'r'):
        super().__init__()
        self.slot_size = args.slot_size
        #self.joint_types = args.joint_types.split(',')
        self.use_art_type_prior = True
        self.num_slots = args.num_slots
        self.joint_types = ['s'] + [joint_type for _ in range(self.num_slots - 1)]
        #self.joint_types = ['s'] + ['r' for _ in range(self.num_slots - 1)]
        # if self.use_art_type_prior:
        #     self.num_slots = len(self.joint_types)
        # else:
        #     self.num_slots = args.num_slots
        #     self.joint_types = ['s'] + ['r' for _ in range(self.num_slots - 1)]

        joints = torch.zeros(self.num_slots - 1, 7) + torch.randn(self.num_slots - 1, 7) * 1e-5
        joints[:, 0] = 1
        self.joints = nn.Parameter(joints)
        self.register_buffer('Ts', torch.eye(4).float())
        self.register_buffer('qr_s', torch.Tensor([1, 0, 0, 0]))
        self.register_buffer('qd_s', torch.Tensor([0, 0, 0, 0]))

        self.seg_model = CenterBasedSeg(self.num_slots, self.slot_size, scale_factor=args.scale_factor,
                                        shift_weight=args.shift_weight)
        self.revolute_constraint = args.revolute_constraint
        self.reg_loss = 0.
        self.tau = 1.0
        self.tau_decay_steps = args.tau_decay_steps

    @torch.no_grad()
    def cal_art_type(self):
        qr, qd = self.get_slot_deform()
        axis_dir, theta = quaternion_to_axis_angle(qr[1:])
        theta = theta.rad2deg()
        self.joint_types = ['s']
        self.joint_types += ['r' if t.item() > 10 else 'p' for t in theta]
        self.use_art_type_prior = True
        print(self.joint_types)
        return ','.join(self.joint_types)

    def slotdq_to_gsdq(self, slot_qr, slot_qd, mask):
        # slot_qr: [K, 4], slot_qd: [K, 4], mask: [N, K]
        qr = torch.einsum('nk, kl->nl', mask, slot_qr)  # [N, 4]
        qd = torch.einsum('nk, kl->nl', mask, slot_qd)  # [N, 4]
        return normalize_dualquaternion(qr, qd)

    def get_slot_deform(self):
        qrs = []
        qds = []
        for i, joint_type in enumerate(self.joint_types):
            if i == 0:
                assert joint_type == 's'
                qr, qd = self.qr_s, self.qd_s
            else:
                joint = self.joints[i - 1]
                qr = F.normalize(joint[:4], p=2, dim=-1)
                t0 = torch.cat([torch.zeros(1).to(qr.device), joint[4:7]])
                if self.use_art_type_prior:
                    if joint_type == 'p':
                        qr = self.qr_s
                        qd = 0.5 * quaternion_mul(t0, qr)
                    elif joint_type == 'r':
                        if self.revolute_constraint:
                            qd = 0.5 * (quaternion_mul(t0, qr) - quaternion_mul(qr,
                                                                                t0))  # better for multi-part real-world objects, but sensitive to initialization
                        else:
                            qd = 0.5 * quaternion_mul(t0, qr)
                else:
                    qd = 0.5 * quaternion_mul(t0, qr)
            qrs.append(qr)
            qds.append(qd)
        qrs, qds = torch.stack(qrs), torch.stack(qds)
        return qrs, qds

    def deform_pts(self, xc, mask, slot_qr, slot_qd, state):
        if state < 0.5:
            slot_qr, slot_qd = dual_quaternion_inverse((slot_qr, slot_qd))
        gs_qr, gs_qd = self.slotdq_to_gsdq(slot_qr, slot_qd, mask)
        xt = dual_quaternion_apply((gs_qr, gs_qd), xc)
        return xt, gs_qr

    def trainable_parameters(self):
        params = [
            {'params': self.joints, 'name': 'mlp'},
            {'params': list(self.seg_model.parameters()), 'name': 'slot'},
        ]
        return params

    def get_mask(self, x, is_training=False):
        # tau = self.tau if is_training else 0.1
        tau = 1
        slots, mask = self.seg_model(x, tau, is_training)
        self.slots = slots
        return mask

    @torch.no_grad()
    def get_joint_param(self, joint_type_list):
        qrs, qds = self.get_slot_deform()
        qrs, qds = qrs[1:], qds[1:]
        joint_info_list = []
        for i, joint_type in enumerate(joint_type_list):
            qr, qd = qrs[i], qds[i]
            qr, t = dual_quaternion_to_quaternion_translation((qr, qd))
            R = quaternion_to_matrix(qr).cpu().numpy()
            t = t.cpu().numpy()

            if joint_type == 'r':
                axis_dir, theta = quaternion_to_axis_angle(qr)
                axis_dir, theta = axis_dir.cpu().numpy(), theta.cpu().numpy()
                theta = 2 * theta
                axis_position = np.matmul(np.linalg.inv(np.eye(3) - R), t.reshape(3, 1)).reshape(-1)
                axis_position += axis_dir * np.dot(axis_dir, -axis_position)
                R = R @ R
                t = R @ t + t
                joint_info = {'type': joint_type,
                              'axis_position': axis_position,
                              'axis_direction': axis_dir,
                              'theta': np.rad2deg(theta),
                              'rotation': R, 'translation': t}
            elif joint_type == 'p':
                t = t * 2
                theta = np.linalg.norm(t)
                axis_dir = t / theta
                joint_info = {'type': joint_type,
                              'axis_position': np.zeros(3),
                              'axis_direction': axis_dir,
                              'theta': theta,
                              'rotation': R, 'translation': t}
            joint_info_list.append(joint_info)
        return joint_info_list

    def one_transform(self, gaussians: GaussianModel, state, is_training):
        xc = gaussians.get_xyz.detach()
        N = xc.shape[0]
        mask = self.get_mask(xc, is_training)  # [N, K]
        qr, qd = self.get_slot_deform()
        xt, rot = self.deform_pts(xc, mask, qr, qd, state)

        # regularization loss for center
        opacity = gaussians.get_opacity.detach()
        m = mask * opacity
        m = m / (m.sum(0, keepdim=True) + 1e-5)
        c = torch.einsum('nk,nj->kj', m, xc)
        self.reg_loss = F.mse_loss(self.seg_model.center, c) * 0.1

        d_xyz = xt - xc
        d_rotation = rot.detach()

        return {
            'd_xyz': d_xyz,
            'd_rotation': d_rotation,
            'xt': xt,
            'mask': mask.argmax(-1),
        }

    def forward(self, gaussians: GaussianModel, is_training=False):
        xc = gaussians._xyz.detach()
        N = xc.shape[0]
        d_values_list = []
        mask = self.get_mask(xc, is_training)  # [N, K]
        # mask = (mask * torch.tensor([1,15]).cuda())
        mask = (mask * torch.tensor([1, 1.5]).cuda())  # 10
        qr, qd = self.get_slot_deform()

        for state in [0, 1]:
            xt, rot = self.deform_pts(xc, mask, qr, qd, state)
            d_xyz = xt - xc
            d_rotation = rot.detach()
            d_values = {
                'd_xyz': d_xyz,
                'd_rotation': d_rotation,
                'xt': xt,
                'mask': mask.argmax(-1),
            }
            d_values_list.append(d_values)

        return d_values_list

    def interpolate_single_state(self, gaussians: GaussianModel, t):
        xc = gaussians._xyz.detach()
        mask = self.get_mask(xc, False)  # [N, K]
        qr1, qd1 = self.get_slot_deform()
        qr0, qd0 = dual_quaternion_inverse((qr1, qd1))

        slot_qr = (1 - t) * qr0 + t * qr1
        slot_qd = (1 - t) * qd0 + t * qd1
        gs_qr, gs_qd = self.slotdq_to_gsdq(slot_qr, slot_qd, mask)
        xt = dual_quaternion_apply((gs_qr, gs_qd), xc)
        dx = xt - xc
        dr = gs_qr
        return dx, dr

    def interpolate(self, gaussians: GaussianModel, time_list):
        xc = gaussians._xyz.detach()
        mask = self.get_mask(xc, False)  # [N, K]
        qr1, qd1 = self.get_slot_deform()
        qr0, qd0 = dual_quaternion_inverse((qr1, qd1))

        dx_list = []
        dr_list = []
        for t in time_list:
            slot_qr = (1 - t) * qr0 + t * qr1
            slot_qd = (1 - t) * qd0 + t * qd1
            gs_qr, gs_qd = self.slotdq_to_gsdq(slot_qr, slot_qd, mask)
            xt = dual_quaternion_apply((gs_qr, gs_qd), xc)
            dx_list.append(xt - xc)
            dr_list.append(gs_qr)
        return dx_list, dr_list


    def update(self, iteration, *args, **kwargs):
        self.tau = self.cosine_anneal(iteration, self.tau_decay_steps, 0, 1.0, 0.1)
        self.seg_model.grid.update_step(global_step=iteration)

    def cosine_anneal(self, step, final_step, start_step=0, start_value=1.0, final_value=0.1):
        if start_value <= final_value or start_step >= final_step:
            return final_value

        if step < start_step:
            value = start_value
        elif step >= final_step:
            value = final_value
        else:
            a = 0.5 * (start_value - final_value)
            b = 0.5 * (start_value + final_value)
            progress = (step - start_step) / (final_step - start_step)
            value = a * math.cos(math.pi * progress) + b
        return value

