#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import trimesh
from utils.vis_utils import save_points
from scene.appearance_network import AppearanceNetwork
import math
from scene.field import SDFNetwork, SingleVarianceNetwork, initialize_weights

def init_quaternions(half_angle, init_dir):
    a = torch.tensor([init_dir[0], init_dir[1], init_dir[2]], dtype=torch.float32)
    a = torch.nn.functional.normalize(a, p=2., dim=0)
    sin_ = math.sin(half_angle)
    cos_ = math.cos(half_angle)
    r = cos_
    i = a[0] * sin_
    j = a[1] * sin_
    k = a[2] * sin_
    q = torch.tensor([r, i, j, k], dtype=torch.float32)
    return q


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance) #  压缩对称矩阵
            return symm


        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, prismatic= False, revolute = False, use_sdf=False):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        #self.spatial_lr_scale = 5
        self.setup_functions()
        # appearance network and appearance embedding
        #self.appearance_network = AppearanceNetwork(3+64, 3).cuda()
        self.prismatic = prismatic
        self.revolute = revolute
        self.use_sdf = use_sdf
        #if prismatic:
        self.dynamic_threshold_ratio = 0.02
        std = 1e-4
        # self._appearance_embeddings = nn.Parameter(torch.empty(2048, 64).cuda())
        # self._appearance_embeddings.data.normal_(0, std)
        self.dynamic_part_mask = None
        self.canonical = 0.5
        self.no_filter = False
        self.xyzs = None


    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def part_seg(self, part_mask, use_mask=True):
        with torch.no_grad():
            self._xyz = self._xyz[part_mask]
            self._features_dc = self._features_dc[part_mask]
            self._features_rest = self._features_rest[part_mask]
            self._scaling = self._scaling[part_mask]
            self._rotation = self._rotation[part_mask]
            self._opacity = self._opacity[part_mask]
            if use_mask:
                self.dynamic_part_mask = np.ones(self._xyz.shape[0], dtype=np.bool_)
            else:
                self.dynamic_part_mask = np.zeros(self._xyz.shape[0], dtype=np.bool_)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling
        
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        if self.use_sdf:
            #return torch.exp(self.belta*self._opacity)/torch.pow((1+torch.exp(self.belta*self._opacity)), 2)
            return self._opacity
        else:
            return self.opacity_activation(self._opacity)
    # @property
    # def get_opacity(self):
    #     return self.opacity_activation(self._opacity)


    @property
    def get_opacity_with_3D_filter(self):
        if self.use_sdf:
            opacity = self.get_opacity
        else:
            opacity = self.opacity_activation(self._opacity)
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]

    def get_apperance_embedding(self, idx):
        return self._appearance_embeddings[idx]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_covariance33(self, scaling_modifier = 1):
        L = build_scaling_rotation(scaling_modifier * self.get_scaling, self._rotation)
        actual_covariance = L @ L.transpose(1, 2)
        #symm = strip_symmetric(actual_covariance) #  压缩对称矩阵
        return actual_covariance


    def get_view2gaussian(self, viewmatrix):
        r = self._rotation
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

        q = r / norm[:, None]
        
        R = torch.zeros((q.size(0), 3, 3), device='cuda')

        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - r*z)
        R[:, 0, 2] = 2 * (x*z + r*y)
        R[:, 1, 0] = 2 * (x*y + r*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - r*x)
        R[:, 2, 0] = 2 * (x*z - r*y)
        R[:, 2, 1] = 2 * (y*z + r*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    
        rots = R
        xyz = self.get_xyz
        N = xyz.shape[0]
        G2W = torch.zeros((N, 4, 4), device='cuda')
        G2W[:, :3, :3] = rots # TODO check if we need to transpose here
        G2W[:, :3, 3] = xyz
        G2W[:, 3, 3] = 1.0
        
        viewmatrix = viewmatrix.transpose(0, 1)
        G2V = viewmatrix @ G2W
        
        R = G2V[:, :3, :3]
        t = G2V[:, :3, 3]
        
        t2 = torch.bmm(-R.transpose(1, 2), t[..., None])[..., 0]
        V2G = torch.zeros((N, 4, 4), device='cuda')
        V2G[:, :3, :3] = R.transpose(1, 2)
        V2G[:, :3, 3] = t2
        V2G[:, 3, 3] = 1.0
        
        # transpose view2gaussian to match glm in CUDA code
        V2G = V2G.transpose(2, 1).contiguous()
        
        # precompute results to reduce computation and IO
        scales = self.get_scaling_with_3D_filter
        S_inv_square = 1.0 / (scales ** 2)
        R = V2G[:, :3, :3].transpose(1, 2)
        t2 = V2G[:, 3:, :3]
        
        C = torch.sum((t2 ** 2) * S_inv_square[:, None, :], dim=2)
        S_inv_square_R = S_inv_square[:, :, None] * R
        B = t2 @ S_inv_square_R
        Sigma = R.transpose(1, 2) @ S_inv_square_R
        merged = torch.cat([Sigma[:, :, 0], Sigma[:, 1:, 1], Sigma[:, 2:, 2], B.squeeze(), C], dim=1)
        
        return merged

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        print("Computing 3D filter")
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2 # TODO remove hard coded value
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x
        
        distance[~valid_points] = distance[valid_points].max()
        
        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = filter_3D[..., None]
        
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.spatial_lr_scale = 5
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def sdf_training_setup(self, training_args):
        self._opacity = None
        self.belta = nn.Parameter(torch.tensor([0.1], dtype=torch.float32).cuda(), requires_grad=True) #0.1
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.prismatic_optimizer = None
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            #{'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self.belta], 'lr': 0.001, "name": "belta"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            #{'params': [self._appearance_embeddings], 'lr': training_args.appearance_embeddings_lr,
            # "name": "appearance_embeddings"},
            # {'params': self.appearance_network.parameters(), 'lr': training_args.appearance_network_lr,
            #  "name": "appearance_network"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.sdf_network = SDFNetwork(d_out=257,
                                                d_in=3,
                                                d_hidden=256,
                                                n_layers=8,
                                                skip_in=[4],
                                                multires=6,
                                                bias=0.5,
                                                #scale=3.0,
                                                scale=1.0,
                                                geometric_init=True,
                                                weight_norm=True).cuda()
        self.deviation_network = SingleVarianceNetwork(init_val=0.3).cuda()
        #self.deviation_network = SingleVarianceNetwork(init_val=0.003).cuda()
        #initialize_weights(self.sdf_network)
        #initialize_weights(self.deviation_network)
        # Define optimizer (shared for encoder and decoder)
        self.sdf_optimizer = torch.optim.Adam(list(self.sdf_network.parameters()) + list(self.deviation_network.parameters()), lr=2e-4) #2e-4
        self.sdf_scheduler_args = get_expon_lr_func(lr_init=2e-4,
                                                    lr_final=2e-5,
                                                    max_steps=training_args.iterations)



    def deform_training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.spatial_lr_scale = 5

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]


        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.position_lr_max_steps)

    def prismatic_update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.prismatic_optimizer.param_groups:
            #if param_group["name"] == "dir":
            lr = self.prismatic_scheduler_args(iteration)
            param_group['lr'] = lr
            return lr

    def revolute_update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.revolute_optimizer.param_groups:
            if param_group["name"] == "axis_o":
                lr = self.revolute_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def sdf_update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.sdf_optimizer.param_groups:
            lr = self.sdf_scheduler_args(iteration)
            param_group['lr'] = lr
            return lr

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            #if param_group["name"] == "xyz":
            if param_group["name"] in ["xyz", "belta"]:
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, exclude_filter=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if not exclude_filter:
            l.append('filter_3D')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        filter_3D = self.filter_3D.detach().cpu().numpy()



        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]


        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, filter_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def vanilla_save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(exclude_filter=True)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_fused_ply(self, path, bbox=None):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        filter_3D = self.filter_3D.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]



        if bbox is not None:
            elements = np.empty(xyz[bbox].shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz[bbox], normals[bbox], f_dc[bbox], f_rest[bbox], opacities[bbox], scale[bbox], rotation[bbox], filter_3D[bbox]), axis=1)
        else:
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, filter_3D), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        print(f"save ply in {path}")


    @torch.no_grad()
    def get_tetra_points(self):
        M = trimesh.creation.box()
        M.vertices *= 2
        
        rots = build_rotation(self._rotation)
        xyz = self.get_xyz

        #scale = self.get_scaling_with_3D_filter * 3. # TODO test
        scale = self.get_scaling_with_3D_filter * 3

        # filter points with small opacity for bicycle scene
        # opacity = self.get_opacity_with_3D_filter
        # #mask = (opacity > 0.1).squeeze(-1)
        # mask = (opacity > 0.4).squeeze(-1)
        # xyz = xyz[mask]
        # scale = scale[mask]
        # rots = rots[mask]
        use_bbox = False
        if use_bbox:
            mask = xyz.abs() < 2.0
            mask = torch.any(mask, dim=-1)
            xyz = xyz[mask]
            scale = scale[mask]
            rots = rots[mask]

        vertices = M.vertices.T    
        vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
        # scale vertices first
        vertices = vertices * scale.unsqueeze(-1)
        vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
        # concat center points
        vertices = torch.cat([vertices, xyz], dim=0)
        
        # scale is not a good solution but use it for now
        scale = scale.max(dim=-1, keepdim=True)[0]
        scale_corner = scale.repeat(1, 8).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)
        
        return vertices, vertices_scale

    @torch.no_grad()
    def get_dynamic_tetra_points(self, xyz, rotation):
        M = trimesh.creation.box()
        M.vertices *= 2
        if rotation is not None:
            rots = build_rotation(rotation)
        else:
            rots = build_rotation(self._rotation)
        #xyz = self.get_xyz
        # scale = self.get_scaling_with_3D_filter * 3. # TODO test
        scale = self.get_scaling_with_3D_filter * 3

        # filter points with small opacity for bicycle scene
        # opacity = self.get_opacity_with_3D_filter
        # #mask = (opacity > 0.1).squeeze(-1)
        # mask = (opacity > 0.4).squeeze(-1)
        # xyz = xyz[mask]
        # scale = scale[mask]
        # rots = rots[mask]
        use_bbox = True
        if use_bbox:
            mask = xyz.abs() < 2.0
            mask = torch.any(mask, dim=-1)
            xyz = xyz[mask]
            scale = scale[mask]
            rots = rots[mask]

        vertices = M.vertices.T
        vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
        # scale vertices first
        vertices = vertices * scale.unsqueeze(-1)
        vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
        # concat center points
        vertices = torch.cat([vertices, xyz], dim=0)

        # scale is not a good solution but use it for now
        scale = scale.max(dim=-1, keepdim=True)[0]
        scale_corner = scale.repeat(1, 8).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)

        return vertices, vertices_scale

    def reset_opacity(self):
        # reset opacity to by considering 3D filter
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter)*0.01)
        
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        opacities_new = opacities_new / coef[..., None]
        opacities_new = self.inverse_opacity_activation(opacities_new)
        # if self.prismatic:
        #     optimizable_tensors = self.replace_tensor_to_prismatic_optimizer(opacities_new, "opacity")
        # else:
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def vanilla_reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]



    def load_ply(self, path, crop_gs =False, bounding_box:list = None):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        if not self.no_filter:
            filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis] #PGSR


        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if crop_gs:
            valid_mask_x = np.abs(xyz[..., 0]) < bounding_box[0]
            valid_mask_y = np.abs(xyz[..., 1]) < bounding_box[1]
            valid_mask_z = np.abs(xyz[..., 2]) < bounding_box[2]
            valid_mask = (valid_mask_x & valid_mask_y & valid_mask_z)

        else:
            valid_mask = torch.ones(xyz.shape[0], dtype=torch.bool)

        self._xyz = nn.Parameter(torch.tensor(xyz[valid_mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc[valid_mask], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra[valid_mask], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities[valid_mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales[valid_mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots[valid_mask], dtype=torch.float, device="cuda").requires_grad_(True))
        if not self.no_filter:
            self.filter_3D = torch.tensor(filter_3D[valid_mask], dtype=torch.float, device="cuda")



        self.active_sh_degree = self.max_sh_degree
        print(f"remain {self.get_xyz.shape[0]} valid points")
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz[valid_mask])
        # axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])  # red x; green y; blue z
        # #vis.add_geometry(axis_pcd)
        # o3d.visualization.draw_geometries([pcd, axis_pcd])


        # if self.prismatic or self.revolute:
        #     self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def replace_tensor_to_optimizer(self, tensor, name, optimizer = None):
        optimizable_tensors = {}
        if not optimizer:
            optimizer = self.optimizer

        #for group in self.optimizer.param_groups:
        for group in optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors





    def _prune_optimizer(self, mask, optimizer= None):
        optimizable_tensors = {}
        if not optimizer:
            optimizer = self.optimizer
        #for group in self.optimizer.param_groups:
        for group in optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network", "pivot", "belta"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _part_prune_optimizer(self, mask, dynamic_part_mask, optimizer= None):
        optimizable_tensors = {}
        if not optimizer:
            optimizer = self.optimizer

        base = np.ones_like(dynamic_part_mask, dtype=np.bool_)
        dynamic_indice = np.where(dynamic_part_mask == 1)[0]
        dynamic_indice_prune = dynamic_indice[~mask.cpu().numpy()] # dynamic_indice.shape[0]>mask.shape[0]
        base[dynamic_indice_prune] = False

        #for group in self.optimizer.param_groups:
        for group in optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network", "pivot"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = stored_state["exp_avg"][base]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][base]

                # stored_state["exp_avg"][dynamic_part_mask] = stored_state["exp_avg"][dynamic_part_mask][mask]
                # stored_state["exp_avg_sq"][dynamic_part_mask] = stored_state["exp_avg_sq"][dynamic_part_mask][mask]

                del self.optimizer.state[group['params'][0]]

                #dynamic_part_mask = dynamic_part_mask[base]

                group["params"][0] = nn.Parameter((group["params"][0][base].requires_grad_(True)))



                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:

                #dynamic_part_mask = dynamic_part_mask[base]

                group["params"][0] = nn.Parameter(group["params"][0][base].requires_grad_(True))


                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        if not self.use_sdf:
            self._opacity = optimizable_tensors["opacity"]
        else:
            self._opacity = self._opacity[valid_points_mask]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def  part_prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._part_prune_optimizer(valid_points_mask, self.dynamic_part_mask)

        base = np.ones_like(self.dynamic_part_mask, dtype=np.bool_)
        dynamic_indice = np.where(self.dynamic_part_mask == 1)[0]
        dynamic_indice_prune = dynamic_indice[~valid_points_mask.cpu().numpy()]
        base[dynamic_indice_prune] = False

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[base]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[base]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[base]
        self.denom = self.denom[base]
        self.max_radii2D = self.max_radii2D[base]
        # if self.prismatic:
        #     prismatic_optimizable_tensors = self._part_prune_optimizer(valid_points_mask, self.dynamic_part_mask, optimizer=self.prismatic_optimizer)
        #     self.d_xyz = prismatic_optimizable_tensors["d_xyz"]
        #     self.d_scaling = prismatic_optimizable_tensors["d_scaling"]
        # if self.revolute:
        #     revolute_optimizable_tensors = self._part_prune_optimizer(valid_points_mask, self.dynamic_part_mask, optimizer=self.revolute_optimizer)
        #     self.d_angle = revolute_optimizable_tensors["d_angle"]

        self.dynamic_part_mask = self.dynamic_part_mask[base]
        #self.static_part_mask = ~self.dynamic_part_mask

    def extract_part_points(self, mask):
        valid_points_mask = mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.prismatic:
            self.d_xyz = optimizable_tensors["d_xyz"]
            self.d_scaling = optimizable_tensors["d_scaling"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        #self.max_radii2D = self.max_radii2D[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict, optimizer = None):
        optimizable_tensors = {}
        if not optimizer:
            optimizer = self.optimizer
        #for group in self.optimizer.param_groups:
        for group in optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            assert len(group["params"]) == 1
            if group["name"] in tensors_dict:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:

                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))

                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors



    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        if not self.use_sdf:
            self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # if self.prismatic or self.revolute:
        #     #new_mask = np.ones(new_d_angle.shape[0], dtype=np.bool_)
        #     new_mask = np.ones(new_xyz.shape[0], dtype=np.bool_)
        #     self.dynamic_part_mask = np.concatenate((self.dynamic_part_mask, new_mask), axis=0)


        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold,  grads_abs, grad_abs_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        # if self.use_sdf:
        #     selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        #     selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if self.use_sdf:
            stds = self.get_scaling[selected_pts_mask].repeat(N,1)
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        else:
            new_xyz = self._xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        if not self.use_sdf:
            new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        else:
            #new_opacity = None
            new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
            self._opacity = torch.cat((self._opacity, new_opacity), dim=0)


        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        #self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))

        self.prune_points(prune_filter)

    def part_densify_and_split(self, grads, grad_threshold,  grads_abs, grad_abs_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz[self.dynamic_part_mask].shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling[self.dynamic_part_mask], dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[self.dynamic_part_mask][selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[self.dynamic_part_mask][selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[self.dynamic_part_mask][selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[self.dynamic_part_mask][selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[self.dynamic_part_mask][selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[self.dynamic_part_mask][selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[self.dynamic_part_mask][selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[self.dynamic_part_mask][selected_pts_mask].repeat(N,1)
        new_d_xyz = None
        new_d_scaling = None
        new_d_angle = None
        # if self.prismatic:
        #     new_d_xyz = self.d_xyz[self.dynamic_part_mask][selected_pts_mask].repeat(N, 1)
        #     new_d_scaling = self.d_scaling[self.dynamic_part_mask][selected_pts_mask].repeat(N, 1)
        # if self.revolute:
        #     new_d_angle = self.d_angle[self.dynamic_part_mask][selected_pts_mask].repeat(N)
        if self.get_xyz.shape[0] < 10_0000:
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

            #self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

            prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
            if self.get_xyz.shape[0] > 5000:
                self.part_prune_points(prune_filter) #bug



    def densify_and_clone(self, grads, grad_threshold,  grads_abs, grad_abs_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # if self.use_sdf:
        #     selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
        #     selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        

        # sample a new gaussian instead of fixing position
        if self.use_sdf:
            stds = self.get_scaling[selected_pts_mask]
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
        else:
            new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        if self.use_sdf:
            self._opacity = torch.cat((self._opacity, new_opacities), dim=0)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)


    def part_densify_and_clone(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling[self.dynamic_part_mask],
                                                        dim=1).values <= self.percent_dense * scene_extent)


        # sample a new gaussian instead of fixing position
        stds = self.get_scaling[self.dynamic_part_mask][selected_pts_mask]
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[self.dynamic_part_mask][selected_pts_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[self.dynamic_part_mask][selected_pts_mask]

        new_features_dc = self._features_dc[self.dynamic_part_mask][selected_pts_mask]
        new_features_rest = self._features_rest[self.dynamic_part_mask][selected_pts_mask]
        new_opacities = self._opacity[self.dynamic_part_mask][selected_pts_mask]
        new_scaling = self._scaling[self.dynamic_part_mask][selected_pts_mask]
        new_rotation = self._rotation[self.dynamic_part_mask][selected_pts_mask]

        new_d_xyz = None
        new_d_scaling = None
        new_d_angle = None
        # if self.prismatic:
        #     new_d_xyz = self.d_xyz[self.dynamic_part_mask][selected_pts_mask]
        #     new_d_scaling = self.d_scaling[self.dynamic_part_mask][selected_pts_mask]
        # if self.revolute:
        #     new_d_angle = self.d_angle[self.dynamic_part_mask][selected_pts_mask]
        if self.get_xyz.shape[0] < 10_0000:
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                       new_rotation)


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
        ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)
        
        before = self._xyz.shape[0]
        if self._xyz.shape[0] < 100000:
            self.densify_and_clone(grads, max_grad, grads_abs, Q, extent)
        clone = self._xyz.shape[0]
        if self._xyz.shape[0] < 100000:
            self.densify_and_split(grads, max_grad, grads_abs, Q, extent)

        # self.densify_and_clone(grads, max_grad, grads_abs, Q, extent)
        # clone = self._xyz.shape[0]
        #
        # self.densify_and_split(grads, max_grad, grads_abs, Q, extent)
        split = self._xyz.shape[0]
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        #if self.use_sdf:
            #prune_mask = (self.get_opacity < min_opacity).squeeze()
            # prune_mask = (torch.exp(-sdf.abs() / ((5e+3) * self.get_opacity)) < min_opacity).squeeze()
            # #prune_mask = (torch.exp(-sdf / ((5e+5) * self.get_opacity)) < min_opacity).squeeze()
            # new_mask = torch.zeros((self.get_xyz.shape[0]-self.get_opacity.shape[0]), dtype=torch.bool).cuda()
            # prune_mask = torch.cat((prune_mask,new_mask), dim=0)
        #else:
            #prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        #if self._xyz[~prune_mask].shape[0] > 3000: #7000
        # if self.use_sdf:
        #     if self._xyz[~prune_mask].shape[0] > 5000:  # 7000
        #         self.prune_points(prune_mask)
        # else:
        self.prune_points(prune_mask)
        prune = self._xyz.shape[0]
        # torch.cuda.empty_cache()
        return clone - before, split - clone, split - prune

    def part_densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = (self.xyz_gradient_accum / self.denom)[self.dynamic_part_mask]
        grads[grads.isnan()] = 0.0

        grads_abs = (self.xyz_gradient_accum_abs / self.denom)[self.dynamic_part_mask]
        grads_abs[grads_abs.isnan()] = 0.0
        ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)

        before = self._xyz.shape[0]
        self.part_densify_and_clone(grads, max_grad, grads_abs, Q, extent)
        clone = self._xyz.shape[0]

        self.part_densify_and_split(grads, max_grad, grads_abs, Q, extent)
        split = self._xyz.shape[0]

        prune_mask = (self.get_opacity[self.dynamic_part_mask] < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D[self.dynamic_part_mask] > max_screen_size
            big_points_ws = self.get_scaling[self.dynamic_part_mask].max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        if self._xyz[self.dynamic_part_mask][~prune_mask].shape[0] > 5000:
            self.part_prune_points(prune_mask)
        prune = self._xyz.shape[0]
        # torch.cuda.empty_cache()
        return clone - before, split - clone, split - prune


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        #TODO maybe use max instead of average
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1


    def load_ply_cano(self, path, state=0.5):
        # state = 0
        if abs(state - 0.5) < 1e-6:
            self.load_ply(path)
            print("Loaded canonical state = 0.5")
        elif abs(state) < 1e-6:
            cano_path = path.replace('point_cloud.ply', 'point_cloud_0.ply')
            self.load_ply(cano_path)
            print("Loaded canonical state = 0")
        elif abs(state - 1) < 1e-6:
            cano_path = path.replace('point_cloud.ply', 'point_cloud_1.ply')
            self.load_ply(cano_path)
            print("Loaded canonical state = 1")
        else:
            raise ValueError("Invalid canonical state")
        xyz_static_0 = torch.tensor(np.load(path.replace('point_cloud_cano.ply', 'xyz_static_0.npy'))).float().to(self._xyz.device)
        xyz_static_1 = torch.tensor(np.load(path.replace('point_cloud_cano.ply', 'xyz_static_1.npy'))).float().to(self._xyz.device)
        xyz_dynamic_0 = torch.tensor(np.load(path.replace('point_cloud_cano.ply', 'xyz_dynamic_0.npy'))).float().to(self._xyz.device)
        xyz_dynamic_1 = torch.tensor(np.load(path.replace('point_cloud_cano.ply', 'xyz_dynamic_1.npy'))).float().to(self._xyz.device)
        self.xyzs = [torch.cat([xyz_static_0, xyz_dynamic_0]), torch.cat([xyz_static_1, xyz_dynamic_1])]
