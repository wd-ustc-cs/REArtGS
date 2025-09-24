import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from models.embedder import get_embedder
#from scene.gaussian_model import GaussianModel
from utils.general_utils import build_rotation
import os
#from scene import Scene, GaussianModel
def initialize_weights(model):
    """
    Initialize weights of the model.
    Args:
    - model (torch.nn.Module): Model to initialize weights.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def stable_sigmoid_derivative(x, beta):
    z = beta * x
    pos_mask = z >= 0
    neg_mask = ~pos_mask
    result = torch.zeros_like(z)
    result[pos_mask] = torch.exp(-z[pos_mask]) / (1 + torch.exp(-z[pos_mask]))**2
    result[neg_mask] = torch.exp(z[neg_mask]) / (1 + torch.exp(z[neg_mask]))**2
    return result

def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            y = self.sdf(x)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        return gradients.unsqueeze(1)

    def save_weights(self, model_path, iteration):
        # out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        # os.makedirs(out_weights_path, exist_ok=True)
        # torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))
        torch.save({
            "model_state_dict": self.state_dict(),
            # "config": {  # 保存网络结构关键参数，确保加载时能正确初始化
            #     "d_in": self.d_in,
            #     "d_out": self.d_out,
            #     "d_hidden": self.d_hidden,
            #     "n_layers": self.n_layers,
            #     "skip_in": self.skip_in,
            #     "multires": self.multires,
            #     "bias": self.bias,
            #     "scale": self.scale,
            #     "geometric_init": self.geometric_init,
            #     "weight_norm": self.weight_norm,
            #     "inside_outside": self.inside_outside
            # }
        }, os.path.join(model_path, f'sdf_{iteration}.pth'))



class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]).cuda() * torch.exp(self.variance * 10.0)

def sdf2opacity(cos_anneal_ratio, sample_dist, sdf_network:SDFNetwork, deviation_network: SingleVarianceNetwork, xyz, viewpoint_cam):

    sdf = sdf_network.sdf(xyz)
    sdf_gradients = sdf_network.gradient(xyz).squeeze()
    inv_s = deviation_network(torch.zeros([1, 3]).cuda())[:, :1].clip(1e-6, 1e6).expand(
        sdf.shape)  # Single parameter
    # inv_s = inv_s.expand(batch_size * n_samples, 1)
    # gaussian_to_camera = torch.nn.functional.normalize(fov_cameras.get_camera_center() - self.points, dim=-1)
    dir_pp = (xyz - viewpoint_cam.camera_center.repeat(xyz.shape[0], 1))
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    true_cos = (dir_pp_normalized * sdf_gradients).sum(-1, keepdim=True)
    # cos_anneal_ratio = 0.0
    iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                 F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
    # rots = build_rotation(self.gaussians._rotation)
    # rot_scale = (rots @ self.gaussians.get_scaling[..., None]).squeeze()
    # # gaussian_standard_deviations = (pc.get_scaling * dir_pp_normalized).norm(dim=-1)[..., None]
    # gaussian_standard_deviations = (rot_scale * dir_pp_normalized).norm(dim=-1)[..., None]
    gaussian_standard_deviations = sample_dist
    # new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

    estimated_next_sdf = sdf + iter_cos * gaussian_standard_deviations * 0.5
    estimated_prev_sdf = sdf - iter_cos * gaussian_standard_deviations * 0.5
    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
    #prev_cdf = estimated_prev_sdf * inv_s
    #next_cdf = estimated_next_sdf * inv_s
    #
    p = prev_cdf - next_cdf
    c = prev_cdf
    estimated_opacity = ((p + 1e-5) / (c + 1e-5))
    #estimated_opacity = torch.sigmoid((p + 1e-5) / (c + 1e-5))
    return estimated_opacity, sdf, sdf_gradients

# def dist_sdf2opacity(cos_anneal_ratio, gaussians, viewpoint_cam):
#     xyz = gaussians.get_xyz.detach().clone()
#     sdf = gaussians.sdf_network.sdf(xyz)
#     sdf_gradients = gaussians.sdf_network.gradient(xyz).squeeze()
#     inv_s = gaussians.deviation_network(torch.zeros([1, 3]).cuda())[:, :1].clip(1e-6, 1e6).expand(
#         sdf.shape)  # Single parameter
#     # inv_s = inv_s.expand(batch_size * n_samples, 1)
#     # gaussian_to_camera = torch.nn.functional.normalize(fov_cameras.get_camera_center() - self.points, dim=-1)
#     dir_pp = (xyz - viewpoint_cam.camera_center.repeat(xyz.shape[0], 1))
#     dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
#     true_cos = (dir_pp_normalized * sdf_gradients).sum(-1, keepdim=True)
#     # cos_anneal_ratio = 0.0
#     iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
#                  F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
#     rots = build_rotation(gaussians._rotation.detach())
#     rot_scale = (rots @ gaussians.get_scaling.detach()[..., None]).squeeze()
#     # gaussian_standard_deviations = (pc.get_scaling * dir_pp_normalized).norm(dim=-1)[..., None]
#     gaussian_standard_deviations = (rot_scale * dir_pp_normalized).norm(dim=-1)[..., None]
#     #gaussian_standard_deviations = sample_dist
#     # new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
#
#     estimated_next_sdf = sdf + iter_cos * gaussian_standard_deviations * 0.5
#     estimated_prev_sdf = sdf - iter_cos * gaussian_standard_deviations * 0.5
#     # prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
#     # next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
#     prev_cdf = estimated_prev_sdf * inv_s
#     next_cdf = estimated_next_sdf * inv_s
#     prev_cdf = torch.exp(gaussians.belta * prev_cdf) / torch.pow((1 + torch.exp(gaussians.belta * prev_cdf)), 2)
#     next_cdf = torch.exp(gaussians.belta * next_cdf) / torch.pow((1 + torch.exp(gaussians.belta * next_cdf)), 2)
#     #prev_cdf = estimated_prev_sdf * inv_s
#     #next_cdf = estimated_next_sdf * inv_s
#     #
#     p = prev_cdf - next_cdf
#     c = prev_cdf
#     estimated_opacity = ((p + 1e-5) / (c + 1e-5))
#     #estimated_opacity = torch.where(estimated_opacity>0, estimated_opacity, 0.0)
#     #estimated_opacity = torch.sigmoid(estimated_opacity)
#     #estimated_opacity = torch.sigmoid((p + 1e-5) / (c + 1e-5))
#     return estimated_opacity, sdf, sdf_gradients

# cos_anneal_ratio, gaussians, viewpoint_cam
def dist_sdf2opacity(cos_anneal_ratio, gaussians, viewpoint_cam, sample_dist = None):
    xyz = gaussians.get_xyz.detach().clone()
    sdf = gaussians.sdf_network.sdf(xyz)
    #sdf_gradients = gaussians.sdf_network.gradient(xyz).squeeze()
    sdf_gradients = gaussians.sdf_network.gradient(xyz).squeeze().clamp(-1e3, 1e3)
    inv_s = gaussians.deviation_network(torch.zeros([1, 3]).cuda())[:, :1].clip(1e-6, 1e6).expand(
        sdf.shape)  # Single parameter
    # inv_s = inv_s.expand(batch_size * n_samples, 1)
    # gaussian_to_camera = torch.nn.functional.normalize(fov_cameras.get_camera_center() - self.points, dim=-1)

    dir_pp = (xyz - viewpoint_cam.camera_center.repeat(xyz.shape[0], 1))
    dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-8)
    true_cos = (dir_pp_normalized * sdf_gradients).sum(-1, keepdim=True)
    # cos_anneal_ratio = 0.0
    iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                 F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
    rots = build_rotation(gaussians._rotation.detach())
    rot_scale = (rots @ gaussians.get_scaling.detach()[..., None]).squeeze()
    # gaussian_standard_deviations = (pc.get_scaling * dir_pp_normalized).norm(dim=-1)[..., None]
    # if sample_dist is not None:
    #     gaussian_standard_deviations = sample_dist
    # else:
    #     gaussian_standard_deviations = (rot_scale * dir_pp_normalized).norm(dim=-1)[..., None]
    gaussian_standard_deviations = (rot_scale * dir_pp_normalized).norm(dim=-1)[..., None]


    estimated_next_sdf = sdf + iter_cos * gaussian_standard_deviations * 0.5
    estimated_prev_sdf = sdf - iter_cos * gaussian_standard_deviations * 0.5

    # scaled_prev = estimated_prev_sdf * inv_s
    # scaled_next = estimated_next_sdf * inv_s
    # prev_cdf = stable_sigmoid_derivative(scaled_prev, gaussians.belta)
    # next_cdf = stable_sigmoid_derivative(scaled_next, gaussians.belta)

    prev_cdf = estimated_prev_sdf * inv_s
    next_cdf = estimated_next_sdf * inv_s
    prev_cdf = torch.exp(gaussians.belta * prev_cdf) / torch.pow((1 + torch.exp(gaussians.belta * prev_cdf)), 2)
    next_cdf = torch.exp(gaussians.belta * next_cdf) / torch.pow((1 + torch.exp(gaussians.belta * next_cdf)), 2)


    #
    p = prev_cdf - next_cdf
    c = prev_cdf
    estimated_opacity = ((p + 1e-5) / (c + 1e-5))
    #estimated_opacity = torch.where(estimated_opacity>0, estimated_opacity, 0.0)
    #estimated_opacity = torch.sigmoid(estimated_opacity)
    #estimated_opacity = torch.sigmoid((p + 1e-5) / (c + 1e-5))
    #return estimated_opacity.clamp(1e-5, 1.0), sdf, sdf_gradients
    return estimated_opacity, sdf, sdf_gradients


def load_sdf_network(weight_path: str, device="cuda") -> SDFNetwork:
    """从权重文件加载SDFNetwork模型"""
    checkpoint = torch.load(weight_path, map_location=device)
    #config = checkpoint["config"]

    # 根据保存的配置重新初始化模型
    # model = SDFNetwork(
    #     d_in=config["d_in"],
    #     d_out=config["d_out"],
    #     d_hidden=config["d_hidden"],
    #     n_layers=config["n_layers"],
    #     skip_in=config["skip_in"],
    #     multires=config["multires"],
    #     bias=config["bias"],
    #     scale=config["scale"],
    #     geometric_init=config["geometric_init"],
    #     weight_norm=config["weight_norm"],
    #     inside_outside=config["inside_outside"]
    # ).to(device)

    model= SDFNetwork(d_out=257,
                      d_in=3,
                      d_hidden=256,
                      n_layers=8,
                      skip_in=[4],
                      multires=6,
                      bias=0.5,
                      # scale=3.0,
                      scale=1.0,
                      geometric_init=True,
                      weight_norm=True).to(device)

    # 加载权重
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def compute_sdf_regularization(
        means3D: torch.Tensor,  # [N, 3] 高斯中心位置
        rotations: torch.Tensor,  # [N, 4] 高斯旋转（四元数）
        scales: torch.Tensor,  # [N, 3] 高斯缩放（对数尺度）
        viewpoint_cam,
        sdf_network: torch.nn.Module,  # SDF网络
        lambda_reg: float = 0.1,  # 正则化系数
        eps: float = 1e-8
):
    """
    计算SDF正则项，约束高斯最大opacity点处的SDF值趋近于零
    """
    ray_origins = viewpoint_cam.camera_center.unsqueeze(dim=0)
    ray_dirs = (means3D - viewpoint_cam.camera_center.repeat(means3D.shape[0], 1))
    ray_dirs= ray_dirs / ray_dirs.norm(dim=1, keepdim=True)
    #N, M = means3D.shape[0], ray_origins.shape[0]

    # 转换四元数到旋转矩阵 [N, 3, 3]
    R = build_rotation(rotations)  # 需实现四元数转矩阵函数

    # 计算缩放矩阵及其逆 [N, 3, 3]
    S = torch.exp(scales)  # 确保缩放为正数
    S_inv = 1.0 / (S + eps)
    S_inv = torch.diag_embed(S_inv)

    # 扩展维度用于广播计算 [N, M, 3, 3]
    # R_exp = R.unsqueeze(1)  # [N, 1, 3, 3]
    # S_inv_exp = S_inv.unsqueeze(1)  # [N, 1, 3, 3]
    # means_exp = means3D.unsqueeze(1)  # [N, 1, 3]

    # 将射线转换到高斯局部坐标系
    # ray_origins_exp = ray_origins.unsqueeze(0)  # [1, M, 3]
    # ray_dirs_exp = ray_dirs.unsqueeze(0)  # [1, M, 3]

    # o_g = S^{-1} * R^T * (o - p_k) [N, M, 3]
    # o_g = torch.einsum('nmpq,nmq->nmp',
    #                    S_inv_exp @ R_exp.transpose(-1, -2),
    #                    ray_origins_exp - means_exp)
    o_g = (S_inv @ R.transpose(-1, -2)@(ray_origins - means3D).unsqueeze(dim=-1)).squeeze(dim=-1)
    # r_g = S^{-1} * R^T * r [N, M, 3]
    # r_g = torch.einsum('nmpq,mq->nmp',
    #                    S_inv_exp @ R_exp.transpose(-1, -2),
    #                    ray_dirs)
    r_g = (S_inv @ R.transpose(-1, -2) @ ray_dirs.unsqueeze(dim=-1)).squeeze(dim=-1)
    # 计算t* = -B/(2A) (公式中的A和B)
    A = torch.sum(r_g ** 2, dim=-1)  # [N, M]
    B = torch.sum(o_g * r_g, dim=-1)  # [N, M]
    #t_star = -B / (2 * A + eps)  # [N, M]
    t_star = -B / (A + eps)  # [N, M]
    # 计算世界坐标系下的交点 x* = o + t* * r
    x_star = ray_origins + t_star.unsqueeze(-1) * ray_dirs  # [N, M, 3]

    # 计算SDF值 [N*M, 1]
    sdf_values = sdf_network.sdf(x_star.clip(-1,1))  # [N, M]

    # 构造正则项：仅对有效交点（A>0且t>0）计算损失
    valid_mask = (A > eps) & (t_star > 0)  # [N, M]
    reg_loss = lambda_reg * torch.mean((sdf_values * valid_mask.float()) ** 2)

    return reg_loss