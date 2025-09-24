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

import os
import time
import torch
from random import randint
from utils.loss_utils import l1_loss
from gaussian_renderer import d_render
import sys
from scene import Scene, GaussianModel
from pytorch3d.loss import chamfer_distance
from utils.general_utils import safe_state

import tqdm

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, SegParams
#from train import training_report
import math
from utils.gui_utils import OrbitCamera
import numpy as np
import dearpygui.dearpygui as dpg
#from scene.app_model import AppModel
from utils.loss_utils import l1_loss, ssim

import uuid
from utils.image_utils import psnr

from gaussian_renderer.mini_renderer import ag_render
from utils.depth_utils import depth_to_normal

from scene.art_model import ArtModel
from utils.rigid_utils import match_gaussians, cal_cluster_centers

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, fid):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.fid = fid

        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()



class GUI:
    def __init__(self, args, dataset, opt, pipe,  testing_iterations, saving_iterations) -> None:
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations

        self.tb_writer = prepare_output_and_logger(dataset)

        self.gaussians = GaussianModel(dataset.sh_degree)

        self.gaussians.no_filter = True

        self.scene = Scene(dataset, self.gaussians, load_iteration= -1, crop_gs=True)
        #self.scene = Scene(dataset, self.gaussians, load_cano=True)
        self.use_mask = dataset.use_mask

        self.gaussians.deform_training_setup(opt)
        self.one_stage_xyz = self.gaussians.get_xyz.detach().cpu().clone()
        self.gaussians.use_canonical = dataset.use_canonical
        self.art_model = ArtModel(dataset, joint_type = args.joint_type)

        self.art_model.train_setting(self.opt)

        self.trainCameras = self.scene.getTrainCameras().copy()
        self.testCameras = self.scene.getTestCameras().copy()
        self.viewpoint_stacks = None
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = 1

        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        self.best_psnr = 0.0
        self.best_iteration = 0
        self.progress_bar = tqdm.tqdm(range(opt.iterations), desc="Training progress")


        # For UI
        self.visualization_mode = 'RGB'

        self.gui = args.gui  # enable gui
        self.W = args.W
        self.H = args.H
        self.cam = OrbitCamera(args.W, args.H, r=args.radius, fovy=args.fovy)

        self.mode = "render"
        self.seed = "random"
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.training = False
        if self.gaussians.no_filter:
            self.gaussians.compute_3D_filter(cameras=self.trainCameras)
        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def init_deform(self, center_init='cgs'):
        if center_init == 'cgs':
            center, scale = self.art_model.deform.seg_model.init_from_file(
                f'{self.dataset.model_path}/center_info.npy')

            print('Init center from coarse gaussian.')
        elif center_init == 'pcd':
            center, scale = self.art_model.deform.seg_model.init_from_file(f'{self.args.source_path}/center_info.npy')
            print('Init center from pcd.')
        else:
            print('Init center randomly.')

    @torch.no_grad()
    def training_report(self, tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                        renderArgs, load2gpu_on_the_fly):
        if tb_writer:
            tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
            tb_writer.add_scalar('iter_time', elapsed, iteration)

        test_psnr = 0.0
        # Report test and samples of training set
        if iteration in testing_iterations:
            torch.cuda.empty_cache()
            validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                                  {'name': 'train',
                                   'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                               range(5, 30, 5)]})

            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    images = torch.tensor([], device="cuda")
                    gts = torch.tensor([], device="cuda")
                    for idx, viewpoint in enumerate(config['cameras']):
                        if load2gpu_on_the_fly:
                            viewpoint.load2device()
                        fid = torch.tensor(viewpoint.fid).cuda()
                        d_xyz, d_rot = self.art_model.deform.interpolate_single_state(self.gaussians, fid)
                        # xyz = scene.gaussians.get_xyz
                        # time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                        # d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)

                        image = torch.clamp(
                            renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz=d_xyz, d_rot=d_rot)["render"],
                            0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        images = torch.cat((images, image.unsqueeze(0)), dim=0)
                        gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                        if load2gpu_on_the_fly:
                            viewpoint.load2device('cpu')
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                                 image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(
                                    config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                    gt_image[None], global_step=iteration)

                    l1_test = l1_loss(images, gts)
                    psnr_test = psnr(images, gts).mean()
                    if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                        test_psnr = psnr_test
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test,
                                                                            psnr_test))
                    if tb_writer:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

            if tb_writer:
                tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
                tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            torch.cuda.empty_cache()

        return test_psnr


    def register_dpg(self):
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window
        # the rendered image, as the primary window
        with dpg.window(
                tag="_primary_window",
                width=self.W,
                height=self.H,
                pos=[0, 0],
                no_move=True,
                no_title_bar=True,
                no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
                label="Control",
                tag="_control_window",
                width=600,
                height=self.H,
                pos=[self.W, 0],
                no_move=True,
                no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                        directory_selector=False,
                        show=False,
                        callback=callback_select_input,
                        file_count=1,
                        tag="file_dialog_tag",
                        width=700,
                        height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Visualization: ")

                    def callback_vismode(sender, app_data, user_data):
                        self.visualization_mode = user_data
                        if user_data == 'Node':
                            self.node_vis_fea = True if not hasattr(self, 'node_vis_fea') else not self.node_vis_fea
                            print("Visualize node features" if self.node_vis_fea else "Visualize node importance")
                            if self.node_vis_fea or True:
                                from motion import visualize_featuremap
                                if True:  # self.renderer.gaussians.motion_model.soft_edge:
                                    if hasattr(self.renderer.gaussians.motion_model, 'nodes_fea'):
                                        node_rgb = visualize_featuremap(
                                            self.renderer.gaussians.motion_model.nodes_fea.detach().cpu().numpy())
                                        self.node_rgb = torch.from_numpy(node_rgb).cuda()
                                    else:
                                        self.node_rgb = None
                                else:
                                    self.node_rgb = None
                            else:
                                node_imp = self.renderer.gaussians.motion_model.cal_node_importance(
                                    x=self.renderer.gaussians.get_xyz)
                                node_imp = (node_imp - node_imp.min()) / (node_imp.max() - node_imp.min())
                                node_rgb = torch.zeros([node_imp.shape[0], 3], dtype=torch.float32).cuda()
                                node_rgb[..., 0] = node_imp
                                node_rgb[..., -1] = 1 - node_imp
                                self.node_rgb = node_rgb

                    dpg.add_button(
                        label="RGB",
                        tag="_button_vis_rgb",
                        callback=callback_vismode,
                        user_data='RGB',
                    )
                    dpg.bind_item_theme("_button_vis_rgb", theme_button)

                    dpg.add_button(
                        label="UV_COOR",
                        tag="_button_vis_uv",
                        callback=callback_vismode,
                        user_data='UV_COOR',
                    )
                    dpg.bind_item_theme("_button_vis_uv", theme_button)
                    dpg.add_button(
                        label="MotionMask",
                        tag="_button_vis_motion_mask",
                        callback=callback_vismode,
                        user_data='MotionMask',
                    )
                    dpg.bind_item_theme("_button_vis_motion_mask", theme_button)

                    dpg.add_button(
                        label="Node",
                        tag="_button_vis_node",
                        callback=callback_vismode,
                        user_data='Node',
                    )
                    dpg.bind_item_theme("_button_vis_node", theme_button)

                    def callback_use_const_var(sender, app_data):
                        self.use_const_var = not self.use_const_var

                    dpg.add_button(
                        label="Const Var",
                        tag="_button_const_var",
                        callback=callback_use_const_var
                    )
                    dpg.bind_item_theme("_button_const_var", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("Scale Const: ")

                    def callback_vis_scale_const(sender):
                        self.vis_scale_const = 10 ** dpg.get_value(sender)
                        self.need_update = True

                    dpg.add_slider_float(
                        label="Log vis_scale_const (For debugging)",
                        default_value=-3,
                        max_value=-.5,
                        min_value=-5,
                        callback=callback_vis_scale_const,
                    )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Temporal Speed: ")
                    self.video_speed = 1.

                    def callback_speed_control(sender):
                        self.video_speed = dpg.get_value(sender)
                        self.need_update = True

                    dpg.add_slider_float(
                        label="Play speed",
                        default_value=1.,
                        max_value=2.,
                        min_value=0.0,
                        callback=callback_speed_control,
                    )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        self.save_model(mode=user_data)

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_button(
                        label="geo",
                        tag="_button_save_mesh",
                        callback=callback_save,
                        user_data='geo',
                    )
                    dpg.bind_item_theme("_button_save_mesh", theme_button)

                    dpg.add_button(
                        label="geo+tex",
                        tag="_button_save_mesh_with_tex",
                        callback=callback_save,
                        user_data='geo+tex',
                    )
                    dpg.bind_item_theme("_button_save_mesh_with_tex", theme_button)

                    dpg.add_button(
                        label="pcl",
                        tag="_button_save_pcl",
                        callback=callback_save,
                        user_data='pcl',
                    )
                    dpg.bind_item_theme("_button_save_pcl", theme_button)

                    def call_back_save_train(sender, app_data, user_data):
                        self.render_all_train_data()

                    dpg.add_button(
                        label="save_train",
                        tag="_button_save_train",
                        callback=call_back_save_train,
                    )

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            # self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_psnr")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("d_part", "render", "depth"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="REArtGS",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        self.gaussians.compute_3D_filter(cameras=self.trainCameras)
        assert self.gui

        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:

                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
        print("1")



    def train_step(self):

        self.iter_start.record()


        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()


        if not self.viewpoint_stacks:
            self.viewpoint_stacks = self.scene.getTrainCameras().copy()
        viewpoint_cam = self.viewpoint_stacks.pop(randint(0, len(self.viewpoint_stacks) - 1))

        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()

        fid = viewpoint_cam.fid

        d_values = self.art_model.deform.one_transform(self.gaussians, fid, is_training=True)
        d_xyz, d_rot = d_values['d_xyz'], d_values['d_rotation']
        render_pkg_re = ag_render(viewpoint_cam, self.gaussians, self.pipe, self.background,
                                  d_xyz=d_xyz, d_rot=d_rot)
        rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        image = rendering


        # rgb Loss
        gt_image = viewpoint_cam.original_image.cuda()


        Ll1 = l1_loss(image, gt_image)
        # use L1 loss for the transformed image if using decoupled appearance
        # if self.dataset.use_decoupled_appearance:
        #    Ll1 = L1_loss_appearance(image, gt_image, self.gaussians, viewpoint_cam.idx)

        rgb_loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss = rgb_loss
        if self.iteration > 3000:
            loss = loss + self.art_model.reg_loss

        loss = torch.nan_to_num(loss)
        loss.backward()

        self.iter_end.record()



        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                self.progress_bar.update(10)
            if self.iteration == self.opt.iterations:
                self.progress_bar.close()

            # Keep track of max radii in image-space for pruning
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter],
                                                                      radii[visibility_filter])

            # Keep track of max radii in image-space for pruning
            if self.gaussians.max_radii2D.shape[0] == 0:
                self.gaussians.max_radii2D = torch.zeros_like(radii)
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter],
                                                                      radii[visibility_filter])
            # Densification
            if self.iteration < self.opt.densify_until_iter:
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    #threshold = 0.01 if self.iteration > 3000 else 0.05
                    threshold = 0.05
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, threshold,
                                                     self.scene.cameras_extent, size_threshold)
                    print(f"Gaussian number:{self.gaussians.get_xyz.shape[0]}")
                if self.iteration % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                    self.gaussians.vanilla_reset_opacity()


            if self.iteration in self.testing_iterations:
                cur_psnr = self.training_report(self.tb_writer, self.iteration, Ll1, loss, l1_loss,
                                           self.iter_start.elapsed_time(self.iter_end), self.testing_iterations,
                                           self.scene, ag_render, (self.pipe, self.background),
                                           self.dataset.load2gpu_on_the_fly)
                if cur_psnr.item() > self.best_psnr:
                    self.best_psnr = cur_psnr.item()
                    self.best_iteration = self.iteration


            if self.iteration in self.saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))


                self.gaussians.compute_3D_filter(cameras=self.trainCameras)
                self.scene.save(self.iteration)
                self.art_model.save_weights(self.args.model_path, self.iteration)

            if self.iteration == self.best_iteration:
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))

                self.gaussians.compute_3D_filter(cameras=self.trainCameras)

                self.scene.save(self.iteration)
                self.art_model.save_weights(self.args.model_path, self.iteration, is_best=True)


            #
            if self.iteration == (self.opt.iterations-200):
                # don't update in the end of training
                self.gaussians.compute_3D_filter(cameras=self.trainCameras)
                #self.gaussians.compute_3D_filter(cameras=self.viewpoint_stack)

            # Optimizer step


            if self.iteration < self.opt.iterations:

                self.gaussians.optimizer.step()
                self.gaussians.update_learning_rate(self.iteration)
                self.gaussians.optimizer.zero_grad(set_to_none=True)

                self.art_model.optimizer.step()
                self.art_model.optimizer.zero_grad()
                self.art_model.update_learning_rate(self.iteration)

                self.art_model.update(max(0, self.iteration))


            if self.iteration % 500 == 0:
                torch.cuda.empty_cache()




        if self.gui:
            dpg.set_value(
                "_log_train_psnr",
                "Best PSNR = {} in Iteration {}".format(self.best_psnr, self.best_iteration)
            )
        else:
            print("Best PSNR = {} in Iteration {}".format(self.best_psnr, self.best_iteration))
        self.iteration += 1

        if self.gui:
            dpg.set_value(
                "_log_train_log",
                f"step = {self.iteration: 5d} loss = {loss.item():.4f}",
            )
        torch.cuda.empty_cache()

    @torch.no_grad()
    def test_step(self):

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        if not hasattr(self, 't0'):
            self.t0 = time.time()
            self.fps_of_fid = 10

        cur_cam = MiniCam(
            self.cam.pose,
            self.W,
            self.H,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
            fid=torch.remainder(torch.tensor((time.time() - self.t0) * self.fps_of_fid).float().cuda() / len(
                self.scene.getTrainCameras()), 1.)
        )
        fid = cur_cam.fid
        # if self.iteration <= self.opt.warm_up:
        #     d_xyz, d_rot = None, None
        # else:
        d_xyz, d_rot = self.art_model.deform.interpolate_single_state(self.gaussians, fid)
        #d_values = self.art_model.deform.one_transform(self.gaussians, fid, is_training=True)
        #d_xyz, d_rot = d_values['d_xyz'], d_values['d_rotation']
        # render_pkg_re = new_render(viewpoint_cam, self.gaussians, self.pipe, self.background, fid=fid)
        # out = d_render(cur_cam, self.gaussians, self.pipe, self.background,
        #                          #self.dataset.kernel_size, d_xyz=d_xyz, d_rot=d_rot)
        #                self.dataset.kernel_size)
        # if self.iteration < self.opt.warm_up:
        #     out = new_render(viewpoint_camera=cur_cam, pc=self.gaussians, pipe=self.pipe, bg_color=self.background)
        # else:
        out = ag_render(viewpoint_camera=cur_cam, pc=self.gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rot=d_rot)


        if self.mode == 'd_part':
            dynamic_part_mask = (self.art_model.step(self.gaussians, is_training=False)[0]['mask'] == 1)
            buffer_image = ag_render(viewpoint_camera=cur_cam, pc=self.gaussians, pipe=self.pipe, bg_color=self.background, d_xyz=d_xyz, d_rot=d_rot, vis_mask=dynamic_part_mask)['render']
        else:
            buffer_image = out[self.mode]  # [3, H, W]

        if self.mode in ['depth', 'alpha']:
            buffer_image = buffer_image.repeat(3, 1, 1)
            if self.mode == 'depth':
                buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)
        try:
            buffer_image = torch.nn.functional.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )



            self.need_update = True

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            if self.gui:
                dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000 / t)} FPS FID: {fid.item()})")
                dpg.set_value(
                    "_texture", self.buffer_image
                )  # buffer must be contiguous, else seg fault!
        except:
            pass


    # no gui mode
    def train(self, iters=5000):
        self.gaussians.compute_3D_filter(cameras=self.trainCameras)
        if iters > 0:
            for i in tqdm.trange(iters):
                self.train_step()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    sp = SegParams(parser)

    parser.add_argument('--gui', action='store_false', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--elevation', type=float, default=0, help="default GUI camera elevation")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int,
    #                     default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 20000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--joint_type", choices=['r', 'p'])
    # parser.add_argument("--revolute", action="store_true")
    # parser.add_argument("--prismatic", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    gui = GUI(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args),
              testing_iterations=args.test_iterations, saving_iterations=args.save_iterations)

    if args.gui:
        gui.render()
    else:
        gui.train(args.iterations)

    # All done
    print("\nTraining complete.")
