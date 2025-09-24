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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, load_root=False, load_cano=False, crop_gs = False, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path+"/s1", "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.train_ours)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "metadata.json")):
            print("Found metadata.json file, assuming multi scale Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Multi-scale"](args.source_path, args.white_background, args.eval, args.load_allres)
        elif os.path.exists(os.path.join(args.source_path, "camera_train.json")):
            print("Found camera_train.json file, assuming PartNet set!")
            scene_info = sceneLoadTypeCallbacks["PartNet"](args.source_path, args.white_background, args.eval, init_num= args.init_num)
        elif os.path.exists(os.path.join(args.source_path, "start")):
            print("Found camera_train.json file, assuming d-PartNet set!")
            scene_info = sceneLoadTypeCallbacks["PartNet"](args.source_path, args.white_background, args.eval, deform=True)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if load_cano:
            self.gaussians.load_ply_cano(os.path.join(self.model_path, "point_cloud_cano.ply"))

            print(f'load cano_points')


        elif self.loaded_iter:
            if not load_root:
                self.gaussians.load_ply(os.path.join(self.model_path+"/s1",
                                                               "point_cloud",
                                                               "iteration_" + str(self.loaded_iter),
                                                               "point_cloud.ply"), crop_gs=crop_gs, bounding_box= [2, 2, 2]) # [1.0,1.0,1.0]
                print(f'load {os.path.join(self.model_path+"/start", "point_cloud","iteration_" + str(self.loaded_iter),"point_cloud.ply")}')
            else:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                     "point_cloud",
                                                     "iteration_" + str(self.loaded_iter),
                                                     "point_cloud.ply"))
                print(f'load {os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply")}')
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_state(self, iteration, state=1):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.vanilla_save_ply(os.path.join(point_cloud_path, f"point_cloud_{state}.ply"))
        print("saving_complete")
        # cano_gs = GaussianModel(self.gaussians.max_sh_degree)
        # large_motion_state = match_gaussians(os.path.join(point_cloud_path, "point_cloud.ply"), cano_gs, num_slots, vis_cano)
        # cal_cluster_centers(os.path.join(point_cloud_path, "point_cloud.ply"), num_slots, vis_center)
        # return large_motion_state


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]