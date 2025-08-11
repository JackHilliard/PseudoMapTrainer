# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from RoGS (a214497)
#   (https://github.com/fzhiheng/RoGS/tree/a21449733c157ca2d58adfc3b8c2225a624ee466)
# Copyright 2024 Zhiheng Feng, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.


import os

import numpy as np
from multiprocessing.pool import ThreadPool as Pool
from .label_mappings.label_mapping import LabelMapping, get_label_mapping
from typing import List, Type

from abc import abstractmethod

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    REF_POSE_IDX = 0  # Default reference pose index

    def __init__(self, cfg, use_label=True, use_depth=False, use_lidar=False):
        # --------------- user-supplied configuration ---------------------
        self.base_dir: str = cfg["base_dir"]
        self.label_dir: str = cfg.get("label_dir", None)
        self.clip_list: List[str] = cfg["clip_list"]
        self.splits: List[str] = cfg["splits"]
        self.camera_names: List[str] = cfg["camera_names"]
        self.resized_image_size = (cfg["image_width"], cfg["image_height"])
        self.rm_stat_scenes: bool = cfg.get("rm_stat_scenes", False)
        self.sens_data_from_stat_scenes: bool = cfg.get("sens_data_from_stat_scenes", True)

        self.use_label: bool = use_label
        self.use_depth: bool = use_depth
        self.use_lidar: bool = use_lidar
        self.label_mapping: LabelMapping = get_label_mapping(cfg["label_mapping"])
        self.camera_times_all     = np.empty([0])     # M ndarray of camera timestamp
        self.camera2world_all     = np.empty([0,4,4]) # Mx4x4 ndarray camera2world transform
        self.chassis2world_all    = np.empty([0,4,4]) # Lx4x4 ndarray
        self.chassis2world_unique = np.empty([0,4,4]) # Kx4x4 ndarray
        self.image_filenames_all = []  # list of image relative path w.r.t to self.base_dir
        self.label_filenames_all = []  # list of label relative path w.r.t to self.base_dir
        self.cameras_K_all = []  # list of 3x3 ndarray camera intrinsics
        self.cameras_idx_all = []  # list of camera idx
        self.road_pointcloud: dict = None  # dict of road pointclouds, if available

        self.filter_clips_by_split()
        if self.rm_stat_scenes:
            self.remove_standing_scenes()
        if len(self.clip_list) == 0:
            raise NoScenesExpection(f"No scenes found")
        
        self.parse_clips()
        self.relativize_poses()

        self.cameras_extent = self.getNerfppNorm()["radius"]

    def __len__(self):
        return len(self.image_filenames_all)

    def getNerfppNorm(self):
        def get_center_and_diag(cam_centers):
            cam_centers = np.hstack(cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal

        cam_centers = [pose[:3, 3:4] for pose in self.camera2world_all]

        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1
        translate = -center
        return {"translate": translate, "radius": radius}

    def filter_by_index(self, index):
        self.image_filenames_all = [self.image_filenames_all[i] for i in index]
        self.label_filenames_all = [self.label_filenames_all[i] for i in index]
        self.cameras_K_all = [self.cameras_K_all[i] for i in index]
        self.cameras_idx_all = [self.cameras_idx_all[i] for i in index]
        if hasattr(self, "depth_filenames_all"):
            self.depth_filenames_all = [self.depth_filenames_all[i] for i in index]


    def remove_standing_scenes(self):
        self.clip_list = [scene for scene in self.clip_list if not self.scene_is_stat(scene)]


    @abstractmethod
    def scene_is_stat(self, scene_name:str, dist_threshold=10.0):
        """
        Return True iff the whole scene is basically static.

        A scene is considered a 'stop scene' iff total horizontal
        distance travelled  < dist_threshold

        Args:
            scene_name (str): Name of the scene to check.
            dist_threshold (float): Distance threshold in metres.
        Returns:
            bool: True if the scene is a stop scene, False otherwise.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    @abstractmethod
    def filter_clips_by_split(self):
        """
        Filter the clips based on the provided dataset splits.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    @abstractmethod
    def parse_clips(self):
        """
        Parse the clips and populate the necessary attributes.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def relativize_poses(self):
        self.ref_pose = self.chassis2world_unique[self.REF_POSE_IDX]

        ref_pose_inv = np.linalg.inv(self.ref_pose)

        self.chassis2world_unique = ref_pose_inv @ self.chassis2world_unique
        self.camera2world_all = ref_pose_inv @ self.camera2world_all
        self.chassis2world_all = ref_pose_inv @ self.chassis2world_all
        self.lidar2world_all = ref_pose_inv @ self.lidar2world_all

    @staticmethod
    def file_valid(file_name):
        if os.path.exists(file_name) and (os.path.getsize(file_name) != 0):
            return True
        else:
            return False

    @staticmethod
    def check_filelist_exist(filelist):
        with Pool(32) as p:
            exist_list = p.map(BaseDataset.file_valid, filelist)
        return exist_list
    
    def remap_semantic(self, semantic_label):
        return self.label_mapping.remap_semantic(semantic_label)

    def label2mask(self, label):
        return self.label_mapping.label2mask(label)

    @property
    def label_remaps(self):
        return self.label_mapping.label_remaps
    
    @property
    def filted_color_map(self):
        return self.label_mapping.color_map
    
    @property
    def num_class(self):
        return np.max(self.label_remaps) + 1


class NoScenesExpection(Exception):
    pass

class NoDataFoundExpection(Exception):
    pass


def get_dataset_cls(name:str) -> Type[BaseDataset]:
    if name == "NuscDataset":
        from .nusc import NuscDataset as DatasetCls
    else:
        raise NotImplementedError("Dataset not implemented")
    
    return DatasetCls