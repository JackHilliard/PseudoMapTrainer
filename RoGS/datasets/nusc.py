# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from RoGS (a214497)
#   (https://github.com/fzhiheng/RoGS/tree/a21449733c157ca2d58adfc3b8c2225a624ee466)
# Copyright 2024 Zhiheng Feng, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import os
from copy import deepcopy
from multiprocessing.pool import ThreadPool as Pool

import cv2
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from plyfile import PlyData
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.splits import create_splits_scenes

from datasets.base import BaseDataset


def all_scene_names() -> list:
    """
    Get all scene names from the NuScenes dataset.
    :return: List of all scene names.
    """
    splits = create_splits_scenes()
    return splits["train"] + splits["val"] + splits["test"]


def load_geo_split_scenes(splits=["train", "val", "test"]):

    all_scenes = []
    for split in splits:
        with open(f"datasets/nusc/geosplits/near/{split}.txt", "r") as f:
            split_set = f.read().splitlines()
        all_scenes += split_set

    return all_scenes




def loda_depth(depth_file):
    loaded_data = np.load(depth_file)
    depth_img = sp.csr_matrix((loaded_data['data'], loaded_data['indices'], loaded_data['indptr']), shape=loaded_data['shape'])
    return depth_img


def worldpoint2camera(points: np.ndarray, WH, cam2world, cam_intrinsic, min_dist: float = 1.0):
    """
    1. transform world points to camera points
    Args:
        points: (N, 3)
        image:  (H, W, 3)
        cam2world: (4, 4)
        cam_intrinsic: (3, 3)
        min_dist: float

    Returns:
        uv: (2, N)
        depths: (N, )
        mask: (N, )

    """
    width, height = WH
    world2cam = np.linalg.inv(cam2world)  # (4, 4)
    points_cam = world2cam[:3, :3] @ points.T + world2cam[:3, 3:4]  # (3, N)
    depths = points_cam[2, :]  # (N, )
    points_uv1 = view_points(points_cam, np.array(cam_intrinsic), normalize=True)  # (3, N)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points_uv1[0, :] > 1)
    mask = np.logical_and(mask, points_uv1[0, :] < width - 1)
    mask = np.logical_and(mask, points_uv1[1, :] > 1)
    mask = np.logical_and(mask, points_uv1[1, :] < height - 1)

    uv = points_uv1[:, mask][:2, :]
    uv = np.round(uv).astype(np.uint16)
    depths = depths[mask]
    return uv, depths, mask


class NuscDataset(BaseDataset):

    def __init__(self, configs, use_label=True, use_depth=False, use_lidar=True, nusc_all=None):
        self.nusc_all = {
            "trainval": NuScenes(version="v1.0-trainval", dataroot=configs["base_dir"]),
            "test": NuScenes(version="v1.0-test", dataroot=configs["base_dir"])
        } if nusc_all is None else nusc_all

        self.use_geo_split: bool = configs["use_geo_split"]
        self.road_gt_dir: str = configs["road_gt_dir"]
        self.lidar_is_keyframe = []
        self.lidar_times_all = []
        self.lidar_filenames_all = []
        self.lidar2world_all = []

        super().__init__(configs, use_label=use_label, use_depth=use_depth, use_lidar=use_lidar)


    def __len__(self):
        return len(self.image_filenames_all)

    def __getitem__(self, idx):
        cam_idx = self.cameras_idx_all[idx]
        cam2world = self.camera2world_all[idx]
        K = self.cameras_K_all[idx]
        camera_name = self.camera_names[cam_idx]
        image_path = os.path.join(self.base_dir, self.image_filenames_all[idx])
        image_name = os.path.basename(image_path).split(".")[0]
        input_image = cv2.imread(image_path)

        crop_cy = int(self.resized_image_size[1] * 0.5)
        origin_image_size = input_image.shape
        resized_image = cv2.resize(input_image, dsize=self.resized_image_size, interpolation=cv2.INTER_LINEAR)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image = resized_image[crop_cy:, :, :]  # crop the sky
        gt_image = (np.asarray(resized_image) / 255.0).astype(np.float32)  # [H, W, 3]
        gt_image = np.clip(gt_image, 0.0, 1.0)
        width, height = gt_image.shape[1], gt_image.shape[0]

        new_K = deepcopy(K)
        width_scale = self.resized_image_size[0] / origin_image_size[1]
        height_scale = self.resized_image_size[1] / origin_image_size[0]
        new_K[0, :] *= width_scale
        new_K[1, :] *= height_scale
        new_K[1][2] -= crop_cy
        R = cam2world[:3, :3]
        T = cam2world[:3, 3]

        sample = {
            "image": gt_image, "idx": idx,
            "cam_idx": cam_idx, "image_name": image_name, "R": R,
            "T": T, "K": new_K, "W": width, "H": height
        }

        if self.use_label:
            label_path = os.path.join(self.label_dir, self.label_filenames_all[idx])
            label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            resized_label = cv2.resize(label, dsize=self.resized_image_size, interpolation=cv2.INTER_NEAREST)
            mask, label = self.label2mask(resized_label)
            if camera_name == "CAM_BACK":
                h = mask.shape[0]
                mask[int(0.83 * h):, :] = 0
            label = self.remap_semantic(label).astype(int)
            mask = mask[crop_cy:, :]
            label = label[crop_cy:, :]
            sample["mask"] = mask
            sample["label"] = label

        if self.use_depth:
            cam_time = self.camera_times_all[idx]
            lidar_idx = np.argmin(np.abs(self.lidar_times_all - cam_time))
            lidar2world = self.lidar2world_all[lidar_idx]
            lidar_path = os.path.join(self.base_dir, self.lidar_filenames_all[lidar_idx])
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
            points_world = lidar2world[:3, :3] @ points.T + lidar2world[:3, 3:4]  # (3, N)
            uv, depths, mask = worldpoint2camera(points_world.T, (width, height), cam2world, new_K)
            sort_idx = np.argsort(depths)[::-1]
            uv = uv[:, sort_idx]
            depths = depths[sort_idx]
            depth_image = np.zeros((height, width), dtype=np.float32)
            depth_image[uv[1], uv[0]] = depths
            sample["depth"] = depth_image

        return sample

    def parse_clips(self):
        road_pointcloud = dict()
        chassis2world_unique = []
        chassis2world_all    = []
        camera2world_all     = []
        camera_times_all     = []

        for scene_name in tqdm(self.clip_list, desc="Loading data clips"):
            nusc = self.get_nusc_obj(scene_name)
            records = [samp for samp in nusc.sample if nusc.get("scene", samp["scene_token"])["name"] in scene_name]
            records.sort(key=lambda x: (x['timestamp']))

            print(f"Loading data from scene {scene_name}")
            cam_info, chassis_info = self.load_cameras(records, nusc)

            chassis2world_unique.extend(chassis_info["unique_poses"])
            chassis2world_all.extend(chassis_info["poses"])

            if self.sens_data_from_stat_scenes or (not self.scene_is_stat(scene_name)):
                camera2world_all.extend(cam_info["poses"])
                camera_times_all.extend(cam_info["times"])
                self.cameras_K_all.extend(cam_info["intrinsics"])
                self.cameras_idx_all.extend(cam_info["idxs"])
                self.image_filenames_all.extend(cam_info["filenames"])

                lidar_info = self.load_lidars(records, nusc)
                self.lidar_is_keyframe.extend(lidar_info["is_key"])
                self.lidar_times_all.extend(lidar_info["times"])
                self.lidar_filenames_all.extend(lidar_info["filenames"])
                self.lidar2world_all.extend(lidar_info["poses"])

                if self.use_lidar:
                    point_gt_path = os.path.join(self.road_gt_dir, f"{scene_name}.ply")
                    xyz, rgb, label = self.load_gt_points(point_gt_path)
                    road_pointcloud[scene_name] = {"xyz": xyz, "rgb": rgb, "label": label}
        
        self.label_filenames_all += [rel_camera_path.replace("/CAM", "/seg_CAM").replace(".jpg", ".png") for rel_camera_path in self.image_filenames_all]

        self.chassis2world_unique = np.array(chassis2world_unique)  # [K, 4, 4]
        self.chassis2world_all = np.array(chassis2world_all)  # [L, 4, 4]
        self.camera2world_all = np.array(camera2world_all)  # [M, 4, 4]
        self.camera_times_all = np.array(camera_times_all)  # [M, ]

        self.lidar2world_all = np.array(self.lidar2world_all)  # [N, 4, 4]
        self.lidar_times_all = np.array(self.lidar_times_all)  # [N, ]

        self.file_check()
        if len(self.image_filenames_all) == 0:
            raise FileNotFoundError("No data found in the dataset")

        self.road_pointcloud = {
            k: np.concatenate([road_pointcloud[s][k] for s in road_pointcloud.keys()], axis=0) for k in ("xyz", "rgb", "label")
        } if self.use_depth and self.use_lidar else None


    def relativize_poses(self):
        super().relativize_poses()
        self.relativize_pointcloud()


    def relativize_pointcloud(self):
        if self.road_pointcloud is not None:
            ref_pose_inv = np.linalg.inv(self.ref_pose)
            xyz = self.road_pointcloud["xyz"]
            new_xyz = ref_pose_inv[:3, :3] @ xyz.T + ref_pose_inv[:3, 3:4]
            self.road_pointcloud["xyz"] = new_xyz.T

    def get_nusc_obj(self, scene_name):
        all_orig_splits = create_splits_scenes()
        trainval_scenes = set(all_orig_splits["train"] + all_orig_splits["val"])
        nusc = self.nusc_all["trainval"] if scene_name in trainval_scenes else self.nusc_all["test"]
        return nusc
    
    def filter_clips_by_split(self):
        if self.use_geo_split:
            scenes_in_split = load_geo_split_scenes(self.splits)
        else:
            all_orig_splits = create_splits_scenes()
            scenes_in_split = sum([all_orig_splits[split] for split in self.splits], [])
        self.clip_list = [clip for clip in self.clip_list if clip in scenes_in_split]

    def load_gt_points(self, ply_path):
        plydata = PlyData.read(ply_path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)  # [N, 3]
        rgb = np.stack((np.asarray(plydata.elements[0]["r"]),
                        np.asarray(plydata.elements[0]["g"]),
                        np.asarray(plydata.elements[0]["b"])), axis=1)
        label = np.asarray(plydata.elements[0]["label"]).astype(np.uint8)
        label = label[..., None]  # [N, 1]
        return xyz, rgb, label

    def load_lidars(self, records, nusc):
        is_key = []
        lidar_times = []
        lidar_files = []
        lidar2worlds = []
        ego2globals = []

        for rec in tqdm(records):
            samp = nusc.get("sample_data", rec["data"]["LIDAR_TOP"])
            while True:

                lidar_times.append(samp["timestamp"])
                lidar_files.append(samp["filename"])
                is_key.append(samp["is_key_frame"])

                lidar2ego = self.compute_extrinsic2chassis(samp, nusc)
                ego2global = self.compute_chassis2world(samp, nusc)
                ego2globals.append(ego2global)

                lidar2global = ego2global @ lidar2ego
                lidar2worlds.append(lidar2global)

                if samp["next"] != "":
                    samp = nusc.get('sample_data', samp["next"])
                    if samp["is_key_frame"]: break
                else:
                    break

        return {
            "times": lidar_times,
            "filenames": lidar_files,
            "poses": lidar2worlds,
            "ego2globals": ego2globals,
            "is_key": is_key
        }


    def load_cameras(self, records, nusc):
        chassis2world_unique = []
        chassis2worlds = []
        cam2worlds = []
        cameras_K = []
        cameras_idxs = []
        cameras_times = []
        image_filenames = []
        wh = dict()

        # interpolate images from 2HZ to 12 HZ (sample + sweep)
        for rec in tqdm(records):
            for camera_idx, cam in enumerate(self.camera_names):
                # compute camera key frame poses
                rec_token = rec["data"][cam]
                samp = nusc.get("sample_data", rec_token)
                wh.setdefault(cam, (samp["width"], samp["height"]))
                while True:
                    rel_camera_path = samp["filename"]
                    cameras_times.append(samp["timestamp"])
                    image_filenames.append(rel_camera_path)

                    camera2chassis = self.compute_extrinsic2chassis(samp, nusc)
                    chas2w = self.compute_chassis2world(samp, nusc)
                    chassis2worlds.append(chas2w)
                    if camera_idx == 0:
                        chassis2world_unique.append(chas2w)
                    cam2w = chas2w @ camera2chassis
                    cam2worlds.append(cam2w.astype(np.float32))

                    calibrated_sensor = nusc.get("calibrated_sensor", samp["calibrated_sensor_token"])
                    intrinsic = np.array(calibrated_sensor["camera_intrinsic"])
                    cameras_K.append(intrinsic.astype(np.float32))

                    cameras_idxs.append(camera_idx)
                    # not key frames
                    if samp["next"] != "":
                        samp = nusc.get('sample_data', samp["next"])
                        if samp["is_key_frame"]: break
                    else:
                        break
        cam_info = {"poses": cam2worlds, "intrinsics": cameras_K, "idxs": cameras_idxs, "filenames": image_filenames, "times": cameras_times, "wh": wh}
        chassis_info = {"poses": chassis2worlds, "unique_poses": chassis2world_unique}

        return cam_info, chassis_info


    @staticmethod
    def compute_chassis2world(samp, nusc):
        """transform sensor in world coordinate"""
        # comput current frame Homogeneous transformation matrix : from chassis 2 global
        pose_chassis2global = nusc.get("ego_pose", samp['ego_pose_token'])
        chassis2global = transform_matrix(pose_chassis2global['translation'],
                                          Quaternion(pose_chassis2global['rotation']),
                                          inverse=False)
        return chassis2global

    @staticmethod
    def compute_extrinsic2chassis(samp, nusc):
        calibrated_sensor = nusc.get("calibrated_sensor", samp["calibrated_sensor_token"])
        rot = np.array(Quaternion(calibrated_sensor["rotation"]).rotation_matrix)
        tran = np.expand_dims(np.array(calibrated_sensor["translation"]), axis=0)
        sensor2chassis = np.hstack((rot, tran.T))
        sensor2chassis = np.vstack((sensor2chassis, np.array([[0, 0, 0, 1]])))  # [4, 4] camera 3D
        return sensor2chassis

    def file_check(self):
        image_paths = [os.path.join(self.base_dir, image_path) for image_path in self.image_filenames_all]
        image_exists = np.asarray(self.check_filelist_exist(image_paths))
        print(f"Drop {len(image_paths) - len(np.where(image_exists)[0])} frames out of {len(image_paths)} by image exists check")
        exists = image_exists
        label_paths = [os.path.join(self.label_dir, label_path) for label_path in self.label_filenames_all]
        label_exists = np.asarray(self.check_filelist_exist(label_paths))
        print(f"Drop {len(image_paths) - len(np.where(label_exists)[0])} frames out of {len(image_paths)} by label exists check")
        exists *= label_exists

        lidar_paths = [os.path.join(self.base_dir, lidar_path) for lidar_path in self.lidar_filenames_all]
        lidar_exists = np.asarray(self.check_filelist_exist(lidar_paths))
        print(f"Drop {len(lidar_paths) - len(np.where(lidar_exists)[0])} lidar out of {len(lidar_paths)} by lidar exists check")
        lidar_available = list(np.where(lidar_exists)[0])
        self.lidar_times_all = [self.lidar_times_all[i] for i in lidar_available]
        self.lidar_filenames_all = [self.lidar_filenames_all[i] for i in lidar_available]
        self.lidar2world_all = [self.lidar2world_all[i] for i in lidar_available]
        self.lidar_is_keyframe = [self.lidar_is_keyframe[i] for i in lidar_available]

        available_index = list(np.where(exists)[0])
        print(f"Drop {len(image_paths) - len(available_index)} frames out of {len(image_paths)} by file exists check")
        self.filter_by_index(available_index)

    def label_valid_check(self):
        label_paths = [os.path.join(self.label_dir, label_path) for label_path in self.label_filenames_all]
        label_valid = np.asarray(self.check_label_valid(label_paths))
        available_index = list(np.where(label_valid)[0])
        print(f"Drop {len(label_paths) - len(available_index)} frames out of {len(label_paths)} by label valid check")
        self.filter_by_index(available_index)

    def label_valid(self, label_name):
        label = cv2.imread(label_name, cv2.IMREAD_UNCHANGED)
        label_movable = label >= 52
        ratio_movable = label_movable.sum() / label_movable.size
        label_off_road = ((0 <= label) & (label <= 1)) | ((3 <= label) & (label <= 6)) | ((10 <= label) & (label <= 12)) \
                         | ((15 <= label) & (label <= 22)) | ((25 <= label) & (label <= 40)) | (label >= 42)
        ratio_static = label_off_road.sum() / label_off_road.size
        if ratio_movable > 0.3 or ratio_static > 0.9:
            return False
        else:
            return True

    def check_label_valid(self, filelist):
        with Pool(32) as p:
            exist_list = p.map(self.label_valid, filelist)
        return exist_list

    def filter_by_index(self, index):
        self.camera2world_all = self.camera2world_all[index]
        self.camera_times_all = self.camera_times_all[index]
        self.image_filenames_all = [self.image_filenames_all[i] for i in index]
        self.cameras_K_all = [self.cameras_K_all[i] for i in index]
        self.cameras_idx_all = [self.cameras_idx_all[i] for i in index]
        self.label_filenames_all = [self.label_filenames_all[i] for i in index]


    def scene_is_stat(self, scene_name:str, dist_threshold=10.0) -> bool:
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
        nusc = self.get_nusc_obj(scene_name)
        # --- gather all ego-poses for this scene ---
        scene_token = next(s["token"] for s in nusc.scene if s["name"] == scene_name)
        scene = nusc.get("scene", scene_token)

        sample = nusc.get("sample", scene["first_sample_token"])
        xy, t = [], []                          # horizontal pos & timestamps (µs)

        while True:
            sd    = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            pose  = nusc.get("ego_pose", sd["ego_pose_token"])
            xy.append(pose["translation"][:2])       # x, y only
            t.append(sd["timestamp"])                # µs
            if sample["next"] == "":
                break
            sample = nusc.get("sample", sample["next"])

        # --- total distance criterion -------------------------------------------
        step_dist   = np.linalg.norm(np.diff(xy, axis=0), axis=1)   # metres
        total_dist  = step_dist.sum()
        return total_dist < dist_threshold



