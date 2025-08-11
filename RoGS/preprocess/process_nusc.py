#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from RoGS (a214497)
#   (https://github.com/fzhiheng/RoGS/tree/a21449733c157ca2d58adfc3b8c2225a624ee466)
# Copyright 2024 Zhiheng Feng, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import os
import argparse
from typing import List


import cv2
import numpy as np
import yaml
import addict
from tqdm import tqdm
import scipy.sparse as sp
from plyfile import PlyData, PlyElement
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from datasets.base import get_dataset_cls
from datasets.nusc import NuscDataset
from datasets.label_mappings.label_mapping import get_label_mapping, LabelMapping
from utils.visualizer import depth2color
from utils.image import render_semantic


def get_pointcloud_to_world(nusc, samp_token, filter_sky=True) -> LidarPointCloud:
    samp = nusc.get('sample_data', samp_token)
    pcl_path = os.path.join(nusc.dataroot, samp['filename'])
    pc = LidarPointCloud.from_file(pcl_path)

    # first step: remove points too close to the lidar and outside x range
    pc.remove_close(1.0)
    if filter_sky:
        mask1 = np.logical_and(pc.points[2, :] > -2.0, pc.points[2, :] < -1.2)
        mask2 = np.logical_and(pc.points[0, :] > -20, pc.points[0, :] < 20)
        mask = np.logical_and(mask1, mask2)
        pc = LidarPointCloud(pc.points[:, mask])

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', samp['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))
    return pc


def worldpoint2camera(nusc, pc: LidarPointCloud, camera_token, min_dist: float = 1.0):
    cam = nusc.get('sample_data', camera_token)
    image = cv2.imread(os.path.join(nusc.dataroot, cam['filename']))
    width = image.shape[1]
    height = image.shape[0]

    pc = LidarPointCloud(pc.points.copy())
    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
    depths = pc.points[2, :]

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < width - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < height - 1)

    points = points[:, mask]
    depths = depths[mask]
    uv = points[:2, :]
    uv = np.round(uv).astype(np.uint16)
    return uv, depths, image, mask


def generate_cam_depth(nusc, cam_name, all_lidars: List[LidarPointCloud], lidar_times, depth_save_root, seg_root,
                       label_mapping: LabelMapping, use_all_frame=False, lidar_frame_range=(-5, 0), vis=False):
    records = [samp for samp in nusc.sample if nusc.get("scene", samp["scene_token"])["name"] in scene_name]
    records.sort(key=lambda x: (x['timestamp']))
    current_point_cloud = LidarPointCloud(points=np.concatenate([lidar.points for lidar in all_lidars], axis=1))
    for rec in records:
        samp = nusc.get("sample_data", rec["data"][cam_name])
        flag = True
        while flag or not samp["is_key_frame"]:
            flag = False
            cam_time = samp["timestamp"]
            # 找到距离相机时间最近的雷达时间
            if not use_all_frame:
                idx = np.argmin(np.abs(lidar_times - cam_time))
                left_idx = max(idx + lidar_frame_range[0], 0)
                right_idx = min(idx + lidar_frame_range[1] + 1, len(all_lidars))
                current_point_cloud = LidarPointCloud(points=np.concatenate([lidar.points for lidar in all_lidars[left_idx:right_idx]], axis=1))

            rel_camera_path = samp["filename"]
            rel_label_path = rel_camera_path.replace("/CAM", "/seg_CAM")
            rel_label_path = rel_label_path.replace(".jpg", ".png")
            label_path = os.path.join(seg_root, rel_label_path)

            rel_depth_path = rel_camera_path.replace("/CAM", "/depth_CAM")
            rel_depth_path = rel_depth_path.replace(".jpg", ".npz")
            depth_path = os.path.join(depth_save_root, rel_depth_path)
            current_depth_root = os.path.dirname(depth_path)
            os.makedirs(current_depth_root, exist_ok=True)

            label_image = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            mask, label = label_mapping.label2mask(label_image)

            uv, depth, im, _ = worldpoint2camera(nusc, current_point_cloud, samp["token"])

            # ===> 从大到小排序, 深度使用最近的点
            idx = np.argsort(depth)[::-1]
            uv = uv[:, idx]
            depth = depth[idx]

            depth_img = np.zeros((im.shape[0], im.shape[1]), dtype=np.float32)
            depth_img[uv[1], uv[0]] = depth
            depth_img[mask == 0] = 0

            sparse_matrix = sp.csr_matrix(depth_img)
            np.savez(depth_path, data=sparse_matrix.data, indices=sparse_matrix.indices, indptr=sparse_matrix.indptr, shape=sparse_matrix.shape)

            # 可视化深度图
            vis_render_depth = depth2color(depth_img, depth_img > 0)
            vis_render_depth = cv2.cvtColor(vis_render_depth, cv2.COLOR_RGB2BGR)
            remap_label = label_mapping.remap_semantic(label).astype(int)
            gt_lable = render_semantic(remap_label, label_mapping.color_map)  # RGB fomat
            gt_lable = cv2.cvtColor(gt_lable, cv2.COLOR_RGB2BGR)
            blend = cv2.addWeighted(im, 0.5, gt_lable, 0.5, 0)
            blend[depth_img > 0] = vis_render_depth[depth_img > 0]

            depth_vis_path = os.path.join(f"{depth_save_root}-vis", rel_camera_path)
            os.makedirs(os.path.dirname(depth_vis_path), exist_ok=True)
            cv2.imwrite(depth_vis_path, blend)

            if vis:
                cv2.imshow("im", blend)
                cv2.imshow("depth", vis_render_depth)
                cv2.waitKey(1)

            if samp["next"] != "":
                samp = nusc.get('sample_data', samp["next"])
            else:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate nuscenes dataset")
    parser.add_argument('--config', help='config yaml path')
    parser.add_argument("--depth", action="store_true", help="generate depth image")
    parser.add_argument("-a", "--depth_use_all_lidar", action="store_true", help="use all lidar frames when generating depth image for a image")
    parser.add_argument("-r", "--lidar_frame_range", type=int, nargs=2, default=(-5, 0), help="lidar frame range")
    parser.add_argument("--skip_processed", action="store_true", help="skip already processed scene")
    parser.add_argument("--skip_missing_labels", action="store_true",
                        help="skip scenes, where no segmentation labels are available. Otherwise, the script will raise an error.")

    args = parser.parse_args()
    with open(args.config) as file:
        configs = yaml.safe_load(file)
    configs["file"] = os.path.abspath(args.config)
    configs = addict.Dict(configs)

    label_mapping = get_label_mapping(configs.dataset.label_mapping)
    Dataset = get_dataset_cls(configs.dataset.dataset)
    assert issubclass(Dataset, NuscDataset), "Dataset must be a NuscDataset"

    nusc_root = configs.dataset.base_dir
    seg_root = configs.dataset.label_dir
    save_root = os.path.dirname(configs.dataset.road_gt_dir)
    gt_root = os.path.join(save_root, "nuScenes_road_gt")
    os.makedirs(gt_root, exist_ok=True)

    all_cam_names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


    nusc_all = {
        "trainval": NuScenes(version="v1.0-trainval", dataroot=nusc_root),
        "test": NuScenes(version="v1.0-test", dataroot=nusc_root)
    }
    splits = create_splits_scenes()
    trainval_scenes = splits["train"] + splits["val"]
    test_scenes = splits["test"]

    scene_names = configs.dataset.clip_list

    if scene_names is None:
        all_scenes = trainval_scenes + test_scenes

        scene_names = [
            scene for scene in
            all_scenes[len(all_scenes)*configs.pre_processing.percent_of_data_start//100 : len(all_scenes)*configs.pre_processing.percent_of_data_end//100]
        ]

    if len(scene_names) < 10:
        print(f"scene_names: {scene_names}")
    else:
        print(f"scene_names: {scene_names[:5]} ... {scene_names[-5:]}")

    for scene_name in tqdm(scene_names):

        if args.skip_processed:
            # check if the scene is already processed and skip processing if it is
            gt_ply_path = os.path.join(gt_root, f"{scene_name}.ply")
            if os.path.exists(gt_ply_path):
                continue

        nusc = nusc_all["trainval"] if scene_name in trainval_scenes else nusc_all["test"]
        records = [samp for samp in nusc.sample if nusc.get("scene", samp["scene_token"])["name"] in scene_name]
        records.sort(key=lambda x: (x['timestamp']))

        # =====> get all cam info
        all_cam_times = dict()
        all_cam_tokens = dict()
        for cam_name in all_cam_names:
            for rec in records:
                samp = nusc.get("sample_data", rec["data"][cam_name])
                flag = True
                while flag or not samp["is_key_frame"]:
                    flag = False
                    all_cam_times.setdefault(cam_name, []).append(samp["timestamp"])
                    all_cam_tokens.setdefault(cam_name, []).append(samp["token"])
                    if samp["next"] != "":
                        samp = nusc.get('sample_data', samp["next"])
                    else:
                        break
        for cam_name, times in all_cam_times.items():
            all_cam_times[cam_name] = np.array(times)

        # =====> get all lidar info
        all_lidars = []
        lidar_times = []
        for rec in records:
            samp = nusc.get("sample_data", rec["data"]["LIDAR_TOP"])
            flag = True
            while flag or not samp["is_key_frame"]:
                flag = False
                pc = get_pointcloud_to_world(nusc, samp["token"])
                all_lidars.append(pc)
                lidar_times.append(samp["timestamp"])
                if samp["next"] != "":
                    samp = nusc.get('sample_data', samp["next"])
                else:
                    break
        lidar_times = np.array(lidar_times)

        # ====> generate road ground truth
        try:
            all_labels_points = dict()
            for i in range(len(all_lidars)):
                lidar = all_lidars[i]
                lidar_time = lidar_times[i]
                point_num = lidar.points.shape[1]
                current_point = lidar.points[:3, :].T

                # 根据雷达投影到多个相机上获取雷达点的rgb和语义label
                labels = []
                masks = []
                final_rgb = np.zeros((point_num, 3), dtype=np.float32)
                front_rgb = None
                front_mask = None
                NAN_FLAG_NUM = 255
                for cam in all_cam_names:
                    cam_time = all_cam_times[cam]
                    idx = np.argmin(np.abs(cam_time - lidar_time))
                    token = all_cam_tokens[cam][idx]

                    uv, depths, image, mask = worldpoint2camera(nusc, lidar, token)  # (2, M), (M,), (H, W, 3), (N,) sum(mask) = M
                    rel_camera_path = nusc.get("sample_data", token)["filename"]
                    rel_label_path = rel_camera_path.replace("/CAM", "/seg_CAM")
                    rel_label_path = rel_label_path.replace(".jpg", ".png")
                    label_path = os.path.join(seg_root, rel_label_path)
                    if not os.path.exists(label_path):
                        raise FileNotFoundError(f"label_path {label_path} not exists!")
                    label_image = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)  # (H, W)
                    road_mask, cam_label = label_mapping.label2mask(label_image)  # (H, W), (H, W)

                    remap_cam_label = label_mapping.remap_semantic(cam_label).astype(np.uint8)
                    render_label = render_semantic(remap_cam_label, label_mapping.color_map)

                    valid_rgb = image[uv[1], uv[0]][:, ::-1] / 255.0
                    valid_label = cam_label[uv[1], uv[0]]

                    road_point_mask = road_mask[uv[1], uv[0]] == 1  # (M,)
                    tmp_mask = np.zeros((point_num,), dtype=bool)
                    tmp_mask[mask] = road_point_mask
                    mask = tmp_mask

                    valid_rgb = valid_rgb[road_point_mask]
                    valid_label = valid_label[road_point_mask]

                    final_rgb[mask] = valid_rgb
                    if cam == "CAM_FRONT":
                        front_rgb = valid_rgb
                        front_mask = mask

                    tmp_label = np.ones((point_num,), dtype=np.uint8) * NAN_FLAG_NUM
                    tmp_label[mask] = valid_label
                    labels.append(tmp_label)
                    masks.append(mask)  # (N,) sum(mask) = M_cam

                # 尽量使用前视相机的rgb
                final_rgb[front_mask] = front_rgb
                mask = np.stack(masks, axis=-1).any(axis=-1)  # (point_num,)
                final_rgb = final_rgb[mask]
                final_point = current_point[mask]
                final_label = np.stack(labels, axis=-1)  # (point_num, cam_num)
                final_label = final_label[mask]  # (M, cam_num)

                final_label = np.apply_along_axis(lambda x: np.bincount(x[x != NAN_FLAG_NUM]).argmax(), axis=1, arr=final_label)  # (point_num,)
                final_cam_label = label_mapping.remap_semantic(final_label).astype(np.uint8)
                final_render_label = render_semantic(final_cam_label, label_mapping.color_map).squeeze(1) / 255.0

                all_labels_points.setdefault("point", []).append(final_point)
                all_labels_points.setdefault("label", []).append(final_label[:, None])
                all_labels_points.setdefault("rgb", []).append(final_rgb)
                all_labels_points.setdefault("label_rgb", []).append(final_render_label)
        except FileNotFoundError as e:
            print(f"scene {scene_name} not processed because of {e}")
            if args.skip_missing_labels:
                continue
            else:
                raise e

        xyz = np.concatenate(all_labels_points["point"], axis=0)  # (n, 3)
        label = np.concatenate(all_labels_points["label"], axis=0)  # (n, 1)
        rgb = np.concatenate(all_labels_points["rgb"], axis=0)  # (n, 3)
        label_rgb = np.concatenate(all_labels_points["label_rgb"], axis=0)  # (n, 3)
        attributes = np.concatenate((xyz, rgb, label_rgb, label), axis=1)
        print(attributes.shape)

        dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z', 'r', 'g', 'b', 'label_r', 'label_g', 'label_b', 'label']]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        ply_path = os.path.join(gt_root, f"{scene_name}.ply")
        PlyData([el]).write(ply_path)

        # ======> generate depth image
        if args.depth:
            lidar_frame_range = args.lidar_frame_range
            use_all_frame = args.depth_use_all_lidar
            if use_all_frame:
                depth_save_root = os.path.join(save_root, "nuScenes_depth_all")
            else:
                depth_save_root = os.path.join(save_root, f"nuScenes_depth_{lidar_frame_range[0]}-{lidar_frame_range[1]}")
            os.makedirs(depth_save_root, exist_ok=True)

            for cam_name in all_cam_names:
                print(f"processing {cam_name} depth ...")
                generate_cam_depth(nusc, cam_name, all_lidars, lidar_times, depth_save_root, seg_root, label_mapping, use_all_frame=use_all_frame,
                                lidar_frame_range=lidar_frame_range, vis=False)
