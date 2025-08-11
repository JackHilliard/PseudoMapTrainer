# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from train import train, get_configs
import addict
import torch
import cv2
import numpy as np
import os
import pickle
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from datasets.nusc import all_scene_names as all_nusc_scenes, NuscDataset
from datasets.base import NoScenesExpection, get_dataset_cls

import post_process
import vectorize_mask
import json

from maptr_camera import MapTRCamera
from utils.render import render, render_label


def produce_maptr_bevs(configs):
    
    #Adjust config for MapTR BEV production
    configs.train.log = False
    configs.train.store_cfg = False
    configs.train.store_artefacts = False
    configs.train.output_render = False

    proc_config = configs.bev_processing
    pad_width = proc_config.pad_width
    mask_cls = post_process.MASK_CLASS

    device = torch.device(configs["device"] if torch.cuda.is_available() else "cpu")

    bg_color_fun = torch.rand if configs.optimization.random_background \
        else (torch.ones if configs.model.white_background else torch.zeros)
    bg_color = bg_color_fun(3, dtype=torch.float32, device=device)

    bg_label = torch.zeros(6, dtype=torch.float32, device=device) # 6 classes
    bg_label[mask_cls] = 1

    scene_coll_path = proc_config.scene_collection_path
    clips = configs.dataset.clip_list
    assert not ((scene_coll_path is not None) and (clips is not None)), "Either provide a scene collection or clips, not both"
    
    subsets = None
    dataset_kwargs = {}
    is_nusc = issubclass(get_dataset_cls(configs.dataset.dataset), NuscDataset)
    if is_nusc:
        # Load the NuScenes dataset once for all scenes
        dataset_kwargs["nusc_all"] = {
            "trainval": NuScenes(version="v1.0-trainval", dataroot=configs.dataset.base_dir),
            "test": NuScenes(version="v1.0-test", dataroot=configs.dataset.base_dir)
        }

    if scene_coll_path:
        with open(scene_coll_path, "r") as f:
            subsets = json.load(f)
        # append scene groups from all data splits
        subsets = sum([subsets[split] for split in subsets.keys()], [])
        assert len(subsets) > 0, "No scenes found in the scene collection file"
    elif clips:
        scenes = clips
    else: # get all scenes
        if is_nusc:
            scenes = all_nusc_scenes()
        else:
            raise NotImplementedError("Only support for NuScenes if all scenes should be processed")

    if subsets is None:
        subsets = [scenes] if proc_config.combine_scenes else [[scene] for scene in scenes]            

    # data chunking for parallel processing
    subsets = subsets[len(subsets)*proc_config.percent_of_data_start//100 : len(subsets)*proc_config.percent_of_data_end//100]

    # Processing loop
    for scene_subset in tqdm(subsets):
        configs.dataset.clip_list = scene_subset

        # Gaussian splatting
        try:
            gaussians, exposure_model, dataset, _ = train(configs, dataset_kwargs=dataset_kwargs)
        except NoScenesExpection as e:
            print(f"Skipping scene subset due to data error: {e}")
            continue

        lidar_key_poses = [torch.tensor(pose, dtype=torch.float32) for pose in dataset.lidar2world_all[dataset.lidar_is_keyframe]]
        lidar_key_filenames = [filepath for i, filepath in enumerate(dataset.lidar_filenames_all) if dataset.lidar_is_keyframe[i]]
        
        ref_pose = torch.tensor(dataset.ref_pose, dtype=torch.float32)
        
        for lidar_pose, lidar_filepath in zip(lidar_key_poses, lidar_key_filenames):
            current_root = os.path.join(configs.output, lidar_filepath.replace(".pcd.bin", ""))

            # Produce BEVs for MapTR training
            cam = MapTRCamera(lidar_pose, ref_pose, device, bev_h=proc_config.height, bev_w=proc_config.width)
            cam = cam.pad_image(pad_width)

            # BEV rendering
            # label
            label_feature = render_label(cam, gaussians, configs.pipeline, bg_label)
            bev_label, bev_mask = label_feature["render"], label_feature["mask"]
            bev_label = np.argmax(bev_label.detach().cpu().numpy(), axis=0)  # (H, W)
            bev_mask = bev_mask.cpu().numpy()
            bev_mask = bev_mask & (bev_label != mask_cls)
            bev_label[~bev_mask] = mask_cls

            #image
            bev_pkg = render(cam, gaussians, configs.pipeline, bg_color)
            src_bev_image = bev_pkg["render"]
            bev_image = exposure_model(0, src_bev_image)
            bev_image = bev_image.permute(1, 2, 0)
            bev_image = bev_image.detach().cpu().numpy() * 255
            bev_image = cv2.cvtColor(bev_image.astype(np.uint8), cv2.COLOR_RGB2BGRA)
            bev_image[~bev_mask] = 0

            _, map_vectors, bev_mask_post = post_process.post_process(bev_label, bev_mask, max_vector_points=20)
            
            # crop the vectors to mitigate border effects
            for key in map_vectors.keys():
                new_vectors = []
                is_polygon = key in ["crosswalk"]
                for vector in map_vectors[key]:
                    new_vectors += vectorize_mask.clip_polyline(vector, bev_mask.shape[1], bev_mask.shape[0], pad_width, polygon=is_polygon)    
                map_vectors[key] = new_vectors


            # cropping
            bev_image = post_process.crop_img(bev_image, pad_width)
            bev_label = post_process.crop_img(bev_label, pad_width)
            bev_mask_post = post_process.crop_img(bev_mask_post, pad_width)

            # Save the BEV images and vectors
            os.makedirs(current_root, exist_ok=True)
            cv2.imwrite(os.path.join(current_root, "bev_image.png"), bev_image)
            cv2.imwrite(os.path.join(current_root, "bev_label.png"), bev_label)
            cv2.imwrite(os.path.join(current_root, "bev_mask_post.png"), bev_mask_post.astype(np.uint8))

            with open(os.path.join(current_root, "map_vectors.pkl"), "wb") as f:
                pickle.dump(map_vectors, f)


if __name__ == "__main__":
    configs = get_configs()
    configs = addict.Dict(configs)
    produce_maptr_bevs(configs)
    