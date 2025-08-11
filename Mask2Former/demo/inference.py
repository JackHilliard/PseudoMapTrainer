# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from RoMe (188fd42)
#   (https://github.com/DRosemei/RoMe/blob/188fd42d0bf5324d6dd7a48758eef6c81a04d8fe/scripts/mask2former_infer/inference.py)
# Copyright (c) 2023 Horizon Robotics, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
from nuscenes_scenes import crawl_nusc_scenes_paths

from detectron2.data import DatasetCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
import torch
# fmt: off
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on


from mask2former import add_maskformer2_config
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.config import get_cfg
from os.path import join
from pathlib import Path
import tqdm
import numpy as np
import cv2
import tempfile
import argparse
import multiprocessing as mp

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def get_parser():
    """
    Create a parser for command-line arguments optionally based on a initial parser.
    If `parser` is None, a new one is created.
    """
    parser = default_argument_parser()
    parser.add_argument(
        "--base_dir",
        default="#####/Nuscenes/sweeps/",  # "samples" contain key frames
        help="nuScenes base dir",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output file if it exists. Default is False",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="nuScenes version (trainval, mini, test). " \
        "If None, the whole dataset will be used.",
    )
    parser.add_argument(
        "--save_dir",
        default="#####/KittiOdom/sequences",
        help="nuScenes base dir",
    )
    parser.add_argument(
        "--scene_names",
        type=str,
        nargs="+",
        default=[],
        help="Names of all scenes to infer (e.g. 'scene-0063' 'scene-0064' 'scene-0200'). Default means that the whole dataset should be labeled"
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def get_dict(file_paths, label_paths, overwrite=False):
    dataset_dicts = []
    for img_path, label_path in zip(file_paths, label_paths):
        if overwrite or not os.path.isfile(label_path):
            dataset_dicts.append({
                "file_name": img_path
            })
    return dataset_dicts


def register_dataset(name:str, file_paths, label_paths, overwrite=False):
    DatasetCatalog.register(
        name, lambda: get_dict(file_paths, label_paths, overwrite=overwrite)
    )


def main(args):
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup(args)
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    model.eval()
    base_dir = str(Path(args.base_dir).expanduser().resolve())
    save_dir = str(Path(args.save_dir).expanduser().resolve())

    camera_names = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    crawl_func = crawl_nusc_scenes_paths
    get_label_path = lambda img_path: img_path.replace(base_dir, save_dir).replace("/CAM", "/seg_CAM").replace(".jpg", ".png")

    file_paths = crawl_func(base_dir, camera_names, args.split, args.scene_names)
    label_save_paths = [get_label_path(img_path) for img_path in file_paths]
    output_folder_paths = set([os.path.dirname(label_path) for label_path in label_save_paths])
    for output_folder_path in output_folder_paths:
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)

    register_dataset('nuscenes', file_paths, label_save_paths, overwrite=args.overwrite)
    data_loader = DefaultTrainer.build_test_loader(cfg, 'nuscenes')
    
    for inputs in tqdm.tqdm(data_loader):
        with torch.no_grad():
            outputs = model(inputs)
        for k, output in enumerate(outputs):
            save_img = output["sem_seg"].argmax(dim=0).cpu()
            img_path = inputs[k]["file_name"]
            seg_save_path = get_label_path(img_path)
            successful_write = cv2.imwrite(seg_save_path, save_img.numpy())
            if not successful_write:
                logger.error(f"Failed to save label to {seg_save_path}")


if __name__ == "__main__":
    args = get_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
