# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from Mask2Former (9b0651c)
#   (https://github.com/facebookresearch/Mask2Former/tree/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a)
# Copyright (c) Facebook, Inc. and its affiliates, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import json


def _get_mapillary_vistas_v2_meta():
    # read in config file
    with open('datasets/mapillary_vistas_config_v2.0.json') as config_file:
        config = json.load(config_file)
    # in this example we are only interested in the labels
    labels = config['labels']
    stuff_name = []
    stuff_color = []
    for label in labels[:-1]:
        stuff_name.append(label["readable"])
        stuff_color.append(label["color"])
    ret = {
        "stuff_classes": stuff_name,
        "stuff_colors": stuff_color,
    }
    return ret


def register_all_mapillary_vistas_v2(root):
    root = os.path.join(root, "mapillary_vistas")
    meta = _get_mapillary_vistas_v2_meta()
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, dirname, "images")
        gt_dir = os.path.join(root, dirname, "v2.0", "labels")
        name = f"mapillary_vistas_v2_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=123,  # different from other datasets, Mapillary Vistas v2 sets ignore_label to 123
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_mapillary_vistas_v2(_root)
