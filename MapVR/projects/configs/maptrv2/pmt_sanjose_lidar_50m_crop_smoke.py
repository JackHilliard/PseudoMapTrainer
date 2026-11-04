# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

"""Smoke-test overlay: the v1_smoke split subset (400 train / 60 val tiles).

Same model and crop kwargs as the parent; only the manifests and the GT json
name change. `val_60.json` is a prefix of `val.json`, so its frozen crops are
the first 60 of the full split. Val doubles as test here.
"""

_base_ = ['./pmt_sanjose_lidar_50m_crop.py']
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

splits_root = 'data/san_jose_splits_v1/v1_smoke/'
ann_file_train = splits_root + 'train_400.json'
ann_file_val = splits_root + 'val_60.json'
map_ann_file_smoke = 'data/carla/sanjose_map_gt_50m_crop_curb_val60_smoke.json'

data = dict(
    train=dict(ann_file=ann_file_train),
    val=dict(ann_file=ann_file_val, map_ann_file=map_ann_file_smoke),
    test=dict(ann_file=ann_file_val, map_ann_file=map_ann_file_smoke),
)
evaluation = dict(interval=1)
