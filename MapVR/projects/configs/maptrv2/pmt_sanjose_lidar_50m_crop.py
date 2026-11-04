# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from MapTRv2 (e03f097)
#   (https://github.com/hustvl/MapTR/tree/e03f097abef19e1ba3fed5f471a8d80fbfa0a064)
# Copyright (c) 2022 Hust Vision Lab, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

"""HRDX San Jose 50 m tiles -> rotated 30 m crops, one class (curb).

PseudoMapTrainer's side of the San Jose benchmark: a thin overlay on
`pmt_carla_lidar_50m_crop.py` mirroring the sibling MapTRv2 repo's
`maptrv2_sanjose_r50_24ep_lidar_50m_crop.py`. The MODEL and the crop recipe
are the CARLA 50 m ones unchanged (30 m crops, uniform yaw + <= 3.79 m shift
+ sigma 0.02 m noise at train; frozen, unrotated, aimed crops at eval;
colour-free 3-channel points; PMT's masked assigner/losses with
`mask_mode='ones'`). What changes is the data:

* **The export is `san_jose_tiles_50m`** (HRDX San Jose, 17723 tiles of
  real LiDAR, `tile_radius` 25, world-frame points). Its layout is FLAT --
  `blocks/` and `reference_lines/` sit directly under the root, no
  `<split>/` level -- and the split manifests live in a separate directory,
  `san_jose_splits_v1/v1/{train,val,test}.json` (12073 / 1312 / 1978 tiles;
  a leakage-free 750 m-block spatial split). `PMTSanJose50mCropDataset`
  reads both.
* **One class, `curb`.** The export's taxonomy is `{1: curb,
  2: curb_parametric}`; the parametric class re-traces the same curbs, so
  only `curb` is trained. `map_classes = ['curb']`, `aux_seg.seg_classes =
  1`; `num_vec_one2one` 50 x 1 class keeps the inherited `max_num=50` legal.
* **The splits were finalised with the MapTRv2 loader's crop code** (the
  same code as this repo's `PMTCarla50mCropDataset`): every val/test tile
  has a non-empty, >= 1000-point crop at the frozen pose
  `default_rng(crop_seed=0 + index)` in manifest order, every train tile a
  non-empty crop for one of four seeds. The dataset checks the manifests'
  `finalize` block against its own kwargs and warns if they drift.
* **val != test here**, so the two frozen splits get their OWN
  `map_ann_file`s -- `_format_gt()` writes a GT json once and never
  regenerates it. Delete both whenever the tile set, crop kwargs or seed
  change.

Data layout (relative to the MapVR root):

    data/san_jose_50m/blocks/<name>.npz
    data/san_jose_50m/reference_lines/<name>_reference_lines.json
    data/san_jose_splits_v1/v1/{train,val,test}.json

The 50 m base already rewrote `data.*` with `_delete_`, so this overlay only
re-points the keys that change (type, ann files, GT jsons, taxonomy, the
flat-layout `split=None`); mmcv merges the rest from the base.
"""

_base_ = ['./pmt_carla_lidar_50m_crop.py']
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# ---- data source -----------------------------------------------------------
dataset_type = 'PMTSanJose50mCropDataset'
data_root = 'data/san_jose_50m/'
splits_root = 'data/san_jose_splits_v1/v1/'
ann_file_train = splits_root + 'train.json'
ann_file_val = splits_root + 'val.json'
ann_file_test = splits_root + 'test.json'
# One GT json PER frozen split (see the docstring).
map_ann_file_val = 'data/carla/sanjose_map_gt_50m_crop_curb_val.json'
map_ann_file_test = 'data/carla/sanjose_map_gt_50m_crop_curb_test.json'

# ---- taxonomy --------------------------------------------------------------
map_classes = ['curb']
num_map_classes = len(map_classes)

aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=False,
    seg_classes=num_map_classes,
    feat_down_sample=32,
    pv_thickness=1,
)

# ---- geometry (restated: config isolation) ---------------------------------
crop_size = 30.0
lidar_point_cloud_range = [-15.0, -15.0, -72.0, 15.0, 15.0, 96.0]
lidar_voxel_size = [0.1, 0.1, 0.4]

# flat export: no <split>/ level. Stated explicitly because mmcv merges dicts
# recursively and the base's split='train'/'test' would survive.
sj_common = dict(
    type=dataset_type,
    data_root=data_root,
    split=None,
    gt_classes=('curb',),
    map_classes=map_classes,
    aux_seg=aux_seg_cfg,
)

# ---- pipelines (== the 50 m base, restated with this file's class list) ----
train_pipeline = [
    dict(type='LoadCarla50mCrop', coord_type='LIDAR', load_dim=3, use_dim=3),
    dict(type='GridSamplePoints', grid_size=lidar_voxel_size,
         point_cloud_range=lidar_point_cloud_range),
    dict(type='DefaultFormatBundle3D', with_gt=False, with_label=False,
         class_names=map_classes),
    dict(type='CustomCollect3D', keys=['points'])
]
test_pipeline = [
    dict(type='LoadCarla50mCrop', coord_type='LIDAR', load_dim=3, use_dim=3),
    dict(type='GridSamplePoints', grid_size=lidar_voxel_size,
         point_cloud_range=lidar_point_cloud_range),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1, 1),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', with_gt=False, with_label=False,
                 class_names=map_classes),
            dict(type='CustomCollect3D', keys=['points'])
        ])
]

model = dict(
    pts_bbox_head=dict(
        num_classes=num_map_classes,
        aux_seg=aux_seg_cfg,
        bbox_coder=dict(num_classes=num_map_classes)))

data = dict(
    train=dict(ann_file=ann_file_train, pipeline=train_pipeline, **sj_common),
    val=dict(ann_file=ann_file_val, map_ann_file=map_ann_file_val,
             pipeline=test_pipeline, **sj_common),
    test=dict(ann_file=ann_file_test, map_ann_file=map_ann_file_test,
              pipeline=test_pipeline, **sj_common),
)

# Restated purely to re-point it at THIS test_pipeline.
evaluation = dict(interval=6, pipeline=test_pipeline, metric='chamfer')
