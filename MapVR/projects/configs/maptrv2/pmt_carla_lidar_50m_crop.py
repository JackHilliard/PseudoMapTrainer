# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from MapTRv2 (e03f097)
#   (https://github.com/hustvl/MapTR/tree/e03f097abef19e1ba3fed5f471a8d80fbfa0a064)
# Copyright (c) 2022 Hust Vision Lab, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

"""50 m CARLA tiles -> rotated 30 m crops, two classes (driving / curb).

PseudoMapTrainer's side of the 50 m crop benchmark, a thin overlay on
`pmt_carla_lidar.py` mirroring the sibling MapTRv2 repo's
`maptrv2_carla_r50_24ep_lidar_50m_crop.py`. The MODEL is the 30 m one
unchanged -- same `point_cloud_range` (+-15), `sparse_shape`, `bev_h_`/
`bev_w_`, renderer size, coder and assigner ranges, colour-free
`SparseEncoder(in_channels=3)`, PMT's masked assigner/losses with
`mask_mode='ones'` -- because every crop the network sees IS a 30 m tile.
What changes is where those tiles come from:

* **The data are 50 m tiles on disk** (`tile_radius` 25 m), read straight from
  the export's `<split>/manifest.json` by `PMTCarla50mCropDataset` -- there
  is no converter pkl. Each `__getitem__` draws a uniform yaw about the tile
  centre plus a centre shift of up to 3.79 m (25 - 15*sqrt2, the slack a
  rotated 30 m square leaves inside a 50 m one, so every crop is fully
  populated), crops a 30 m square, and adds sigma = 0.02 m Gaussian noise to
  the points. GT polylines get the SAME pose and are clipped to the same box
  with the tiler's own `clip_polyline_to_box`.
* **Val/test crops are frozen by `crop_seed`**, not centred: `_format_gt()`
  caches the eval GT to `map_ann_file` once, so the crop each tile is scored
  on must be identical on every run. A *centred* crop would leave ~24% of the
  test tiles with no GT at all (the manifest was filtered for a *reachable*
  non-empty crop, found by aiming); seeded aiming drops that to <1%. No yaw
  at eval. Same seed as MapTRv2's config, so both repos score identical crops.
* **The crop frame replaces the tile_center frame.** The crop centre is the
  origin by construction, so `LoadCarla50mCrop` does the reframing itself and
  there is no `lidar_recenter_shift` in these pipelines. Points' z is
  re-zeroed per crop on its own 2nd-percentile ground.
* **Two classes, `driving` and `curb`**, in that (label) order -- the export's
  own names. Labels come from `map_classes` order via
  `VectorizedCarlaLocalMap` (this repo's nuScenes `VectorizedLocalMap`
  hardcodes its label table and would vectorise these names to nothing).
  `aux_seg.seg_classes` is raised to 2 as well so the auxiliary BEV mask is
  per class; it is restated on BOTH sides (model `aux_seg` and
  `data.train.aux_seg`) because the head builds its seg conv from one and
  the dataset rasterises from the other.
* **Colour-free points**: `LoadCarla50mCrop` defaults to `load_dim=3,
  use_dim=3`, matching the inherited `in_channels=3`.

`samples_per_gpu`/`workers_per_gpu`, the optimiser and the schedule are
PMT's own (inherited); MapTRv2's benchmark pinned 16 per GPU on an 80 GB
H100 -- raise it with `--cfg-options data.samples_per_gpu=16` to match.

Data layout expected under `data_root` (the export root, NOT `data/carla/`):

    <data_root>/<split>/manifest.json               tiles: [{name, center}]
    <data_root>/<split>/blocks/<name>.npz           points (world), features,
                                                    tile_center, tile_radius
    <data_root>/<split>/reference_lines/<name>_reference_lines.json

Set `data_root` to wherever the 50 m export is mounted; the default below
assumes a symlink or bind-mount at `data/carla50m/`. The eval GT json is
written next to the other CARLA GT files and must not be shared with any
other config.

Why the pipelines and `data.*` are restated in full rather than patched: mmcv
evaluates each config file in isolation and resolves `data.*.pipeline`
against the BASE's pipeline objects, so a derived config that merely
redefines `train_pipeline` would leave the datasets on the parent's
`LoadCarlaPointsFromFile` pipeline (the trap `pmt_carla_lidar_25m.py`'s
header describes). Every dataset dict below therefore names its pipeline
explicitly, and `evaluation` is re-pointed at this file's `test_pipeline`.
"""

_base_ = ['./pmt_carla_lidar.py']
# mmcv executes each config file in isolation before _base_ merging, so the
# plugin loader in custom_tools/train.py reads these from THIS file.
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# ---- data source -----------------------------------------------------------
dataset_type = 'PMTCarla50mCropDataset'
data_root = 'data/carla50m/'
ann_file_train = data_root + 'train/manifest.json'
ann_file_val = data_root + 'test/manifest.json'
ann_file_test = data_root + 'test/manifest.json'
# Own GT json: _format_gt() writes it once and never regenerates, and the
# frozen eval crops below are what it records.
map_ann_file = 'data/carla/carla_map_gt_50m_crop_2cls.json'

# ---- taxonomy --------------------------------------------------------------
map_classes = ['driving', 'curb']
num_map_classes = len(map_classes)

# Per-class auxiliary BEV mask (the 30 m base keeps seg_classes=1). Same dict
# on both sides -- see the docstring.
aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=False,
    seg_classes=num_map_classes,
    feat_down_sample=32,
    pv_thickness=1,
)

# ---- geometry (restated: config isolation) ---------------------------------
# Must equal the base's values. crop_size is the base's tile size: +-15 m.
crop_size = 30.0
point_cloud_range = [-15.0, -15.0, -30.0, 15.0, 15.0, 20.0]
lidar_point_cloud_range = [-15.0, -15.0, -72.0, 15.0, 15.0, 96.0]
lidar_voxel_size = [0.1, 0.1, 0.4]
bev_h_ = 100
bev_w_ = 100
renderer_H = 128
renderer_W = 128
fixed_ptsnum_per_gt_line = 20
eval_use_same_gt_sample_num_flag = True
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

crop_common = dict(
    crop_size=crop_size,
    require_gt=True,
    gt_tries=32,
    min_polyline_len=2.0,
    # export class names, positionally paired with map_classes above
    gt_classes=('driving', 'curb'),
    reference_dir='reference_lines',
)
crop_train = dict(split='train', rotate=True, translate=True, noise_std=0.02,
                  **crop_common)
# Frozen by seed, no yaw, no noise -- see the docstring.
crop_eval = dict(split='test', rotate=False, translate=True, noise_std=0.0,
                 crop_seed=0, **crop_common)

# ---- pipelines -------------------------------------------------------------
# LoadCarla50mCrop replaces LoadCarlaPointsFromFile: it reads the world-frame
# points, applies the pose drawn by the dataset (so points and GT cannot
# decouple) and crops. load_dim/use_dim=3 is the colour-free convention and
# must match SparseEncoder.in_channels=3 in the base.
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
        # num_vec_one2one stays at 50, so the one2one branch flattens to
        # 50 * 2 = 100 and the inherited max_num=50 is within topk's range.
        bbox_coder=dict(num_classes=num_map_classes)))

# The dataset dicts are rewritten in full (_delete_) so nothing pkl-specific
# from the base (ann_file, raw_data_root, map_ann_file) leaks through.
_eval_common = dict(
    type=dataset_type,
    data_root=data_root,
    map_ann_file=map_ann_file,
    pipeline=test_pipeline,
    raster=[renderer_W, renderer_H],
    bev_size=(bev_h_, bev_w_),
    pc_range=point_cloud_range,
    lidar_pc_range=lidar_point_cloud_range,
    fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
    eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
    padding_value=-10000,
    map_classes=map_classes,
    aux_seg=aux_seg_cfg,
    classes=[],
    modality=input_modality,
    test_mode=True,
    box_type_3d='LiDAR',
    **crop_eval)

data = dict(
    train=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_train,
        map_ann_file=map_ann_file,
        pipeline=train_pipeline,
        # PMT mask: 'ones' = whole crop observed (like-for-like baseline).
        raster=[renderer_W, renderer_H],
        mask_mode='ones',
        mask_thresh=0.0,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        lidar_pc_range=lidar_point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        aux_seg=aux_seg_cfg,
        classes=[],
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR',
        **crop_train),
    val=dict(_delete_=True, ann_file=ann_file_val, samples_per_gpu=1,
             **_eval_common),
    test=dict(_delete_=True, ann_file=ann_file_test, **_eval_common),
)

# Restated purely to re-point it at THIS test_pipeline; otherwise the eval
# hook inherits the base's copy and loads pkl-style points into a crop
# dataset.
evaluation = dict(interval=6, pipeline=test_pipeline, metric='chamfer')
