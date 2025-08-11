# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from MapTRv2 (e03f097)
#   (https://github.com/hustvl/MapTR/tree/e03f097abef19e1ba3fed5f471a8d80fbfa0a064)
# Copyright (c) 2022 Hust Vision Lab, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.


_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-15.0, -30.0,-10.0, 15.0, 30.0, 10.0]
voxel_size = [0.15, 0.15, 20.0]
dbound=[1.0, 35.0, 0.5]

grid_config = {
    'x': [-30.0, -30.0, 0.15], # useless
    'y': [-15.0, -15.0, 0.15], # useless
    'z': [-10, 10, 20],        # useless
    'depth': [1.0, 35.0, 0.5], # useful
}


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# map has classes: divider, ped_crossing, boundary
map_classes = ['divider', 'ped_crossing','boundary']
# fixed_ptsnum_per_line = 20
# map_classes = ['divider',]
num_vec=50
fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag=True
num_map_classes = len(map_classes)

renderer_H = 256
renderer_W = 128

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
# bev_h_ = 50
# bev_w_ = 50
bev_h_ = 200
bev_w_ = 100
queue_length = 1 # each sequence contains `queue_length` frames.

aux_seg_cfg = dict(
    use_aux_seg=False,
    bev_seg=False,
    pv_seg=False,
    seg_classes=1,
    feat_down_sample=32,
    pv_thickness=1,
)

dataset_type = 'CustomNuScenesOfflineLocalMapDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
   
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D', 
                with_gt=False, 
                with_label=False,
                class_names=map_classes),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4, # TODO
    pseudo_test=dict(
        type='PseudoMapDataset',
        raster=[renderer_W, renderer_H],
        data_root=data_root,
        mask_thresh=0.0,
        use_pv_pseudo_seg=False,
        ann_file=data_root + 'nuscenes_pseudo_single_map_infos_temporal_val.pkl',
        map_ann_file=data_root + 'nuscenes_map_anns_pseudo_single_val.json',
        pipeline=test_pipeline, 
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        classes=class_names, 
        modality=input_modality),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_map_infos_temporal_val.pkl',
        map_ann_file=data_root + 'nuscenes_map_anns_val.json',
        pipeline=test_pipeline, 
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        classes=class_names, 
        modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
fp16 = dict(loss_scale=512.)