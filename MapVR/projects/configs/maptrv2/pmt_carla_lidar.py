# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from MapTRv2 (e03f097)
#   (https://github.com/hustvl/MapTR/tree/e03f097abef19e1ba3fed5f471a8d80fbfa0a064)
# Copyright (c) 2022 Hust Vision Lab, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

#
# PseudoMapTrainer on the CARLA road-polyline tile dataset, LiDAR only.
# Tile size: 30 x 30 m. For the 25 x 25 m example export, use
# pmt_carla_lidar_25m.py, which is this config with the five tile-size-derived
# values changed.
#
# This is pmt_single.py with the camera swapped out for LiDAR. The LiDAR path
# is the one proven in the sibling MapTRv2/GeMap codebases
# (Voxelization -> SparseEncoder -> ConvFuser channel projection, with the
# camera BEV encoder bypassed entirely -- see MapTRPerceptionTransformer.
# get_bev_features); everything that makes this PseudoMapTrainer rather than
# plain MapTRv2 is kept exactly as pmt_single.py has it: MaskedMapTRAssigner
# with allow_split, RenderedMaskDiceLoss/RenderedMaskDiceCost, MaskedBCE, the
# renderer sizes, the optimizer and the 24-epoch schedule.
#
# CARLA ships complete map GT and no visibility mask, so PMT's mask is
# synthesised by PMTCarlaMapDataset. This config uses mask_mode='ones' (the
# whole patch observed), which makes every masked operation degenerate to its
# unmasked form -- the like-for-like baseline. pmt_carla_lidar_cov.py switches
# to a mask derived from real LiDAR coverage, where the masked assignment does
# actual work.
#
# Deltas from pmt_single.py, all forced by the change of sensor/dataset:
#   * no image backbone/neck/pretrained weights, no image transforms, no LSS
#     depth branch (and therefore no gt_depth: MapTRv2.forward_train would
#     otherwise call transformer.encoder.get_depth_loss on a None encoder)
#   * aux_seg pv_seg=False -- there is no imagery to rasterize into
#   * one map class ('divider') instead of three
#   * square 100x100 BEV over the square 30 x 30 m tile (0.3 m/cell, the same
#     BEV resolution pmt_single.py uses on nuScenes)
#
# carlasim_map.py already pulls in ../_base_/default_runtime.py; listing it
# here as well makes mmcv reject the config for duplicate base keys.
_base_ = [
    '../carla/carlasim_map.py',
]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# Map / coder range: the square CARLA tile (30 x 30 m, tile_radius 15). GT
# polylines are XY-only, so the z half only has to contain them.
#
# Tile size is the one thing to change for a differently-sized export, and it
# touches five values: this range's xy, lidar_point_cloud_range's xy,
# sparse_shape's x/y, post_center_range, and (if you want to hold the cell
# size) bev_h_/bev_w_ and renderer_H/W. pmt_carla_lidar_25m.py is the same
# config with all of them set for a 25 x 25 m export.
point_cloud_range = [-15.0, -15.0, -30.0, 15.0, 15.0, 20.0]
voxel_size = [0.15, 0.15, 20.0]

# LiDAR branch geometry, deliberately separate from the map range above.
#
# z is [-72, 96] with a z_max=96 early filter on the loaders, matching the
# MapTRv2/GeMap 30m benchmark configs exactly, so all repos voxelize the
# same point set from a shared pkl. This replaces this config's original
# [-98, 92] with no z filtering; that wider span was measured on the 25 m
# export, where 98 town03/town05 overpass tiles dip below -72 (2.4% of
# tiles) -- on the 30 m tile-centre export the upstream repos observe no
# point outside [-72, 96] at all, so nothing is clipped here. If a future
# export breaches it, widen z in ALL sibling repos together (and re-measure
# sparse_shape / lidar_bev_proj.in_channels) rather than diverging again.
lidar_point_cloud_range = [-15.0, -15.0, -72.0, 15.0, 15.0, 96.0]
lidar_voxel_size = [0.1, 0.1, 0.4]

map_classes = ['divider']
num_vec = 50
fixed_ptsnum_per_gt_line = 20  # now only support fixed_pts > 0
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag = True
num_map_classes = len(map_classes)

# Rasterization canvas for RenderedMaskDiceLoss and for the BEV mask, which
# must be exactly (renderer_H, renderer_W) -- the loss reshapes the mask to
# renderer_H * renderer_W. Square, matching the square tile; over 30 m, 128
# gives 0.234 m/cell, the same rendering resolution pmt_single.py uses on
# nuScenes (256 x 128 over 60 x 30 m).
renderer_H = 128
renderer_W = 128

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
# Square BEV for a square tile. pmt_single.py's 200x100 mirrors nuScenes'
# 30 x 60 m patch; using it here would give non-square cells and mismatch the
# dataset's own gt_seg_mask canvas. Over 30 m, 100 x 100 is 0.3 m/cell --
# exactly pmt_single.py's BEV resolution on nuScenes.
bev_h_ = 100
bev_w_ = 100
queue_length = 1  # each sequence contains `queue_length` frames.

aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=False,   # no camera imagery to rasterize into
    seg_classes=1,
    feat_down_sample=32,
    pv_thickness=1,
)

model = dict(
    type='MapTRv2',
    use_grid_mask=True,     # inert without images; kept for diff minimality
    video_test_mode=False,
    modality='lidar',
    # LiDAR BEV encoder. SparseEncoder returns a dense (B, C*D, H, W) tensor,
    # so no pooling/LSS step is needed -- the transformer only channel-projects
    # it. sparse_shape is (x, y, z+1) for the range/voxel size above:
    # 301 = 30 m / 0.1 + 1, 421 = 168 m / 0.4 + 1 -- the same values the
    # sibling GeMap/MapTRv2 30m configs pair with this identical range,
    # voxel size and encoder (same vendored fork, same [x, y, z] axis
    # order).
    lidar_encoder=dict(
        voxelize=dict(
            max_num_points=10,
            point_cloud_range=lidar_point_cloud_range,
            voxel_size=lidar_voxel_size,
            max_voxels=[90000, 120000]),
        backbone=dict(
            type='SparseEncoder',
            # x, y, z only. The points also carry a "strength" channel
            # (BT.709 luma of the per-point rgb), dropped via use_dim=3 in
            # the pipelines below to match the MapTRv2 30m HM benchmark
            # convention (colour-free). This value and use_dim MUST move
            # together -- a mismatch fails at the first sparse conv.
            # sparse_shape and lidar_bev_proj.in_channels do not depend on
            # the input channel width.
            in_channels=3,
            sparse_shape=[301, 301, 421],
            output_channels=128,
            order=('conv', 'norm', 'act'),
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
            encoder_paddings=([0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]),
            block_type='basicblock'),
    ),
    pts_bbox_head=dict(
        type='MapTRv2Head',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_vec_one2one=50,
        num_vec_one2many=0,
        k_one2many=0,
        num_pts_per_vec=fixed_ptsnum_per_pred_line, # one bbox
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        dir_interval=1,
        query_embed_type='instance_pts',
        transform_method='minmax',
        gt_shift_pts_pattern='v2',
        num_classes=num_map_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        weight_mask=False,
        constant_pts_avg_factor=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        aux_seg=aux_seg_cfg,
        transformer=dict(
            type='MapTRPerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            modality='lidar',
            # 3200 = SparseEncoder output_channels (128) x its residual z
            # depth (25). Only the z half of lidar_point_cloud_range moves
            # this number -- tile size changes the spatial dims but not the
            # channel count. This value was MEASURED (not derived -- the z
            # downsample factor is not linear in the input extent) in the
            # sibling GeMap/MapTRv2 repos for exactly this z geometry
            # (z [-72, 96], voxel 0.4) and encoder; it was 3712 when this
            # config's z span was the wider [-98, 92]. Re-measure with a
            # dummy extract_lidar_feat() call if the z range or voxel size
            # ever changes; a wrong value fails loudly as a Conv2d mismatch.
            lidar_bev_proj=dict(
                type='ConvFuser',
                in_channels=[3200],
                out_channels=_dim_),
            # No `encoder`: the LiDAR path never runs a camera BEV encoder,
            # and MapTRPerceptionTransformer now treats it as optional rather
            # than building a dead LSSTransform for DDP to trip over.
            decoder=dict(
                type='MapTRDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DecoupledDetrTransformerDecoderLayer',
                    num_vec=num_vec,
                    num_pts_per_vec=fixed_ptsnum_per_pred_line,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'self_attn', 'norm','cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='MapTRNMSFreeCoder',
            # Tile-sized, mirroring pmt_single.py's relationship to its own
            # point_cloud_range.
            post_center_range=[-20.0, -20.0, -20.0, -20.0, 20.0, 20.0, 20.0, 20.0],
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=num_map_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_pts=dict(type='PtsL1Loss', loss_weight=1.0),
        loss_rendered_mask=dict(
            type='RenderedMaskDiceLoss',
            weight=15.0,
            renderer_H=renderer_H,
            renderer_W=renderer_W,
            sample_weighting_with_mask=False),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.002),
        loss_seg=dict(type='MaskedBCE',
            pos_weight=4.0,
            loss_weight=1.0),),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='MaskedMapTRAssigner',
            allow_split=True,
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost', weight=1.0),
            rendered_mask_cost=dict(type='RenderedMaskDiceCost', weight=10.0),
            pc_range=point_cloud_range))))

dataset_type = 'PMTCarlaMapDataset'
data_root = 'data/carla/'
# None = resolve the LiDAR paths against the absolute data_root recorded in
# the annotation pkl (what lidar_path is relative to). Set explicitly only
# when the tile export lives at a different path than at conversion time.
raw_data_root = None

# GridSamplePoints is not optional here: 18 train tiles hit the converter's
# 5,000,000-point ceiling (median is ~115k), and at that scale mmdet3d's
# legacy Voxelization kernel is both very slow and silently under-reports
# occupied voxels. Its grid matches lidar_voxel_size, so it costs no spatial
# precision the voxelizer would not have taken anyway.
train_pipeline = [
    # load_dim stays 4 (the loader builds the strength column before
    # selecting); use_dim=3 keeps only xyz -- see in_channels=3 above. The
    # loader also recentres the points into the tile-centred frame whenever
    # the pkl records a lidar_recenter_shift (--gt-frame tile_center, the
    # converter's default), keeping points and GT in the same frame.
    dict(type='LoadCarlaPointsFromFile', coord_type='LIDAR',
         load_dim=4, use_dim=3, z_max=96.0),
    dict(type='GridSamplePoints', grid_size=lidar_voxel_size,
         point_cloud_range=lidar_point_cloud_range),
    dict(type='DefaultFormatBundle3D', with_gt=False, with_label=False,
         class_names=map_classes),
    dict(type='CustomCollect3D', keys=['points'])
]

# MultiScaleFlipAug3D with flip=False is required even though it augments
# nothing: MapTRv2.forward_test indexes img_metas[0][0], which needs mmcv's
# test-time nesting.
test_pipeline = [
    dict(type='LoadCarlaPointsFromFile', coord_type='LIDAR',
         # colour-free and z-filtered, matching train_pipeline
         load_dim=4, use_dim=3, z_max=96.0),
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

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        raw_data_root=raw_data_root,
        ann_file=data_root + 'carla_map_infos_train.pkl',
        map_ann_file=data_root + 'carla_map_gt.json',
        pipeline=train_pipeline,
        # PMT mask. 'ones' = whole patch observed, so the masked assigner and
        # masked losses stay wired but degenerate to their unmasked form.
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
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        raw_data_root=raw_data_root,
        ann_file=data_root + 'carla_map_infos_test.pkl',
        map_ann_file=data_root + 'carla_map_gt.json',
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
        samples_per_gpu=1),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        raw_data_root=raw_data_root,
        ann_file=data_root + 'carla_map_infos_test.pkl',
        map_ann_file=data_root + 'carla_map_gt.json',
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
        box_type_3d='LiDAR'),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

# pmt_single.py's optimizer, minus the img_backbone lr_mult (no backbone).
optimizer = dict(
    type='AdamW',
    lr=3e-4,
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=6, pipeline=test_pipeline, metric='chamfer')

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
fp16 = dict(loss_scale=512.)
checkpoint_config = dict(interval=6)
# Load-bearing: on the LiDAR path the head's bev_embedding, positional
# encoding and the transformer's can_bus/cams/level embeddings never receive
# a gradient.
find_unused_parameters = True
