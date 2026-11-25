"""PseudoMapTrainer (PMT) baseline on the CARLA 3-class whole-50 m-tile benchmark
(carla_3cls_50m).

The HRDX config (pmt_hrdx_16ep_lidar_50m_tile.py) value for value -- model, losses,
matching, optimizer, schedule, augmentation, eval -- with the CARLA data in place of the HRDX
cities; the data part is the MapTRv2 repo's maptrv2_carla3cls_16ep_lidar_50m_tile.py, value for
value. THE BENCHMARK (identical in every sister baseline: MapTR, GeMap, PseudoMapTrainer,
mapdiffusion, maptracker, SuperMapNet, CurbNet -- change it in all of them or in none):

  data      data/carla_3cls_50m/<Town>/sweep/tiles_50m/{blocks,reference_lines}
            splits data/carla_splits_v1/{train,val,test,test_towns}.json
            (18,522 / 2,236 / 2,309 / 688 tiles; test = Town12/13 1 km blocks,
            test_towns = Town01-05 held out whole), smoke/{train200,val30,test30}.json
  sample    the WHOLE 50 m tile, +-25 m, no crop, no translation
  GT        three classes: centerline, curb_model, road_edge; the straight/arc pieces of one
            class along one road joined (carla_chain_gt.py), clipped to the +-25 m box, pieces
            shorter than 2 m dropped
  input     (x, y, z, strength): z re-zeroed on the tile's 2nd-percentile ground, strength = 0
            (CARLA has no intensity; a zero channel leaves the HRDX model unchanged);
            grid-sampled at 0.1 x 0.1 x 0.4 m; shuffled
  train aug as HRDX: rotate z U(-pi, pi) p 0.5, scale U(0.9, 1.1), flip x / y p 0.5,
            jitter sigma 0.005 clip 0.02; tiles with < 5000 in-range points or no GT resampled
  val/test  never augmented; evaluated every 2 epochs on val, chamfer AP per class
  schedule  16 epochs, AdamW (weight decay 0.01), cosine annealing with a 4000-iteration linear
            warmup, batch 2 x 8 accumulated iterations (effective 16), fp16 with a fixed loss
            scale, gradient clipping at 35, checkpoint every epoch (keep 4)

PMT specifics: as the HRDX config (MapTRv2Head without one-to-many, RenderedMaskDiceLoss on a
128 x 128 raster, MaskedBCE aux seg, MaskedMapTRAssigner with the rendered-mask cost,
mask_mode='ones', lr 3e-4). One change the class count forces: the rendered-mask loss and
cost render the labels in ``polygon_class_labels`` as filled polygons, default (1,) = nuScenes'
ped_crossing. HRDX (one class, label 0) never hits it; here label 1 is curb_model, a line, so
both get polygon_class_labels=[] (every class rendered as a line, as HRDX's boundary is).

Three classes: num_classes = aux_seg.seg_classes = 3 (one BEV mask per class); max_num 50 <=
num_vec x 3. GT caches (map_ann_file) are written at the first evaluation and checked against
the dataset at every later construction.
"""
custom_imports = dict(
    imports=['projects.mmdet3d_plugin.datasets.sanjose50m_crop_dataset',
             'projects.mmdet3d_plugin.datasets.hrdx50m_dataset',
             'projects.mmdet3d_plugin.datasets.carla3cls50m_dataset'],
    allow_failed_imports=False)

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# ---- data source -----------------------------------------------------------
dataset_type = 'PMTCarla3cls50mDataset'
data_root = 'data/carla_3cls_50m/'
splits_root = 'data/carla_splits_v1/'
ann_file_train = splits_root + 'train.json'
ann_file_val = splits_root + 'val.json'
ann_file_test = splits_root + 'test.json'
ann_file_test_towns = splits_root + 'test_towns.json'
map_ann_file_val = splits_root + 'val_gt_3cls.json'
map_ann_file_test = splits_root + 'test_gt_3cls.json'
map_ann_file_test_towns = splits_root + 'test_towns_gt_3cls.json'   # tools/test.py --cfg-options data.test.ann_file=... data.test.map_ann_file=...

# ---- taxonomy / geometry ---------------------------------------------------
map_classes = ['centerline', 'curb_model', 'road_edge']
num_map_classes = len(map_classes)

point_cloud_range = [-25.0, -25.0, -30.0, 25.0, 25.0, 20.0]          # BEV / head range
lidar_point_cloud_range = [-25.0, -25.0, -72.0, 25.0, 25.0, 96.0]    # voxelisation range
lidar_voxel_size = [0.1, 0.1, 0.4]
voxel_size = [0.15, 0.15, 20.0]
load_dim = 4
use_dim = 4

num_vec = 50
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag = True

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 200
bev_w_ = 200
queue_length = 1

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=False,
    seg_classes=num_map_classes,
    feat_down_sample=32,
    pv_thickness=1)

# ---- augmentation (train only) ---------------------------------------------
carla_aug = dict(rotate_p=0.5, rotate_angle=(-1.0, 1.0), scale=(0.9, 1.1), flip_p=0.5,
                jitter_sigma=0.005, jitter_clip=0.02)

# ---- pipelines -------------------------------------------------------------
train_pipeline = [
    dict(type='LoadCarla3cls50mTile', coord_type='LIDAR', load_dim=load_dim, use_dim=use_dim,
         strength='zero'),
    dict(
        type='GridSamplePoints',
        grid_size=lidar_voxel_size,
        point_cloud_range=lidar_point_cloud_range,
        min_points=5000),
    dict(type='PointShuffle'),
    dict(
        type='DefaultFormatBundle3D',
        with_gt=False,
        with_label=False,
        class_names=map_classes),
    dict(type='CustomCollect3D', keys=['points'])
]
test_pipeline = [
    dict(type='LoadCarla3cls50mTile', coord_type='LIDAR', load_dim=load_dim, use_dim=use_dim,
         strength='zero'),
    dict(
        type='GridSamplePoints',
        grid_size=lidar_voxel_size,
        point_cloud_range=lidar_point_cloud_range),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1, 1),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                with_gt=False,
                with_label=False,
                class_names=map_classes),
            dict(type='CustomCollect3D', keys=['points'])
        ])
]

carla_common = dict(
    type=dataset_type,
    data_root=data_root,
    classes=[],
    map_classes=map_classes,
    pc_range=point_cloud_range,
    modality=input_modality,
    # PMT: BEV visibility mask raster (== the rendered-mask loss size); 'ones' = all observed
    raster=[128, 128],
    padding_value=-10000,
    aux_seg=aux_seg_cfg,
    box_type_3d='LiDAR',
    fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
    eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
    bev_size=(bev_h_, bev_w_),
    min_lidar_points=1,
    lidar_pc_range=lidar_point_cloud_range,
    split='',
    require_gt=False,
    gt_tries=32,
    min_polyline_len=2.0)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6,
    train=dict(
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        filter_empty_gt=True,
        test_mode=False,
        mask_mode='ones',
        mask_thresh=0.0,
        aug=carla_aug,
        **carla_common),
    val=dict(
        samples_per_gpu=1,
        ann_file=ann_file_val,
        map_ann_file=map_ann_file_val,
        pipeline=test_pipeline,
        test_mode=True,
        crop_seed=0,
        **carla_common),
    test=dict(
        ann_file=ann_file_test,
        map_ann_file=map_ann_file_test,
        pipeline=test_pipeline,
        test_mode=True,
        crop_seed=0,
        **carla_common),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

# ---- model (== pmt_hrdx_16ep_lidar_50m_tile.py) ------------------
renderer_H = 128
renderer_W = 128
model = dict(
    type='MapTRv2',
    use_grid_mask=True,
    video_test_mode=False,
    modality='lidar',
    lidar_encoder=dict(
        voxelize=dict(
            max_num_points=10,
            point_cloud_range=lidar_point_cloud_range,
            voxel_size=lidar_voxel_size,
            max_voxels=[700000, 1000000]),
        backbone=dict(
            type='SparseEncoder',
            in_channels=use_dim,
            sparse_shape=[501, 501, 421],
            output_channels=128,
            order=('conv', 'norm', 'act'),
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128),
                              (128, 128)),
            encoder_paddings=([0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]),
            block_type='basicblock')),
    pts_bbox_head=dict(
        type='MapTRv2Head',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_vec_one2one=num_vec,
        num_vec_one2many=0,
        k_one2many=0,
        num_pts_per_vec=fixed_ptsnum_per_pred_line,
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
            lidar_bev_proj=dict(
                type='ConvFuser', in_channels=[3200], out_channels=_dim_),
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
                            num_levels=_num_levels_)
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'self_attn', 'norm',
                                     'cross_attn', 'norm', 'ffn', 'norm')))),
        bbox_coder=dict(
            type='MapTRNMSFreeCoder',
            post_center_range=[
                -30.0, -30.0, -30.0, -30.0, 30.0, 30.0, 30.0, 30.0
            ],
            pc_range=point_cloud_range,
            max_num=num_vec,
            voxel_size=voxel_size,
            num_classes=num_map_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_),
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
            sample_weighting_with_mask=False,
            # all three CARLA classes are lines; the default (1,) is nuScenes' ped_crossing and
            # would render label 1 (curb_model) as a filled polygon
            polygon_class_labels=[]),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.002),
        loss_seg=dict(type='MaskedBCE', pos_weight=4.0, loss_weight=1.0)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='MaskedMapTRAssigner',
                allow_split=True,
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(
                    type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                pts_cost=dict(type='OrderedPtsL1Cost', weight=1.0),
                rendered_mask_cost=dict(
                    type='RenderedMaskDiceCost', weight=10.0,
                    polygon_class_labels=[]),       # see loss_rendered_mask
                pc_range=point_cloud_range))))

# ---- optimisation / schedule (== pmt_hrdx_16ep_lidar_50m_tile.py) --------------------------
file_client_args = dict(backend='disk')
optimizer = dict(type='AdamW', lr=3e-4, weight_decay=0.01)   # lr: PMT's own
optimizer_config = dict(
    type='GradientCumulativeFp16OptimizerHook',
    cumulative_iters=8,
    loss_scale=512.0,
    grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=4000,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 16
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

evaluation = dict(interval=2, pipeline=test_pipeline, metric='chamfer',
                  save_best='CarlaMap_chamfer/mAP', rule='greater')
checkpoint_config = dict(interval=1, max_keep_ckpts=4)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/pmt_carla3cls_16ep_lidar_50m_tile'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
gpu_ids = range(0, 1)
