"""Smoke subset of pmt_hrdx_16ep_lidar_50m_tile.py: 200 train / 30 val tiles
(hrdx_splits_v1/smoke). Same benchmark, same model; own GT cache and work dir."""
_base_ = ['./pmt_hrdx_16ep_lidar_50m_tile.py']

splits_root = 'data/hrdx_splits_v1/'
data = dict(
    train=dict(ann_file=splits_root + 'smoke/train200.json'),
    val=dict(ann_file=splits_root + 'smoke/val30.json',
             map_ann_file=splits_root + 'smoke/val30_gt_param.json'),
    test=dict(ann_file=splits_root + 'smoke/val30.json',
              map_ann_file=splits_root + 'smoke/val30_gt_param.json'))
# 200 tiles / batch 2 = 100 iterations per epoch: keep the warmup inside the first epochs
lr_config = dict(warmup_iters=100)
work_dir = './work_dirs/pmt_hrdx_16ep_lidar_50m_tile_smoke'
