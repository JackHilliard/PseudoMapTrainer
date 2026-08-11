_base_ = [
    '../_base_/default_runtime.py'
]
#
# CARLA simulator LiDAR + vectorized map-GT data config.
#
# Wires up `PMTCarlaMapDataset` (CustomCarlaLocalMapDataset + PseudoMapTrainer's
# BEV visibility mask) against the pkl produced by
# custom_tools/maptrv2/custom_carla_map_converter.py.
#
# Generate the pkls first, one per split:
#   python custom_tools/maptrv2/custom_carla_map_converter.py \
#       --data-root /path/to/carla --out-dir data/carla/ --split train
#   python custom_tools/maptrv2/custom_carla_map_converter.py \
#       --data-root /path/to/carla --out-dir data/carla/ --split test
#
# Regenerating is not optional if you were handed pkls built by the sibling
# MapTRv2 repo before its GT-frame fix: those place polylines in the
# `tile_center` frame rather than each block's own `offset`, which misaligns
# every GT line against its own point cloud by a median 0.19 m (vs 0.045 m
# after the fix) -- material against chamfer thresholds of 0.5/1.0/1.5 m.
# They also store absolute container paths in `lidar_path`. A pkl predating
# the fix is recognisable by its samples having no `annotation_origin` key.
# Delete carla_map_gt.json whenever you regenerate: _format_gt() skips
# regeneration if it exists, so eval would silently score against stale GT.
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

dataset_type = 'PMTCarlaMapDataset'
data_root = 'data/carla/'
# Where the tiles themselves live. Kept separate from data_root (which holds
# the pkls/GT json) and joined against each sample's relative lidar_path, so
# the same pkl works wherever the raw dataset is mounted.
raw_data_root = 'data/carla/'

ann_file_train = data_root + 'carla_map_infos_train.pkl'
ann_file_val = data_root + 'carla_map_infos_test.pkl'
ann_file_test = data_root + 'carla_map_infos_test.pkl'
map_ann_file = data_root + 'carla_map_gt.json'

# Matches the square CARLA tile (tile_radius=15 -> 30m x 30m). The z half
# only needs to contain the map GT, which is XY-only here; the LiDAR branch
# has its own, much wider `lidar_point_cloud_range` (see pmt_carla_lidar.py).
#
# This is only a default: every trainable config sets its own geometry, so
# changing tile size means editing that config (pmt_carla_lidar.py for
# 30 x 30 m, pmt_carla_lidar_25m.py for 25 x 25 m), not this file. The
# converter derives its own xy range from the manifest's tile_radius and
# records it in the pkl, so the dataset warns when the two disagree.
point_cloud_range = [-15.0, -15.0, -30.0, 15.0, 15.0, 20.0]
map_classes = ['divider']

# LiDAR points are [x, y, z, strength].
load_dim = 4
use_dim = 4

# NOTE: there is deliberately no `z_max` here. The sibling MapTRv2/GeMap
# configs filter points above a z ceiling inside LoadCarlaPointsFromFile;
# this repo does not filter on z at all, so the only thing that can drop a
# point on the z axis is `lidar_point_cloud_range`, which pmt_carla_lidar.py
# sets wide enough to cover the whole measured extent.

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
