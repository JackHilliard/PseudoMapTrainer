import os
from typing import Any, Dict, Tuple

import mmcv
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from PIL import Image


from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations

from .loading_utils import load_augmented_point_cloud, reduce_LiDAR_beams

import torch
from pyquaternion import Quaternion

@PIPELINES.register_module()
class CustomLoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, padding=True,pad_val=128, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.padding = padding
        self.pad_val = pad_val

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        # img = np.stack(
        #     [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        img_list = [mmcv.imread(name, self.color_type) for name in filename]
        img_shape_list = [img.shape for img in img_list]
        max_h = max([shape[0] for shape in img_shape_list])
        max_w = max([shape[1] for shape in img_shape_list])
        size = (max_h, max_w)
        # import pdb;pdb.set_trace()
        img_list = [mmcv.impad(
                    img, shape=size, pad_val=self.pad_val) for img in img_list]

        img = np.stack(img_list,axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class CustomLoadPointsFromMultiSweeps:
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results["points"]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e6
        if self.pad_empty_sweeps and len(results["sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                # NOTE: seems possible to load frame -11?
                if not self.load_augmented:
                    choices = np.random.choice(
                        len(results["sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    # don't allow to sample the earliest frame, match with Tianwei's implementation.
                    choices = np.random.choice(
                        len(results["sweeps"]) - 1, self.sweeps_num, replace=False
                    )
            for idx in choices:
                sweep = results["sweeps"][idx]
                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                # TODO: make it more general
                if self.reduce_beams and self.reduce_beams < 32:
                    points_sweep = reduce_LiDAR_beams(points_sweep, self.reduce_beams)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results["points"] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"



@PIPELINES.register_module()
class CustomLoadPointsFromFile:
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.

        Args:
            lidar_path (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        mmcv.check_file_exist(lidar_path)
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        lidar_path = results["lidar_path"]
        points = self._load_points(lidar_path)
        points = points.reshape(-1, self.load_dim)
        # TODO: make it more general
        if self.reduce_beams and self.reduce_beams < 32:
            points = reduce_LiDAR_beams(points, self.reduce_beams)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["points"] = points

        return results


@PIPELINES.register_module()
class CustomPointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs = np.stack(results['img'])
        img_aug_matrix  = results['img_aug_matrix']
        post_rots = [torch.tensor(single_aug_matrix[:3, :3]).to(torch.float) for single_aug_matrix in img_aug_matrix]
        post_trans = torch.stack([torch.tensor(single_aug_matrix[:3, 3]).to(torch.float) for single_aug_matrix in img_aug_matrix])
        # import pdb;pdb.set_trace()
        intrins = results['camera_intrinsics']
        depth_map_list = []
        
        for cid in range(len(imgs)):
            # import pdb;pdb.set_trace()
            lidar2lidarego = torch.tensor(results['lidar2ego']).to(torch.float32)
            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(results['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)
            cam2camego = torch.tensor(results['camera2ego'][cid])

            camego2global = results['camego2global'][cid]

            cam2img = torch.tensor(intrins[cid]).to(torch.float32)
            
            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)

            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T.to(torch.float)) + lidar2img[:3, 3].to(torch.float).unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[1],
                                             imgs.shape[2])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        
        ##################################################################
        # global i
        # import cv2
        # for image_id in range(imgs.shape[0]):
        #     i+=1
        #     image = imgs[image_id]
        #     gt_depth_image = depth_map[image_id].numpy()
            
        #     gt_depth_image = np.expand_dims(gt_depth_image,2).repeat(3,2)
            
        #     #apply colormap on deoth image(image must be converted to 8-bit per pixel first)
        #     im_color=cv2.applyColorMap(cv2.convertScaleAbs(gt_depth_image,alpha=15),cv2.COLORMAP_JET)
        #     #convert to mat png
        #     image[gt_depth_image>0] = im_color[gt_depth_image>0]
        #     im=Image.fromarray(np.uint8(image))
        #     #save image
        #     im.save('visualize_1/visualize_{}.png'.format(i))
        #################################################################

        results['gt_depth'] = depth_map
        return results


class EmptyLidarTileError(RuntimeError):
    """A tile has no LiDAR point inside the voxelizer's range.

    Such a tile voxelizes to zero voxels, which crashes
    ``MapTRv2.extract_lidar_feat``. Raised by ``GridSamplePoints`` so that
    ``CustomCarlaLocalMapDataset`` can skip the sample instead (train mode
    only -- see its ``prepare_train_data``). Tiles like this are normally
    already dropped at conversion time; this exists for annotation files
    that predate that check.
    """


@PIPELINES.register_module()
class LoadCarlaPointsFromFile(object):
    """Load CARLA-simulator LiDAR point clouds from an ``.npz`` block.

    Each ``.npz`` block stores a ``features`` array of shape ``(N, 6)``
    (xyz + rgb) and a ``labels`` array. Here we only build the LiDAR point
    cloud: xyz coordinates plus a scalar "strength" derived from the RGB
    channels (ITU-R BT.709 luma), matching the original
    ``strength = rgb @ [0.2126, 0.7152, 0.0722]`` formula.

    Unlike the upstream MapTRv2/GeMap version there is deliberately **no
    ``z_max`` filter**: nothing is dropped on the z axis here. Tiles in this
    dataset legitimately span roughly z in [-90, +90] m (highway overpasses
    inside a 25 x 25 m footprint), and the training config's
    ``lidar_point_cloud_range`` is set wide enough to cover that, so the
    voxelizer keeps those returns too.

    Args:
        coord_type (str): Coordinate frame of the points. One of
            ``'LIDAR'``, ``'DEPTH'``, ``'CAMERA'``. Defaults to ``'LIDAR'``.
        load_dim (int): Number of columns produced before selection
            (``x, y, z, strength``). Defaults to 4.
        use_dim (int | list[int]): Which of those columns to keep. Defaults to
            4 (all of them).
    """

    def __init__(self,
                 coord_type='LIDAR',
                 load_dim=4,
                 use_dim=4):
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        # ITU-R BT.709 luma weights, as in the original Pointcept dataset.
        self._rgb2strength = np.array([0.2126, 0.7152, 0.0722],
                                      dtype=np.float32)

    def _load_points(self, pts_filename):
        mmcv.check_file_exist(pts_filename)
        block = np.load(pts_filename)
        features = block['features']
        coord = features[:, 0:3].astype(np.float32)
        strength = (features[:, 3:6].astype(np.float32)
                    @ self._rgb2strength).reshape([-1, 1])
        return np.concatenate([coord, strength], axis=1)

    def __call__(self, results):
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points[:, self.use_dim]

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=None)
        results['points'] = points
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'coord_type={self.coord_type}, '
                f'load_dim={self.load_dim}, use_dim={self.use_dim})')


@PIPELINES.register_module()
class GridSamplePoints(object):
    """Pointcept-style grid downsampling: keep one representative point per
    occupied (grid_size)^3 cell, via integer coordinate packing + a single
    1D ``torch.unique`` (vectorized, no Python loop over points).

    Some CARLA tiles have raw point counts up to 5,000,000 (a converter
    artifact -- an unbounded number of scan passes merged into one static
    block, unlike AV2/nuScenes' hardware-bounded ~10-sweep aggregation).
    Feeding that directly into the LiDAR voxelizer is both very slow
    (dominates GPU time end to end) and, confirmed empirically upstream
    against a known-ground-truth synthetic point cloud, produces genuinely
    wrong output: mmdet3d's legacy ``Voxelization`` CUDA kernel silently
    under-reports occupied voxels by ~36% at this scale (2000 known
    distinct cells, 5,000,000 points -> only 1,280 reported). Grid sampling
    first collapses the redundant, oversampled raw points to ~1-per-cell
    *before* voxelization, which is both ~26x faster and recovers 100% of
    the occupied voxels on the real worst-case tile, vs. only 8.6%-22.4%
    for naive random subsampling to a similar point budget -- grid sampling
    is density-uniform, not density-proportional, so it does not
    disproportionately thin out sparse regions (e.g. divider lines).

    Args:
        grid_size (float | tuple[float, float, float]): cell size in
            meters. Defaults to (0.1, 0.1, 0.4), matching this project's
            LiDAR ``voxel_size`` -- this collapses raw-point redundancy
            with no *additional* spatial precision loss beyond what the
            model's own voxelizer already imposes.
        point_cloud_range (list[float]): must match the range passed to
            the LiDAR voxelizer downstream (``lidar_point_cloud_range`` in
            the training config) -- used to bound/offset the integer grid
            coordinates and to detect tiles with nothing in range, but not
            to filter points (the voxelizer does that itself).
        min_points (int): raise ``EmptyLidarTileError`` when fewer than
            this many points fall inside ``point_cloud_range``, since the
            voxelizer would then produce zero (or near-zero) voxels and
            crash ``extract_lidar_feat``. Set to 0 to disable the check.
            Defaults to 1.
    """

    def __init__(self,
                 grid_size=(0.1, 0.1, 0.4),
                 point_cloud_range=None,
                 min_points=1):
        if point_cloud_range is None:
            raise ValueError('GridSamplePoints requires point_cloud_range '
                             '(must match the downstream LiDAR voxelizer\'s '
                             'point_cloud_range).')
        if isinstance(grid_size, (int, float)):
            grid_size = (grid_size, grid_size, grid_size)
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.min_points = min_points
        self.dims = [
            int(round((point_cloud_range[3 + i] - point_cloud_range[i])
                      / grid_size[i])) + 1
            for i in range(3)
        ]

    def __call__(self, results):
        points = results['points']
        tensor = points.tensor
        xyz = tensor[:, :3]
        lo = xyz.new_tensor(self.point_cloud_range[:3])
        hi = xyz.new_tensor(self.point_cloud_range[3:])
        if tensor.shape[0] == 0:
            # torch.unique(...).max() below is undefined on an empty tensor,
            # so bail out before it: nothing to downsample either way.
            self._check_not_empty(results, 0, 0)
            return results
        # Counted before the clamp below, which would otherwise pull
        # out-of-range points into edge cells and hide the fact that the
        # voxelizer is about to drop every one of them.
        n_in_range = int(((xyz >= lo) & (xyz < hi)).all(1).sum())
        self._check_not_empty(results, tensor.shape[0], n_in_range)
        gsize = xyz.new_tensor(self.grid_size)
        gcoord = torch.floor((xyz - lo) / gsize).long()
        gcoord[:, 0].clamp_(0, self.dims[0] - 1)
        gcoord[:, 1].clamp_(0, self.dims[1] - 1)
        gcoord[:, 2].clamp_(0, self.dims[2] - 1)
        key = (gcoord[:, 0] * self.dims[1] + gcoord[:, 1]) * self.dims[2] \
            + gcoord[:, 2]

        _, inverse = torch.unique(key, return_inverse=True)
        order = torch.arange(tensor.shape[0], device=tensor.device)
        rep_idx = order.new_full((int(inverse.max()) + 1,), tensor.shape[0])
        rep_idx.scatter_reduce_(0, inverse, order, reduce='amin',
                                include_self=True)

        results['points'] = points[rep_idx]
        return results

    def _check_not_empty(self, results, n_raw, n_in_range):
        if self.min_points <= 0 or n_in_range >= self.min_points:
            return
        raise EmptyLidarTileError(
            f'tile {results.get("sample_idx")} has {n_in_range} of {n_raw} '
            f'point(s) inside point_cloud_range={self.point_cloud_range} '
            f'(need >= {self.min_points}); it would voxelize to zero voxels. '
            'Regenerate the annotation pkl with '
            'custom_tools/maptrv2/custom_carla_map_converter.py to drop tiles '
            'like this up front, or widen the range.')

    def __repr__(self):
        return (f'{self.__class__.__name__}(grid_size={self.grid_size}, '
                f'point_cloud_range={self.point_cloud_range}, '
                f'min_points={self.min_points})')
