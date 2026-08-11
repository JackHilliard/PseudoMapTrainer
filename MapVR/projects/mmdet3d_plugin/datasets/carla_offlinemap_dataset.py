"""CARLA road-polyline tile dataset with vectorized map ground truth.

Each sample is one static square LiDAR tile produced by
custom_tools/maptrv2/custom_carla_map_converter.py -- unlike the nuScenes
offline map dataset there is no ego trajectory/temporal queue or camera
imagery to handle, so this subclasses ``Custom3DDataset`` directly rather
than ``CustomNuScenesDataset``. In particular it never goes through
``union2one()``, so ``img_metas`` reaches the detector as a flat per-sample
list (which is what ``MapTRv2.forward_train``'s ``modality == 'lidar'``
branch expects).

Ported from the sibling MapTRv2 codebase. The vectorization machinery
(``VectorizedLocalMap``/``LiDARInstanceLines``) and ``output_to_vecs`` are
reused from this repo's own nuScenes offline map module instead of being
duplicated -- registering a second copy of those classes would collide in
mmcv's registries, and their bodies have no nuScenes dependency.

``PMTCarlaMapDataset`` at the bottom of this file adds PseudoMapTrainer's
BEV visibility mask on top, which is what keeps ``MaskedMapTRAssigner``,
``RenderedMaskDiceLoss`` and ``MaskedBCE`` meaningful on a dataset that
ships complete ground truth.
"""

import json
import os
import tempfile
import warnings
from os import path as osp

import cv2
import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets.custom_3d import Custom3DDataset

from .nuscenes_offlinemap_dataset import (LiDARInstanceLines,
                                          VectorizedLocalMap, output_to_vecs)
from .pipelines.loading import EmptyLidarTileError


@DATASETS.register_module()
class CustomCarlaLocalMapDataset(Custom3DDataset):
    """CARLA simulator dataset with vectorized map (divider) ground truth."""

    CLASSES = None
    MAPCLASSES = ('divider', )

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 raw_data_root=None,
                 map_ann_file=None,
                 bev_size=(200, 200),
                 pc_range=[-12.5, -12.5, -2.0, 12.5, 12.5, 24.0],
                 fixed_ptsnum_per_line=-1,
                 eval_use_same_gt_sample_num_flag=False,
                 padding_value=-10000,
                 map_classes=None,
                 aux_seg=dict(
                     use_aux_seg=False,
                     bev_seg=False,
                     pv_seg=False,
                     seg_classes=1,
                     feat_down_sample=32,
                 ),
                 code_size=2,
                 eval_nproc=8,
                 min_lidar_points=1,
                 lidar_pc_range=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 **kwargs):
        # Must be set before super().__init__, since load_annotations() /
        # vectormap_pipeline() (indirectly, via prepare_train_data) rely on
        # them and the base __init__ calls load_annotations() itself.
        self.map_ann_file = map_ann_file
        # Where the raw tiles live. data_root holds the pkl / GT json, which
        # the dataset the converter was pointed at with --data-root need not.
        # Defaults to data_root for the common case where they coincide.
        self.raw_data_root = raw_data_root if raw_data_root is not None else data_root
        self.code_size = code_size
        self.bev_size = bev_size
        self.MAPCLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.pc_range = pc_range
        patch_h = pc_range[4] - pc_range[1]
        patch_w = pc_range[3] - pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.min_z = pc_range[2]
        self.max_z = pc_range[5]
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.eval_use_same_gt_sample_num_flag = eval_use_same_gt_sample_num_flag
        self.aux_seg = aux_seg
        self.eval_nproc = eval_nproc
        self.min_lidar_points = min_lidar_points
        self.lidar_pc_range = lidar_pc_range
        # Counts consecutive samples skipped by the runtime empty-tile guard
        # in prepare_train_data -- see the comment there.
        self._consecutive_empty_skips = 0
        # This repo's VectorizedLocalMap has no code_size/min_z/max_z args (it
        # is always 2D, matching pred_z_flag=False in every config here);
        # self.code_size is still used to slice GT points in _format_gt().
        self.vector_map = VectorizedLocalMap(
            canvas_size=bev_size,
            patch_size=self.patch_size,
            map_classes=self.MAPCLASSES,
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
            padding_value=self.padding_value,
            aux_seg=aux_seg)
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format='pkl')
        samples = sorted(data['samples'], key=lambda e: e['sample_idx'])
        return self._filter_empty_lidar_tiles(samples,
                                              data.get('lidar_check'))

    def _filter_empty_lidar_tiles(self, samples, lidar_check):
        """Drop tiles whose LiDAR points would voxelize to zero voxels.

        The converter (custom_tools/maptrv2/custom_carla_map_converter.py) already
        drops these and records ``num_lidar_points_in_range`` on every
        sample it keeps; this is the second line of defence, so an
        already-generated pkl (or one converted against a different
        ``lidar_point_cloud_range``) still can't take a run down with the
        ``extract_lidar_feat`` zero-voxel RuntimeError.

        Filtering here rather than in ``__getitem__`` is deliberate: it
        happens before ``Custom3DDataset.__init__`` calls
        ``_set_group_flag()``, so ``self.flag``, ``len(self)``,
        ``format_results()``'s length assert and ``_format_bbox()``'s
        positional ``data_infos[sample_id]`` indexing all stay consistent.
        Skipping at ``__getitem__`` time would only work in train mode
        (which resamples on ``None``) and would silently desynchronise
        eval.
        """
        if lidar_check is not None and self.lidar_pc_range is not None:
            recorded = lidar_check.get('point_cloud_range')
            if recorded is not None and \
                    not np.allclose(recorded, self.lidar_pc_range):
                warnings.warn(
                    f'{self.__class__.__name__}: the annotation file\'s '
                    f'point counts were measured against '
                    f'lidar_point_cloud_range={list(recorded)}, but this '
                    f'config uses {list(self.lidar_pc_range)}. The recorded '
                    'num_lidar_points_in_range values do not describe what '
                    'this run will voxelize -- regenerate the pkl with a '
                    'matching --lidar-point-cloud-range.')

        if not any(
                s.get('num_lidar_points_in_range') is not None
                for s in samples):
            warnings.warn(
                f'{self.__class__.__name__}: no sample in this annotation '
                'file records num_lidar_points_in_range, so empty (zero-'
                'voxel) tiles cannot be filtered out. Regenerate it with '
                'python custom_tools/maptrv2/custom_carla_map_converter.py '
                '--data-root <path> --out-dir data/carla/ --split <split>')
            return samples

        kept, dropped = [], []
        for s in samples:
            n = s.get('num_lidar_points_in_range')
            if n is not None and n < self.min_lidar_points:
                dropped.append(s)
            else:
                kept.append(s)

        if dropped:
            names = ', '.join(s['sample_idx'] for s in dropped[:10])
            if len(dropped) > 10:
                names += f', ... (+{len(dropped) - 10} more)'
            print_log(
                f'{self.__class__.__name__}: dropped {len(dropped)} of '
                f'{len(samples)} tiles with fewer than '
                f'{self.min_lidar_points} in-range LiDAR point(s): {names}',
                logger='current')
        return kept

    @classmethod
    def get_map_classes(cls, map_classes=None):
        if map_classes is None:
            return cls.MAPCLASSES
        if isinstance(map_classes, str):
            return mmcv.list_from_file(map_classes)
        elif isinstance(map_classes, (tuple, list)):
            return map_classes
        raise ValueError(
            f'Unsupported type {type(map_classes)} of map classes.')

    def get_data_info(self, index):
        info = self.data_infos[index]
        # lidar_path is stored relative to raw_data_root (see the converter's
        # --data-root) so the pkl stays valid across containers and mounts
        # instead of baking in an absolute path from wherever conversion ran.
        # os.path.join is a no-op on an absolute path, so pkls from the
        # sibling MapTRv2 repo (which stored absolute paths) still load.
        return dict(
            pts_filename=osp.join(self.raw_data_root, info['lidar_path']),
            sample_idx=info['sample_idx'],
            timestamp=info.get('timestamp', index),
            # Static tiles have no ego motion / temporal chain; these are
            # placeholders so unconditional img_metas reads elsewhere in the
            # detector don't KeyError (video_test_mode=False makes them
            # otherwise inert).
            scene_token=info['sample_idx'],
            can_bus=np.zeros(18, dtype=np.float32),
            annotation=info['annotation'],
            ann_info=info['annotation'],
        )

    @staticmethod
    def _annotation_2d(annotation):
        """Drop the z column from every GT polyline.

        The converter stores polylines as (N, 3); this repo's
        VectorizedLocalMap is strictly 2D (unlike the AV2 variant in the
        sibling MapTRv2 repo, which carries code_size/min_z/max_z). Handing
        it 3D lines does not raise -- shapely's interpolate() returns
        3-tuples, and LiDARInstanceLines' `.reshape(-1, 2)` then silently
        reinterprets 20 xyz points as 30 garbage xy points, which surfaces
        far away as a tensor-size mismatch (or, worse, as plausible-looking
        nonsense GT). Slice z off here, at the dataset boundary.
        """
        return {
            cls_name: [np.asarray(line)[:, :2] for line in lines]
            for cls_name, lines in annotation.items()
        }

    def vectormap_pipeline(self, example, input_dict):
        annotation = (input_dict['annotation'] if 'annotation' in input_dict
                      else input_dict['ann_info'])
        anns_results = self.vector_map.gen_vectorized_samples(
            self._annotation_2d(annotation),
            example=example,
            feat_down_sample=self.aux_seg['feat_down_sample'])

        gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
        if isinstance(anns_results['gt_vecs_pts_loc'], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
        else:
            gt_vecs_pts_loc = to_tensor(anns_results['gt_vecs_pts_loc'])
            try:
                gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(
                    dtype=torch.float32)
            except Exception:
                # Empty tensor -- passed through untouched (train filters
                # this sample out via filter_empty_gt; test path keeps it).
                gt_vecs_pts_loc = gt_vecs_pts_loc
        example['gt_labels_3d'] = DC(gt_vecs_label, cpu_only=False)
        example['gt_bboxes_3d'] = DC(gt_vecs_pts_loc, cpu_only=True)
        if anns_results['gt_semantic_mask'] is not None:
            example['gt_seg_mask'] = DC(
                to_tensor(anns_results['gt_semantic_mask']), cpu_only=False)
        if anns_results['gt_pv_semantic_mask'] is not None:
            example['gt_pv_seg_mask'] = DC(
                to_tensor(anns_results['gt_pv_semantic_mask']),
                cpu_only=False)
        return example

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        try:
            example = self.pipeline(input_dict)
        except EmptyLidarTileError as e:
            # Last resort for a pkl that predates the converter's own check
            # (or one converted against a different range). Returning None
            # makes Custom3DDataset.__getitem__ resample another index --
            # but if the *config's* range is wrong then every tile is empty
            # and that would spin forever, so give up after a run of them.
            self._consecutive_empty_skips += 1
            if self._consecutive_empty_skips > 100:
                raise
            warnings.warn(f'{self.__class__.__name__}: skipping sample '
                          f'{input_dict["sample_idx"]} -- {e}')
            return None
        self._consecutive_empty_skips = 0
        example = self.vectormap_pipeline(example, input_dict)
        if self.filter_empty_gt and \
                (example is None or
                 ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        return self.pipeline(input_dict)

    def _format_gt(self):
        gt_annos = []
        print('Start to convert gt map format...')
        assert self.map_ann_file is not None
        if not os.path.exists(self.map_ann_file):
            dataset_length = len(self)
            prog_bar = mmcv.ProgressBar(dataset_length)
            mapped_class_names = self.MAPCLASSES
            for sample_id in range(dataset_length):
                sample_token = self.data_infos[sample_id]['sample_idx']
                gt_sample_dict = self.vectormap_pipeline(
                    {}, self.data_infos[sample_id])
                gt_labels = gt_sample_dict['gt_labels_3d'].data.numpy()
                gt_vecs = gt_sample_dict['gt_bboxes_3d'].data.instance_list
                gt_vec_list = []
                for gt_label, gt_vec in zip(gt_labels, gt_vecs):
                    name = mapped_class_names[gt_label]
                    gt_vec_list.append(
                        dict(
                            pts=np.array(list(
                                gt_vec.coords))[:, :self.code_size],
                            pts_num=len(list(gt_vec.coords)),
                            cls_name=name,
                            type=int(gt_label),
                        ))
                gt_annos.append(
                    dict(sample_token=sample_token, vectors=gt_vec_list))
                prog_bar.update()
            print('\n GT anns writes to', self.map_ann_file)
            mmcv.dump(dict(GTs=gt_annos), self.map_ann_file)
        else:
            print(f'{self.map_ann_file} exist, not update')

    def _format_bbox(self, results, jsonfile_prefix=None):
        assert self.map_ann_file is not None
        pred_annos = []
        mapped_class_names = self.MAPCLASSES
        print('Start to convert map detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            vecs = output_to_vecs(det)
            sample_token = self.data_infos[sample_id]['sample_idx']
            pred_vec_list = []
            for vec in vecs:
                pred_vec_list.append(
                    dict(
                        pts=vec['pts'],
                        pts_num=len(vec['pts']),
                        cls_name=mapped_class_names[vec['label']],
                        type=int(vec['label']),
                        confidence_level=vec['score']))
            pred_annos.append(
                dict(sample_token=sample_token, vectors=pred_vec_list))

        if not os.path.exists(self.map_ann_file):
            self._format_gt()
        else:
            print(f'{self.map_ann_file} exist, not update')

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'carlamap_results.json')
        print('Results writes to', res_path)
        mmcv.dump(dict(meta=self.modality, results=pred_annos), res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None):
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='chamfer',
                         result_name='pts_bbox'):
        from projects.mmdet3d_plugin.datasets.map_utils.mean_ap import (
            eval_map, format_res_gt_by_classes)
        result_path = osp.abspath(result_path)
        detail = dict()

        print('Formating results & gts by classes')
        with open(result_path, 'r') as f:
            pred_results = json.load(f)
        gen_results = pred_results['results']
        with open(self.map_ann_file, 'r') as ann_f:
            gt_anns = json.load(ann_f)
        annotations = gt_anns['GTs']
        cls_gens, cls_gts = format_res_gt_by_classes(
            result_path,
            gen_results,
            annotations,
            cls_names=self.MAPCLASSES,
            num_pred_pts_per_instance=self.fixed_num,
            eval_use_same_gt_sample_num_flag=self.
            eval_use_same_gt_sample_num_flag,
            pc_range=self.pc_range,
            nproc=self.eval_nproc)

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['chamfer', 'iou']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        for metric in metrics:
            if metric == 'chamfer':
                thresholds = [0.5, 1.0, 1.5]
            elif metric == 'iou':
                thresholds = np.linspace(
                    .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1,
                    endpoint=True)
            cls_aps = np.zeros((len(thresholds), self.NUM_MAPCLASSES))

            for i, thr in enumerate(thresholds):
                _, cls_ap = eval_map(
                    gen_results,
                    annotations,
                    cls_gens,
                    cls_gts,
                    threshold=thr,
                    cls_names=self.MAPCLASSES,
                    logger=logger,
                    num_pred_pts_per_instance=self.fixed_num,
                    pc_range=self.pc_range,
                    metric=metric,
                    nproc=self.eval_nproc)
                for j in range(self.NUM_MAPCLASSES):
                    cls_aps[i, j] = cls_ap[j]['ap']

            for i, name in enumerate(self.MAPCLASSES):
                print('{}: {}'.format(name, cls_aps.mean(0)[i]))
                detail[f'CarlaMap_{metric}/{name}_AP'] = cls_aps.mean(0)[i]
            print('map: {}'.format(cls_aps.mean(0).mean()))
            detail[f'CarlaMap_{metric}/mAP'] = cls_aps.mean(0).mean()

            for i, name in enumerate(self.MAPCLASSES):
                for j, thr in enumerate(thresholds):
                    if metric == 'chamfer':
                        detail[f'CarlaMap_{metric}/{name}_AP_thr_{thr}'] = \
                            cls_aps[j][i]

        return detail

    def evaluate(self,
                results,
                metric='chamfer',
                logger=None,
                jsonfile_prefix=None,
                result_names=['pts_bbox'],
                show=False,
                out_dir=None,
                pipeline=None):
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(
                    result_files[name], metric=metric)
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files, metric=metric)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return results_dict


@DATASETS.register_module()
class PMTCarlaMapDataset(CustomCarlaLocalMapDataset):
    """CARLA tiles plus PseudoMapTrainer's BEV visibility mask.

    PMT's contributions -- ``MaskedMapTRAssigner`` (with ``allow_split``),
    ``RenderedMaskDiceLoss``/``RenderedMaskDiceCost`` and ``MaskedBCE`` -- all
    consume a per-sample BEV mask marking which part of the patch is actually
    observed. On nuScenes that mask comes from RoGS (``bev_mask_post.png``);
    CARLA ships complete ground truth and no mask, so it is synthesised here:

    ``mask_mode='ones'``
        Everything observed. Every masked operation degenerates to its
        unmasked form, which makes this the like-for-like baseline against
        which the coverage mask below can be judged.
    ``mask_mode='lidar_coverage'``
        Observed = where the tile's LiDAR actually returned points. PMT's
        masked assignment then does real work: GT polyline segments crossing
        unobserved ground stop being charged against predictions, and
        ``allow_split`` lets several predictions cover one GT line across a
        coverage gap.

    The mask is built *after* the transform pipeline has run, so it reflects
    the same (grid-sampled) points the model is given.

    Args:
        raster (list[int]): ``[renderer_W, renderer_H]``. Must match the
            ``renderer_W``/``renderer_H`` of ``RenderedMaskDiceLoss``, which
            reshapes the mask to ``renderer_H * renderer_W``.
        mask_mode (str): ``'ones'`` or ``'lidar_coverage'``.
        mask_thresh (float): drop samples whose mask covers less than this
            fraction of the patch (PMT's low-information sample filter).
            Meaningless for ``'ones'``; leave at 0.
        min_points_per_cell (int): a cell counts as observed once this many
            points fall in it. Higher values erode thinly-scanned area.
        close_kernel (int): size of the morphological closing applied to the
            raw occupancy, to bridge the gaps between individual returns.
            Set to 0 to disable.
    """

    MASK_MODES = ('ones', 'lidar_coverage')

    def __init__(self,
                 raster,
                 mask_mode='ones',
                 mask_thresh=0.0,
                 min_points_per_cell=1,
                 close_kernel=3,
                 *args,
                 **kwargs):
        if mask_mode not in self.MASK_MODES:
            raise ValueError(
                f'mask_mode must be one of {self.MASK_MODES}, got {mask_mode!r}')
        self.raster_W, self.raster_H = raster
        self.mask_mode = mask_mode
        self.mask_thresh = mask_thresh
        self.min_points_per_cell = min_points_per_cell
        self.close_kernel = close_kernel
        self._n_dropped_by_mask = 0
        super().__init__(*args, **kwargs)

    def _lidar_coverage_mask(self, points):
        """BEV occupancy of ``points`` over the patch, as (raster_H, raster_W).

        Row index is y and column index is x, matching how the assigner reads
        the mask (``gt_bev_mask.T[x, y]`` after ``denormalize_2d_pts`` against
        the mask's own shape) and how ``normalize_2d_pts`` maps GT coordinates.

        ``points`` is whatever the pipeline left in ``example['points']``:
        DefaultFormatBundle3D replaces the BasePoints object with its raw
        tensor, so accept either.
        """
        tensor = points.tensor if hasattr(points, 'tensor') else points
        xy = tensor[:, :2].numpy()
        x0, y0, x1, y1 = (self.pc_range[0], self.pc_range[1],
                          self.pc_range[3], self.pc_range[4])
        col = np.floor((xy[:, 0] - x0) / (x1 - x0) * self.raster_W)
        row = np.floor((xy[:, 1] - y0) / (y1 - y0) * self.raster_H)
        inside = ((col >= 0) & (col < self.raster_W)
                  & (row >= 0) & (row < self.raster_H))
        counts = np.zeros((self.raster_H, self.raster_W), dtype=np.int32)
        np.add.at(counts,
                  (row[inside].astype(np.int64), col[inside].astype(np.int64)),
                  1)
        mask = counts >= self.min_points_per_cell
        if self.close_kernel and self.close_kernel > 1:
            kernel = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                    kernel).astype(bool)
        return mask

    def vectormap_pipeline(self, example, input_dict):
        example = super().vectormap_pipeline(example, input_dict)

        if self.mask_mode == 'ones' or 'points' not in example:
            # 'points' is absent when _format_gt() calls this with an empty
            # example purely to rasterize GT; an all-ones mask is also the
            # right answer there, since nothing is being masked.
            mask = np.ones((self.raster_H, self.raster_W), dtype=bool)
        else:
            mask = self._lidar_coverage_mask(example['points'].data)
            if mask.mean() < self.mask_thresh:
                self._n_dropped_by_mask += 1
                return None

        # float32, not bool: the head resizes this mask to the BEV grid with
        # F.interpolate(mode='nearest'), which has no bool kernel, and the
        # rendered-mask loss uses it as a per-pixel weight. Values stay 0/1 so
        # mask.mean() keeps its "fraction observed" meaning either way.
        example['bev_mask'] = to_tensor(mask.astype(np.float32))
        # bev_label is plumbed from the detector into the head alongside
        # bev_mask but never read by any loss in this codebase; emit zeros
        # rather than inventing semantics for it.
        example['bev_label'] = to_tensor(
            np.zeros((self.raster_H, self.raster_W), dtype=np.uint8))
        return example

