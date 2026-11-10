"""HRDX 50 m tiles (San Jose, San Francisco, San Mateo) as WHOLE tiles, with ground truth
sampled on the parametric design equations and the point-cloud augmentation of Pointcept's
configs/nuscenes/semseg-pt-v3m1-0-base.py.

This is the HRDX benchmark of the MapTR_hrdx reference repo, ported so that every sister
baseline (MapTR, GeMap, PseudoMapTrainer, mapdiffusion, SuperMapNet) trains and evaluates on
exactly the same samples. The numbers below are the benchmark's definition; keep them equal in
all five repos.

* Layout. ``data_root`` holds one directory per city (``<city>_tiles_50m_lines/{blocks,
  reference_lines}``); each manifest tile names its ``city`` (split manifests
  ``hrdx_splits_v1/{train,val,test}.json``: 20,499 / 2,227 / 3,392 tiles). Every city has its
  OWN local frame, so a tile is only ever read in its city's frame (tile-local = world - tile
  centre).
* Ground truth. ``gt_source='parametric'`` (default): one instance per continuous in-tile
  stretch of a design alignment, its vertices the equation samples (hrdx_param_gt.py);
  ``gt_source='hand'``: the hand-drawn 'curb' polylines (for reference evaluations).
* No crops. The whole 50 m tile is the sample (crop_size=50, no translation), so the model
  ranges are +-25 m.
* Input. ``(x, y, z, strength)``: xy tile-local, z re-zeroed on the tile's own 2nd-percentile
  ground, strength = the uint8 LiDAR ``intensity`` / 255.
* Augmentation (train only, ``aug`` dict; val/test are never augmented). Pointcept's active
  transforms, in its order:
      RandomRotate(angle=[-1, 1] * pi, axis z, about the tile centre, p=0.5)
      RandomScale([0.9, 1.1])          isotropic, x, y and z
      RandomFlip(p=0.5)                x and y flipped independently, each with p=0.5
      RandomJitter(sigma=0.005, clip=0.02)   points only
  The rotation, scale and flips form ONE 2x2 map A applied to the points (in the loader) and to
  the GT polylines (here) through the inherited pose (crop_R = A), then both are clipped to the
  +-25 m box (a rotated tile leaves its corners outside the box and empty corners inside it,
  exactly what the rotation does to a nuScenes sweep). z is scaled by the same factor; jitter
  is applied to the points only. If an augmented tile would hold no GT, the draw is repeated
  (up to ``aug_tries``), then the tile is used unaugmented.
* Evaluation. val/test tiles are frozen at load time (identity pose), so the GT cache
  (``map_ann_file``) is a pure function of the split manifest and ``gt_source``. An existing
  cache is compared with this dataset's own GT at construction and refused when it differs
  (predictions are paired with the cache by position); it is written atomically.
"""
from __future__ import annotations

import json
import os

import numpy as np
from mmdet.datasets import DATASETS
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import get_points_type

from .carla50m_crop_dataset import LoadCarla50mCrop
from .hrdx_param_gt import chain_parametric
from .pipelines.loading import EmptyLidarTileError
from .sanjose50m_crop_dataset import PMTSanJose50mCropDataset

CITY_DIRS = {'san_jose': 'san_jose_tiles_50m_lines',
             'san_francisco': 'san_francisco_tiles_50m_lines',
             'san_mateo': 'san_mateo_tiles_50m_lines'}

PTV3_NUSCENES_AUG = dict(rotate_p=0.5, rotate_angle=(-1.0, 1.0), scale=(0.9, 1.1),
                         flip_p=0.5, jitter_sigma=0.005, jitter_clip=0.02)

# fixed by the benchmark (whole tiles); a config that passes one of these is a mistake
_FIXED_KWARGS = ('crop_size', 'rotate', 'translate', 'noise_std', 'noise_xy_only', 'gt_classes',
                 'check_finalize', 'reference_dir')


@DATASETS.register_module()
class PMTHrdx50mDataset(PMTSanJose50mCropDataset):
    """Whole HRDX tiles, parametric (or hand-drawn) curb GT, PTv3-nuScenes augmentation,
    in PMT's data contract (``raster`` / ``mask_mode`` pass through to ``PMTCarlaMapDataset``).

    Args (beyond ``PMTSanJose50mCropDataset``'s):
        data_root (str): directory holding the ``<city>_tiles_50m_lines`` directories.
        ann_file (str): a split manifest whose tiles carry ``name``, ``city`` and ``center``.
        gt_source (str): ``'parametric'`` (default) or ``'hand'``.
        aug (dict | None): train augmentation, merged over ``PTV3_NUSCENES_AUG``; ``None``
            (val/test) disables it.
        aug_tries (int): redraws when an augmented tile holds no GT.
        min_crop_points (int | 'manifest'): train-only floor on the raw points inside the
            +-25 m box (0 = off, the benchmark default; ``GridSamplePoints(min_points=...)``
            in the train pipeline is the benchmark's sparse-tile guard).
    """

    def __init__(self, data_root, ann_file, gt_source='parametric', aug=None, aug_tries=8,
                 min_polyline_len=2.0, min_crop_points=0, **kwargs):
        if gt_source not in ('parametric', 'hand'):
            raise ValueError(f"gt_source must be 'parametric' or 'hand', got {gt_source!r}")
        # read by load_annotations(), which the base __init__ calls
        self.gt_source = gt_source
        self.aug = None if aug is None else {**PTV3_NUSCENES_AUG, **dict(aug)}
        self.aug_tries = int(aug_tries)
        self.hrdx_root = data_root
        self._gt_path = {}
        for k in _FIXED_KWARGS:
            if k in kwargs:
                raise ValueError(f'{k} is fixed for whole HRDX tiles; use aug=... for augmentation')
        super().__init__(data_root=data_root, ann_file=ann_file, crop_size=50.0, rotate=False,
                         translate=False, noise_std=0.0, min_polyline_len=min_polyline_len,
                         gt_classes=('curb_parametric',) if gt_source == 'parametric' else ('curb',),
                         check_finalize=False, min_crop_points=min_crop_points, **kwargs)
        if self.test_mode and self.map_ann_file:
            self._check_gt_cache()

    # ------------------------------------------------------------------
    def load_annotations(self, ann_file):
        with open(ann_file) as f:
            man = json.load(f)
        self.min_crop_points = self._resolve_min_points(self.min_crop_points, man, ann_file)
        self.max_shift = 0.0            # whole tiles: nothing to shift
        self.finalize = {}
        tiles = man.get('tiles', [])
        if not tiles:
            raise RuntimeError(f'no tiles in manifest {ann_file}')
        infos = []
        for i, t in enumerate(tiles):
            name, city = t['name'], t['city']
            if city not in CITY_DIRS:
                raise KeyError(f'{ann_file}: tile {name} has unknown city {city!r} '
                               f'(known: {sorted(CITY_DIRS)})')
            d = os.path.join(self.hrdx_root, CITY_DIRS[city])
            infos.append(dict(
                sample_idx=name,
                city=city,
                timestamp=i,
                lidar_path=os.path.join(d, 'blocks', f'{name}.npz'),
                tile_center=np.asarray(t['center'], dtype=np.float64)[:2]))
            self._gt_path[name] = os.path.join(d, 'reference_lines', f'{name}_reference_lines.json')
        self._polys = {info['sample_idx']: self._load_polylines(info['sample_idx']) for info in infos}
        if self.crop_seed is not None:          # val/test: identity pose, annotation frozen
            eye, zero = np.eye(2), np.zeros(2)
            for info in infos:
                info['annotation'] = self._annotation(info['sample_idx'], eye, zero, info['tile_center'])
                info['crop_theta'], info['crop_R'], info['crop_shift'] = 0.0, eye, zero
        return infos

    def _load_polylines(self, name):
        f = self._gt_path.get(name)
        if f is None or not os.path.exists(f):
            self._tc_z[name] = 0.0
            return []
        with open(f) as fh:
            gt = json.load(fh)
        tc = gt.get('tile_center')
        self._tc_z[name] = float(tc[2]) if tc is not None and len(tc) > 2 else 0.0
        if self.gt_source == 'parametric':
            return [{'points': c['points'].astype(np.float32), 'class_id': 0, 'is_arc': bool(c['is_arc'])}
                    for c in chain_parametric(gt.get('polylines', []))]
        out = []
        for pl in gt.get('polylines', []):
            if pl.get('class') != 'curb':
                continue
            p = np.asarray(pl['points'], dtype=np.float32)
            if p.ndim == 2 and len(p) >= 2:
                out.append({'points': p, 'class_id': 0, 'is_arc': bool(pl.get('is_arc', False))})
        return out

    # ------------------------------------------------------------------
    def _draw_aug(self):
        """One draw of Pointcept's rotate / scale / flip, as the 2x2 map A = F * s * R(theta)."""
        a = self.aug
        rng = np.random
        th = float(rng.uniform(a['rotate_angle'][0], a['rotate_angle'][1]) * np.pi) \
            if rng.random() < a['rotate_p'] else 0.0
        s = float(rng.uniform(a['scale'][0], a['scale'][1]))
        fx = -1.0 if rng.random() < a['flip_p'] else 1.0
        fy = -1.0 if rng.random() < a['flip_p'] else 1.0
        c, sn = np.cos(th), np.sin(th)
        A = np.diag([fx, fy]) @ (s * np.array([[c, -sn], [sn, c]], dtype=np.float64))
        return A, s, th

    def get_data_info(self, index):
        info = self.data_infos[index]
        name, centre = info['sample_idx'], info['tile_center']
        zero = np.zeros(2, dtype=np.float64)
        A, zs, th = np.eye(2), 1.0, 0.0
        if 'annotation' in info:                 # frozen val/test
            ann = info['annotation']
        elif self.aug is None:
            ann = self._annotation(name, A, zero, centre)
        else:
            ann = None
            for _ in range(max(1, self.aug_tries)):
                A, zs, th = self._draw_aug()
                ann = self._annotation(name, A, zero, centre)
                if any(len(v) for v in ann.values()):
                    break
            else:
                A, zs, th = np.eye(2), 1.0, 0.0
                ann = self._annotation(name, A, zero, centre)
        a = self.aug if (self.aug is not None and 'annotation' not in info) else {}
        return dict(
            pts_filename=info['lidar_path'],
            sample_idx=name,
            timestamp=info.get('timestamp', index),
            scene_token=name,
            can_bus=np.zeros(18, dtype=np.float32),
            annotation=ann,
            ann_info=ann,
            # consumed by LoadHrdx50mTile so the points get the SAME map as the GT
            crop_centre=np.asarray(centre, dtype=np.float64),
            crop_R=np.asarray(A, dtype=np.float64),
            crop_shift=zero,
            crop_half=self.half,
            crop_z_pct=self.z_ground_percentile,
            aug_z_scale=float(zs),
            aug_theta=float(th),
            aug_jitter_sigma=float(a.get('jitter_sigma', 0.0)),
            aug_jitter_clip=float(a.get('jitter_clip', 0.0)),
            # train-only raw-point floor (0 = off); frozen tiles are never skipped
            crop_min_points=0 if 'annotation' in info else self.min_crop_points,
            # No pkl frame keys: the tile centre is the origin by construction.
            annotation_origin=None,
            gt_frame=None,
            lidar_recenter_shift=None,
        )

    # ------------------------------------------------------------------
    # GT cache (map_ann_file): content check + atomic write
    # ------------------------------------------------------------------
    def _gt_vectors(self, idx):
        """This dataset's GT for sample idx, exactly as _format_gt writes it."""
        d = self.vectormap_pipeline({}, self.data_infos[idx])
        labels = d['gt_labels_3d'].data.numpy()
        vecs = d['gt_bboxes_3d'].data.instance_list
        return [(int(lab), np.array(list(v.coords))[:, :self.code_size])
                for lab, v in zip(labels, vecs)]

    def _check_gt_cache(self):
        """Refuse an existing GT cache whose CONTENT differs from this dataset's."""
        path = self.map_ann_file
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                gts = json.load(f)['GTs']
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f'unreadable GT cache {path} ({e}); delete it to rebuild') from e
        problem = None
        if len(gts) != len(self.data_infos):
            problem = f'{len(gts)} cached samples vs {len(self.data_infos)} in this dataset'
        else:
            for i, g in enumerate(gts):
                name = self.data_infos[i]['sample_idx']
                if g.get('sample_token') != name:
                    problem = f'sample {i} is {g.get("sample_token")!r} in the cache, {name!r} here'
                    break
                mine = self._gt_vectors(i)
                theirs = g.get('vectors', [])
                if len(mine) != len(theirs):
                    problem = f'sample {i} ({name}): {len(theirs)} cached GT vs {len(mine)} here'
                    break
                for (lab, pts), v in zip(mine, theirs):
                    cpts = np.asarray(v.get('pts'), dtype=np.float64)
                    if int(v.get('type', -1)) != lab or v.get('cls_name') != self.MAPCLASSES[lab] \
                            or cpts.shape != pts.shape or not np.allclose(cpts, pts, atol=1e-4):
                        problem = f'sample {i} ({name}): GT geometry or class differs'
                        break
                if problem:
                    break
        if problem:
            raise RuntimeError(
                f'stale GT cache {path}: {problem}. Predictions are paired with the '
                'cache by position, so this would score against the wrong ground truth. '
                'Delete the file (it is rebuilt at the next evaluation) or point '
                'map_ann_file at a new path.')

    def _format_gt(self):
        """Parent's GT cache writer, made atomic (write a temp file, then rename)."""
        final = self.map_ann_file
        if os.path.exists(final):
            return super()._format_gt()
        os.makedirs(os.path.dirname(os.path.abspath(final)), exist_ok=True)
        root, ext = os.path.splitext(final)
        tmp = f'{root}.tmp{os.getpid()}{ext}'      # keep the extension: mmcv.dump infers the format
        self.map_ann_file = tmp
        try:
            super()._format_gt()
        finally:
            self.map_ann_file = final
        os.replace(tmp, final)
        print(f' GT cache moved into place: {final}')


@PIPELINES.register_module()
class LoadHrdx50mTile(LoadCarla50mCrop):
    """Whole HRDX tile -> ``(x, y, z, strength)`` under the dataset's 2x2 map A.

    xy: ``A (world_xy - tile centre)``, kept inside the +-25 m box; z: re-zeroed on the kept
    points' 2nd percentile, then scaled by the augmentation's scale factor; clipped Gaussian
    jitter on xyz (train only); strength = uint8 ``intensity`` / 255 (never augmented).
    ``use_dim=3`` drops the strength column.
    """

    def __init__(self, coord_type='LIDAR', load_dim=4, use_dim=4):
        super().__init__(coord_type=coord_type, load_dim=load_dim, use_dim=use_dim)

    def __call__(self, results):
        with np.load(results['pts_filename']) as d:
            world = d['points'].astype(np.float64)
            inten = d['intensity'] if self._need_strength else None
        A = results['crop_R']
        half = results['crop_half']
        local = (world[:, :2] - results['crop_centre']) @ A.T - results['crop_shift']
        keep = (np.abs(local[:, 0]) <= half) & (np.abs(local[:, 1]) <= half)
        n = int(keep.sum())
        need = int(results.get('crop_min_points', 0) or 0)
        if n < need:
            raise EmptyLidarTileError(
                f'tile {results.get("sample_idx")}: {n} raw point(s) in the '
                f'{2 * half:g} m tile (need >= {need})')
        xyz = np.empty((n, 3), dtype=np.float32)
        xyz[:, :2] = local[keep].astype(np.float32)
        xyz[:, 2] = world[keep, 2].astype(np.float32)
        if n:
            xyz[:, 2] -= np.percentile(xyz[:, 2], float(results.get('crop_z_pct', 2.0)))
        zs = float(results.get('aug_z_scale', 1.0))
        if zs != 1.0:
            xyz[:, 2] *= np.float32(zs)
        sig = float(results.get('aug_jitter_sigma', 0.0))
        if sig > 0 and n:
            clip = float(results.get('aug_jitter_clip', 0.0)) or np.inf
            xyz += np.clip(sig * np.random.randn(n, 3), -clip, clip).astype(np.float32)
        if self._need_strength:
            strength = (inten[keep].astype(np.float32) / np.float32(255.0)).reshape(-1, 1)
            points = np.concatenate([xyz, strength], axis=1)
        else:
            points = xyz
        points = points[:, self.use_dim]
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        return results
