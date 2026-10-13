"""CARLA 50 m tiles -> rotated 30 m crops, augmented at load time.

Ported from the sibling MapTRv2 repo (``carla50m_crop_dataset.py``) onto
``PMTCarlaMapDataset``, so PseudoMapTrainer's BEV visibility mask
(``bev_mask``/``bev_label`` from ``vectormap_pipeline``) is layered on top of
the crop GT exactly as it is on the pkl tiles. The crop geometry is kept
identical to MapTRv2's so the two repos score the same frozen eval crops.

The tiles on disk are 50 m squares (``tile_radius`` 25 m). Augmentation happens
HERE, per __getitem__, exactly as in ``Pointcept/data/tile_dataset_50m.py``:

  1. random yaw about the tile centre + random centre shift, then a 30 m square
     CROP. A 30 m square rotated arbitrarily needs 30*sqrt(2) = 42.43 m, which
     fits inside the 50 m tile, so every crop is fully populated -- no empty
     corners. Leftover slack (25 - 21.21 = 3.79 m) is the max centre shift.
  2. additive Gaussian noise on the point coordinates
  3. GT polylines rotated and clipped to the SAME 30 m box, via the tiler's own
     ``clip_polyline_to_box`` (vendored below) so crop GT is built identically
     to how the tiler builds tile GT.

FRAMES (verified against the npz/json on disk, not assumed):
  * ``points``            world-absolute (N,3)
  * ``features[:, :3]``   points - offset (block-local)   -- unused here
  * ``features[:, 3:6]``  rgb in [0,1] -> BT.709 luma "strength" (only read
                          when a colour-carrying ``use_dim`` asks for it)
  * GT ``polylines[].points``  WORLD-absolute (N,3)
  * ``tile_center``       world centre of the 50 m tile
So the whole augmentation is done in world frame about ``tile_center``, and the
crop-local output frame is what the network sees. This is the crop-time
equivalent of the pkl datasets' ``tile_center`` frame: the crop centre IS the
origin, so there is no ``lidar_recenter_shift`` for ``LoadCarlaPointsFromFile``
to apply -- that loader is not part of this pipeline at all.

EVAL DETERMINISM. MapTRv2 caches GT to ``map_ann_file`` once and reuses it for
every evaluation. If the val split were randomly augmented, the cached GT would
not match the crops actually fed to the model and every AP would be wrong. So
``rotate``/``noise_std`` must be OFF for val/test and ``crop_seed`` set; the
annotation is then computed once at load time and stored on ``data_infos``,
which is where ``_format_gt`` reads it from.

This is a manifest-driven dataset: ``ann_file`` is the export's
``<split>/manifest.json``, not a converter pkl, so none of the pkl-side
bookkeeping in ``CustomCarlaLocalMapDataset`` (empty-tile filter,
``raw_data_root``, the pkl frame keys) applies. ``data_root`` is the export root itself.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import List

import numpy as np
from mmdet.datasets import DATASETS
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import get_points_type

from .carla50m_metrics import precision_and_chamfer
from .carla_offlinemap_dataset import PMTCarlaMapDataset


# ---------------------------------------------------------------------------
# Vendored from Pointcept/data/grid_tile_export.py so the crop GT is clipped by
# EXACTLY the tiler's logic (boundary vertex placed on the edge, runs of <2
# vertices dropped). Copied verbatim rather than imported: the tiler lives in
# the Pointcept tree, which is not on PYTHONPATH inside the container.
# ---------------------------------------------------------------------------
def _clip_segment_to_box(p0, p1, box):
    """Liang-Barsky. Returns (q0, q1, t0, t1) clipped to box, or None."""
    xmin, ymin, xmax, ymax = box
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin),
                 (dy, ymax - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return None
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return None
                if r < t1:
                    t1 = r
    q0 = p0 + t0 * (p1 - p0)
    q1 = p0 + t1 * (p1 - p0)
    return q0, q1, t0, t1


def clip_polyline_to_box(points: np.ndarray, curved: np.ndarray,
                         box) -> List[dict]:
    """Clip a polyline to a square -> continuous sub-polylines reaching the edge."""
    out: List[dict] = []
    cur_pts: List[np.ndarray] = []
    cur_cur: List[bool] = []

    def flush():
        if len(cur_pts) >= 2:
            out.append({
                'points': np.asarray(cur_pts, dtype=np.float32),
                'curved': np.asarray(cur_cur, dtype=bool)
            })
        cur_pts.clear()
        cur_cur.clear()

    for i in range(len(points) - 1):
        res = _clip_segment_to_box(points[i], points[i + 1], box)
        if res is None:
            flush()
            continue
        q0, q1, t0, t1 = res
        c_i = bool(curved[i])
        if not cur_pts:
            cur_pts.append(q0)
            cur_cur.append(c_i)
        elif np.linalg.norm(cur_pts[-1] - q0) > 1e-4:
            flush()
            cur_pts.append(q0)
            cur_cur.append(c_i)
        cur_pts.append(q1)
        cur_cur.append(bool(curved[i + 1]) if t1 >= 1.0 - 1e-9 else c_i)
        if t1 < 1.0 - 1e-9:
            flush()
    flush()
    return out


@DATASETS.register_module()
class PMTCarla50mCropDataset(PMTCarlaMapDataset):
    """50 m CARLA tiles -> augmented 30 m crops, in PMT's data contract.

    Args (beyond ``PMTCarlaMapDataset``'s -- ``raster``/``mask_mode`` etc.
    are passed straight through):
        split (str): ``<data_root>/<split>/{blocks,<reference_dir>}`` layout.
        crop_size (float): side of the square crop, metres.
        rotate / translate / noise_std: the augmentation. Train: all on.
            Val/test: ``rotate=False``, ``noise_std=0.0`` and a ``crop_seed``.
        require_gt / gt_tries: aim each crop at a random GT vertex so it is
            non-empty (see ``_draw_crop``).
        min_polyline_len (float): drop clipped stubs shorter than this.
        gt_classes (tuple[str] | None): the export's class names, in the
            order they map onto ``map_classes`` (positional). ``None`` means
            the export uses the config's own ``map_classes`` names, which is
            the case for the `driving`/`curb` exports and is the convention
            the pkl datasets follow. Polylines carrying no class at all are
            treated as ``gt_classes[0]``.
        reference_dir (str): polyline directory under ``<split>/``. Newer
            exports ship several (``reference_lines``,
            ``reference_curb_driving_lines``, ...) holding different GT.
        z_ground_percentile (float): points' z is re-zeroed on this
            percentile of the crop's own z, per crop.
        crop_seed (int | None): freeze the crop pose per index (val/test).
    """

    def __init__(self,
                 data_root,
                 ann_file,
                 split='train',
                 crop_size=30.0,
                 rotate=True,
                 translate=True,
                 noise_std=0.02,
                 noise_xy_only=False,
                 require_gt=True,
                 gt_tries=32,
                 min_polyline_len=2.0,
                 gt_classes=None,
                 reference_dir='reference_lines',
                 z_ground_percentile=2.0,
                 crop_seed=None,
                 map_classes=None,
                 raster=None,
                 **kwargs):
        # --- all of these are read by load_annotations(), which the base
        # __init__ calls, so they must exist BEFORE super().__init__ ---
        self.split = split
        self.blocks_dir = os.path.join(data_root, split, 'blocks')
        self.gt_dir = os.path.join(data_root, split, reference_dir)
        self.half = float(crop_size) / 2.0
        self.rotate = bool(rotate)
        self.translate = bool(translate)
        self.noise_std = float(noise_std)
        self.noise_xy_only = bool(noise_xy_only)
        self.require_gt = bool(require_gt)
        self.gt_tries = int(gt_tries)
        self.min_polyline_len = float(min_polyline_len)
        self.z_ground_percentile = float(z_ground_percentile)
        self.crop_seed = crop_seed
        # Export class name -> output label. Labels follow map_classes order
        # (VectorizedCarlaLocalMap builds CLASS2LABEL from that same list), so
        # gt_classes[i] is what the export calls map_classes[i].
        names = self.get_map_classes(map_classes)
        if raster is None:
            raise ValueError('PMTCarla50mCropDataset needs raster='
                             '[renderer_W, renderer_H] like PMTCarlaMapDataset')
        self.gt_classes = tuple(gt_classes) if gt_classes else tuple(names)
        if len(self.gt_classes) != len(names):
            raise ValueError(
                f'gt_classes {self.gt_classes} must pair positionally with '
                f'map_classes {list(names)}')
        self._src_keep = {n: i for i, n in enumerate(self.gt_classes)}
        self._tc_z = {}   # tile centre elevation, filled by _load_polylines
        # a 30 m square rotated arbitrarily needs half*sqrt(2)
        self._need = self.half * (np.sqrt(2.0) if self.rotate else 1.0)
        self.max_shift = 0.0  # set in load_annotations once tile_radius is known
        super().__init__(raster, data_root=data_root, ann_file=ann_file,
                         map_classes=map_classes, **kwargs)

    # ------------------------------------------------------------------
    def load_annotations(self, ann_file):
        with open(ann_file) as f:
            man = json.load(f)
        tiles = man.get('tiles', [])
        names = [t['name'] if isinstance(t, dict) else t for t in tiles]
        if not names:
            raise RuntimeError(f'no tiles in manifest {ann_file}')

        # tile radius -> how much centre shift the crop can afford
        with np.load(os.path.join(self.blocks_dir, f'{names[0]}.npz')) as d:
            radius = float(d['tile_radius'])
        if self._need > radius + 1e-6:
            raise ValueError(
                f'crop_size={2*self.half} with rotate={self.rotate} needs '
                f'tile_radius >= {self._need:.2f} m, but tiles have radius '
                f'{radius:.2f} m')
        self.max_shift = max(0.0, radius - self._need) if self.translate else 0.0

        # Tile centres come from the manifest (already there as 'center'), NOT
        # from opening 14.9k npz files -- np.load is lazy but the open() calls
        # alone cost minutes at this count.
        infos = []
        for i, t in enumerate(tiles):
            name = t['name'] if isinstance(t, dict) else t
            infos.append(dict(
                sample_idx=name,
                lidar_path=os.path.join(self.blocks_dir, f'{name}.npz'),
                timestamp=i,
                tile_center=np.asarray(t['center'], dtype=np.float64)[:2]
                if isinstance(t, dict) and 'center' in t else None,
            ))

        # Polylines are needed on EVERY __getitem__ (a fresh crop each time), so
        # read the json files once here instead of once per iteration.
        self._polys = {}
        for info in infos:
            self._polys[info['sample_idx']] = self._load_polylines(
                info['sample_idx'])
            if info['tile_center'] is None:
                with np.load(info['lidar_path']) as d:
                    info['tile_center'] = np.asarray(
                        d['tile_center'], dtype=np.float64)[:2]

        # DETERMINISM FOR EVAL. MapTRv2 caches GT to map_ann_file once, so the
        # val crops must be identical on every run -- but "identical" must not
        # mean "centred and unrotated": 24.4% of test tiles have NO GT in their
        # centred 30 m crop (the manifest filtered for a *reachable* non-empty
        # crop, i.e. one found by aiming, not a centred one). Those tiles would
        # contribute nothing but false positives.
        # So: crop_seed set -> draw the pose ONCE from a per-index seeded rng
        # and freeze it. Reproducible across runs AND non-empty.
        if self.crop_seed is not None:
            for idx, info in enumerate(infos):
                name = info['sample_idx']
                rng = np.random.default_rng(self.crop_seed + idx)
                th, R, sh = self._draw_crop(self._polys.get(name, []),
                                            info['tile_center'], rng)
                info['annotation'] = self._annotation(name, R, sh,
                                                      info['tile_center'])
                info['crop_theta'] = th
                info['crop_R'] = R
                info['crop_shift'] = sh
        return infos

    # ------------------------------------------------------------------
    def _load_polylines(self, name):
        f = os.path.join(self.gt_dir, f'{name}_reference_lines.json')
        if not os.path.exists(f):
            self._tc_z[name] = 0.0
            return []
        with open(f) as fh:
            gt = json.load(fh)
        lookup = gt.get('classes') or {}
        tc = gt.get('tile_center')
        self._tc_z[name] = float(tc[2]) if tc is not None and len(tc) > 2 else 0.0
        out = []
        for pl in gt.get('polylines', []):
            cname = pl.get('class')
            if cname is None and 'class_id' in pl:
                cname = lookup.get(str(pl['class_id']))
            if cname is None:
                cname = self.gt_classes[0]
            if cname not in self._src_keep:
                continue
            p = np.asarray(pl['points'], dtype=np.float32)
            if p.ndim != 2 or len(p) < 2:
                continue
            out.append({'points': p, 'class_id': self._src_keep[cname],
                        'is_arc': bool(pl.get('is_arc', False))})
        return out

    # ------------------------------------------------------------------
    # Augmentation geometry -- mirrors tile_dataset_50m.py
    # ------------------------------------------------------------------
    @staticmethod
    def _transform(xy, centre, R, shift=None):
        out = (xy - centre) @ R.T
        return out if shift is None else out - shift

    def _crop_polylines(self, polys, centre, R, shift=None):
        """Rotate + shift into crop frame, then clip to the +-half box."""
        box = (-self.half, -self.half, self.half, self.half)
        pts_out, cls_out = [], []
        for pl in polys:
            p = pl['points']
            q = p.copy().astype(np.float64)
            q[:, :2] = self._transform(p[:, :2].astype(np.float64), centre, R,
                                       shift)
            curved = np.full(len(q), pl['is_arc'], dtype=bool)
            for sub in clip_polyline_to_box(q.astype(np.float32), curved, box):
                s = sub['points']
                if len(s) < 2:
                    continue
                # drop degenerate stubs the clip leaves at corners
                if float(np.linalg.norm(np.diff(s[:, :2], axis=0),
                                        axis=1).sum()) < self.min_polyline_len:
                    continue
                pts_out.append(np.asarray(s, dtype=np.float32))
                cls_out.append(pl['class_id'])
        return pts_out, cls_out

    def _rand_pose(self, rng):
        th = float(rng.uniform(0.0, 2.0 * np.pi)) if self.rotate else 0.0
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        if self.max_shift > 0 and self.translate:
            rad = self.max_shift * np.sqrt(rng.random())
            ang = rng.uniform(0.0, 2.0 * np.pi)
            sh = np.array([rad * np.cos(ang), rad * np.sin(ang)],
                          dtype=np.float64)
        else:
            sh = np.zeros(2, dtype=np.float64)
        return th, R, sh

    def _draw_crop(self, polys, centre, rng):
        """Pick (yaw, shift), AIMING the crop at a random GT vertex.

        Rejection sampling would waste draws on tiles whose GT sits near the
        tile edge; instead rotate freely then pull the crop centre along the
        chosen vertex direction (bounded by max_shift) so the vertex lands
        inside the box. Each try is verified; falls back to the GT vertex
        closest to the tile centre, then to a plain random pose.
        """
        if not (self.require_gt and polys):
            return self._rand_pose(rng)
        for _ in range(max(1, self.gt_tries)):
            pl = polys[int(rng.integers(len(polys)))]
            q = pl['points']
            p = q[int(rng.integers(len(q)))][:2].astype(np.float64)
            th, R, _ = self._rand_pose(rng)
            v = (p - centre) @ R.T
            n = float(np.hypot(v[0], v[1]))
            if n > 1e-9:
                pull = min(self.max_shift, max(0.0, n - 0.6 * self.half))
                sh = v * (pull / n)
            else:
                sh = np.zeros(2, dtype=np.float64)
            if self._crop_polylines(polys, centre, R, sh)[0]:
                return th, R, sh
        best = None
        for pl in polys:
            q = np.asarray(pl['points'], dtype=np.float64)[:, :2]
            d = np.hypot(q[:, 0] - centre[0], q[:, 1] - centre[1])
            j = int(np.argmin(d))
            if best is None or d[j] < best[0]:
                best = (float(d[j]), q[j])
        if best is not None:
            p = best[1]
            for th in (0.0,) + tuple(
                    np.linspace(0, 2 * np.pi, 8, endpoint=False)[1:]):
                c, s_ = np.cos(th), np.sin(th)
                R = np.array([[c, -s_], [s_, c]], dtype=np.float64)
                v = (p - centre) @ R.T
                n = float(np.hypot(v[0], v[1]))
                sh = v * (min(self.max_shift, max(0.0, n - 0.6 * self.half)) / n) \
                    if n > 1e-9 else np.zeros(2, dtype=np.float64)
                if self._crop_polylines(polys, centre, R, sh)[0]:
                    return float(th), R, sh
        return self._rand_pose(rng)

    # ------------------------------------------------------------------
    def _annotation(self, name, R, shift, centre):
        """Crop GT -> the {map_class: [ (N,3) ... ]} annotation contract.

        Instances keep 3 columns, like the converter's pkl polylines, so the
        base class's ``_annotation_2d`` slices z off at the same boundary it
        does for pkl tiles (this repo's ``VectorizedLocalMap`` is strictly
        2D). z is re-zeroed on the tile centre elevation purely for
        host-side tooling parity with the sibling repos; nothing here
        supervises it.
        """
        pts, cls = self._crop_polylines(self._polys.get(name, []), centre, R,
                                        shift)
        z0 = self._tc_z.get(name, 0.0)
        ann = {c: [] for c in self.MAPCLASSES}
        for p, c in zip(pts, cls):
            if c >= len(self.MAPCLASSES):
                continue
            q = np.zeros((len(p), 3), dtype=np.float32)
            q[:, :2] = p[:, :2]
            if p.shape[1] > 2:
                q[:, 2] = p[:, 2] - z0
            ann[self.MAPCLASSES[c]].append(q)
        return ann

    def _evaluate_single(self, result_path, logger=None, metric='chamfer',
                         result_name='pts_bbox'):
        """Parent's per-class AP, PLUS precision/recall/F1 and chamfer distance.

        The base logs AP only -- eval_map computes the per-class PR curve and
        throws precision away, and the chamfer that drives matching is never
        surfaced. Both are added here so they reach TensorBoard. The result
        json schema (``sample_token`` / ``vectors[{pts, type,
        confidence_level}]``) is the same as MapTRv2's, so
        ``carla50m_metrics.py`` is shared verbatim.

        Wrapped in try/except on purpose: a metrics bug must never kill a 24 h
        training job.
        """
        detail = super()._evaluate_single(result_path, logger=logger,
                                          metric=metric,
                                          result_name=result_name)
        if metric != 'chamfer':
            return detail
        try:
            with open(result_path) as f:
                res = json.load(f)['results']
            with open(self.map_ann_file) as f:
                gts = json.load(f)['GTs']
            extra = precision_and_chamfer(res, gts, list(self.MAPCLASSES))
            for k, v in extra.items():
                detail[f'CarlaMap_{metric}/{k}'] = v
        except Exception as e:  # noqa: BLE001
            warnings.warn(f'{self.__class__.__name__}: precision/chamfer '
                          f'metrics failed ({e}); AP is unaffected')
        return detail

    def get_data_info(self, index):
        info = self.data_infos[index]
        name = info['sample_idx']
        centre = info['tile_center']
        if 'annotation' in info:   # frozen (seeded) split -- see load_annotations
            theta, shift = info['crop_theta'], info['crop_shift']
            R = info['crop_R']
            ann = info['annotation']
        else:                      # train: fresh crop every epoch
            rng = np.random.default_rng(
                None if self.crop_seed is None else self.crop_seed + index)
            theta, R, shift = self._draw_crop(self._polys.get(name, []),
                                              centre, rng)
            ann = self._annotation(name, R, shift, centre)
        return dict(
            pts_filename=info['lidar_path'],
            sample_idx=name,
            timestamp=info.get('timestamp', index),
            scene_token=name,
            can_bus=np.zeros(18, dtype=np.float32),
            annotation=ann,
            ann_info=ann,
            # No pkl frame keys: the crop centre is the origin by construction.
            annotation_origin=None,
            gt_frame=None,
            lidar_recenter_shift=None,
            # consumed by LoadCarla50mCrop so points get the SAME pose as the GT
            crop_centre=np.asarray(centre, dtype=np.float64),
            crop_R=np.asarray(R, dtype=np.float64),
            crop_shift=np.asarray(shift, dtype=np.float64),
            crop_half=self.half,
            crop_noise_std=self.noise_std,
            crop_noise_xy_only=self.noise_xy_only,
            crop_z_pct=self.z_ground_percentile,
        )


@PIPELINES.register_module()
class LoadCarla50mCrop(object):
    """Load a 50 m block and apply the SAME crop pose the GT was cropped with.

    The pose is drawn once in ``CustomCarla50mCropDataset.get_data_info`` and
    passed through ``results``; doing it here independently would silently
    decouple the points from their labels.

    Output points are ``(x, y, z[, strength])`` in the crop-local frame, with
    z re-zeroed on the crop's own ground (``z_pct`` percentile) so the network
    sees a consistent vertical frame across tiles of differing elevation.

    ``load_dim``/``use_dim`` follow ``LoadCarlaPointsFromFile``'s convention:
    the default 3/3 is the benchmark's colour-free setting (xyz only, and the
    block's ``features`` array is never read); ``load_dim=4, use_dim=4`` keeps
    the BT.709 strength channel. Whatever is chosen must match the model's
    ``SparseEncoder.in_channels``. ``GridSamplePoints`` and the PMT coverage
    mask run on the output exactly as they do on pkl tiles.
    """

    def __init__(self, coord_type='LIDAR', load_dim=3, use_dim=3):
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ('CAMERA', 'LIDAR', 'DEPTH')
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self._need_strength = max(self.use_dim) >= 3
        self._rgb2strength = np.array([0.2126, 0.7152, 0.0722],
                                      dtype=np.float32)

    def __call__(self, results):
        with np.load(results['pts_filename']) as d:
            world = d['points'].astype(np.float64)   # (N,3) world-absolute
            rgb = (d['features'][:, 3:6].astype(np.float32)
                   if self._need_strength else None)

        centre = results['crop_centre']
        R = results['crop_R']
        shift = results['crop_shift']
        half = results['crop_half']

        local = (world[:, :2] - centre) @ R.T - shift
        keep = (np.abs(local[:, 0]) <= half) & (np.abs(local[:, 1]) <= half)

        n = int(keep.sum())
        xyz = np.empty((n, 3), dtype=np.float32)
        xyz[:, :2] = local[keep].astype(np.float32)
        xyz[:, 2] = world[keep, 2].astype(np.float32)

        std = float(results.get('crop_noise_std', 0.0))
        if std > 0 and n:
            xy_only = bool(results.get('crop_noise_xy_only', False))
            noise = np.random.normal(
                0.0, std, size=(n, 2 if xy_only else 3)).astype(np.float32)
            xyz[:, :2] += noise[:, :2]
            if not xy_only:
                xyz[:, 2] += noise[:, 2]

        # ground-relative z, per crop
        if n:
            xyz[:, 2] -= np.percentile(xyz[:, 2],
                                       float(results.get('crop_z_pct', 2.0)))

        if self._need_strength:
            strength = (rgb[keep] @ self._rgb2strength).reshape(-1, 1)
            points = np.concatenate([xyz, strength], axis=1)
        else:
            points = xyz
        points = points[:, self.use_dim]

        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(
            points, points_dim=points.shape[-1], attribute_dims=None)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(coord_type={self.coord_type}, '
                f'load_dim={self.load_dim}, use_dim={self.use_dim})')
