"""HRDX San Jose 50 m tiles -> rotated 30 m crops (curb only).

The San Jose export (``san_jose_tiles_50m``) carries the same per-tile files
as the CARLA 50 m export -- ``blocks/<name>.npz`` with world-frame ``points``,
``tile_center`` and ``tile_radius`` 25, and ``reference_lines/<name>_
reference_lines.json`` with world-frame polylines -- so the whole crop
machinery of :class:`PMTCarla50mCropDataset` applies unchanged. Three
things differ, and they are all this subclass does:

1. **Flat layout, external split manifests.** There is no ``<split>/``
   level: ``blocks/`` and ``reference_lines/`` sit directly under the export
   root, and the train/val/test manifests live in a separate directory
   (``san_jose_splits_v1/v1/{train,val,test}.json``). ``ann_file`` is any
   of those; ``data_root`` is the export root. The manifests are
   ``manifest.json`` subsets (same ``tiles: [{name, center, ...}]`` schema)
   plus a ``finalize`` block recording how they were filtered.
2. **One class, ``curb``.** The export's ``class_lookup`` is
   ``{1: curb, 2: curb_parametric}``; ``curb_parametric`` re-traces the same
   curbs as fitted alignment elements and must NOT be trained alongside
   (duplicate GT on the same geometry). ``gt_classes`` defaults to
   ``('curb',)``.
3. **The split filter is checked against the crop kwargs.** ``v1`` was
   finalised by running the MapTRv2 repo's ``_draw_crop``/``_crop_polylines``
   (the same code as this repo's ``PMTCarla50mCropDataset``)
   (``san_jose_splits_v1/tools/sj_finalize_splits.py``): every val/test tile
   is guaranteed a non-empty, >= 1000-point crop at the frozen pose
   ``default_rng(crop_seed + index)`` with ``rotate=False, translate=True``
   (``max_shift`` 10 m) -- in MANIFEST ORDER -- and every train tile a
   non-empty crop for at least one of four seeds at ``max_shift`` 3.79 m.
   That guarantee only holds for the kwargs it was computed with, so
   ``load_annotations`` compares the manifest's ``finalize`` block against
   the dataset's own ``crop_seed``/``max_shift``/``gt_classes`` and warns
   loudly on a mismatch (a GT-empty frozen crop contributes only false
   positives; a 0-point one raises in ``prepare_test_data``).

Everything else -- the frozen eval pose, the ``LoadCarla50mCrop`` pipeline
step, the per-crop ground re-zeroing, ``_format_gt``'s ``map_ann_file``
cache (own file per config!) -- is inherited. ``val_60.json`` is a prefix of
``val.json``, so its 60 frozen crops are identical to the first 60 of the
full split; ``v1_smoke/val60_gt.json`` is the ``map_ann_file`` those crops
produce and can be diffed against a freshly written one.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
from mmdet.datasets import DATASETS

from .carla50m_crop_dataset import PMTCarla50mCropDataset

SANJOSE_GT_CLASSES = ('curb',)


@DATASETS.register_module()
class PMTSanJose50mCropDataset(PMTCarla50mCropDataset):
    """San Jose 50 m tiles -> 30 m crops, in PMT's data contract (``raster``/``mask_mode`` pass through).

    Args (beyond the parent's):
        data_root (str): the export root, holding ``blocks/`` and
            ``<reference_dir>/`` directly (``split`` is accepted for API
            compatibility and must be empty / None).
        ann_file (str): a split manifest json from ``san_jose_splits_v1``.
        gt_classes: defaults to ``('curb',)``; ``map_classes`` must have the
            same length (positional pairing, as in the parent).
        check_finalize (bool): compare the manifest's ``finalize`` block with
            the crop kwargs and warn on mismatch (default True).
    """

    def __init__(self,
                 data_root,
                 ann_file,
                 split=None,
                 gt_classes=None,
                 check_finalize=True,
                 **kwargs):
        if split:
            raise ValueError(
                f'{self.__class__.__name__}: the San Jose export is flat '
                f'(<data_root>/blocks, <data_root>/<reference_dir>); got '
                f'split={split!r}')
        self.check_finalize = bool(check_finalize)
        self.finalize = {}
        super().__init__(data_root=data_root, ann_file=ann_file, split='',
                         gt_classes=gt_classes or SANJOSE_GT_CLASSES,
                         **kwargs)

    # ------------------------------------------------------------------
    def load_annotations(self, ann_file):
        with open(ann_file) as f:
            man = json.load(f)
        self.finalize = dict(man.get('finalize') or {})
        lookup = man.get('class_lookup') or {}
        if lookup:
            missing = [c for c in self.gt_classes if c not in lookup.values()]
            if missing:
                raise ValueError(
                    f'{ann_file}: gt_classes {missing} not in the export '
                    f'class_lookup {lookup}')
        infos = super().load_annotations(ann_file)
        if self.check_finalize:
            self._check_finalize(ann_file, len(infos))
        return infos

    def _check_finalize(self, ann_file, n):
        fz = self.finalize
        if not fz:
            warnings.warn(
                f'{ann_file} has no "finalize" block: it was not filtered '
                f'with the loader, so frozen crops may be GT-empty or '
                f'point-free (see sj_finalize_splits.py)')
            return
        problems = []
        if tuple(fz.get('gt_classes', ())) != tuple(self.gt_classes):
            problems.append(f'gt_classes {tuple(self.gt_classes)} != '
                            f'finalised {tuple(fz.get("gt_classes", ()))}')
        ms = fz.get('max_shift_m')
        if ms is not None and abs(float(ms) - self.max_shift) > 1e-3:
            problems.append(f'max_shift {self.max_shift:.3f} m != finalised '
                            f'{float(ms):.3f} m (rotate/translate/crop_size '
                            f'differ)')
        if 'crop_seed' in fz:            # a frozen (val/test) split
            if self.crop_seed != fz['crop_seed']:
                problems.append(f'crop_seed {self.crop_seed} != finalised '
                                f'{fz["crop_seed"]}')
            if self.rotate:
                problems.append('rotate=True on a split finalised unrotated')
        elif self.crop_seed is not None:  # train split used frozen
            warnings.warn(f'{ann_file} is a train split but crop_seed='
                          f'{self.crop_seed} freezes its crops')
        if problems:
            warnings.warn(
                f'{self.__class__.__name__}: {ann_file} was finalised with '
                f'different crop settings -- its non-empty-crop guarantee '
                f'does not hold: ' + '; '.join(problems))
        else:
            print(f'[{self.__class__.__name__}] {os.path.basename(ann_file)}: '
                  f'{n} tiles, finalize block matches '
                  f'(gt_classes={tuple(self.gt_classes)}, '
                  f'max_shift={self.max_shift:.2f} m, crop_seed={self.crop_seed})')
