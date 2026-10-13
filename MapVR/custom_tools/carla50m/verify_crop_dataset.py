"""Verify the 50 m -> 30 m crop dataset: alignment, coverage, determinism.

Builds the train and val datasets straight from a config (so it checks the
crop kwargs the training job will actually use), pulls crops through
`LoadCarla50mCrop`, and reports the invariants that matter:

  * GT vertices sit ON LiDAR returns (points and GT received the same pose);
  * every crop is fully populated and everything is inside the +-half box;
  * the train yaw is uniform;
  * the val crop is identical across calls (the cached map_ann_file depends
    on it) while the train crop is not.

Run inside the container, from MapVR/ (train.py adds the cwd to sys.path)::

    python3 custom_tools/carla50m/verify_crop_dataset.py \
        projects/configs/maptrv2/pmt_carla_lidar_50m_crop.py \
        [--data-root <export root>] [--n 40] [--train-split test]
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.getcwd())  # custom_tools/train.py does the same

import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset

import projects.mmdet3d_plugin  # noqa: F401  (registers the dataset)
from projects.mmdet3d_plugin.datasets.carla50m_crop_dataset import \
    LoadCarla50mCrop


def build(cfg, key, data_root, split=None):
    d = dict(cfg.data[key])
    d['pipeline'] = []
    if split:
        d['split'] = split
    if data_root or split:
        d['data_root'] = data_root or d['data_root']
        d['ann_file'] = f"{d['data_root'].rstrip('/')}/{d['split']}/manifest.json"
    return build_dataset(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--data-root', default=None,
                    help='override the config\'s data_root (export root)')
    ap.add_argument('--n', type=int, default=40, help='crops to sample')
    ap.add_argument('--train-split', default=None,
                    help='split to build the TRAIN-side dataset from (e.g. '
                         '"test" for a test-only export); crop kwargs stay '
                         'the train ones')
    args = ap.parse_args()
    warnings.filterwarnings('ignore')
    cfg = Config.fromfile(args.config)
    classes = list(cfg.data.train.map_classes)

    print('=' * 78 + '\n1. BUILD\n' + '=' * 78)
    tr = build(cfg, 'train', args.data_root, args.train_split)
    te = build(cfg, 'val', args.data_root)
    print(f'  train: {len(tr)} samples   val: {len(te)} samples   '
          f'classes: {classes}')
    print(f'  train max_shift={tr.max_shift:.3f} m  need={tr._need:.3f} m')
    print(f'  val   max_shift={te.max_shift:.3f} m  need={te._need:.3f} m')
    assert 'annotation' in te.data_infos[0], 'val crops must be frozen'
    assert 'annotation' not in tr.data_infos[0], 'train crops must be fresh'
    empty = sum(1 for i in te.data_infos
                if not any(i['annotation'][c] for c in classes))
    print(f'  val crops with NO GT (frozen): {empty}/{len(te)}')

    load_cfg = dict(cfg.train_pipeline[0])
    load_cfg.pop('type')
    loader = LoadCarla50mCrop(**load_cfg)
    half = tr.half
    # the point noise is added AFTER the crop, so allow a few sigma of slack
    slack = 4.0 * float(cfg.data.train.noise_std) + 1e-3
    n = min(args.n, len(tr))
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(tr), n, replace=False)

    print('\n' + '=' * 78 + '\n2. POINT/GT ALIGNMENT\n' + '=' * 78)
    d_all, occ, npts, ngt, bad_p, bad_g, dims = [], [], [], [], 0, 0, set()
    for i in idxs:
        info = tr.get_data_info(int(i))
        res = dict(info)
        loader(res)
        P = res['points'].tensor.numpy()
        dims.add(P.shape[1])
        npts.append(len(P))
        G = [g for c in classes for g in info['annotation'][c]]
        ngt.append(len(G))
        if len(P) and np.abs(P[:, :2]).max() > half + slack:
            bad_p += 1
        bad_g += sum(np.abs(g[:, :2]).max() > half + 1e-3 for g in G)
        if len(P) == 0 or not G:
            continue
        V = np.concatenate(G, 0)
        d_all.append(np.linalg.norm(V[:, None, :2] - P[None, :, :2],
                                    axis=-1).min(1))
        q = [((P[:, 0] > 0) & (P[:, 1] > 0)), ((P[:, 0] <= 0) & (P[:, 1] > 0)),
             ((P[:, 0] > 0) & (P[:, 1] <= 0)),
             ((P[:, 0] <= 0) & (P[:, 1] <= 0))]
        occ.append(sum(1 for m in q if m.sum() > 0))
    d = np.concatenate(d_all)
    print(f'  point dims: {sorted(dims)} (3 = colour-free convention)')
    print(f'  GT vertex -> nearest LiDAR point over {n} train crops: '
          f'median {np.median(d):.3f} m | p90 {np.percentile(d, 90):.3f} | '
          f'max {d.max():.3f}')
    print(f'  VERDICT: {"ALIGNED" if np.median(d) < 0.5 else "MISALIGNED"}')
    print(f'  all 4 quadrants populated: {sum(o == 4 for o in occ)}/{len(occ)}')
    print(f'  points/crop: median {int(np.median(npts)):,} '
          f'min {min(npts):,} max {max(npts):,}')
    print(f'  GT lines/crop: median {np.median(ngt):.1f} min {min(ngt)} '
          f'max {max(ngt)} empty: {sum(g == 0 for g in ngt)}')
    print(f'  crops with points outside +-{half} m (+noise slack): {bad_p}/{n}; '
          f'GT polylines outside: {bad_g}')

    print('\n' + '=' * 78 + '\n3. ROTATION\n' + '=' * 78)
    m = min(300, len(tr))
    Rs = [tr.get_data_info(int(i))['crop_R']
          for i in rng.choice(len(tr), m, replace=True)]
    ang = np.array([np.arctan2(R[1, 0], R[0, 0]) for R in Rs]) % (2 * np.pi)
    h, _ = np.histogram(ang, bins=8, range=(0, 2 * np.pi))
    print(f'  yaw histogram over {m} draws (8 bins): {h.tolist()}')

    print('\n' + '=' * 78 + '\n4. EVAL DETERMINISM\n' + '=' * 78)
    k = min(5, len(te))
    a = [te.get_data_info(i) for i in range(k)]
    b = [te.get_data_info(i) for i in range(k)]
    same = all(
        np.array_equal(x['crop_R'], y['crop_R']) and all(
            len(x['annotation'][c]) == len(y['annotation'][c]) and all(
                np.array_equal(u, v)
                for u, v in zip(x['annotation'][c], y['annotation'][c]))
            for c in classes) for x, y in zip(a, b))
    print(f'  val get_data_info identical across calls: {same}')
    print(f'  val yaw is identity: {np.allclose(a[0]["crop_R"], np.eye(2))}')
    ta, tb = tr.get_data_info(0)['crop_R'], tr.get_data_info(0)['crop_R']
    print(f'  train crop varies across calls: {not np.array_equal(ta, tb)}')
    assert same and not np.array_equal(ta, tb)
    print('\nOK')


if __name__ == '__main__':
    main()
