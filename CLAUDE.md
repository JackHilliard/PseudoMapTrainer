# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Research prototype for *"PseudoMapTrainer: Learning Online Mapping without HD Maps"* (ICCV 2025): trains an online HD-map model on nuScenes without HD-map GT. No tests, no linter, no CI.

## Three vendored forks, three conda environments

Nothing imports across these directories — each has its own env (see `README.md`, `Mask2Former/INSTALL.md`, `MapVR/docs/install.md`).

- `Mask2Former/` — perspective-view semantic segmentation (py3.9, torch 1.10.1+cu113, detectron2)
- `RoGS/` — 2D-Gaussian road reconstruction → BEV labels → vectorized pseudo-labels (py3.8.20, torch 1.13.1+cu116)
- `MapVR/` — MapTRv2 + MapVR online mapping model (py3.8, mmcv-full 1.4.0, mmdet 2.14.0, local `mmdetection3d/`)

Git history isolates each layer (upstream import → port → `Our modifications of X`), so `git show <commit> -- <path>` gives an exact upstream diff.

## Pipeline

```
nuScenes images → Mask2Former PV segmentation PNGs
  → RoGS per-scene Gaussian road surface → BEV semantic + visibility mask per LiDAR keyframe
  → post_process.py / vectorize_mask.py → map_vectors.pkl + bev_*.png
  → MapVR custom_nusc_map_converter.py → *_map_infos_temporal_{train,val}.pkl
  → MapTRv2 training with masked losses/assigner
```

Everything comes in **single-trip** / **multi-trip** pairs (`single_trip.yaml`/`multi_trip.yaml`, `pmt_single.py`/`pmt_multi.py`); multi-trip sets `combine_scenes: True` + `scene_collection_path: geo_split.json`.

6-class BEV label space from `RoGS/datasets/label_mappings/mapillary_vistas.py` (0=mask, 1=lane marking, 2=crosswalk, 3=road, 4=boundary, 5=background). `MASK_CLASS = 0` marks unobserved area and propagates into training as the visibility mask; the label rasterizer must be built with `NUM_CHANNELS=6`.

## Commands

```sh
# Mask2Former (DETECTRON2_DATASETS must be set)
python train_net.py --num-gpus 2 --config-file configs/mapillary-vistas-v2/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml
python demo/inference.py --config-file <same> --base_dir path/to/nuscenes --save_dir data/m2f_infer MODEL.WEIGHTS output/model_xyz.pth

# RoGS
python -m preprocess.process_nusc --config configs/nusc/single_trip.yaml   # optional LiDAR
python pseudo_label_generation.py --config configs/nusc/single_trip.yaml

# MapVR — convert pseudo labels then GT; --use-geo-split must match between the two
python custom_tools/maptrv2/custom_nusc_map_converter.py --root-path ./data/nuscenes \
  --pseudo-labels-dir ../RoGS/output/single_trip --out-dir ./data/nuscenes \
  --extra-tag nuscenes_pseudo_single --version v1.0 --canbus ./data --use-geo-split
custom_tools/dist_train.sh ./projects/configs/maptrv2/pmt_single.py 2
custom_tools/dist_test_map.sh ./projects/configs/maptrv2/pmt_single.py ./ckpt.pth 2
custom_tools/dist_test_pseudo_labels.sh ./projects/configs/pseudo_eval/single_trip.py 2 --masked
```

`dist_test_pseudo_labels.sh` scores the pseudo-labels themselves (`--masked` = observed area only, Table 1). Override `PORT=` for concurrent jobs. `RoGS/mwe.ipynb` is a two-trip minimal example, useful for visualization and quick iteration.

## Consistency traps

- Same directory, different names per stage: Mask2Former `--save_dir` == RoGS `label_dir` == MapVR `data_root_seg`; `--base_dir` == RoGS `base_dir` == MapVR `--root-path`; RoGS `output` == MapVR `--pseudo-labels-dir`. All hard-coded, none derived.
- Geographic splits (ported LiljaAdam/geographical-splits) exist in **two copies** — `RoGS/datasets/nusc/geosplits/` and `MapVR/custom_tools/maptrv2/geosplits/`. Mixing geo and default splits between pseudo-label and GT infos silently yields meaningless eval numbers.
- Skipping LiDAR requires `z_weight: 0` in *both* RoGS configs.
- `--extra-tag` determines the `ann_file` names the training config expects.

## What this work adds to MapTRv2/MapVR

In `MapVR/projects/mmdet3d_plugin/` — the parts that tolerate incomplete labels:

- `datasets/nuscenes_pseudolabel_dataset.py` — `PseudoMapDataset`, loads RoGS vectors + visibility mask
- `maptr/assigners/maptr_assigner.py` — `MaskedMapTRAssigner`, ignores unobserved regions, `allow_split=True` lets several predictions match one GT polyline across a mask gap; PuLP LP in `solver.py`
- `maptr/losses/simple_loss.py` — `MaskedBCE`, `MaskFocalLoss`; `map_loss.py` — `RenderedMaskDiceLoss` with `sample_weighting_with_mask`
- `custom_tools/test_pseudo_labels.py` — evaluates pseudo-labels with no model

`pmt_*.py` are the paper's main results (diff against `maptrv2_gt.py` for the GT baseline); `maptrv2_pseudo_*` and `maptrv2_semi_0.1` are ablations. New datasets/losses/assigners are only reachable from configs once exported through the relevant `__init__.py` under `plugin_dir`.

## HRDX whole-50 m-tile benchmark, three cities (2026-11-09)

Whole-50 m-tile benchmark over three cities (San Jose, San Francisco, San Mateo), defined
by the reference repo `../MapTR_hrdx_wt` (`hrdx_ORIG_L1.py`, `datasets/hrdx50m_dataset.py`)
and ported to all six sister repos (MapTR, GeMap, PseudoMapTrainer, mapdiffusion,
SuperMapNet, maptracker). The benchmark part is the same everywhere; change it in all six or
in none:

- data `data/hrdx_tiles_50m/<city>_tiles_50m_lines/{blocks,reference_lines}`, splits
  `data/hrdx_splits_v1/{train,val,test}.json` (20,499 / 2,227 / 3,392 tiles, each with
  `name`, `city`, `center`), smoke subsets `smoke/{train200,val30}.json`
- sample = the whole tile, +-25 m, no crop, no translation
- GT: one class, `gt_source='parametric'` (`curb_parametric` design samples chained per
  `alignment_id` by `hrdx_param_gt.chain_parametric`), clipped to the box, pieces < 2 m dropped
- input `(x, y, z, strength)`: z re-zeroed on the tile's 2nd percentile, strength =
  intensity / 255; `GridSamplePoints` 0.1 x 0.1 x 0.4 m; shuffled in train
- train aug `PTV3_NUSCENES_AUG`: rotate U(-pi, pi) p 0.5, scale U(0.9, 1.1), flip x / flip y
  p 0.5 each as ONE 2x2 map on points and GT, jitter sigma 0.005 clip 0.02 on points only;
  up to 8 redraws when the augmented tile has no GT; tiles under 5000 in-range points or
  without GT are resampled. RNG order matters for cross-repo identity: rotate coin, angle,
  scale, flip x, flip y, then the loader's `randn(n, 3)` jitter
- val/test never augmented (`crop_seed=0`); eval every 2 epochs, chamfer AP
- schedule: 16 epochs, AdamW wd 0.01, cosine with 4000-iteration linear warmup, batch 2 x 8
  accumulated iterations (`GradientCumulativeFp16OptimizerHook`, loss scale 512, grad clip
  35), checkpoint every epoch keep 4
- GT caches `data/hrdx_splits_v1/{val,test}_gt_param.json` are written at the first
  evaluation (atomically) and checked against the dataset at construction
  (`RuntimeError('stale GT cache ...')` means delete the cache)

Proof of identity: `~/submit_scripts/hrdx_checks/hrdx_dump_mmdet.py CONFIG OUT.npz` dumps
augmentation matrices, GT, raw and piped points for fixed seeds; `hrdx_compare.py REF OTHER`
must print `IDENTICAL`. Verified 2026-11-09 against the reference repo on 4 local San Jose
tiles posing as an HRDX city (the three-city data is only on the cluster).

Training on the cluster no longer uses a container: one shared python environment
(`~/submit_scripts/pc_env/README.md`) and `~/submit_scripts/run_*_hrdx.sh` on top of
`hrdx_common.sh` (repo rsynced from `/scratch/$USER/PC_encoding/<repo>` to the node).

Here (all under `MapVR/`): `datasets/hrdx50m_dataset.py` (`PMTHrdx50mDataset`, the MapTR
file with PMT's parent class and the three extra `get_data_info` keys) and
`hrdx_param_gt.py`. Config `projects/configs/maptrv2/pmt_hrdx_16ep_lidar_50m_tile.py`
(+ `_smoke`): benchmark data and schedule, PMT's own model (no one-to-many branch, no BEV
encoder, `RenderedMaskDiceLoss` on a 128 x 128 raster, `MaskedBCE`, `MaskedMapTRAssigner`),
`mask_mode='ones'`, lr 3e-4, `post_center_range` +-30. Train with `custom_tools/train.py`
(this repo has no `tools/`). Submit script `run_pmt_hrdx.sh`.
