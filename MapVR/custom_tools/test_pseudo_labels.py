# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from MapVR (eec23fe)
#   (https://github.com/ZhangGongjie/MapVR/tree/eec23fe86215b82f7630c0e983777bcfe6677939)
# Copyright (c) 2022 Hust Vision Lab, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import (get_dist_info, init_dist)

from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test, custom_single_gpu_test, vec_list2inter_tensor
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import datetime
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and config')
    parser.add_argument('--masked', action='store_true')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # torch.distributed.launch/torchrun inject --local-rank (hyphen) as of
    # newer torch, but older ones use --local_rank (underscore); accept both.
    parser.add_argument('--local-rank', '--local_rank', dest='local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True


    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))


    # in case the test dataset is concatenated
    samples_per_gpu = 1
    for dataset_cfg in [cfg.data.test, cfg.data.pseudo_test]:
        if isinstance(dataset_cfg, dict):
            dataset_cfg.test_mode = True
            samples_per_gpu = dataset_cfg.pop('samples_per_gpu', 1)
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                dataset_cfg.pipeline = replace_ImageToTensor(
                    dataset_cfg.pipeline)
        elif isinstance(dataset_cfg, list):
            for ds_cfg in dataset_cfg:
                ds_cfg.test_mode = True
            samples_per_gpu = max(
                [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in dataset_cfg])
            if samples_per_gpu > 1:
                for ds_cfg in dataset_cfg:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)


    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=0,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )


    # PSEUDO LABELS
    pseudo_dataset = build_dataset(cfg.data.pseudo_test)

    for dataset_i in [dataset, pseudo_dataset]:
        if not osp.exists(dataset_i.map_ann_file):
            dataset_i._format_gt() # Fix the bug in the original MapVR implementation


    class PseudoDatasetModel:
        """
        Pretends to be a model, but derives labels from Dataset.
        Designed to be compatible with the interface of the model test function.
        """
        def __init__(self, map_ann_file, fixed_ptsnum_per_pred_line, dataset):
            with open(map_ann_file, "r") as f:
                pseudo_data = json.load(f)["GTs"]
            self.data_dict = {}
            self.fixed_ptsnum_per_pred_line = fixed_ptsnum_per_pred_line
            self.dataset = dataset
            for item in pseudo_data:
                self.data_dict[item["sample_token"]] = item

        def __call__(self, img_metas, *args, **kwargs):
            token = img_metas[0].data[0][0]["sample_idx"]
            if token not in self.data_dict:
                fn = img_metas[0].data[0][0]["pts_filename"]
                raise ValueError(f"Token {token} not found in pseudo data with filename: {fn}.")
            else:
                vectors = self.data_dict[token]["vectors"]
                idx = [i for i, info in enumerate(self.dataset.data_infos) if info["token"] == token][0]
                bev_mask = self.dataset.get_data_info(idx)["bev_mask"]

            labels = torch.tensor([vector["type"] for vector in vectors])
            scores = torch.ones(len(vectors))
            pts = vec_list2inter_tensor(vectors, self.fixed_ptsnum_per_pred_line)
            boxes_3d = torch.zeros((len(vectors), 4))

            return_dict = {
                "pts_bbox": {
                    "pts_3d": pts,
                    "boxes_3d": boxes_3d,
                    "scores_3d": scores,
                    "labels_3d": labels,
                },
                "bev_mask": bev_mask
            }

            return [return_dict]
        
        def eval(self):
            pass

    model = PseudoDatasetModel(pseudo_dataset.map_ann_file, cfg.fixed_ptsnum_per_pred_line, pseudo_dataset)

    if not distributed:
        outputs, outputs_coco = custom_single_gpu_test(model, data_loader, args.masked)
    else:
        outputs, outputs_coco = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                        args.gpu_collect, args.masked)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            assert False
            #mmcv.dump(outputs['bbox_results'], args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            eval_result = {}
            for output in outputs_coco:
                eval_result[output["type"]] = output["evaluator"].summarize(output["results"])
            date = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            output_path = os.path.join(cfg.work_dir, f"{'masked_' if args.masked else ''}results_{date}.json")
            with open(output_path, "w") as f:
                json.dump(eval_result, f)
            print(f"Results are written to {output_path}")
            if args.masked:
                chamfer_res = dataset.evaluate_with_pseudo_mask(outputs, **eval_kwargs)
            else:
                chamfer_res = dataset.evaluate(outputs, **eval_kwargs)
            print(chamfer_res)
            chamfer_output_path = os.path.join(cfg.work_dir, f"{'masked_' if args.masked else ''}chamfer_results_{date}.json")
            with open(chamfer_output_path, "w") as f:
                json.dump(chamfer_res, f)



if __name__ == '__main__':
    main()
