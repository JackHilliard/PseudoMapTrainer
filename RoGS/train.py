# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from RoGS (a214497)
#   (https://github.com/fzhiheng/RoGS/tree/a21449733c157ca2d58adfc3b8c2225a624ee466)
# Copyright 2024 Zhiheng Feng, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import os
import datetime
import random
import argparse
import math

GUI_FLAG = False
try:
    import mayavi.mlab as mlab
    from utils.vis import vis_gaussian_3d

    GUI_FLAG = True
except ImportError:
    pass
import cv2
import yaml
import pytz
import torch
import addict
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.structures import Pointclouds
from torch.utils.data import DataLoader
from diff_gaussian_rasterization.scene.cameras import PerspectiveCamera

from utils.logging import create_logger, DummyLogger
from utils.image import render_semantic
from utils.render import render, render_label
from utils.visualizer import loss2color, depth2color, CustomPointVisualizer
from models.road import Road
from models.loss import L1MaskedLoss, CELossWithMask
from models.exposure_model import ExposureModel
from models.gaussian_model import GaussianModel2D
from eval import eval_bev_metric, eval_z_metric
from datasets.base import get_dataset_cls

from post_process import MASK_CLASS


def set_randomness(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_configs():
    parser = argparse.ArgumentParser(description='G4M config')
    parser.add_argument('--config', default="configs/local_nusc_mini.yaml", help='config yaml path')
    args = parser.parse_args()
    with open(args.config) as file:
        configs = yaml.safe_load(file)
    configs["file"] = os.path.abspath(args.config)
    return configs


def gt_render(dataset, min_xy, max_xy, bev_cam_height, wh=None, resolution=None, save_root=None, device="cuda:0"):
    road_pointcloud = dataset.road_pointcloud
    for key, value in road_pointcloud.items():
        road_pointcloud[key] = torch.from_numpy(value.astype(np.float32)).to(device)

    if save_root is not None:
        features = torch.cat((road_pointcloud["rgb"], road_pointcloud["label"]), dim=1)
        pointclouds = Pointclouds(points=[road_pointcloud["xyz"]], features=[features])
        pointclouds.extend(1)
        gt_bev_visualizer = CustomPointVisualizer(device, min_xy, max_xy, bev_cam_height, wh=wh, resolution=resolution)
        point_feature, depth = gt_bev_visualizer(pointclouds)
        point_feature = point_feature[0].detach().cpu().numpy()

        point_rgb = point_feature[..., :3]  # (H,W,3)
        bev_gt_label = dataset.remap_semantic(point_feature[..., -2])  # (H,W)
        bev_point_mask = point_feature[..., -1] > 0  # (H,W)
        point_height = bev_cam_height - depth[0, :, :, 0]  # (H,W)
        point_height = point_height.detach().cpu().numpy()  # (H,W)

        os.makedirs(save_root, exist_ok=True)
        np.save(os.path.join(save_root, f"bev_height.npy"), point_height)

        point_rgb[~bev_point_mask] = [0, 0, 0]
        vis_bev_gt_seg = render_semantic(bev_gt_label, dataset.filted_color_map)
        vis_bev_gt_seg[~bev_point_mask] = [0, 0, 0]
        cv2.imwrite(os.path.join(save_root, "bev_image.png"), cv2.cvtColor((point_rgb * 225).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_root, "bev_label.png"), bev_gt_label)
        cv2.imwrite(os.path.join(save_root, "bev_mask.png"), bev_point_mask.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(save_root, "bev_label_vis.png"), cv2.cvtColor(vis_bev_gt_seg, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_root, "bev_height_vis.png"), cv2.cvtColor(depth2color(point_height, bev_point_mask), cv2.COLOR_RGB2BGR))


def train(configs, dataset_kwargs={}):
    dataset_cfg = configs.dataset
    model_cfg = configs.model
    pipe = configs.pipeline
    opt = configs.optimization
    train_cfg = configs.train

    set_randomness(configs.seed)
    torch.autograd.set_detect_anomaly(False)

    tz = pytz.timezone('Asia/Shanghai')
    output_root = os.path.join(configs.output, f'{datetime.datetime.now(tz).strftime("%m-%d-%H-%M")}')
    if opt.z_weight > 0:
        output_root += "-z"
    else:
        output_root += "-no_z"
    if train_cfg.log or train_cfg.store_artifacts or train_cfg.output_render or train_cfg.store_cfg:
        os.makedirs(output_root, exist_ok=True)
    logger = create_logger(f"RoGS", os.path.join(output_root, "train.log")) if train_cfg.log else DummyLogger()
    img_root = os.path.join(output_root, "images")

    # backup
    if train_cfg.store_cfg:
        os.system(f"cp {configs.file} {output_root}")

    device = torch.device(configs["device"] if torch.cuda.is_available() else "cpu")
    Dataset = get_dataset_cls(dataset_cfg["dataset"])

    use_lidar = opt.depth_loss_weight > 0 or opt.z_weight > 0
    dataset = Dataset(dataset_cfg, use_label=opt.seg_loss_weight > 0,
        use_depth=opt.depth_loss_weight > 0, use_lidar=use_lidar, **dataset_kwargs)
    logger.info(f"Dataset cameras_extent: {dataset.cameras_extent} - size: {len(dataset)}")
    road = Road(model_cfg, dataset, device=device, vis=train_cfg.vis and GUI_FLAG)
    gaussians = GaussianModel2D(model_cfg)
    gaussians.init_2d_gaussian(road.vertices, road.rotation, road.rgb, road.label, road.resolution, dataset.ref_pose, dataset.cameras_extent)
    opt["position_lr_max_steps"] = len(dataset) * opt.epochs
    gaussians.training_setup(opt)

    bev_cam = road.bev_camera
    bev_cam_height = bev_cam.cam2world[2, 3]

    road_pointcloud = dataset.road_pointcloud
    if road_pointcloud is not None:
        road_point_root = os.path.join(output_root, "road_point") if train_cfg.store_artifacts else None
        gt_render(dataset, road.min_xy, road.max_xy, bev_cam_height, wh=(bev_cam.image_width, bev_cam.image_height), save_root=road_point_root, device=device)

    exposure_model = ExposureModel(num_camera=len(dataset.camera_names)).to(device)
    exposure_optimizer = torch.optim.Adam(exposure_model.parameters(), lr=opt.exposure_lr)

    first_iter = 0
    if train_cfg.start_checkpoint:
        (model_params, first_iter) = torch.load(train_cfg.start_checkpoint)
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if model_cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    loss_function = L1MaskedLoss()
    depth_loss_function = L1MaskedLoss()
    CE_loss_with_mask = CELossWithMask()

    gaussian_xy = gaussians.get_xyz[:, :2]
    if opt.smooth_loss_weight > 0:
        # =====> smooth xyz
        # near_idx = knn_points(gaussian_xy[None], gaussian_xy[None], K=5, return_nn=False).idx  # (1, n, k)
        # near_idx = near_idx.squeeze(0)  # (n,k)
        near_idx = road.four_indices  # (n,4) faster
    if opt.z_weight > 0 and road_pointcloud is not None:
        if gaussian_xy.shape[0] < road_pointcloud["xyz"].shape[0]:
            sample_idx = torch.randint(0, road_pointcloud["xyz"].shape[0], (gaussian_xy.shape[0],))
            road_xyz = road_pointcloud["xyz"][sample_idx]
        else:
            road_xyz = road_pointcloud["xyz"]
        road_xyz_near_idx = knn_points(gaussian_xy[None], road_xyz[None, :, :2], K=1, return_nn=False).idx  # (1, n, 1)
        road_xyz_near_idx = road_xyz_near_idx.squeeze(0)  # (n,1)

    logger.info(f"render loss: ON")
    logger.info(f"smooth loss: {'ON' if opt.smooth_loss_weight > 0 else 'OFF'}")
    logger.info(f"semantic loss: {'ON' if opt.seg_loss_weight > 0 else 'OFF'}")
    logger.info(f"z loss: {'ON' if opt.z_weight > 0 and road_pointcloud is not None else 'OFF'}")
    logger.info(f"depth loss: {'ON' if opt.depth_loss_weight > 0 else 'OFF'}")

    cost_time = 0
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.epochs * len(dataset)), desc="Training progress")
    first_iter += 1
    iteration = first_iter
    max_iter = opt.epochs * len(dataset)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True, drop_last=True)
    for epoch in range(opt.epochs):
        for sample in dataloader:
            iter_start.record()
            for key, value in sample.items():
                if key != "image_name":
                    sample[key] = value[0].to(device)
                else:
                    sample[key] = value[0]
            gaussians.update_learning_rate(iteration)
            if not gaussians.use_rgb and iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            image_name = sample["image_name"]
            gt_image = sample["image"]
            image_idx = sample["idx"].item()
            cam_idx = sample["cam_idx"].item()
            R, T = sample["R"], sample["T"]

            NEAR, FAR = 1, 20
            viewpoint_cam = PerspectiveCamera(R, T, sample["K"], sample["W"], sample["H"], NEAR, FAR, device)

            bg = torch.rand((3), device="cuda") if opt.random_background else background
            bg_label = torch.zeros(dataset.num_class, dtype=torch.float32, device=bg.device) # 6 classes
            bg_label[MASK_CLASS] = 1

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            src_render_image, render_depth, render_mask = render_pkg["render"], render_pkg["depth"][0], render_pkg["mask"]
            visibility_filter = render_pkg["visibility_filter"]
            hit_num = torch.sum(visibility_filter)
            render_image = exposure_model(cam_idx, src_render_image)
            render_image = render_image.permute(1, 2, 0)

            valid_mask = torch.bitwise_and(render_depth.detach() > viewpoint_cam.znear, render_depth.detach() < viewpoint_cam.zfar)
            loss_mask = valid_mask.float()
            if "mask" in sample:
                seg_mask = sample["mask"]
                loss_mask *= seg_mask
            # erode and dilate
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            loss_mask = loss_mask.cpu().numpy().astype(np.uint8)
            loss_mask = cv2.erode(loss_mask, kernel)
            loss_mask = cv2.dilate(loss_mask, kernel)
            loss_mask = torch.tensor(loss_mask, device=device)

            render_loss = loss_function(render_image, gt_image, loss_mask[:, :, None])
            total_loss = render_loss.mean()

            current_gaussian_xyz = gaussians.get_xyz
            if opt.depth_loss_weight > 0:
                gt_depth = sample["depth"]
                gt_mask_depth = gt_depth > 0
                depth_loss = depth_loss_function(render_depth, gt_depth, loss_mask * gt_mask_depth.float()) * opt.depth_loss_weight
                total_loss += depth_loss.mean()

            if opt.seg_loss_weight > 0:
                gt_seg = sample["label"]
                label_feature = render_label(viewpoint_cam, gaussians, pipe, bg)
                render_seg = label_feature["render"]
                render_seg = render_seg.permute(1, 2, 0)
                seg_loss = CE_loss_with_mask(render_seg.reshape(-1, render_seg.shape[-1]), gt_seg.reshape(-1), loss_mask.reshape(-1)) * opt.seg_loss_weight
                total_loss += seg_loss

            if opt.smooth_loss_weight > 0:
                vis_z = current_gaussian_xyz[:, 2:][visibility_filter]  # (m,1)
                cur_near_idx = near_idx[visibility_filter]  # (m,k)
                near_z = knn_gather(current_gaussian_xyz[:, 2:].unsqueeze(0), cur_near_idx.unsqueeze(0))  # (1, m, k, 1)
                near_z = near_z.squeeze(0)  # (m,k,1)
                z_smooth_loss = torch.norm(near_z - vis_z[:, None, :]).mean() * opt.smooth_loss_weight
                total_loss += z_smooth_loss

            if opt.z_weight > 0 and road_pointcloud is not None:
                cam_xy = T[:2]
                surround_min_xy = cam_xy - 10
                surround_max_xy = cam_xy + 10
                surround_filter1 = torch.logical_and(current_gaussian_xyz[:, 0] > surround_min_xy[0], current_gaussian_xyz[:, 0] < surround_max_xy[0])
                surround_filter2 = torch.logical_and(current_gaussian_xyz[:, 1] > surround_min_xy[1], current_gaussian_xyz[:, 1] < surround_max_xy[1])
                surround_filter = torch.logical_and(surround_filter1, surround_filter2)
                vis_z = gaussians.get_xyz[:, 2:][surround_filter]  # (m,1)
                cur_near_idx = road_xyz_near_idx[surround_filter]  # (m,k)
                near_z = knn_gather(road_xyz[:, 2:3].unsqueeze(0), cur_near_idx.unsqueeze(0))  # (1, m, k, 1)
                near_z = near_z.squeeze(0)  # (m,k,1)
                z_loss = torch.norm(near_z - vis_z[:, None, :]).mean() * opt.z_weight
                total_loss += z_loss

            total_loss.backward()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            exposure_optimizer.step()
            exposure_optimizer.zero_grad(set_to_none=True)

            iter_end.record()
            torch.cuda.synchronize()
            cost_time += iter_start.elapsed_time(iter_end)

            with torch.no_grad():
                ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "hit_num": f"{hit_num.item()}/{len(visibility_filter)}"})
                    progress_bar.update(10)
                if iteration == max_iter - 1:
                    progress_bar.close()
            iteration += 1

        if train_cfg.output_render: # Visualization
            current_root = os.path.join(img_root, f"EPOCH-{epoch}_IDX-{image_idx}", f"{image_name}")
            if epoch == opt.epochs - 1:
                final_root = os.path.join(img_root, "final")
                current_root = final_root
            os.makedirs(current_root, exist_ok=True)
            gt_image = gt_image * 255
            gt_image = gt_image.cpu().numpy().astype(np.uint8)
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR)
            gt_label = render_semantic(gt_seg.cpu().numpy(), dataset.filted_color_map)  # RGB fomat
            gt_label = cv2.cvtColor(gt_label, cv2.COLOR_RGB2BGR)
            mask = seg_mask.cpu().numpy() * 255
            mask_ = np.stack([mask, mask, mask], axis=-1).astype(np.uint8)
            gt_image_mask = cv2.addWeighted(gt_image, 0.5, mask_, 0.5, 0)
            gt_blend = cv2.addWeighted(gt_image, 0.7, gt_label, 0.3, 0)

            render_mask = render_mask.detach().cpu().numpy()
            vis_render_seg = render_semantic(np.argmax(render_seg.detach().cpu().numpy(), axis=-1), dataset.filted_color_map)
            vis_render_seg = cv2.cvtColor(vis_render_seg, cv2.COLOR_RGB2BGR)
            vis_render_seg[~render_mask] = 0
            render_image = render_image.detach().cpu().numpy() * 255
            render_image = cv2.cvtColor(render_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            render_image[~render_mask] = 0
            render_blend = cv2.addWeighted(render_image, 0.7, vis_render_seg, 0.3, 0)

            vis_loss_mask = loss_mask.cpu().numpy().astype(np.uint8) * 255
            cv2.imwrite(os.path.join(current_root, f"loss_mask.png"), vis_loss_mask)

            vis_render_loss = loss2color(render_loss.detach().cpu().numpy())
            vis_render_depth = depth2color(render_depth.detach().cpu().numpy(), render_mask)
            vis_render_depth = cv2.cvtColor(vis_render_depth, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(current_root, f"gt_image.png"), gt_image)
            cv2.imwrite(os.path.join(current_root, f"gt_mask.png"), gt_image_mask)
            cv2.imwrite(os.path.join(current_root, f"gt_label.png"), gt_label)
            cv2.imwrite(os.path.join(current_root, f"gt_blend.png"), gt_blend)

            src_render_image = src_render_image.permute(1, 2, 0)  # (H, W, 3)
            src_render_image = src_render_image.detach().cpu().numpy() * 255
            src_render_image = cv2.cvtColor(src_render_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(current_root, f"render_src_image.png"), src_render_image)
            cv2.imwrite(os.path.join(current_root, f"render_image.png"), render_image)
            cv2.imwrite(os.path.join(current_root, f"render_label_vis.png"), vis_render_seg)
            cv2.imwrite(os.path.join(current_root, f"render_blend.png"), render_blend)
            cv2.imwrite(os.path.join(current_root, f"render_loss.png"), vis_render_loss)
            cv2.imwrite(os.path.join(current_root, f"render_depth_vis.png"), vis_render_depth)

            # BEV render
            # TODO for very large scene, should render chunk by chunk
            bev_pkg = render(bev_cam, gaussians, pipe, bg)
            src_bev_image, bev_depth, bev_mask = bev_pkg["render"], bev_pkg["depth"], bev_pkg["mask"]
            bev_height = bev_cam_height - bev_depth
            bev_height = bev_height[0].detach().cpu().numpy()

            bev_mask = bev_mask.cpu().numpy()
            bev_mask = bev_mask.astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            bev_mask = cv2.erode(bev_mask, kernel)
            bev_mask = bev_mask > 0

            # save space
            if bev_mask.sum() < 0.3 * bev_mask.size:
                bev_height[~bev_mask] = 0
                sparse_matrix = sp.csr_matrix(bev_height)
                np.savez(os.path.join(current_root, f"bev_height.npz"), data=sparse_matrix.data, indices=sparse_matrix.indices, indptr=sparse_matrix.indptr,
                         shape=sparse_matrix.shape)
            else:
                np.save(os.path.join(current_root, f"bev_height.npy"), bev_height)

            bev_image = exposure_model(0, src_bev_image)
            bev_image = bev_image.permute(1, 2, 0)
            bev_image = bev_image.detach().cpu().numpy() * 255
            bev_image = cv2.cvtColor(bev_image.astype(np.uint8), cv2.COLOR_RGB2BGRA)
            bev_image[~bev_mask] = 0

            src_bev_image = src_bev_image.permute(1, 2, 0)
            src_bev_image = src_bev_image.detach().cpu().numpy() * 255
            src_bev_image = cv2.cvtColor(src_bev_image.astype(np.uint8), cv2.COLOR_RGB2BGRA)
            src_bev_image[~bev_mask] = 0

            vis_bev_height = depth2color(bev_height, mask=bev_mask)
            vis_bev_height = cv2.cvtColor(vis_bev_height, cv2.COLOR_RGB2BGRA)


    logger.info(f"Opt has end! It cost time: {cost_time / 1000} s")

    return gaussians, exposure_model, dataset, road


if __name__ == "__main__":
    configs = get_configs()
    configs = addict.Dict(configs)
    train(configs)
