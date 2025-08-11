# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
from diff_gaussian_rasterization.scene.cameras import OrthographicCamera

def project2z(R: torch.Tensor) -> torch.Tensor:
    # Extract the upper-left 2x2 submatrix
    R2 = R[:2, :2]

    # Perform SVD on R2
    U, _, Vt = torch.linalg.svd(R2)

    # Reconstruct the orthogonal 2x2 matrix
    R2_ortho = U @ Vt

    # Construct the re-orthogonalized 3x3 rotation matrix
    R_ortho = torch.eye(3)
    R_ortho[:2, :2] = R2_ortho
    return R_ortho


class MapTRCamera(OrthographicCamera):
    """BEV Camera to generate pseudo labels for MapTR pre-training"""
    
    # Rotation matrix flips z matrix
    R_z_flip = torch.tensor([[1, 0,  0],
                             [0, 1,  0],
                             [0, 0, -1]], dtype=torch.float32)

    def __init__(self, lidar_pose, ref_pose, device, bev_h:int=60, bev_w:int=30, pixel_per_meter:int=20, apply_z_proj=True):
        """
        Args:
            lidar_pose: 4x4 torch.Tensor, the lidar pose of the vehicle
            ref_pose: 4x4 torch.Tensor, the reference pose of the dataset
            device: torch.device, the device to create the camera on
            bev_h: float, the height of the BEV image in meters, default 60
            bev_w: float, the width of the BEV image in meters, default 30
            pixel_per_meter: int, for the resolution of the BEV image in pixel/meters, default 20
            apply_z_proj: bool, whether to project the rotation matrix to the z plane
        """
        self.pixel_per_meter = pixel_per_meter
        self.lidar_pose = lidar_pose
        self.ref_pose = ref_pose
        self.apply_z_proj = apply_z_proj

        width  = bev_w * self.pixel_per_meter
        height = bev_h * self.pixel_per_meter

        self.top = -bev_h * 0.5
        self.bottom = bev_h * 0.5
        self.right = bev_w * 0.5
        self.left = -bev_w * 0.5

        R = (self.ref_pose@self.lidar_pose)[:3, :3]
        if self.apply_z_proj:
            R = project2z(R) # project to the z plane of the map coordinate system
        R = torch.linalg.inv(self.ref_pose)[:3, :3] @ R

        R = MapTRCamera.R_z_flip @ R

        t = lidar_pose[:3, 3]
        # move camera back along its own z-axis.
        # Using 1000, so its sufficiently large.
        t = t - 1000 * R[:, 2]

        super().__init__(
            R=R, T=t, W=width, H=height, znear=0, zfar=torch.inf,
            top=self.top, bottom=self.bottom, right=self.right,
            left=self.left, device=device
        )
    
    @staticmethod
    def create_orthogonal_to_ground_camera(lidar_pose, device, bev_h:int=60, bev_w:int=30, pixel_per_meter:int=20):
        """MapTRCamera that is orthogonal to the ground plane"""
        return MapTRCamera(lidar_pose=lidar_pose, ref_pose=torch.eye(4), device=device, \
                      bev_h=bev_h, bev_w=bev_w, pixel_per_meter=pixel_per_meter, apply_z_proj=False)


    def pad_image(self, pad_width: int, pad_height: int=None):
        """Pad the final image by the camera by the given amount of width and height in pixels"""
        if pad_height is None:
            pad_height = pad_width
        assert (pad_width * 2) % self.pixel_per_meter == 0 and (pad_height * 2) % self.pixel_per_meter == 0, \
            f"Padding must be a multiple of pixel_per_meter, i.e. {self.pixel_per_meter}."

        bev_w = (self.image_width  + pad_width  * 2) // self.pixel_per_meter
        bev_h = (self.image_height + pad_height * 2) // self.pixel_per_meter
        return MapTRCamera(lidar_pose=self.lidar_pose, ref_pose=self.ref_pose, device=self.data_device, \
                      bev_h=bev_h, bev_w=bev_w, pixel_per_meter=self.pixel_per_meter, apply_z_proj=self.apply_z_proj)
