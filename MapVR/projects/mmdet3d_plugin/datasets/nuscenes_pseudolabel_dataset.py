# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from mmdet.datasets import DATASETS
import numpy as np
from PIL import Image
from shapely.geometry import LineString
import pathlib
import os.path as osp

from .nuscenes_offlinemap_dataset import CustomNuScenesOfflineLocalMapDataset, LiDARInstanceLines, VectorizedLocalMap

from skimage.measure import block_reduce
from PIL import Image
import numpy as np
import cv2

CLS_DIVIDER = 1
CLS_PED_CROSS = 2
CLS_ROAD = 3
CLS_BOUNDARY = 4
CLS_BKG = 5



def get_nusc_onmap_label_remaps():
    """Maps Mapillary Vistas V2 labels to useful classes for Online Mapping supervision."""
    colors = np.ones((256, 1), dtype="uint8")
    colors *= CLS_BKG  # background  # 0 mask # 1 lane # 2 crosswalk # 3 road # 4 boundary # 5 background
    
    colors[4,  :] = CLS_BOUNDARY  # construction--barrier--curb  -> boundary
    colors[8,  :] = CLS_BOUNDARY  # construction--barrier--road-median  -> boundary
    colors[9,  :] = CLS_BOUNDARY  # construction--barrier--road-side -> boundary
    colors[10, :] = CLS_DIVIDER  # construction--barrier--separator -> lane
    colors[13, :] = CLS_ROAD  # construction--flat--bike-lane -> road
    colors[14, :] = CLS_PED_CROSS  # Crosswalk - Plain -> crosswalk
    colors[15, :] = CLS_BOUNDARY  # Curb Cut -> Boundary
    colors[16, :] = CLS_BOUNDARY  # Driveway -> Boundary
    colors[21:23, :] = CLS_ROAD  # Road,  Road Shoulder, Service Lane  -> Road
    colors[24, :] = CLS_BOUNDARY  # Sidewalk -> Boundary
    colors[25, :] = CLS_BOUNDARY  # Traffic Island -> Boundary
    colors[35:38, :] = CLS_DIVIDER  # marking--continuous--[dashed,solid,zigzag] -> lane
    colors[38, :] = CLS_DIVIDER  # marking--continuous--ambiguous -> lane
    colors[39:45, :] = CLS_ROAD  # arrow -> Road
    colors[45, :] = CLS_PED_CROSS  # Lane Marking - Crosswalk -> crosswalk
    colors[46, :] = CLS_ROAD  # marking--discrete--give-way-row  -> road
    colors[47, :] = CLS_ROAD  # marking--discrete--give-way-single  -> road
    colors[48, :] = CLS_BOUNDARY  # Lane Marking - Hatched (Chevron)  -> boundary
    colors[49, :] = CLS_BOUNDARY  # Lane Marking - Hatched (Diagonal)  -> boundary
    colors[50:55, :] = CLS_ROAD  # Lane Marking - [Other, Stop Line, Symbol (Bicycle), Symbol (Other), Text]  -> road
    colors[55, :] = CLS_DIVIDER  # marking-only--continuous--dashed  -> lane
    colors[56, :] = CLS_PED_CROSS  # Lane Marking (only) - Crosswalk  -> crosswalk
    colors[[57, 58], :] = CLS_ROAD  # Lane Marking (only) - Other, Lane Marking (only) - Test -> Road
    colors[63, :] = CLS_BOUNDARY  # Terrain -> boundary
    colors[69, :] = CLS_ROAD  # Catch Basin -> road
    colors[74, :] = CLS_ROAD  # Manhole -> road
    
    return colors


def remove_small_segs_binary(mask, max_area=400):
    """
    Remove small connected components from a binary mask. The function iterates over each connected component and
    performs a check to determine if the region contains fewer than 'max_area' pixels. If the region is smaller
    than the threshold, the entire connected component is removed by setting its pixels to 0.
    """
    mask_post = mask.copy()

    # Label connected components
    num_components, components, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)

    # Iterate over each connected component (excluding background label 0)
    for i in range(1, num_components):
        mask_post[components == i] = stats[i, cv2.CC_STAT_AREA] > max_area

    return mask_post.astype(bool)


def extract_border(segmentation, kernel_size=3):
    """
    Extract the border between CLS_ROAD and CLS_BOUNDARY in a segmentation map.

    Parameters:
    segmentation (numpy.ndarray): The segmentation map with shape (H, W).
    kernel_size (int): The size of the kernel used for morphological operations.

    Returns:
    numpy.ndarray: An image highlighting the border between CLS_ROAD and CLS_BOUNDARY.
    """
    # Create binary masks for CLS_ROAD and CLS_BOUNDARY
    class3_mask = (segmentation == CLS_ROAD).astype(np.uint8)
    class4_mask = (segmentation == CLS_BOUNDARY).astype(np.uint8)

    # Define a 3x3 kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Find the edges of the CLS_ROAD regions
    class3_edges = cv2.morphologyEx(class3_mask, cv2.MORPH_GRADIENT, kernel)

    # Find where CLS_ROAD edges touch CLS_BOUNDARY regions
    border = cv2.bitwise_and(class3_edges, class4_mask)

    border = remove_small_segs_binary(border, max_area=300).astype(bool)

    return border




class PseudoVectorizedMap(VectorizedLocalMap):
    def __init__(self, data_root=None, data_root_seg=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_root = data_root
        self.data_root_seg = data_root_seg

    def extract_pv_seg(self, img_metas, cam_idx, feat_down_sample):
        """
        Extracts the pseudo segmentation map for the given camera index.
        It assumes that the segmentation map is stored in a specific structure, such as
        data_root_seg/samples/seg_CAM_x/xxxx__CAM_x__xxxx.png
        """
        # extract segmentation path. filepath must have a specific structure
        img_path = img_metas.data["filename"][cam_idx]
        assert self.data_root in img_path, f"img_path ({img_path}) should come from data_root ({self.data_root}) for PV segmentation extraction"
        seg_path = str(pathlib.Path(img_path).relative_to(self.data_root))
        seg_path = osp.join(self.data_root_seg, seg_path).replace("/CAM", "/seg_CAM").replace(".jpg", ".png")

        seg = Image.open(seg_path)

        H, W, _ = img_metas.data["ori_shape"][cam_idx]
        pH, pW, _ = img_metas.data["pad_shape"][cam_idx]

        seg = seg.resize([W, H], resample=Image.NEAREST)
        seg = np.array(seg)
        seg = np.array(cv2.LUT(seg, get_nusc_onmap_label_remaps()))
        seg_lane_div = seg == CLS_DIVIDER
        seg_boundary = extract_border(seg)
        seg_ped_cross = seg == CLS_PED_CROSS
        seg = seg_lane_div | seg_boundary | seg_ped_cross

        seg = np.pad(seg, ((0, pH-H), (0, pW-W)), mode='constant', constant_values=0)
        maxpool_arr = block_reduce(seg, block_size=(feat_down_sample, feat_down_sample), func=np.max) 

        return maxpool_arr.astype(np.uint8)

    def gen_vectorized_samples(self, map_annotation, example=None, feat_down_sample=32):
        '''
        use lidar2global to get gt map layers
        '''

        vectors = []
        for vec_class in self.vec_classes:
            instance_list = map_annotation[vec_class]
            for instance in instance_list:
                vectors.append((LineString(np.array(instance)), self.CLASS2LABEL.get(vec_class, -1)))
        gt_labels = []
        gt_instance = []
        if self.aux_seg['use_aux_seg']:
            if self.aux_seg['seg_classes'] == 1:
                if self.aux_seg['bev_seg']:
                    gt_semantic_mask = np.zeros((1, self.canvas_size[0], self.canvas_size[1]), dtype=np.uint8)
                else:
                    gt_semantic_mask = None
                if self.aux_seg['pv_seg']:
                    num_cam  = len(example['img_metas'].data['pad_shape'])
                    img_shape = example['img_metas'].data['pad_shape'][0]
                    gt_pv_semantic_mask = np.zeros((num_cam, 1, img_shape[0] // feat_down_sample, img_shape[1] // feat_down_sample), dtype=np.uint8)
                    for cam_index in range(num_cam):
                        gt_pv_semantic_mask[cam_index,0] = self.extract_pv_seg(example['img_metas'], cam_index, feat_down_sample)
                else:
                    gt_pv_semantic_mask = None
                for instance, instance_type in vectors:
                    if instance_type != -1:
                        gt_instance.append(instance)
                        gt_labels.append(instance_type)
                        if instance.geom_type == 'LineString':
                            if self.aux_seg['bev_seg']:
                                self.line_ego_to_mask(instance, gt_semantic_mask[0], color=1, thickness=self.thickness)
                        else:
                            print(instance.geom_type)
            else:
                raise NotImplementedError("only binary segmentation supported for pseudo labels for now")
        else:
            for instance, instance_type in vectors:
                if instance_type != -1:
                    gt_instance.append(instance)
                    gt_labels.append(instance_type)
            gt_semantic_mask=None
            gt_pv_semantic_mask=None
        gt_instance = LiDARInstanceLines(gt_instance, gt_labels, self.sample_dist,
                        self.num_samples, self.padding, self.fixed_num,self.padding_value, patch_size=self.patch_size)


        anns_results = dict(
            gt_vecs_pts_loc=gt_instance,
            gt_vecs_label=gt_labels,
            gt_semantic_mask=gt_semantic_mask,
            gt_pv_semantic_mask=gt_pv_semantic_mask,
        )
        return anns_results


@DATASETS.register_module()
class PseudoMapDataset(CustomNuScenesOfflineLocalMapDataset):
    def __init__(self, raster:list, mask_thresh=0.3, data_root_seg=None, use_pv_pseudo_seg=True, mask_only_for_sample_filter=False, *args, **kwargs):
        """
        Args:
            raster (list[int]): List with following entries: [raster_W, raster_H].
        """
        super().__init__(*args, **kwargs)
        self.raster = raster
        self.mask_thresh = mask_thresh
        self.mask_only_for_sample_filter = mask_only_for_sample_filter

        if use_pv_pseudo_seg: # use pseudo segmentation labels produced by Mask2Former
            assert (data_root_seg is not None) and (self.data_root is not None), \
                    "data_root and data_root_seg must be provided for pseudo segmentation extraction"
            self.vector_map = PseudoVectorizedMap(data_root=self.data_root,
                                                data_root_seg=data_root_seg,
                                                canvas_size=kwargs["bev_size"],
                                                patch_size=self.patch_size, 
                                                map_classes=self.MAPCLASSES, 
                                                fixed_ptsnum_per_line=kwargs["fixed_ptsnum_per_line"],
                                                padding_value=self.padding_value,
                                                aux_seg=kwargs["aux_seg"])

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]

        # filter out samples with low information
        bev_mask = Image.open(info["pseudo_mask_path"])
        bev_mask = bev_mask.resize(self.raster, resample=Image.NEAREST)
        bev_mask = np.array(bev_mask)
        if bev_mask.mean() < self.mask_thresh:
            return None

        data = super().get_data_info(index)

        if self.mask_only_for_sample_filter:
            # use the mask only for sample filtering and not for training
            return data

        # load BEV raster from bev_label.png as an numpy array
        pseudo_rast = Image.open(info["pseudo_rast_path"])

        pseudo_rast = pseudo_rast.resize(self.raster, resample=Image.NEAREST)
        pseudo_rast = np.array(pseudo_rast)

        data["bev_mask"] = bev_mask
        data["bev_label"] = pseudo_rast

        return data
