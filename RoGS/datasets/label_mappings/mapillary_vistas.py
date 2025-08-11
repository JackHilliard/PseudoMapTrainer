# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from RoGS (a214497)
#   (https://github.com/fzhiheng/RoGS/tree/a21449733c157ca2d58adfc3b8c2225a624ee466)
# Copyright 2024 Zhiheng Feng, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

from .label_mapping import LabelMapping
import numpy as np
import cv2


class MapillaryVistas(LabelMapping):
    """
    [Mapillary Vistas](https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/mask2former/data/datasets/register_mapillary_vistas_panoptic.py#L9)
    label mapping.
    """
    
    def __init__(self, label_remaps=None, color_map=None):
        if label_remaps is None:
            label_remaps = np.ones((256, 1), dtype="uint8")
            label_remaps *= 5  # background
            label_remaps[7,  :] = 3  # Bike Lane -> Road
            label_remaps[8,  :] = 2  # Crosswalk - Plain -> Crosswalk
            label_remaps[14, :] = 3  # Service Lane -> Road
            label_remaps[23, :] = 2  # Lane Marking - Crosswalk -> Crosswalk
            label_remaps[24, :] = 1  # Lane Marking - General -> Lane marking
            label_remaps[2,  :] = 4  # Curb -> Boundary
            label_remaps[9,  :] = 4  # Curb Cut -> Boundary
            label_remaps[41, :] = 3  # Manhole -> Road
            label_remaps[13, :] = 3  # Road -> Road
            label_remaps[15, :] = 4  # Sidewalk -> Boundary
            label_remaps[29, :] = 4  # Terrain -> Boundary

        if color_map is None:
            color_map = np.zeros((256, 1, 3), dtype='uint8')
            color_map[0, :, :] = [0, 0, 0]  # mask
            color_map[1, :, :] = [0, 0, 255]  # all lane
            color_map[2, :, :] = [255, 0, 0]  # curb
            color_map[3, :, :] = [211, 211, 211]  # road and manhole
            color_map[4, :, :] = [0, 191, 255]  # sidewalk
            color_map[5, :, :] = [152, 251, 152]  # terrain
            color_map[6, :, :] = [157, 234, 50]  # background
        
        super().__init__(label_remaps, color_map)
    
    @staticmethod
    def label2mask(label):
        # Bird, Ground Animal, Curb, Fence, Guard Rail,
        # Barrier, Wall, Bike Lane, Crosswalk - Plain, Curb Cut,
        # Parking, Pedestrian Area, Rail Track, Road, Service Lane,
        # Sidewalk, Bridge, Building, Tunnel, Person,
        # Bicyclist, Motorcyclist, Other Rider, Lane Marking - Crosswalk, Lane Marking - General,
        # Mountain, Sand, Sky, Snow, Terrain,
        # Vegetation, Water, Banner, Bench, Bike Rack,
        # Billboard, Catch Basin, CCTV Camera, Fire Hydrant, Junction Box,
        # Mailbox, Manhole, Phone Booth, Pothole, Street Light,
        # Pole, Traffic Sign Frame, Utility Pole, Traffic Light, Traffic Sign (Back),
        # Traffic Sign (Front), Trash Can, Bicycle, Boat, Bus,
        # Car, Caravan, Motorcycle, On Rails, Other Vehicle,
        # Trailer, Truck, Wheeled Slow, Car Mount, Ego Vehicle
        mask = np.ones_like(label)
        label_off_road = ((label <=  1)) \
            | ((3  <= label) & (label <=  6)) \
            | ((10 <= label) & (label <= 12)) \
            | ((16 <= label) & (label <= 22)) \
            | ((25 <= label) & (label <= 28)) \
            | ((30 <= label) & (label <= 40)) \
            | (label >= 42)

        # dilate iteration 2 for moving objects
        label_movable = label >= 52
        kernel = np.ones((10, 10), dtype=np.uint8)
        label_movable = cv2.dilate(label_movable.astype(np.uint8), kernel, 2).astype(bool)

        label_off_road = label_off_road | label_movable
        mask[label_off_road] = 0
        label[~(mask.astype(bool))] = 64
        mask = mask.astype(np.float32)
        return mask, label


class MapillaryVistasV2(MapillaryVistas):
    """
    Maps Mapillary Vistas V2 labels to useful classes for Online Map Construction supervision.
    """

    def __init__(self, label_remaps=None, color_map=None):
        if label_remaps is None:
            label_remaps = np.ones((256, 1), dtype="uint8")
            label_remaps *= 5  # background  # 0 mask # 1 lane # 2 crosswalk # 3 road # 4 boundary # 5 background
            
            label_remaps[4,  :] = 4  # construction--barrier--curb  -> boundary
            label_remaps[8,  :] = 4  # construction--barrier--road-median  -> boundary
            label_remaps[9,  :] = 4  # construction--barrier--road-side -> boundary
            label_remaps[10, :] = 1  # construction--barrier--separator -> lane
            label_remaps[13, :] = 3  # construction--flat--bike-lane -> road
            label_remaps[14, :] = 2  # Crosswalk - Plain -> crosswalk
            label_remaps[15, :] = 4  # Curb Cut -> Boundary
            label_remaps[16, :] = 4  # Driveway -> Boundary
            label_remaps[21:23, :] = 3  # Road,  Road Shoulder, Service Lane  -> Road
            label_remaps[24, :] = 4  # Sidewalk -> Boundary
            label_remaps[25, :] = 4  # Traffic Island -> Boundary
            label_remaps[35:38, :] = 1  # marking--continuous--[dashed,solid,zigzag] -> lane
            label_remaps[38, :] = 1  # marking--continuous--ambiguous -> lane
            label_remaps[39:45, :] = 3  # arrow -> Road
            label_remaps[45, :] = 2  # Lane Marking - Crosswalk -> crosswalk
            label_remaps[46, :] = 3  # marking--discrete--give-way-row  -> road
            label_remaps[47, :] = 3  # marking--discrete--give-way-single  -> road
            label_remaps[48, :] = 4  # Lane Marking - Hatched (Chevron)  -> boundary
            label_remaps[49, :] = 4  # Lane Marking - Hatched (Diagonal)  -> boundary
            label_remaps[50:55, :] = 3  # Lane Marking - [Other, Stop Line, Symbol (Bicycle), Symbol (Other), Text]  -> road
            label_remaps[55, :] = 1  # marking-only--continuous--dashed  -> lane
            label_remaps[56, :] = 2  # Lane Marking (only) - Crosswalk  -> crosswalk
            label_remaps[[57, 58], :] = 3  # Lane Marking (only) - Other, Lane Marking (only) - Test -> Road
            label_remaps[63, :] = 4  # Terrain -> boundary
            label_remaps[69, :] = 3  # Catch Basin -> road
            label_remaps[74, :] = 3  # Manhole -> road
        
        super().__init__(label_remaps, color_map)

    @staticmethod
    def label2mask(label):
        """Masks out the non-road classes in the Mapillary Vistas V2 dataset."""
        mask = np.ones_like(label)
        label_off_road = ((0 <= label) & (label <= 3)) | ((5 <= label) & (label <= 7)) | \
            ((11 <= label) & (label <= 12)) | ((17 <= label) & (label <= 20)) | ((26 <= label) & (label <= 34)) | \
            ((59 <= label) & (label <= 62)) | ((64 <= label) & (label <= 73)) | (75 <= label) 

        # dilate iteration 2 for moving objects
        label_movable = ((105 <= label) & (label <= 117))
        kernel = np.ones((10, 10), dtype=np.uint8)
        label_movable = cv2.dilate(label_movable.astype(np.uint8), kernel, 2).astype(bool)

        label_off_road = label_off_road | label_movable
        mask[label_off_road] = 0
        label[~(mask.astype(bool))] = 120
        mask = mask.astype(np.float32)
        return mask, label