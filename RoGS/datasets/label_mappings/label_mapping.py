# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from abc import abstractmethod
import numpy as np
import cv2



class LabelMapping:
    def __init__(self, label_remaps, color_map):
        self.label_remaps: np.ndarray = label_remaps
        self.color_map: np.ndarray = color_map

    @staticmethod
    @abstractmethod
    def label2mask(self, label):
        """
        Convert label to mask.
        :param label: label image
        :return: mask, label
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def remap_semantic(self, semantic_label):
        semantic_label = semantic_label.astype('uint8')
        remaped_label = np.array(cv2.LUT(semantic_label, self.label_remaps))
        return remaped_label
    

def get_label_mapping(name:str) -> LabelMapping:
    if name == "MapillaryVistas":
        from .mapillary_vistas import MapillaryVistas as LabelMappingCls
    elif name == "MapillaryVistasV2":
        from .mapillary_vistas import MapillaryVistasV2 as LabelMappingCls
    else:
        raise ValueError(f"Unknown label mapping: {name}")
    
    return LabelMappingCls()