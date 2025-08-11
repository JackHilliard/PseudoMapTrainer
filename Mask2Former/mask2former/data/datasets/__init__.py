# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from Mask2Former (9b0651c)
#   (https://github.com/facebookresearch/Mask2Former/tree/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a)
# Copyright (c) Facebook, Inc. and its affiliates, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

from . import (
    register_ade20k_full,
    register_ade20k_panoptic,
    register_coco_stuff_10k,
    register_mapillary_vistas,
    register_mapillary_vistas_v2,
    register_coco_panoptic_annos_semseg,
    register_ade20k_instance,
    register_mapillary_vistas_panoptic,
)
