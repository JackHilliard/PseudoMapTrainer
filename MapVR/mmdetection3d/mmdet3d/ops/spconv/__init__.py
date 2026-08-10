# Copyright 2019 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Thin re-export shim over the maintained spconv2 (cumm) package.

The vendored spconv 1.x CUDA kernels formerly defined by conv.py/modules.py/
pool.py/structure.py in this directory crash ("cuda execution failed with
error 2") even on trivial inputs under CUDA 11.8 + sm_86/sm_90 -- they
predate those architectures and are no longer usable. Those files are left
in place (unused) only in case something imports them by direct submodule
path; everything that imports this package (`mmdet3d.ops.spconv`) now gets
spconv2's implementation instead.
"""

from mmcv.cnn.bricks.registry import CONV_LAYERS
from spconv.pytorch import (
    ConvAlgo,
    SparseConv2d,
    SparseConv3d,
    SparseConvTensor,
    SparseConvTranspose2d,
    SparseConvTranspose3d,
    SparseInverseConv2d,
    SparseInverseConv3d,
    SparseMaxPool2d,
    SparseMaxPool3d,
    SparseModule,
    SparseSequential,
    SubMConv2d,
    SubMConv3d,
)

# make_sparse_convmodule()/build_conv_layer() resolve conv layers by these
# string names via mmcv's registry; force=True since mmcv's own
# mmcv.ops.sparse_conv ships classes registered under the same names.
for _name, _cls in [
    ("SparseConv2d", SparseConv2d),
    ("SparseConv3d", SparseConv3d),
    ("SparseConvTranspose2d", SparseConvTranspose2d),
    ("SparseConvTranspose3d", SparseConvTranspose3d),
    ("SparseInverseConv2d", SparseInverseConv2d),
    ("SparseInverseConv3d", SparseInverseConv3d),
    ("SubMConv2d", SubMConv2d),
    ("SubMConv3d", SubMConv3d),
]:
    CONV_LAYERS.register_module(name=_name, force=True, module=_cls)

__all__ = [
    "ConvAlgo",
    "SparseConv2d",
    "SparseConv3d",
    "SubMConv2d",
    "SubMConv3d",
    "SparseConvTranspose2d",
    "SparseConvTranspose3d",
    "SparseInverseConv2d",
    "SparseInverseConv3d",
    "SparseModule",
    "SparseSequential",
    "SparseMaxPool2d",
    "SparseMaxPool3d",
    "SparseConvTensor",
]
