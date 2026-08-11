# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

#
# pmt_carla_lidar.py with a real visibility mask instead of an all-ones one.
#
# The mask is the tile's own LiDAR coverage in BEV: cells with returns are
# "observed", the rest are not. This is what makes PseudoMapTrainer's masked
# machinery do actual work on CARLA -- GT polyline segments crossing
# unobserved ground stop being charged against predictions, and the
# assigner's allow_split lets several predictions cover one GT line across a
# coverage gap. Everything else is identical to pmt_carla_lidar.py, so the
# two runs isolate exactly the contribution of the mask.
#
# The two knobs below decide what "observed" means, and they are a research
# choice rather than a fact about the data: min_points_per_cell too high
# erodes thinly-scanned area and silently throws supervision away, while a
# large close_kernel fills real gaps back in and makes the mask a no-op.
# Dump a few masks and look at them before trusting any numbers.
#
# Inherits the 30 x 30 m geometry. For the same thing on the 25 x 25 m export,
# there is no second file: add
#   --cfg-options data.train.mask_mode=lidar_coverage data.train.mask_thresh=0.3
# to a pmt_carla_lidar_25m.py run.
_base_ = ['./pmt_carla_lidar.py']

data = dict(
    train=dict(
        mask_mode='lidar_coverage',
        min_points_per_cell=1,
        close_kernel=3,
        # PMT's low-information sample filter: skip tiles whose observed area
        # is below this fraction of the patch. Matches pmt_single.py's
        # default for single-trip pseudo-labels.
        mask_thresh=0.3))
