# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from MapVR (eec23fe)
#   (https://github.com/ZhangGongjie/MapVR/tree/eec23fe86215b82f7630c0e983777bcfe6677939)
# Copyright (c) 2022 Hust Vision Lab, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.


import torch
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
import torch.nn.functional as F
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

from .solver import solve_mixed_assign_problem
import itertools


def _detect_segments_crossing_invalid_mask(
    gt_bev_mask: torch.BoolTensor,
    norm_pts: torch.FloatTensor,
    num_samples: int = 10
) -> torch.BoolTensor:
    """
    A approximate version that checks if line segments cross any False cell
    in 'gt_bev_mask' by sampling points along each segment.

    Args:
      gt_bev_mask: [H,W] bool  (index as gt_bev_mask.T[x, y])
      norm_pts: [N, L, 2] float, each point is (x, y) in [0..W]x[0..H].
      num_samples: number of sample points per segment.

    Returns:
      crosses_points: [N, L] bool. crosses_points[n, i] = True if
        the segment between (i-1 -> i) or (i -> i+1) passes through any False cell.

    NOTE: This is an approximation and may miss small False cells if the line is
          long or the sampling is sparse. Increase 'num_samples' for better coverage.
    """
    gt_bev_mask = gt_bev_mask.T
    denorm_pts = denormalize_2d_pts(norm_pts, [0]*3 + list(gt_bev_mask.shape) + [0])

    device = denorm_pts.device
    W, H = gt_bev_mask.shape  # W in dim0, H in dim1
    N, L, _ = denorm_pts.shape

    # If we have fewer than 2 points, no segments exist:
    if L < 2:
        return torch.zeros((N, L), dtype=torch.bool, device=device)

    # 1) Extract segment start/end: shape [N, L-1, 2]
    start_segments = denorm_pts[:, :-1, :]  # (x, y) of each start
    end_segments   = denorm_pts[:,  1:, :]  # (x, y) of each end

    # 2) Create a linspace [0..1] with `num_samples` steps
    t = torch.linspace(0, 1, num_samples, device=device)        # shape [num_samples]
    t = t.view(1, 1, num_samples, 1)                            # shape [1,1,num_samples,1]

    # 3) Interpolate: positions = start + t*(end - start)
    #    shape => [N, L-1, num_samples, 2]
    start_segments = start_segments.unsqueeze(2)  # [N, L-1, 1, 2]
    end_segments   = end_segments.unsqueeze(2)    # [N, L-1, 1, 2]
    positions = start_segments + t * (end_segments - start_segments)

    # 4) Floor & clamp to valid integer indices
    positions_int = positions.floor().long()  # still [N, L-1, num_samples, 2]
    # positions_int[..., 0] = x, clamp in [0..W-1]
    # positions_int[..., 1] = y, clamp in [0..H-1]
    positions_int[..., 0].clamp_(0, W - 1)
    positions_int[..., 1].clamp_(0, H - 1)

    # 5) Flatten for indexing
    #    shape => [N*(L-1)*num_samples, 2]
    positions_flat = positions_int.reshape(-1, 2)
    x_flat = positions_flat[:, 0]
    y_flat = positions_flat[:, 1]

    # Gather mask values: True => valid, False => blocked
    mask_values = gt_bev_mask[x_flat, y_flat]  # shape [N*(L-1)*num_samples], bool

    # 6) Reshape back to [N, (L-1), num_samples], check if any sample is False
    mask_values = mask_values.reshape(N, L-1, num_samples)
    # Instead of (~mask_values).any(), use logical_not to ensure we stay in bool.
    crosses_segment = torch.logical_not(mask_values).any(dim=-1)  # [N, L-1] bool

    # 7) Spread to endpoints: if segment j crosses, points j and j+1 are flagged
    crosses_points = torch.zeros((N, L), dtype=torch.bool, device=device)
    # For segment j = (point j -> point j+1):
    #   crosses_points[:, j]   |= crosses_segment[:, j]
    #   crosses_points[:, j+1] |= crosses_segment[:, j]
    crosses_points[:, :-1] |= crosses_segment
    crosses_points[:,  1:] |= crosses_segment

    return crosses_points



def compute_border_mask_no_out_of_bounds(gt_bev_mask: torch.BoolTensor) -> torch.BoolTensor:
    """
    Compute a 2D boolean `border_mask` of the same shape as `gt_bev_mask`,
    where a cell (x, y) is marked True if:
      - gt_bev_mask[x, y] is True
      - It has at least one *in-bounds* 8-connected neighbor that is False.
        (Out-of-bounds neighbors are ignored, not treated as False.)
    """
    W, H = gt_bev_mask.shape

    # We'll accumulate two counts per cell:
    #   neighbor_counts[y, x]: number of in-bounds neighbors (0..8)
    #   true_counts[y, x]:     among those neighbors, how many are True
    neighbor_counts = torch.zeros_like(gt_bev_mask, dtype=torch.float)
    true_counts = torch.zeros_like(gt_bev_mask, dtype=torch.float)

    # Offsets for the 8 neighbors
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
    ]

    # For each neighbor offset (dy, dx), update neighbor_counts and true_counts
    for dx, dy in directions:
        # Source slices
        src_x_min = max(0, -dx)
        src_x_max = W - max(0,  dx)
        src_y_min = max(0, -dy)
        src_y_max = H - max(0,  dy)

        # Target slices
        tgt_x_min = max(0,  dx)
        tgt_x_max = W - max(0, -dx)
        tgt_y_min = max(0,  dy)
        tgt_y_max = H - max(0, -dy)

        # Increase the count for valid neighbors
        neighbor_counts[tgt_x_min:tgt_x_max, tgt_y_min:tgt_y_max,] += 1.0

        # Among valid neighbors, count how many are True
        true_counts[tgt_x_min:tgt_x_max, tgt_y_min:tgt_y_max] += (
            gt_bev_mask[src_x_min:src_x_max, src_y_min:src_y_max].float()
        )

    # A cell is on the border if:
    #   - It's True
    #   - true_counts[x, y] < neighbor_counts[x, y]
    #     (i.e., it has at least one valid in-bounds neighbor that is False)
    border_mask = gt_bev_mask & (true_counts < neighbor_counts)
    return border_mask

def detect_polylines_touching_border(
    gt_bev_mask: torch.BoolTensor,
    gt_norm_pts: torch.FloatTensor
) -> torch.BoolTensor:
    """
    Returns a boolean tensor of shape [N], where each entry is True
    if the corresponding polyline in gt_pts touches a 'border' cell.

    - 'Border cell': A True cell in gt_bev_mask that has at least one
                     in-bounds neighbor that is False.
    - gt_bev_mask: [H, W] boolean
    - gt_norm_pts: [N, L, 2] normalized float coordinates of polyline points (y,x).
    
    Procedure:
      1) Compute border_mask with correct out-of-bounds handling.
      2) Convert polyline coords to int indices: floor & clamp.
      3) Flatten border_mask to 1D with .reshape(-1).
      4) Convert (x,y) -> linear index.
      5) Check if any point hits a border cell.
    """
    # 1) Compute border cells
    gt_bev_mask = gt_bev_mask.T
    denorm_pts = denormalize_2d_pts(gt_norm_pts, [0]*3 + list(gt_bev_mask.shape) + [0])
    border_mask = compute_border_mask_no_out_of_bounds(gt_bev_mask)
    W, H = gt_bev_mask.shape

    # 2) Convert coords to int indices
    pts_int = denorm_pts.long()  # floor for positive coords
    pts_int[..., 0].clamp_(0, W - 1)  # clamp x
    pts_int[..., 1].clamp_(0, H - 1)  # clamp y

    # 3) Flatten border_mask to 1D
    border_mask_flat = border_mask.reshape(-1)

    # 4) Convert (x,y) -> linear index x*H + y
    linear_idx = pts_int[..., 0] * H + pts_int[..., 1]  # shape [N, L]

    # 5) Gather which points fall on border cells -> [N, L] boolean
    is_border_point = border_mask_flat[linear_idx]

    # Reduce along points dimension (L) to see if the polyline has any border point
    touches_border = is_border_point.any(dim=-1)  # shape [N]

    return touches_border


def mask_and_resample_polylines(
    pred_pts: torch.FloatTensor,
    bev_mask: torch.BoolTensor,
    min_pts: int = 2,
    allow_split: bool = True,
    resample: bool = True
) -> torch.FloatTensor:
    """
    Masks each polyline into segments that lie fully within gt_bev_mask,
    then resamples each segment to L points via F.interpolate. Segments
    that are too short are discarded. If 'allow_split' is True, polylines
    can be split into multiple segments.
    
    Args:
      pred_pts: [N, L, 2] float, the polylines in (x,y) format.
      gt_bev_mask: [H,W] bool  (index as gt_bev_mask.T[x, y])
      min_pts:  minimum number of points in a valid sub-chain. Default 2.
      allow_split: if True, allow splitting of sub-chains. Default True.
      resample: if True, resample each sub-chain to 'L' points. Default True.

    Returns:
      subsegments: [M, L, 2] float, resampled sub-chains, if 'resample' is True. Empty if no valid sub-chains.
                   If 'resample' is False, subsegments are not resampled and the number of points may vary.
      partially_masked_idx: [M] int, indices of polylines that are partially masked.      
      idx_map: [M] int, mapping from new_pred_pts to original pred_pts.
      only_o2o: [M] bool, True if the sub-chain is not split and can only assinged to one GT.
      fully_masked_idx: [K] int, indices of polylines that are fully masked or have too few points with
                    valid bev.
    """
    _, L, _ = pred_pts.shape

    # 1) Identify which segments are crossing:
    #    segment j is crossing if both endpoints j and j+1 are flagged
    crosses = _detect_segments_crossing_invalid_mask(bev_mask, pred_pts)
    crosses_segment = crosses[:, :-1] & crosses[:, 1:]  # [N, L-1] bool

    # 2) Valid segment = not crossing
    valid_segment = ~crosses_segment  # [N, L-1] bool

    all_valid_idx = valid_segment.all(1).nonzero()[:,0]
    partially_masked_idx = ((crosses_segment.any(1)) & (~crosses_segment.all(1))).nonzero()[:,0]

    diff = torch.diff(F.pad(valid_segment[partially_masked_idx],(1,1)).to(int))
    start_idxs = (diff>0).nonzero()
    stop_idxs = (diff<0).nonzero()
    stop_idxs[:,1] += 1

    # filter out the case where there are too few points
    long_segment = (stop_idxs-start_idxs)[:,1] >= min_pts
    start_idxs, stop_idxs = start_idxs[long_segment], stop_idxs[long_segment]

    #remap
    start_idxs[:,0] = partially_masked_idx[start_idxs[:,0]]
    stop_idxs[:,0] = partially_masked_idx[stop_idxs[:,0]]

    partially_masked_idx = start_idxs[:,0] # filter out

    unique_idx, counts = torch.unique(partially_masked_idx, return_counts=True)
    only_o2o = torch.isin(partially_masked_idx, unique_idx[counts==1])

    if not allow_split:
        start_idxs = start_idxs[only_o2o]
        stop_idxs = stop_idxs[only_o2o]
        partially_masked_idx = partially_masked_idx[only_o2o]
        only_o2o = torch.ones_like(partially_masked_idx, dtype=bool)

    # We'll accumulate resampled sub-chains in a Python list
    subsegments = []

    for start_idx, stop_idx in zip(start_idxs, stop_idxs):
        subsegment = pred_pts[start_idx[0]][start_idx[1]:stop_idx[1]]  # shape [k, 2]
        if resample:
            subsegment = _resample_1d(subsegment, L).unsqueeze(0)
        subsegments.append(subsegment)

    if resample:
        if len(subsegments) > 0:
            # Concatenate all sub-chains
            subsegments = torch.cat(subsegments, dim=0)
        else:
            # No valid sub-chains found in any polyline
            subsegments = torch.empty((0, L, 2), dtype=pred_pts.dtype, device=pred_pts.device)

    # Create an index for all polylines that are fully masked or have too few points
    fully_masked = torch.ones(len(pred_pts), dtype=bool, device=pred_pts.device)
    fully_masked[all_valid_idx] = False
    fully_masked[partially_masked_idx] = False
    fully_masked_idx = fully_masked.nonzero()[:,0]

    return subsegments, partially_masked_idx, only_o2o, fully_masked_idx, all_valid_idx


def _resample_1d(points: torch.Tensor, M: int) -> torch.Tensor:
    """
    Interpolates a polyline defined by 'points' (a tensor of shape [N, 2])
    to produce exactly M points evenly spaced by arc-length using vectorized operations.
    
    Args:
        points (torch.Tensor): Tensor of shape [N, 2] containing the polyline points.
        M (int): The desired number of interpolated points.
    
    Returns:
        torch.Tensor: A tensor of shape [M, 2] with the interpolated points.
    """
    # Compute differences and segment lengths between consecutive points.
    deltas = points[1:] - points[:-1]            # Shape: [N-1, 2]
    seg_lengths = torch.sqrt((deltas ** 2).sum(dim=1))  # Shape: [N-1]
    
    # Compute the cumulative arc-lengths. Start with 0.
    cum_length = torch.cat([
        torch.zeros(1, device=points.device, dtype=points.dtype),
        torch.cumsum(seg_lengths, dim=0)
    ])  # Shape: [N]
    
    # Generate M evenly spaced target arc-lengths.
    target_lengths = torch.linspace(0, cum_length[-1], M, device=points.device, dtype=points.dtype)
    
    # For each target length, find the segment it falls into.
    # torch.searchsorted returns an index such that cum_length[index-1] <= target < cum_length[index]
    indices = torch.searchsorted(cum_length, target_lengths, right=True) - 1
    # Clamp indices to valid range (0 to N-2)
    indices = torch.clamp(indices, 0, len(cum_length) - 2)
    
    # Get the arc-lengths at the start and end of each segment.
    t0 = cum_length[indices]        # Shape: [M]
    t1 = cum_length[indices + 1]      # Shape: [M]
    
    # Compute the fraction along each segment. Handle the case when t1 == t0.
    denom = t1 - t0
    fraction = torch.where(denom > 0, (target_lengths - t0) / denom, torch.zeros_like(target_lengths))
    
    # Now, linearly interpolate between the points.
    p0 = points[indices]            # Shape: [M, 2]
    p1 = points[indices + 1]        # Shape: [M, 2]
    
    new_points = p0 + fraction.unsqueeze(-1) * (p1 - p0)  # Shape: [M, 2]
    return new_points



def normalize_2d_bbox(bboxes, pc_range):

    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[...,0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[...,1:2] = cxcywh_bboxes[...,1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h,patch_w,patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes

def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts

def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])

    return bboxes
def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    return new_pts

@BBOX_ASSIGNERS.register_module()
class MapTRAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.
    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', weight=0.0),
                 pts_cost=dict(type='ChamferDistance',loss_src_weight=1.0,loss_dst_weight=1.0),
                 rendered_mask_cost=dict(type='RenderedMaskDiceCost', weight=10.0),
                 pc_range=None):
        self.cls_cost = build_match_cost(cls_cost)
        assert reg_cost["weight"] == 0.0 and iou_cost["weight"] == 0.0, \
            "Only pts_cost and rendered_mask_cost are supported"
        self.pts_cost = build_match_cost(pts_cost)
        if rendered_mask_cost is not None and rendered_mask_cost["weight"] > 0.0:
            self.rendered_mask_cost = build_match_cost(rendered_mask_cost)
        else:
            self.rendered_mask_cost = lambda *args, **kwargs: 0.0
        self.pc_range = pc_range

    def assign(self,
               bbox_pred,
               cls_pred,
               pts_pred,
               gt_bboxes, 
               gt_labels,
               gt_pts,
               gt_shift_pts,
               gt_bev_mask=None, # for compatibility
               dup_gt_idx=None, # for compatibility
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.
        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.
        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        assert bbox_pred.shape[-1] == 4, \
            'Only support bbox pred shape is 4 dims'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels), None

        # 2. compute the weighted costs
        # classification cost
        cls_cost = self.cls_cost(cls_pred, gt_labels)

        # pts costs
        _, num_orders, num_pts_per_gtline, num_coords = gt_shift_pts.shape

        normalized_gt_pts = normalize_2d_pts(gt_shift_pts, self.pc_range)
        num_pts_per_predline = pts_pred.size(1)
        if num_pts_per_predline != num_pts_per_gtline:
            pts_pred_interpolated = F.interpolate(pts_pred.permute(0,2,1),size=(num_pts_per_gtline),
                                                  mode='linear', align_corners=True)
            pts_pred_interpolated = pts_pred_interpolated.permute(0,2,1).contiguous()
        else:
            pts_pred_interpolated = pts_pred
        
        # num_q, num_pts, 2 <-> num_gt, num_pts, 2
        pts_cost_ordered = self.pts_cost(pts_pred_interpolated, normalized_gt_pts)
        pts_cost_ordered = pts_cost_ordered.view(num_bboxes, num_gts, num_orders)
        pts_cost, order_index = torch.min(pts_cost_ordered, 2)

        # rendered_mask_cost
        rendered_mask_cost = self.rendered_mask_cost(cls_pred, pts_pred_interpolated, gt_labels, normalize_2d_pts(gt_pts, self.pc_range))

        # weighted sum of above three costs
        cost = cls_cost + pts_cost + rendered_mask_cost
        
        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels), order_index



@BBOX_ASSIGNERS.register_module()
class MaskedMapTRAssigner(MapTRAssigner):
    """
    MapTR Assinger in the presence of a BEV mask, where GT vectors are only
    defined within the mask.
    """
    def __init__(self, ign_pred_outside_mask_for_touch_gt=False, allow_split=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ign_pred_outside_mask_for_touch_gt = ign_pred_outside_mask_for_touch_gt
        self.allow_split=allow_split

    def inter_shift_pts_cost(self, gt_shift_pts, pts_pred):
        num_gts, num_orders, num_pts_per_gtline, _ = gt_shift_pts.shape
        num_q, num_pts_per_predline, _ = pts_pred.shape
        if num_gts == 0 or num_q == 0:
            return torch.empty((num_q, num_gts), device=pts_pred.device), \
                torch.empty((num_q, num_gts), dtype=torch.int64, device=pts_pred.device)

        normalized_gt_shift_pts = normalize_2d_pts(gt_shift_pts, self.pc_range)
        if num_pts_per_predline != num_pts_per_gtline:
            pts_pred_interpolated = F.interpolate(pts_pred.permute(0,2,1),size=(num_pts_per_gtline),
                                                  mode='linear', align_corners=True)
            pts_pred_interpolated = pts_pred_interpolated.permute(0,2,1).contiguous()
        else:
            pts_pred_interpolated = pts_pred
        
        # num_q, num_pts, 2 <-> num_gt, num_pts, 2
        pts_cost_ordered = self.pts_cost(pts_pred_interpolated, normalized_gt_shift_pts)
        pts_cost_ordered = pts_cost_ordered.view(num_q, num_gts, num_orders)
        pts_cost, order_index = torch.min(pts_cost_ordered, 2)
        return pts_cost, order_index


    def assign(self,
               bbox_pred,
               cls_pred,
               pts_pred,
               gt_bboxes, 
               gt_labels,
               gt_pts,
               gt_shift_pts,
               gt_bev_mask,
               dup_gt_idx=None,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.
        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.
        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        assert bbox_pred.shape[-1] == 4, \
            'Only support bbox pred shape is 4 dims'

        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy to install scipy first.')

        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels), None

        # Handle k_many2one
        if dup_gt_idx is None:
            dup_gt_idx = torch.arange(num_gts, device=gt_labels.device)
        num_org_gts = dup_gt_idx.max().item() + 1
        k_many2one = num_gts // num_org_gts
        dup_group = torch.repeat_interleave(torch.arange(k_many2one, device=dup_gt_idx.device), repeats=num_org_gts)

        # 2. compute the weighted costs

        # classification cost
        cls_cost = self.cls_cost(cls_pred, gt_labels)

        # split GT into touching mask and non-touching mask for pts and rendered mask cost        
        gt_norm_pts = normalize_2d_pts(gt_pts, self.pc_range)
        touches_border = detect_polylines_touching_border(gt_bev_mask.to(bool), gt_norm_pts)
        split_pred_pts, idx_map, only_o2o, fully_masked_idx, _ = mask_and_resample_polylines(pts_pred, gt_bev_mask.to(bool), min_pts=4, allow_split=self.allow_split) # min 4 for polygons

        unmasked_pts_cost, unmasked_order_index = self.inter_shift_pts_cost(gt_shift_pts, pts_pred)
        masked_pts_cost, masked_order_index = self.inter_shift_pts_cost(gt_shift_pts[touches_border], split_pred_pts)
        o2o_masked_order_index = torch.zeros_like(unmasked_order_index)
        o2o_masked_order_index[idx_map[:,None][only_o2o], touches_border] = masked_order_index[only_o2o]

        # rendered mask cost
        unmasked_rnd_cost = self.rendered_mask_cost(None, pts_pred, gt_labels, gt_norm_pts)
        masked_rnd_cost = self.rendered_mask_cost(None, split_pred_pts, gt_labels[touches_border], gt_norm_pts[touches_border])

        # weighted sum of above three costs
        unmasked_cost = cls_cost + unmasked_pts_cost + unmasked_rnd_cost
        masked_cost = cls_cost[idx_map[:,None], touches_border] + masked_pts_cost + masked_rnd_cost
        o2o_masked_cost = torch.full_like(unmasked_cost, float('inf'))
        o2o_masked_cost[idx_map[:, None][only_o2o], touches_border] = masked_cost[only_o2o]

        if self.ign_pred_outside_mask_for_touch_gt:
            # ignore the matching between prediction that is fully outside the mask and GT that touches the mask
            unmasked_cost[fully_masked_idx[:, None], touches_border] = float('inf')
        
        # merge both o2o cost matrices by minimum
        o2o_cost, masked_applied = torch.min(torch.stack([unmasked_cost, o2o_masked_cost]),0)
        masked_applied = masked_applied.bool()
        o2o_order_index = torch.where(masked_applied, o2o_masked_order_index, unmasked_order_index)
        
        # calculate o2m cost matrix of shape [num_split_pred, num_subsets]
        o2m_idx_map = idx_map[~only_o2o]
        o2m_idx_uni, counts = o2m_idx_map.unique(return_counts=True)    
        max_splits = counts.max().item() if not only_o2o.all() else 1
        # touches_border lives wherever gt_bev_mask does (CUDA during training),
        # so the arange has to be created there too rather than on the CPU.
        gt_touches_brd_idx_list = torch.arange(
            0, num_gts, device=touches_border.device)[touches_border].tolist()
        num_gt_touch_bor = len(gt_touches_brd_idx_list)

        subsets = []
        for size in range(2, min(max_splits, num_gt_touch_bor)+1):
            subsets += itertools.combinations(gt_touches_brd_idx_list, size)
        subsets = [list(subset) for subset in subsets]
        subsets = [subset for subset in subsets if
                   len(subset) > 1 and
                   (gt_labels[subset] == gt_labels[subset][0]).all() and
                   (dup_group[subset] == dup_group[subset][0]).all() and # use only subset idx from the same duplicate group to be faster
                   (len(dup_gt_idx[subset].unique()) == len(dup_gt_idx[subset])) # all subsets have to be from different duplicates of gt
        ]


        if self.allow_split and len(subsets) > 0:
            o2m_cost = torch.full((len(o2m_idx_uni), len(subsets)), float('inf'), device=masked_cost.device) # sparse matrix
            o2m_order_index = torch.zeros(len(idx_map), len(subsets), dtype=torch.long, device=masked_cost.device)
            o2m_local_assign_matrix = torch.full_like(o2m_order_index,-1)

            for p, idx_org in enumerate(o2m_idx_uni):
                idx_mask = idx_map == idx_org
                count = idx_mask.sum()
                # create new cost matrix for one-to-many matching
                candidate_subsets = [(s, subset) for s, subset in enumerate(subsets) if len(subset) == count]
                for s, subset in candidate_subsets:
                    subset = torch.tensor(subset, device=touches_border.device)
                    subset_subidx = (touches_border.cumsum(0)-1)[subset]
                    multi_match_cost = masked_cost[idx_mask, subset_subidx[:,None]].detach().cpu()
                    local_row, local_col = linear_sum_assignment(multi_match_cost) # small hungarian matching
                    # Two copies on purpose: multi_match_cost was moved to the
                    # CPU for scipy, while the tensors indexed below live on
                    # the GPU, and an index tensor has to sit on the same
                    # device as what it indexes.
                    local_row_cpu, local_col_cpu = torch.from_numpy(local_row), torch.from_numpy(local_col)
                    local_row, local_col = local_row_cpu.to(subset.device), local_col_cpu.to(subset.device)
                    o2m_cost[p,s] = multi_match_cost[local_row_cpu, local_col_cpu].sum()

                    o2m_order_index[idx_mask.nonzero()[local_row,0],s] = masked_order_index[idx_mask, subset_subidx[:,None]][local_row, local_col]
                    o2m_local_assign_matrix[idx_mask.nonzero()[local_row,0],s] = local_col


        # assign all indices to backgrounds first (= 0)
        assigned_gt_inds[:] = 0
        # fully masked should be ignored (= -1), except they got assigned in the next step
        assigned_gt_inds[fully_masked_idx] = -1

        o2m_assign_matrix = torch.zeros_like(masked_applied)
        cost = o2o_cost.detach().cpu()
 
        # 3. Perform matching algorithm
        if not self.allow_split or len(subsets) == 0:
            # Pure one-to-one assignment -> solve with hungarian algorithm
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
            matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
            matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)

        else:
            # Mixed problem of one-to-one and one-to-many assingment -> solve with linear programming
            o2m_cost = o2m_cost.detach().cpu()

            opt_solution = solve_mixed_assign_problem(o2o_cost, o2m_cost, o2m_idx_uni.cpu().numpy(), subsets)
            if len(opt_solution["one2one"]) > 0:
                matched_row_inds, matched_col_inds = torch.tensor(opt_solution["one2one"]).to(bbox_pred.device).T
            else:
                matched_row_inds, matched_col_inds = torch.empty(2,0, dtype=torch.long, device=bbox_pred.device)

            # order_index and local_assignment are currently not used, because pts loss is not applied in the one-to-many case
            # If in future, the pts loss could be applied. Then the following two tensors should be used after checking
            # them on correctness.
            del o2m_order_index
            del o2m_local_assign_matrix

            for org_idx, s in opt_solution["one2subset"]:
                o2m_matched_col_inds = torch.tensor(subsets[s], device=o2m_assign_matrix.device)
                o2m_matched_row_inds = torch.full_like(o2m_matched_col_inds, org_idx)
                o2m_assign_matrix[o2m_matched_row_inds, o2m_matched_col_inds] = True
                assigned_gt_inds[org_idx] = -1 # ignore in initial AssignResult -> o2m is handled in extra property

        
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        masked_applied = masked_applied | o2m_assign_matrix

        # initial result stores only o2o assignment. o2m assignment is stored in extra property
        # o2m matched pred/GT indices are ignored in the initial result
        result = AssignResult(len(matched_row_inds), assigned_gt_inds, None, labels=assigned_labels)

        result.set_extra_property(
            "masked_applied", masked_applied
        )
        result.set_extra_property(
            "o2m_assign_matrix", o2m_assign_matrix
        )

        return result, o2o_order_index
