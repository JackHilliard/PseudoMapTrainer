# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import cv2
import numpy as np
from collections import deque
from shapely.geometry import LineString, Polygon, box


def vectorize_mask(
    mask,
    shape_type='line',           # 'line' or 'polygon'
    max_points=None,             # If not None, iteratively approx until <= this number
    initial_epsilon=0.25,        # Starting epsilon for approxPolyDP
    epsilon_step=0.25,           # Step to increase epsilon each iteration
    connectivity=8,              # Connectivity for connectedComponents
    min_area=400                 # Minimum area for polygons
):
    """
    Vectorizes a binary mask as either line(s) or polygon(s).

    1) Runs connectedComponents on 'mask'.
    2) For each connected region:
       - If shape_type='line', do BFS-based ordering of skeleton pixels.
       - If shape_type='polygon', do findContours-based boundary extraction.
    3) Iteratively call approxPolyDP, increasing epsilon, 
       until we have <= max_points (if provided).
    
    Returns a list of NumPy arrays, each array is shape (N,2) for one line or polygon,
    in (x,y) coordinate format.
    """

    # Ensure mask is binary 0/1 or 0/255
    bin_mask = (mask > 0).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(bin_mask, connectivity=connectivity)
    all_shapes = []

    for label_id in range(1, num_labels):
        component_mask = np.uint8(labels == label_id) * 255

        if shape_type.lower() == 'line':
            # Use BFS-based approach on skeleton / line
            shapes_in_component = _extract_line(component_mask)
        elif shape_type.lower() == 'polygon':
            # Use contour-based approach for filled regions
            shapes_in_component = _extract_polygon(component_mask)
        else:
            raise ValueError(f"Unknown shape_type: {shape_type}")

        # Now refine each shape (polyline) with iterative approx + point-limit
        refined_shapes = []
        for polyline in shapes_in_component:
            refined = _iterative_approx(
                polyline, 
                max_points=max_points,
                initial_epsilon=initial_epsilon,
                epsilon_step=epsilon_step,
                closed=(shape_type.lower() == 'polygon')  # polygons are closed
            )
            refined_shapes.append(refined)
        all_shapes.extend(refined_shapes)

    if shape_type.lower() == 'polygon':
        # Clean up polygons (remove self-intersecting artifacts)
        all_shapes = cleanup_polygons(all_shapes, min_area)

    return all_shapes

def cleanup_polygons(polygons, min_area):
    """
    Some polygons are self-intersecting and therefore invalid.
    This function cleans them up by buffering them with 0 distance and removing small artifacts from the fix.
    """
    polygons = [Polygon(polygon).buffer(0) for polygon in polygons]
    polygons = sum([[polygon] if polygon.geom_type == "Polygon" else list(polygon.geoms) for polygon in polygons], [])
    return [np.array(polygon.exterior.coords) for polygon in polygons if polygon.area > min_area]



def _extract_polygon(component_mask):
    """
    For a filled region (like a pedestrian crossing), 
    find its external boundary using cv2.findContours.
    Returns a list of Nx2 arrays (one per contour).
    """
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polylines = []
    for cnt in contours:
        # cnt is shape (N,1,2)
        # convert to (N,2)
        poly = cnt.reshape(-1,2).astype(np.int32)
        polylines.append(poly)
    return polylines


def close_to_any_red(lane_divider, boundaries, dist_thresh, coverage_ratio=0.9):
    """
    Returns True if `lane_divider` is within dist_thresh distance of
    at least one boundary for >= coverage_ratio of its length.
    
    coverage_ratio=0.9 means 90% or more of the lane_divider's length
    is inside the red buffer.
    """
    lane_len = lane_divider.length
    
    for r_line in boundaries:
        # Buffer the boundary by the distance threshold
        # This creates a 'corridor' around the boundary.
        r_buffer = r_line.buffer(dist_thresh)
        
        # Intersection of the blue line with that corridor
        intersection = lane_divider.intersection(r_buffer)
        
        # Compute how much of the blue line is inside the corridor
        # (intersection might be smaller geometry, possibly MultiLineString)
        inside_length = intersection.length
        
        # Check the fraction of the blue line inside the buffer
        if inside_length / lane_len >= coverage_ratio:
            return True
    
    return False

def filter_lane_divider(lane_dividers:list, boundaries:list, dist_thresh=0.1, coverage_ratio=0.9):
    """
    Removes any lane_dividers that is 'close enough' to a boundary
    for most of its length.
    """
    filtered = []
    lane_dividers = [LineString(line) for line in lane_dividers]
    boundaries = [LineString(line) for line in boundaries]
    for lane_divider in lane_dividers:
        if not close_to_any_red(lane_divider, boundaries, dist_thresh, coverage_ratio):
            filtered.append(lane_divider)

    filtered = [np.array(line.coords) for line in filtered]
    return filtered



def _extract_line(component_mask, min_branch_length=40):
    """
    Given a thinned component_mask==255, extract multiple polylines:
      - The single longest path
      - Additional branches (>= 40 pixels) in the skeleton, if any
    Returns a list of polylines (each polyline is an Nx2 array of [x,y] coords).
    """
    # 1) Gather all skeleton pixels
    coords = np.argwhere(component_mask == 255)  # shape (N,2) => [y,x]
    if len(coords) == 0:
        return []

    coords_xy = [(x, y) for (y, x) in coords]
    pixel_set = set(coords_xy)
    
    # 2) Build adjacency
    adjacency = _build_adjacency(pixel_set)
    
    # 3) Repeatedly extract longest paths
    polylines_xy = _extract_long_paths(adjacency, min_length=min_branch_length)
    
    # 4) Convert polylines back to np.array shape (N,2) => [x, y]
    polylines = []
    for pl_xy in polylines_xy:
        poly_np = np.array(pl_xy, dtype=np.int32)
        polylines.append(poly_np)

    return polylines

# Helper functions used above:
def _build_adjacency(pixel_set):
    adjacency = {}
    for px in pixel_set:
        x, y = px
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (nx, ny) in pixel_set:
                    neighbors.append((nx, ny))
        adjacency[px] = neighbors
    return adjacency

def _find_farthest_and_path(start, adjacency):
    from collections import deque
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    farthest_node = start

    while queue:
        current = queue.popleft()
        farthest_node = current
        for nbr in adjacency[current]:
            if nbr not in visited:
                visited.add(nbr)
                parent[nbr] = current
                queue.append(nbr)

    # Reconstruct path
    path = []
    node = farthest_node
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return farthest_node, path

def _find_longest_path_in_component(adjacency, start_pixel=None):
    if not adjacency:
        return []

    if start_pixel is None:
        start_pixel = next(iter(adjacency))  # pick any key

    farthest1, _ = _find_farthest_and_path(start_pixel, adjacency)
    farthest2, path = _find_farthest_and_path(farthest1, adjacency)
    return path

def _extract_long_paths(adjacency, min_length=40):
    polylines = []

    while adjacency:
        start_pixel = next(iter(adjacency))
        path = _find_longest_path_in_component(adjacency, start_pixel)
        if len(path) < min_length:
            # no sufficiently long path remains
            break
        polylines.append(path)
        # remove used pixels from adjacency
        for p in path:
            neighbors = adjacency.pop(p, [])
            for nbr in neighbors:
                if nbr in adjacency:
                    adjacency[nbr] = [x for x in adjacency[nbr] if x != p]

    return polylines


def _iterative_approx(polyline, max_points, initial_epsilon, epsilon_step, closed):
    """
    Iteratively call approxPolyDP with increasing epsilon until 
    the resulting polyline has <= max_points (if max_points is not None).
    
    polyline: shape (N,2)
    closed:   True if polygon, False if open polyline
    """
    if max_points is None or max_points < 3:
        # Just do one pass or none
        return _approx_polyline(polyline, initial_epsilon, closed=closed)

    eps = initial_epsilon
    approx_coords = polyline.copy()

    while True:
        # Apply approx
        new_approx = _approx_polyline(approx_coords, eps, closed=closed)
        if len(new_approx) <= max_points or len(new_approx) == len(approx_coords):
            # If it's small enough, or didn't get smaller, stop
            approx_coords = new_approx
            break
        else:
            # Keep refining
            approx_coords = new_approx
            eps += epsilon_step

    return approx_coords


def _approx_polyline(line_coords, epsilon, closed=False):
    """
    Use cv2.approxPolyDP to reduce number of points, preserving shape.
    line_coords: (N,2) in (x,y) format
    closed=True => treat as closed polygon
    """
    if len(line_coords) < 3:
        return line_coords

    # Convert to contour format (N,1,2) as float32
    contour = line_coords.reshape(-1, 1, 2).astype(np.float32)
    approx = cv2.approxPolyDP(contour, epsilon, closed=closed)
    approx_coords = approx.reshape(-1, 2).astype(np.int32)
    return approx_coords



def clip_polyline(points, width: int, height: int, delta_w, delta_h=None, polygon=False):
    """
    Clips an polyline or polygon of the original rectangular region [0:width, 0:height] 
    to [delta_w:-delta_w, delta_h:-delta_h]. Returns the result in shifted coordinates,
    where (x_min,y_min) maps to (0,0).
    
    If the line is entirely outside, returns [].
    If it's partially inside, may return multiple disjoint segments.
    
    Returns
    -------
    list of list of (float, float)
        Each element is a list of the clipped segment's vertices in "local" coords:
        (X - x_min, Y - y_min).
    """

    if delta_h is None:
        delta_h = delta_w

    # If both are zero, interpret that as "no clipping".
    if delta_w == 0 and delta_h == 0:
        return [points.copy()]

    assert delta_w > 0 and delta_h > 0, \
        "delta_w and delta_h must be positive (or both 0 for no-op)."
    
    # Clipping box boundaries
    x_min = delta_w
    x_max = width - delta_w
    y_min = delta_h
    y_max = height - delta_h

    shape = Polygon(points) if polygon else LineString(points)
    clip_rect = box(x_min, y_min, x_max, y_max)
    clipped_multi = shape.intersection(clip_rect)

    if clipped_multi.is_empty:
        return []

    result = []
    if hasattr(clipped_multi, "geoms"):
        clipped_multi = list(clipped_multi.geoms)
    else:
        clipped_multi = [clipped_multi]

    for clipped in clipped_multi:
        if clipped.geom_type == "LineString" and not polygon:
            # clipped line segment
            seg = [(px - x_min, py - y_min) for (px, py) in clipped.coords]
            result.append(seg)
        elif clipped.geom_type == "Polygon" and polygon:
            # clipped polygon segment
            exterior = [(px - x_min, py - y_min) for (px, py) in clipped.exterior.coords]
            result.append(exterior)
    
    # convert to numpy arrays
    result = [np.array(poly) for poly in result]

    return result
