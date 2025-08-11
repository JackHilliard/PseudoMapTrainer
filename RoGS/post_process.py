# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import numpy as np
import cv2

from vectorize_mask import vectorize_mask, filter_lane_divider

# Define the class labels
MASK_CLASS = 0
LANE_CLASS = 1
OUTSIDE_CLASS = 4
ROAD_CLASS = 3
CROSSWALK_CLASS = 2


def crop_img(img, width:int, height:int=None):
    if height is None:
        height = width
    if width == 0 and height == 0:
        return img
    assert height > 0 and width > 0, "width and height should be positive or both 0"
    return img[height:-height, width:-width]


def extract_border(segmentation, kernel_size=3):
    """
    Extract the border between class 3 and class 4 in a segmentation map.

    Parameters:
    segmentation (numpy.ndarray): The segmentation map with shape (H, W).
    kernel_size (int): The size of the kernel used for morphological operations.

    Returns:
    numpy.ndarray: An image highlighting the border between class 3 and class 4.
    """
    # Create binary masks for class 3 and class 4
    class3_mask = (segmentation == 3).astype(np.uint8)
    class4_mask = (segmentation == 4).astype(np.uint8)

    # Define a 3x3 kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Find the edges of the class 3 regions
    class3_edges = cv2.morphologyEx(class3_mask, cv2.MORPH_GRADIENT, kernel)

    # Find where class 3 edges touch class 4 regions
    border = cv2.bitwise_and(class3_edges, class4_mask)

    border = remove_small_segs_binary(border, max_area=50).astype(bool)

    return border


def remove_enclosed_segs(segmentation, enclosing_class, ignore_classes=[], max_area=2e4):
    """
    The function iterates over each connected component and checks if it is enclosed by the `enclosing_class`.
    If the connected component is enclosed, the entire connected component is replaced by the `enclosing_class` label.

    Parameters:
        segmentation (numpy.ndarray): The segmentation mask with shape (H, W).
        enclosing_class (int): The class label that encloses the segments.
        ignore_classes (list): A list of class labels to ignore when removing enclosed segments.
        max_area (int): The maximum area of a single segment to be removed.
    """
    # Copy of segmentation to modify enclosed components
    result = segmentation.copy()

    # derive segmenation mask
    seg_mask = segmentation != enclosing_class
    for ignore_class in ignore_classes:
        seg_mask &= segmentation != ignore_class

    # Find all connected components in the segmentation mask
    num_labels, labels, stats, _  = cv2.connectedComponentsWithStats(seg_mask.astype(np.uint8))

    for i in range(1, num_labels):  # Skip the background (label 0)
        # Create a mask for the current component
        component_mask = (labels == i).astype(np.uint8)
        
        # Dilate the component mask to create a boundary buffer
        dilated = cv2.dilate(component_mask, kernel=np.ones((3, 3), np.uint8), iterations=1)
        
        # Check if the dilated mask touches only the `enclosing_class`
        boundary_mask = (dilated - component_mask).astype(bool)
        border_with_encl_class = len(segmentation[boundary_mask]) > 0 and np.all(segmentation[boundary_mask] == enclosing_class)
        touches_img_bound = np.any(component_mask[0, :] == 1) or np.any(component_mask[-1, :] == 1) or np.any(component_mask[:, 0] == 1) or np.any(component_mask[:, -1] == 1)
        too_large = stats[i, cv2.CC_STAT_AREA] > max_area

        if border_with_encl_class and not touches_img_bound and not too_large:
            # Replace the component with the enclosing class label
            result[component_mask.astype(bool)] = enclosing_class
    
    return result


def long_line_skeleton(bev_mask, min_line_length=50):
    """
    Remove short elements in the binary mask and return the skeleton of
    the remaining main lane lines.
    """
    binary_mask = bev_mask.astype(np.uint8)*255
    thinned_mask = cv2.ximgproc.thinning(binary_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # Now 'thinned_mask' is a 1-pixel-wide skeleton of the lines.
    # Next, remove small connected components so that only the
    # main lane lines remain.

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thinned_mask, connectivity=8)
    filtered_mask = np.zeros_like(thinned_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_line_length:
            filtered_mask[labels == i] = 255

    return filtered_mask.astype(bool)


def remove_large_segs(segmentation, remove_class, replace_class, kernel_size=6):
    """
    Remove large connected components from a segmentation mask, where the connected components are defined by the
    'remove_class' label. The function iterates over each connected component and performs morphological erosion
    using a square structuring element of size 'kernel_size'. If the eroded mask contains any non-zero pixels, the
    entire connected component is removed by setting its pixels to 'replace_class'.
    """

    mask = (segmentation == remove_class).astype(np.uint8)
    # Label connected components
    num_labels, labeled_mask = cv2.connectedComponents(mask)

    selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Iterate over each connected component (excluding background label 0)
    for label in range(1, num_labels):
        # Create a mask for the current region
        region_mask = (labeled_mask == label).astype(np.uint8)

        # Perform morphological erosion
        eroded_mask = cv2.erode(region_mask, selem)
        large_object_detected = np.any(eroded_mask) # The region is large enough to contain a kernel_sizexkernel_size square

        if large_object_detected:
            # Remove the entire region by setting its pixels to 'replace_class'
            segmentation[region_mask.astype(bool)] = replace_class

    return segmentation


import numpy as np
import cv2

def remove_small_seg_touching_mask(segmentation, mask_cls=MASK_CLASS, ignore_classes=[], max_area=400, max_distance=50):
    """
    Removes small segments that either touch the mask or are entirely close to it.
    "Close" is defined as the distance between the farthest pixel in the segment and its closest mask pixel.
    The reason behind this is that at the border of the mask, the labels are most uncertain.
    Therefore, we remove small segments that either touch or lie fully within a specified max distance
    from the mask to avoid misclassified artifacts.

    Parameters:
        segmentation (numpy.ndarray): The segmentation mask with shape (H, W).
        mask_cls (int): The class label of the mask.
        ignore_classes (list): A list of class labels to ignore when processing components.
        max_area (int): The maximum number of pixels in a segment eligible for removal.
        max_distance (int or float): The maximum allowed distance (in pixels) from the mask to the farthest pixel
                                     in the component for it to be removed. If set to 0, only segments that directly
                                     touch the mask will be removed.
    Returns:
        numpy.ndarray: The updated segmentation mask.
    """
    # Create a copy to modify
    result = segmentation.copy()

    # Determine the classes to process (exclude the mask and any ignore classes)
    classes_of_interest = np.unique(segmentation)
    ignore_classes = ignore_classes + [mask_cls]
    classes_of_interest = classes_of_interest[~np.isin(classes_of_interest, ignore_classes)]

    # If max_distance is specified, precompute a distance transform.
    # The distance transform will give for each pixel the Euclidean distance to the nearest mask pixel.
    if max_distance > 0:
        # Build a binary mask where mask_cls pixels are 1, and then invert it so that mask pixels become 0.
        mask_binary = (segmentation == mask_cls).astype(np.uint8)
        inv_mask = 1 - mask_binary
        # Compute the distance transform (using the Euclidean (L2) norm).
        dist_transform = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 3)

    # Process each class of interest.
    for cls in classes_of_interest:
        # Create a binary image for the current class.
        binary_for_cls = (segmentation == cls).astype(np.uint8)
        
        # Get connected components and their statistics.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_for_cls)
        
        for i in range(1, num_labels):  # Skip the background (label 0).
            # Create a mask for the current component.
            component_mask = (labels == i).astype(np.uint8)
            
            # Dilate the component mask to create a one-pixel-wide boundary.
            dilated = cv2.dilate(component_mask, kernel=np.ones((3, 3), np.uint8), iterations=1)
            
            # The boundary is defined as the dilated mask minus the component.
            boundary_mask = (dilated - component_mask).astype(bool)
            touches_mask = np.any(segmentation[boundary_mask] == mask_cls) and segmentation[boundary_mask].sum() > 0
            
            # Check if the component touches the image border.
            touches_img_bound = (np.any(component_mask[0, :] == 1) or 
                                 np.any(component_mask[-1, :] == 1) or 
                                 np.any(component_mask[:, 0] == 1) or 
                                 np.any(component_mask[:, -1] == 1))
            
            # Check if the component is too large to be removed.
            too_large = stats[i, cv2.CC_STAT_AREA] > max_area
            
            # Compute the maximum distance from any pixel in the component to the nearest mask pixel.
            # This represents the distance from the farthest pixel in the component to the mask.
            if max_distance > 0:
                max_comp_distance = np.max(dist_transform[component_mask.astype(bool)])
                within_max_distance = max_comp_distance <= max_distance
            else:
                within_max_distance = False

            # Remove the component if:
            #  - It either touches the mask OR its farthest pixel is within the max_distance,
            #  - It does not touch the image border, and
            #  - It is not too large.
            if touches_mask and within_max_distance and not touches_img_bound and not too_large:
                result[component_mask.astype(bool)] = mask_cls

    return result


def remove_seg_touching_other_cls(segmentation, touching_cls:list, rm_cls=MASK_CLASS, replace_class=ROAD_CLASS, max_area=40000):
    """
    Removes segments of class `rm_cls` only if the border of the segment touches all classes in `touching_cls`.
    For example, a lane segment that touches both road and crosswalk regions will be removed because it is likely not
    a true lane divider. Removed segments (originally labeled as rm_cls) are replaced with `replace_class`.

    Parameters:
        segmentation (np.ndarray): The segmentation mask with shape (H, W).
        touching_cls (list): List of class labels that must all be present in the border of a segment to trigger removal.
        rm_cls (int): Class label of the segments to potentially remove (default: MASK_CLASS).
        replace_class (int): Class label that will replace removed segments (default: ROAD_CLASS).
                             Must be one of the classes in touching_cls.
        max_area (int): Maximum area (in pixels) for a segment to be considered for removal.

    Returns:
        np.ndarray: The updated segmentation mask.
    """
    assert replace_class in touching_cls, "replace_class should be in touching_cls"

    result = segmentation.copy()
    
    # Create a binary mask for the segments we want to potentially remove.
    rm_mask = (segmentation == rm_cls).astype(np.uint8)
    
    # Find connected components in rm_mask.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(rm_mask, connectivity=8)
    
    # Loop over each component (skip background label 0)
    for i in range(1, num_labels):
        component_mask = (labels == i).astype(np.uint8)
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Skip if the component is too large.
        if area > max_area:
            continue
        
        # Skip if the component touches the image boundary.
        touches_img_bound = (
            np.any(component_mask[0, :] == 1) or 
            np.any(component_mask[-1, :] == 1) or 
            np.any(component_mask[:, 0] == 1) or 
            np.any(component_mask[:, -1] == 1)
        )
        if touches_img_bound:
            continue
        
        # Dilate the component mask to obtain a border region.
        dilated = cv2.dilate(component_mask, np.ones((3, 3), np.uint8), iterations=1)
        boundary_mask = (dilated - component_mask).astype(bool)
        
        # Check if the boundary touches all classes in touching_cls (logical AND).
        if all(np.any(segmentation[boundary_mask] == cls) for cls in touching_cls):
            result[component_mask.astype(bool)] = replace_class

    return result


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


def post_process(segmentation, bev_mask, max_vector_points:int=None):
    """
    Post-processing v3 is the same as v2, but with a different function to remove lane dividers
    that are close and parallel to the crosswalks.
    """
    #oc = opening, closing
    extent_close_outside = 20
    extent_open_outside = 10
    extent_dilate_lane = 15
    extent_erode = 5

    min_area_crosswalk = 400

    vectors = {}
    segmentation_post = segmentation.copy()

    # Remove enclosed segments
    segmentation_post = remove_enclosed_segs(segmentation_post, OUTSIDE_CLASS)
    segmentation_post = remove_enclosed_segs(segmentation_post, CROSSWALK_CLASS)
    segmentation_post = remove_enclosed_segs(segmentation_post, MASK_CLASS, max_area=5e4)
    segmentation_post = remove_enclosed_segs(segmentation_post, ROAD_CLASS, ignore_classes=[LANE_CLASS, CROSSWALK_CLASS])

    # Remove large/small segments
    segmentation_post = remove_large_segs(segmentation_post, LANE_CLASS, ROAD_CLASS, kernel_size=40)
    segmentation_post = remove_small_seg_touching_mask(segmentation_post)
    bev_mask_post = bev_mask & (segmentation_post != MASK_CLASS)

    # Smothing outside boundary
    label_mask = (segmentation_post == OUTSIDE_CLASS) | (segmentation_post == MASK_CLASS)
    kernel_close_outside = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (extent_close_outside, extent_close_outside))
    kernel_open_outside = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (extent_open_outside, extent_open_outside))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (extent_erode, extent_erode))

    label_mask_post = label_mask.astype(np.uint8)
    label_mask_post = cv2.morphologyEx(label_mask_post, cv2.MORPH_OPEN,  kernel_open_outside)
    label_mask_post = cv2.morphologyEx(label_mask_post, cv2.MORPH_CLOSE, kernel_close_outside)
    label_mask_post = cv2.erode(label_mask_post, kernel_erode)
    label_mask_post = label_mask_post.astype(bool)

    outside_mask = segmentation_post == OUTSIDE_CLASS
    segmentation_post[label_mask_post] = OUTSIDE_CLASS
    segmentation_post[(~label_mask_post) & outside_mask] = ROAD_CLASS
    segmentation_post[~bev_mask_post] = MASK_CLASS

    # Do the remove step again after smoothing
    segmentation_post = remove_enclosed_segs(segmentation_post, OUTSIDE_CLASS, ignore_classes=[LANE_CLASS, CROSSWALK_CLASS])
    segmentation_post = remove_small_seg_touching_mask(segmentation_post, max_area=7e4)

    # Lane marking segmentation
    label_mask_post = (segmentation_post == LANE_CLASS).astype(np.uint8)
    kernel_dilate_lane = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (extent_dilate_lane, extent_dilate_lane))
    label_mask_post = cv2.dilate(label_mask_post, kernel_dilate_lane)
    label_mask_post = long_line_skeleton(label_mask_post)
    segmentation_post[label_mask_post] = LANE_CLASS # recover lane markings
    segmentation_post[(~label_mask_post) & (segmentation_post == LANE_CLASS)] = ROAD_CLASS
    
    # lane divider vectorization
    vectors["divider"] = vectorize_mask(label_mask_post, shape_type='line', max_points=max_vector_points)

    # Crosswalks (remove small artifacts)
    label_mask_post = segmentation_post == CROSSWALK_CLASS
    label_mask_post = remove_small_segs_binary(label_mask_post, min_area_crosswalk)
    segmentation_post[(~label_mask_post) & (segmentation_post == CROSSWALK_CLASS)] = ROAD_CLASS
    vectors["ped_crossing"] = vectorize_mask(label_mask_post, shape_type='polygon', max_points=max_vector_points, min_area=min_area_crosswalk)

    # Extract the boundary (only for vectorization)
    label_mask_post = extract_border(segmentation_post.astype(np.uint8), 3).astype(np.uint8)*255
    label_mask_post = cv2.ximgproc.thinning(label_mask_post, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    vectors["boundary"] = vectorize_mask(label_mask_post, shape_type='line', max_points=20)

    # remove lane dividers that are close and parallel to the boundary or crosswalks
    vectors["divider"] = filter_lane_divider(vectors["divider"], vectors["boundary"], dist_thresh=17, coverage_ratio=0.8)
    vectors["divider"] = filter_lane_divider(vectors["divider"], vectors["ped_crossing"], dist_thresh=17, coverage_ratio=0.8)


    # Upate bev_mask
    bev_mask_post = bev_mask & (segmentation_post != MASK_CLASS)
    segmentation_post[~bev_mask_post] = MASK_CLASS

    return segmentation_post, vectors, bev_mask_post