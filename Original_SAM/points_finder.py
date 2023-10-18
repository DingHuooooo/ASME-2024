import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import erosion, disk
from scipy.ndimage import label, center_of_mass
from skimage.measure import find_contours, points_in_poly
from skimage.morphology import erosion, disk

def shrink_mask(mask):
    """
    Erodes the mask until it reduces to a very small region or even to a single point.
    """
    last_non_empty_mask = mask.copy()  # Keep track of the last non-empty state
    while np.sum(mask) > 1:
        mask = erosion(mask, disk(1))
        
        # Check if mask is empty after erosion
        if np.sum(mask) == 0:
            mask = last_non_empty_mask
            break

        last_non_empty_mask = mask.copy()

    return np.argwhere(mask)


def find_centroids_shrink_mask(pre_mask, ratio=0.1):
    labeled_mask, num_features = label(pre_mask)
    
    # Calculate area sizes
    area_sizes = np.bincount(labeled_mask.ravel())[1:]
    
    # Calculate the total mask area
    total_mask_area = np.sum(pre_mask)
    
    # Compute the threshold based on the ratio
    area_threshold = ratio * total_mask_area

    # 获取面积大于 area_threshold 的连通区域标签
    large_region_labels = np.where(area_sizes > area_threshold)[0] + 1  # +1 to offset the background label

    # 初始化中心点列表
    large_region_centroids = []

    # 获取这些连通区域的中心坐标
    for region_label in large_region_labels:
        mask_for_region = labeled_mask == region_label
        
        # Shrinking the mask for this region to get the point
        point = shrink_mask(mask_for_region)
        
        # The point is a list of coordinates where the mask is 1. 
        # Since we've shrunk it down to a single point, we just take the first item.
        centroid = point[0]
        
        # 将新的中心点添加到中心点列表
        large_region_centroids.append(centroid)
            
    return labeled_mask, large_region_labels, large_region_centroids




from skimage.draw import line

def intersection_points(mask, start_point, direction_vector):
    end_point = start_point + 1000 * direction_vector
    rr, cc = line(int(start_point[0]), int(start_point[1]), int(end_point[0]), int(end_point[1]))
    
    # Ensure the values are within the mask boundaries
    valid_indices = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
    rr = rr[valid_indices]
    cc = cc[valid_indices]

    # Check if the arrays are empty
    if not rr.size or not cc.size:
        return np.array([])

    intersected = mask[rr, cc]
    intersected_indices = np.where(intersected)[0]
    rr_intersected = rr[intersected_indices]
    cc_intersected = cc[intersected_indices]

    intersection_coords = np.column_stack((rr_intersected, cc_intersected))
    
    return intersection_coords


def adjusted_centroid(mask, gravity_centroid):
    contours = find_contours(mask, 0.5)
    if len(contours) == 0:
        return gravity_centroid

    if mask[int(gravity_centroid[0]), int(gravity_centroid[1])] == 1:
        return gravity_centroid
    
    nearest_boundary_point = closest_boundary_point(mask, gravity_centroid)
    direction_vector = nearest_boundary_point - gravity_centroid
    direction_vector /= np.linalg.norm(direction_vector)
    intersections = intersection_points(mask, gravity_centroid, direction_vector)
    
    def custom_round(direction_vector):
        # 根据规则，如果元素的绝对值大于或等于0.5，就取最近的整数；否则，就保留原值
        rounded_vector = np.where(np.abs(direction_vector) >= 0.5, np.round(direction_vector), direction_vector)
        return rounded_vector.astype(int)  # 转换为整数

    if len(intersections) < 2:
        return gravity_centroid  # Safety fallback
    else:
        for point in intersections:
            if mask[int(point[0]) + custom_round(direction_vector)[0], int(point[1]) + custom_round(direction_vector)[1]] == 0:
                end_point = point
                break
    
    return np.mean([intersections[0], end_point], axis=0)

def closest_boundary_point(mask, point):
    # 1. Identify the contours of the mask.
    contours = find_contours(mask, 0.5)
    
    # 2. If no contours are found, simply return the given point.
    if len(contours) == 0:
        return point
    
    # 3. Sort the contours based on their length and pick the longest one.
    longest_contour = sorted(contours, key=lambda x: len(x))[-1]

    # 4. For each point on the contour, calculate its distance from the given point.
    distances = np.linalg.norm(longest_contour - point, axis=1)
    
    # 5. Select the contour point that has the shortest distance to the given point.
    nearest_boundary_point = longest_contour[np.argmin(distances)]

    return nearest_boundary_point


def find_centroids_gravity_center_adjusted(pre_mask, ratio=0.1):
    labeled_mask, num_features = label(pre_mask)
    area_sizes = np.bincount(labeled_mask.ravel())[1:]
    total_mask_area = np.sum(pre_mask)
    area_threshold = ratio * total_mask_area
    large_region_labels = np.where(area_sizes > area_threshold)[0] + 1  # +1 to offset the background label
    large_region_centroids = []

    for region_label in large_region_labels:
        gravity_centroid = center_of_mass(pre_mask, labeled_mask, region_label)
        ensured_point = adjusted_centroid(pre_mask , gravity_centroid)
        large_region_centroids.append(ensured_point)

    return labeled_mask, large_region_labels, large_region_centroids




def bisect_region(mask, centroid, direction_vector):
    """Bisects the mask along the direction_vector starting from the centroid."""
    new_mask = mask.copy()
    
    end_point1 = centroid + 1000 * direction_vector
    end_point2 = centroid - 1000 * direction_vector
    rr, cc = line(int(end_point1[0]), int(end_point1[1]), int(end_point2[0]), int(end_point2[1]))
    
    valid_indices = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
    rr = rr[valid_indices]
    cc = cc[valid_indices]

    # Set to 0 (cut) along the line
    new_mask[rr, cc] = 0
    return new_mask

def closest_boundary_point(mask, point):
    # 1. Identify the contours of the mask.
    contours = find_contours(mask, 0.5)
    
    # 2. If no contours are found, simply return the given point.
    if len(contours) == 0:
        return point
    
    # 3. Sort the contours based on their length and pick the longest one.
    longest_contour = sorted(contours, key=lambda x: len(x))[-1]

    # 4. For each point on the contour, calculate its distance from the given point.
    distances = np.linalg.norm(longest_contour - point, axis=1)
    
    # 5. Select the contour point that has the shortest distance to the given point.
    nearest_boundary_point = longest_contour[np.argmin(distances)]

    return nearest_boundary_point

def is_on_boundary(point, contour):
    """Check if a point is on a contour."""
    return any(np.all(point == contour_pt) for contour_pt in contour)

def check_and_adjust_centroids(mask, depth=0, max_depth=3):
    """Check and adjust centroids recursively with a maximum depth."""
    
    # Base case: if we reach the maximum depth, terminate the recursion
    if depth >= max_depth:
        return []
    
    labeled_mask, num_features = label(mask)
    centroids = []

    for region_label in range(1, num_features + 1):
        region_mask = (labeled_mask == region_label)
        gravity_centroid = center_of_mass(region_mask)

        # Visualization
        contours = find_contours(region_mask, 0.1)[0]

        y, x = int(gravity_centroid[0]), int(gravity_centroid[1])
        is_inside = region_mask[y, x]
        on_boundary = is_on_boundary(gravity_centroid, contours)

        if not is_inside:
            if on_boundary:
                # Find another boundary point and compute the perpendicular direction
                nearest_boundary_point = closest_boundary_point(region_mask, gravity_centroid)
                direction_vector = nearest_boundary_point - gravity_centroid
                perpendicular_vector = np.array([-direction_vector[1], direction_vector[0]])
                cut_mask = bisect_region(region_mask, gravity_centroid, perpendicular_vector)
            else:
                nearest_boundary_point = closest_boundary_point(region_mask, gravity_centroid)
                direction_vector = nearest_boundary_point - gravity_centroid
                cut_mask = bisect_region(region_mask, gravity_centroid, direction_vector)
            centroids += check_and_adjust_centroids(cut_mask, depth + 1)
        else:
            centroids.append(gravity_centroid)

    return centroids

# The function will now recursively adjust centroids up to a maximum depth of 3.



def find_centroids_gravity_center_recursive(pre_mask, ratio=0.1):
    labeled_mask, num_features = label(pre_mask)
    area_sizes = np.bincount(labeled_mask.ravel())[1:]
    total_mask_area = np.sum(pre_mask)
    area_threshold = ratio * total_mask_area
    large_region_indices = np.where(area_sizes > area_threshold)[0] + 1

    large_region_centroids = []
    large_region_labels = []
    for region_label in large_region_indices:
        region_mask = (labeled_mask == region_label)
        current_centroids = check_and_adjust_centroids(region_mask)
        large_region_centroids += current_centroids
        large_region_labels += [region_label] * (len(current_centroids))

    return labeled_mask, large_region_labels, large_region_centroids
