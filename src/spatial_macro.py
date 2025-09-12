import numpy as np
import gudhi
from scipy.spatial import Delaunay, distance
from src.root import *

def get_branch_coors(rsa: Root) -> list:
    '''
    Get lists of root nodes and the assicated time of appearance

    Args:
        rsa: Root Object.

    Returns:
        all_coors: A list of list for each branch's nodes, represented as a tuple of (time, (x,y)).
    '''

    all_corrs = []

    # Extract coordinates and time data from primary:
    corrs = [(node.x, node.y) for node in rsa.primary.nodes]

    times_hr = [node.hr for node in  rsa.primary.nodes]

    all_corrs.append(list(zip(times_hr, corrs)))

    # Check if laterals exist
    if not rsa.laterals:
        return all_corrs
       
    def branch_coor(branch):
        corrs = [(node.x, node.y) for node in branch.nodes]
        times_hr = [node.hr for node in  branch.nodes]
        return times_hr, corrs
    
    #  Extract coordinates and time data from laterals:
    for lr in rsa.laterals:
        times_hr, corrs = branch_coor(lr)
        all_corrs.append(list(zip(times_hr, corrs)))
    
    return all_corrs


def find_closest_timepoint_index(time, corrected_timepoints):
    """
    Find the index of the closest timepoint in corrected_timepoints to the given time.
    If time exactly matches a timepoint, return the index of that timepoint.
    If there are two equally close timepoints, return the index of the larger one.
    
    Parameters:
    -----------
    time : float or int
        The time to find the closest timepoint for
    corrected_timepoints : list
        A sorted list of integers representing timepoints
    
    Returns:
    --------
    int
        The index of the closest timepoint in corrected_timepoints
    """
    rounded_time = round(time)
    
    # Check if the rounded time is in the list
    if rounded_time in corrected_timepoints:
        return corrected_timepoints.index(rounded_time)
    
    # Find the timepoint with the smallest absolute difference
    closest_diff = float('inf')
    closest_index = -1
    
    for i, timepoint in enumerate(corrected_timepoints):
        diff = abs(timepoint - rounded_time)
        
        # If this timepoint is closer than the current closest
        if diff < closest_diff:
            closest_diff = diff
            closest_index = i
        # If this timepoint is equally close but larger than the current closest
        elif diff == closest_diff and timepoint > corrected_timepoints[closest_index]:
            closest_index = i
    
    return closest_index


def get_pointcloud(all_corrs, corrected_timepoints):
    """
    Bin points by their associated (corrected) hour. 'corrected_timepoints' must be a sorted list of ints.
    """
    if corrected_timepoints is None:
        raise ValueError("get_pointcloud requires 'corrected_timepoints' (sorted ints).")

    T = len(corrected_timepoints)
    root_points = [[] for _ in range(T)]
    for branch in all_corrs:
        for t_hr, xy in branch:
            idx = find_closest_timepoint_index(t_hr, corrected_timepoints)
            root_points[idx].append(xy)
    return root_points

def centroid_over_time(root_points):
    """
    Given a list of lists of points per timepoint, return an array (T, 2)
    of cumulative centroids. If no points yet, emit np.nan for (x,y).
    """
    cumul = []
    cents = []
    for pts_t in root_points:
        cumul.extend(pts_t)
        if len(cumul) == 0:
            cents.append([np.nan, np.nan])
        else:
            arr = np.asarray(cumul, dtype=float)
            # centroid = mean of all observed points up to this time
            cents.append(arr.mean(axis=0).tolist())
    return np.asarray(cents, dtype=float)  # (T,2)


def area_over_time(root_points, alphas):
    areas = []
    cumul = []
    for i, pts_t in enumerate(root_points):
        cumul.extend(pts_t)
        vertices = np.array(cumul, dtype=float)
        if len(vertices) < 3:
            areas.append([0.0]*len(alphas)); continue

        ac = gudhi.AlphaComplex(points=vertices)
        st = ac.create_simplex_tree()
        st.compute_persistence()  # builds filtration

        tri_areas = []
        # Pre-extract triangles with their filtration
        tris = [(simplex, filt) for simplex, filt in st.get_filtration() if len(simplex) == 3]
        for alpha in alphas:
            thr = alpha**2  # AlphaComplex uses squared radius
            A = 0.0
            for tri, filt in tris:
                if filt <= thr:
                    a, b, c = vertices[list(tri)]
                    A += 0.5 * abs(a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
            tri_areas.append(A)
        areas.append(tri_areas)
    return np.array(areas, dtype=float)
