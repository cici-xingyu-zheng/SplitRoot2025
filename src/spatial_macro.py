import numpy as np
import gudhi
from scipy.spatial import Delaunay, distance
from root import *

interval = 12 # hrs; is used only when the inverval is REGULAR through out the experiment.

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

    # times = [node.time for node in  rsa.primary.nodes]
    times = [node.hr for node in  rsa.primary.nodes]

    all_corrs.append(list(zip(times, corrs)))

    # Check if laterals exist
    if not rsa.laterals:
        return all_corrs
       
    def branch_coor(branch):
        corrs = [(node.x, node.y) for node in branch.nodes]
        # times = [node.time for node in  branch.nodes]
        times = [node.hr for node in  branch.nodes]
        return times, corrs
    

    
    #  Extract coordinates and time data from laterals:
    for lr in rsa.laterals:
        times, corrs = branch_coor(lr)
        all_corrs.append(list(zip(times, corrs)))
    
    return all_corrs


# def nearest_bigger_multiple_of(num:float) -> int:
#     '''
#     helper func. we had a simper version for temporal
#     '''
#     rounded_number = round(num)
#     if rounded_number % interval == 0:
#         return rounded_number 
#     else:
#         return rounded_number + (interval - rounded_number % interval)

# Potential failure mode (10/14/24): 
# lack of enough points on Day 0: PR # of nodes < 3, as there would be zero simplex
# Current solution: modify the graph directly 

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

def get_pointcloud(all_corrs:list, 
                   max_time:int, corrected_timepoints = None) -> list:
    '''
    Bin points by their associated imaging time.

    Args:
        all_corrs: A list of list for each branch's nodes, represented as a tuple of (time, (x,y)).
        max_time:  Max imaging time.

    Returns:
        root_points: A list of list of the newly imaged nodes' coordinates for each imaging time point.
    '''

    if corrected_timepoints is not None:
        T = len(corrected_timepoints)
    else:
        T = max_time//interval + 1

    root_points = [ [] for _ in range(T)]
    for root in all_corrs:
        for time, corr in root:
            idx = find_closest_timepoint_index(time, corrected_timepoints)
            root_points[idx].append(corr)

        # for time, corr in root:
        #     # if time <= max_time:
        #     # for all points falling on the time measurement (the first node):
        #     t_round = nearest_bigger_multiple_of(time)
        #     if time == round(time):
        #         root_points[(t_round)//interval].append(corr)
        #     # for all points falling earlier, belong to the previous time point:
        #     else:
        #         root_points[(t_round-interval)//interval].append(corr)
                        
    # return root_points[0:16]
    return root_points

def get_area(simplex_tree: gudhi.simplex_tree.SimplexTree, 
             vertices: np.ndarray,  
             alpha: float) -> float:
    '''
    helper to get total area for a particular filtration alpha
    '''
    # Get the alpha complex filtration
    alpha_complex = simplex_tree.get_filtration()

    # Filter the simplices based on the alpha value
    filtered_simplices = [simplex for simplex, filt in alpha_complex if filt <= alpha]
    
    # Extract the triangles from the filtered simplices
    alpha_triangles = [vertices[tri] for tri in filtered_simplices if len(tri) == 3]
    
    # Calculate the area of each triangle and sum them up
    area = 0
    for triangle in alpha_triangles:
        a, b, c = triangle
        area += 0.5 * abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    
    return area


def area_over_time(root_points: list, 
                   alphas:list) -> np.ndarray:
    '''
    Function that calculate the alpha complex area overtime, with different filtration threshold.

    Args:
        root_points: A list of list of root points organized by apparent time.
        alphas: A list of filtration thresholds for Delaunay triangles.

    Returns:
        areas: a 2D nparray (dim_alpha, dim_timepoints) of alpha complex area over time for each alpha.
    '''
    areas = [] # will be storing the areas over time for each alpha

    all_points = [] # we extend this list to include all points emerged during and before the imaging time point

    for i, points_t in enumerate(root_points):
        all_points.extend(points_t)
        print(f'the {i}th time point \n')

        vertices = np.array(all_points)

        # Compute the Delaunay triangulation
        delaunay = Delaunay(vertices)

        # Create a simplex tree
        simplex_tree = gudhi.SimplexTree()

        # Insert the triangles into the simplex tree with filtration value equal to the circumradius of the triangle
        for tri in delaunay.simplices:
            triangle_id = list(tri)
            triangle_vertices = vertices[triangle_id]
            circumcenter = np.mean(triangle_vertices, axis=0)
            circumradius = np.max(distance.cdist(triangle_vertices, [circumcenter]))
            simplex_tree.insert(triangle_id, filtration=circumradius)

        # Compute the persistence of the simplex tree up to dimension 2 (triangles)
        simplex_tree.persistence(persistence_dim_max=2)

        areas_alpha = []

        for alpha in alphas:
            area = get_area(simplex_tree, vertices,  alpha)
            print(area)
            areas_alpha.append(area)
            
        areas.append(areas_alpha)
    
    return np.array(areas) 

