from root import *
import numpy as np


def get_PR_length(rsa: Root, times: list) -> np.ndarray:
    'all lengths containing P2 is a variable of time'

    if not rsa.laterals:   
        Day0_LR_node_0 = []
    else:
        Day0_LR_node_0 = [[lr.nodes[0].x, lr.nodes[0].y] for lr in rsa.laterals if lr.nodes[0].hr == 0]

    if len(Day0_LR_node_0) == 1:
        lowest = Day0_LR_node_0[0]
    elif len(Day0_LR_node_0) > 1:
        lowest = Day0_LR_node_0[np.argmax([coord[1] for coord in Day0_LR_node_0])]
    else:
        lowest = None

    if lowest:
        primary_nodes = np.array([[n.x, n.y] for n in rsa.primary.nodes])
        distances = np.linalg.norm(primary_nodes - lowest, axis=1)
        boundary_index = np.argmin(distances)
        P0 = rsa.primary.nodes[boundary_index].length
    else:
        P0 = 0

    # variables not dependent on time:
    P01 = rsa.primary.tips[0].length
    P1 = P01 - P0

    # get info dependent on time:
    tip_lengths = np.array([tip.length for tip in rsa.primary.tips])

    # tip_times = np.array([tip.time for tip in rsa.primary.tips])
    tip_times = np.array([round(tip.hr) for tip in rsa.primary.tips])

    # Initialize the result array
    P_arr = np.zeros((len(times), 6))

    # Assuming tip_times and tip_lengths are numpy arrays and tip_times is sorted
    for i, time in enumerate(times):
        if time in tip_times:
            Ptotal = tip_lengths[tip_times == time][0]
        else:
            # Find the index of the largest time in tip_times that's smaller than 'time'
            prev_time_index = np.searchsorted(tip_times, time) - 1
            
            Ptotal = tip_lengths[prev_time_index]
            print(f"Time {time} not found. Using closest previous time: {tip_times[prev_time_index]}")
    
        P2 = Ptotal - P01
        P12 = Ptotal - P0
        P_arr[i] = [P0, P1, P2, P01, P12, Ptotal]

    return P_arr



def get_lateral_stage(rsa: Root, include_primary = False) -> list:

    '''
    From the root object get a list to indicate the lateral root stage.

    Args: 
        rsa: Root Object
        include_primary: Boolean, whether to include primary or not.

    Returns:
        lateral_stages: A list of len(# of laterals), with 0: exsit; 1: primary exist; 2: primary has not exist
    '''
        
    # Get the lowest PR node
    primary_depth_t0 = rsa.primary.tips[0].y

    lateral_stages = []

    if include_primary:
        lateral_stages.append(0)
    
    # Check if laterals exist
    if not rsa.laterals:
        return lateral_stages
    
    # Get the coordinate of the first node for the stage 0 LRs

    Day0_LR_node_0 = [[lr.nodes[0].x, lr.nodes[0].y] for lr in rsa.laterals if lr.nodes[0].hr == 0]
    # Get the lowest:
    if len(Day0_LR_node_0) == 1:
        lowest = Day0_LR_node_0[0]
    elif len(Day0_LR_node_0) > 1:
        lowest = Day0_LR_node_0[np.argmax([coord[1] for coord in Day0_LR_node_0])]
    else:
        lowest = None

    for lateral in rsa.laterals:

        init_depth = lateral.nodes[0].y 
        
        # if the depth is above lowest Day0 node:
        if lowest and init_depth <= lowest[1]:
            lateral_stages.append(0)
        # if depth is above last node of primary:
        elif init_depth <= primary_depth_t0:
            lateral_stages.append(1)
        else:
            lateral_stages.append(2)

    return lateral_stages


def get_branchlengths(rsa:Root, include_primary = False) -> list:
    '''
    Get branch lengths (LRs or PR + LRs) over time for all roots.

    Arg:
        rsa: Root object
        include_primary (boolean):, set to `True` when total root lengths' being considered

    Returns:
        all_roots: A list of list. Each branch is represented by a list of (time, length) tuples.
    '''

    all_roots = []

    # Extract length, time data from primary:
    if include_primary:
        tips = rsa.primary.tips
        lengths = [tip.length for tip in tips]
        # times = [tip.time for tip in tips]
        times = [tip.hr for tip in tips]
        all_roots.append(list(zip(times, lengths)))

    # Check if laterals exist
    if not rsa.laterals:
        return all_roots
    
    # Extract length, time data from lateral:
    def branch_length(branch):
        tips = branch.tips
        lengths = [tip.length for tip in tips]
        # times = [tip.time for tip in tips]  
        times = [tip.hr for tip in tips]

        return times, lengths
    
    for lr in rsa.laterals:
        times, lengths = branch_length(lr)
        all_roots.append(list(zip(times, lengths)))
    
    return all_roots


def stagewise_len_t(all_roots:list, 
                    max_time:int, 
                    lateral_stages:list, 
                    stage_choice:int,
                    corrected_timepoints = None)  -> np.ndarray:

    '''
    Get the stage-wise total LR lengths or total root length (including PR) at each imaging time points.
    
    Args:
        all_roots: A list of list for tip length and the associated time for each branch
        max_time: the max image time, used to set the number of snapshots we take.
        lateral_stages: A list of stages with len(lrs)
        stage_choice(0, 1, 10, 2, 21, 210)

    Returns:
        tot_lengths (nparray): total length at each imaged timepoint

    '''

    # tot_lengths = np.zeros((max_time//interval + 1))
    tot_lengths = np.zeros(len(corrected_timepoints))

    # For every imaging time point:
    for idx, t in enumerate(corrected_timepoints):
        # For every branch to add:
        for i, root in enumerate(all_roots):

            # If it is the primary root, or stage 1 or 2 roots
            if stage_choice == 210: # if include all
                include = True

            elif stage_choice == 21: # if include stage 1 and 2
                include = lateral_stages[i]
                
            elif stage_choice == 2: # if include stage 2 only
                include = (lateral_stages[i]==2)

            elif stage_choice == 1:
                include = (lateral_stages[i]==1)
            
            elif stage_choice == 10:
                include = (lateral_stages[i]!=2)
            
            elif stage_choice == 0:
                include = (lateral_stages[i]==0)

            if include:
                # 1. if root exist before t (meaning that we can add):
                if root[0][0] <= t:
                    # we will had to the growth before this imaging time
                    print(f'{i}th root at {t} hr to be added')
                    added = False  
                    #  1.1 find the (hopefully) nearest time point:
                    for (t_prime, root_prime) in root:
                        if round(t_prime) == t:
                            tot_lengths[idx] += root_prime
                        # if closest_t(t_prime) == t:
                        #     tot_lengths[t//interval] += root_prime
                            added = True
                            break

                    # 1.2 if there is no near timepoint (no growth), 
                    # then find the the most recent prev. one as the length:
                    # if skip slide is not allowed then this is not needed at all
                    if not added:
                        for (t_prime, root_prime) in root:
                            if t_prime < t:
                                root_prev = root_prime           
                        # tot_lengths[t//interval] += root_prev
                        tot_lengths[idx] += root_prev
                        added = True
    
                    print(f'{i}th root at {t} hr added:', added)

    # return tot_lengths[:16]
    return tot_lengths

def stagewise_num_lat_t(rsa: Root, 
                        max_time:int, 
                        lateral_stages:list, 
                        stage_choice:int,
                        corrected_timepoints = None) -> np.ndarray:
    '''
    Get the stage-wise # of LR at each imaging time points.
    
    Args:
        rsa: Root object
        max_time: the max image time, used to set the number of snapshots we take.
        lateral_stages: A list of stages with len(lrs)
        stage_choice(0, 1, 2): 0: all LR, or if all roots include PR, all branches; 1: stage 1+2; 2: just stage 2

    Returns:
        tot_laterals(nparray): total # of laterals at each imaged timepoint

    '''
    # tot_laterals = np.zeros((max_time//interval)+1)
    tot_laterals = np.zeros(len(corrected_timepoints))

    # Check if laterals exist
    if not rsa.laterals:
        return tot_laterals
    
    for i, lateral in enumerate(rsa.laterals):
        
        # If it is the primary root, or stage 1 or 2 roots
        if stage_choice == 210: # if include all
            include = True

        elif stage_choice == 21: # if include stage 1 and 2
            include = lateral_stages[i]
            
        elif stage_choice == 2: # if include stage 2 only
            include = (lateral_stages[i]==2)

        elif stage_choice == 1:
            include = (lateral_stages[i]==1)
        
        elif stage_choice == 10:
            include = (lateral_stages[i]!=2)
        
        elif stage_choice == 0:
            include = (lateral_stages[i]==0)
    
        if include:
            # for t in range(0, max_time, interval):
            for idx, t in enumerate(corrected_timepoints):                    
                # If birth time before t:
                if t >= round(lateral.nodes[0].hr):
                    tot_laterals[idx:] += 1
                    break
    # return tot_laterals[:16]
    return tot_laterals


def normalize_len_df(len_df, roots, snapshots, i):
    '''
    i is the index of which pr to use for normalization;
    # will change to just normalize_df()
    '''
    normalized_df = len_df.copy()
    
    for root in roots:
        index = root[0]
        # print(index)
        
        # Get the left and right rows
        left_mask = (normalized_df.index == index) & (normalized_df['side'] == 'L')
        right_mask = (normalized_df.index == index) & (normalized_df['side'] == 'R')
        
        # Extract the snapshot values
        row_left = normalized_df.loc[left_mask, snapshots].to_numpy()
        row_right = normalized_df.loc[right_mask, snapshots].to_numpy()

        # Calculate the P arrays
        # Pick the type of primary matched to this len_df at all timepoints
        P_arr_left = get_PR_length(root[1], snapshots)[:, i] 
        P_arr_right = get_PR_length(root[2], snapshots)[:, i]

        # Normalize the values
        normalized_left = row_left / P_arr_left
        normalized_right = row_right / P_arr_right
        
        # Update the DataFrame with normalized values
        normalized_df.loc[left_mask, snapshots] = normalized_left
        normalized_df.loc[right_mask, snapshots] = normalized_right
    
    normalized_df = normalized_df[snapshots + ['label','condition','side', 'condition-side', 'uniq-condition']]

    return normalized_df

# add for getting average LR size.