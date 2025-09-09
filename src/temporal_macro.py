from src.root import *
import numpy as np


def get_PR_length(rsa: Root, snapshots: list) -> np.ndarray:
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
    tip_times = np.array([round(tip.hr) for tip in rsa.primary.tips])

    # Initialize the result array
    P_arr = np.zeros((len(snapshots), 6))

    # Assuming tip_times and tip_lengths are numpy arrays and tip_times is sorted
    for i, time in enumerate(snapshots):
        # if time in tip_times:
        #     Ptotal = tip_lengths[tip_times == time][0]
        # else:
        #     # Find the index of the largest time in tip_times that's smaller than 'time'
        #     prev_time_index = np.searchsorted(tip_times, time) - 1
            
        #     Ptotal = tip_lengths[prev_time_index]
        #     print(f"Time {time} not found. Using closest previous time: {tip_times[prev_time_index]}")

        # return the insert index after equality, so -1 if time < first element
        idx = np.searchsorted(tip_times, time, side="right") - 1
        if idx < 0:
            Ptotal = np.nan # shouldn't happen; first time in snapshots should be zero
        else:
            Ptotal = tip_lengths[idx]

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


def stagewise_len_t(all_roots: list, 
                    lateral_stages: list, 
                    stage_choice: int,
                    corrected_timepoints=None,
                    verbose: bool = False) -> np.ndarray:
    """
    Get the stage-wise total LR lengths or total root length (including PR) at each imaging time point.

    Args:
        all_roots: list[list[tuple(time, length)]]
            One list per branch; each inner list is [(t0, L0), (t1, L1), ...].
        lateral_stages: list[int]
            Stage per branch (0/1/2); if primary included, you likely set its stage to 0 in the caller.
        stage_choice: int
            One of {0, 1, 2, 10, 21, 210}; see your original semantics.
        corrected_timepoints: list[int]
            Consensus imaging hours (sorted).
        verbose: bool
            If True, prints detailed aggregation info.

    Returns:
        np.ndarray
            Total length at each timepoint.
    """
    if corrected_timepoints is None:
        raise ValueError("stagewise_len_t requires 'corrected_timepoints' (sorted ints).")

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    T = len(corrected_timepoints)
    tot_lengths = np.zeros(T, dtype=float)

    vprint(f"[stagewise_len_t] branches={len(all_roots)}, timepoints={T}, choice={stage_choice}")

    for idx, t in enumerate(corrected_timepoints):
        included = 0
        vprint(f"\n[t={t}] aggregating...")

        for i, root in enumerate(all_roots):
            # Decide inclusion based on stage choice
            s = lateral_stages[i] if i < len(lateral_stages) else None
            if stage_choice == 210:         # include all
                include = True
            elif stage_choice == 21:        # include stage 1 and 2
                include = bool(s)           # 0->False, 1/2->True
            elif stage_choice == 2:
                include = (s == 2)
            elif stage_choice == 1:
                include = (s == 1)
            elif stage_choice == 10:        # include stage 0 and 1
                include = (s != 2)
            elif stage_choice == 0:
                include = (s == 0)
            else:
                include = False  # unknown code -> exclude

            if not include:
                vprint(f"  - branch {i}: stage={s} -> excluded by choice={stage_choice}")
                continue

            # Only add if the branch exists by time t
            if root and root[0][0] <= t:
                included += 1
                added = False
                used_time = None
                val = 0.0

                # Try exact (rounded) time match first
                for (t_prime, root_prime) in root:
                    if round(t_prime) == t:
                        tot_lengths[idx] += root_prime
                        val = root_prime
                        used_time = t_prime
                        added = True
                        vprint(f"  ✓ branch {i}: stage={s} exact match at {t_prime} -> +{root_prime:.3f}")
                        break

                # Otherwise use the most recent previous value
                if not added:
                    prev_val = None
                    prev_t = None
                    for (t_prime, root_prime) in root:
                        if t_prime < t:
                            prev_val = root_prime
                            prev_t = t_prime
                    if prev_val is not None:
                        tot_lengths[idx] += prev_val
                        val = prev_val
                        used_time = prev_t
                        vprint(f"  ✓ branch {i}: stage={s} prev match at {prev_t} -> +{prev_val:.3f}")
                    else:
                        # Shouldn't happen because root[0][0] <= t, but guard anyway
                        vprint(f"  ! branch {i}: stage={s} no usable sample ≤ {t}; adding +0")

            else:
                first_t = root[0][0] if root else None
                vprint(f"  - branch {i}: stage={s} not yet present at t={t} (first={first_t}); skip")

        vprint(f"[t={t}] included {included} branches -> total={tot_lengths[idx]:.3f}")

    vprint("\n[stagewise_len_t] done.")
    return tot_lengths


# def stagewise_len_t(all_roots:list, 
#                     lateral_stages:list, 
#                     stage_choice:int,
#                     corrected_timepoints = None,
#                     verbose = False)  -> np.ndarray:

#     '''
#     Get the stage-wise total LR lengths or total root length (including PR) at each imaging time points.
    
#     Args:
#         all_roots: A list of list for tip length and the associated time for each branch
#         max_time: the max image time, used to set the number of snapshots we take.
#         lateral_stages: A list of stages with len(lrs)
#         stage_choice(0, 1, 10, 2, 21, 210)

#     Returns:
#         tot_lengths (nparray): total length at each imaged timepoint

#     '''

#     # tot_lengths = np.zeros((max_time//interval + 1))
#     tot_lengths = np.zeros(len(corrected_timepoints))

#     # For every imaging time point:
#     for idx, t in enumerate(corrected_timepoints):
#         # For every branch to add:
#         for i, root in enumerate(all_roots):

#             # If it is the primary root, or stage 1 or 2 roots
#             if stage_choice == 210: # if include all
#                 include = True

#             elif stage_choice == 21: # if include stage 1 and 2
#                 include = lateral_stages[i]
                
#             elif stage_choice == 2: # if include stage 2 only
#                 include = (lateral_stages[i]==2)

#             elif stage_choice == 1:
#                 include = (lateral_stages[i]==1)
            
#             elif stage_choice == 10:
#                 include = (lateral_stages[i]!=2)
            
#             elif stage_choice == 0:
#                 include = (lateral_stages[i]==0)

#             if include:
#                 # 1. if root exist before t (meaning that we can add):
#                 if root[0][0] <= t:
#                     # we will had to the growth before this imaging time
#                     print(f'{i}th root at {t} hr to be added')
#                     added = False  
#                     #  1.1 find the (hopefully) nearest time point:
#                     for (t_prime, root_prime) in root:
#                         if round(t_prime) == t:
#                             tot_lengths[idx] += root_prime
#                         # if closest_t(t_prime) == t:
#                         #     tot_lengths[t//interval] += root_prime
#                             added = True
#                             break

#                     # 1.2 if there is no near timepoint (no growth), 
#                     # then find the the most recent prev. one as the length:
#                     # if skip slide is not allowed then this is not needed at all
#                     if not added:
#                         for (t_prime, root_prime) in root:
#                             if t_prime < t:
#                                 root_prev = root_prime           
#                         # tot_lengths[t//interval] += root_prev
#                         tot_lengths[idx] += root_prev
#                         added = True
    
#                     print(f'{i}th root at {t} hr added:', added)

#     # return tot_lengths[:16]
#     return tot_lengths

def stagewise_num_lat_t(rsa: Root, 
                        lateral_stages:list, 
                        stage_choice:int,
                        corrected_timepoints = None) -> np.ndarray:
    '''
    Get the stage-wise # of LR at each imaging time points.
    
    Args:
        rsa: Root object
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

