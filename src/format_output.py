# Streamline analysis

from root import *
from temporal_macro import *
from spatial_macro import *

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

import xmltodict

### Global variables and dictionaries to format dataframes

full_side_dict = {
    'LL-L': 'homo-low', 'LL-R': 'homo-low',
    'LH-L': 'hetero-low', 'LH-R': 'hetero-high',
    'HL-L': 'hetero-high', 'HL-R': 'hetero-low',
    'HH-L': 'homo-high', 'HH-R': 'homo-high',
    'MM-L': 'homo-medium', 'MM-R': 'homo-medium',
    'LM-L': 'hetero-low(medium)', 'LM-R': 'hetero-medium',
    'ML-L': 'hetero-medium', 'ML-R': 'hetero-low(medium)',
}

label_dict = {0: 'LL', 1: 'LH', 2: 'HL', 3: 'HH', 4: 'LM', 5: 'ML', 6: 'MM'}
full_condition_order = ['homo-high', 'hetero-high', 'hetero-low', 'homo-low', 'homo-medium', 'hetero-medium', 'hetero-low(medium)'] 


# def correct_timepoints(roots, dataset):
#     """
#     Extract corrected timepoints from rsml files and ensure no duplicates when rounded.
    
#     Parameters:
#     -----------
#     roots : list
#         List of root data where each element contains root file information
#     dataset : str
#         Dataset identifier used for path construction
        
#     Returns:
#     --------
#     list
#         Corrected timepoints with no duplicates when rounded to integers
#     """
#     meta_corrections_roots = []
#     for root in roots:
#         file_name = f'../Data/Hirros{dataset}/{root[0]}'
#         # convert xml to dict:
#         with open(file_name, 'r', encoding='utf-8') as file:
#             rsml = file.read()

#         RSA_dict = xmltodict.parse(rsml)
#         hr_correction = RSA_dict['rsml']['metadata']['observation-hours']
#         hr_correction = [round(float(x)) for x in hr_correction.split(',')] 
        
#         meta_corrections_roots.append(hr_correction)

#     meta_corrections_roots = np.array(meta_corrections_roots)

#     # Get the mode for each timepoint
#     modes = stats.mode(meta_corrections_roots, axis=0)
#     mode_values = modes.mode.flatten()  # Get the mode values as a flat array
    
#     # Some printing to inform if there is a halt in the middle of the process:
#     n_rows = meta_corrections_roots.shape[0]
#     mode_counts = modes.count
#     # Check each column for variations
#     for col in range(meta_corrections_roots.shape[1]):
#         if mode_counts[col] != n_rows:
#             unique_values = np.unique(meta_corrections_roots[:, col])
#             print(f"\nColumn {col} has variations:")
#             print(f"Unique values: {unique_values}")
#             # Get counts for each unique value
#             value_counts = [(val, np.sum(meta_corrections_roots[:, col] == val)) 
#                             for val in unique_values]
#             print("Value counts:")
#             for val, count in value_counts:
#                 print(f"  Value {val}: {count} occurrences")
    
#     # Check for potential duplicates when rounded
#     duplicate_indices = []
    
#     # Find indices of potential duplicates
#     for i in range(len(mode_values) - 1):
#         if mode_values[i] == mode_values[i + 1]:
#             duplicate_indices.append((i, i + 1))
    
#     # Resolve duplicates
#     for i, j in duplicate_indices:
#         print(f"Found potential duplicate timepoints: {mode_values[i]} and {mode_values[j]} (both round to {mode_values[i]})")
        
#         # Decide which one to adjust based on which is closer to its integer
#         dist_i = abs(mode_values[i] - mode_values[i])
#         dist_j = abs(mode_values[j] - mode_values[j])
        
#         if dist_i <= dist_j:
#             # Adjust the second timepoint
#             if mode_values[j] + 1 != mode_values[j+1] if j+1 < len(mode_values) else True:
#                 # If incrementing doesn't create a new duplicate, increment
#                 mode_values[j] = mode_values[j] + 1
#                 print(f"  Adjusted second timepoint to {mode_values[j]}")
#             else:
#                 # If incrementing creates a new duplicate, decrement
#                 mode_values[j] = mode_values[j] - 1
#                 print(f"  Adjusted second timepoint to {mode_values[j]}")
#         else:
#             # Adjust the first timepoint
#             if mode_values[i] - 1 != mode_values[i-1] if i > 0 else True:
#                 # If decrementing doesn't create a new duplicate, decrement
#                 mode_values[i] = mode_values[i] - 1
#                 print(f"  Adjusted first timepoint to {mode_values[i]}")
#             else:
#                 # If decrementing creates a new duplicate, increment
#                 mode_values[i] = mode_values[i] + 1
#                 print(f"  Adjusted first timepoint to {mode_values[i]}")
    
#     final_timepoints = mode_values
    
#     # Verify no duplicates remain
#     if len(final_timepoints) != len(set(final_timepoints)):
#         print("Warning: Duplicates still exist after adjustment!")
#         # Find remaining duplicates
#         seen = set()
#         duplicates = []
#         for i, val in enumerate(final_timepoints):
#             if val in seen:
#                 duplicates.append(i)
#             seen.add(val)
        
#         # Force resolve any remaining duplicates
#         for i in duplicates:
#             final_timepoints[i] += 1
#             print(f"Forced resolution: Incremented timepoint at index {i} to {final_timepoints[i]}")
    
#     return final_timepoints.tolist()

import numpy as np
import xmltodict
from collections import Counter

def correct_timepoints_from_files(filepaths, *, verbose=False, enforce_strict=True):
    """
    Read `rsml/metadata/observation-hours` from each RSML, round to ints,
    then take the per-column mode across files. If there are ties, pick the
    value closest to the column median (still rounded); if still tied, pick
    the smallest value. Optionally enforce strict monotonic increase (+1 steps).

    Returns
    -------
    corrected : list[int]
        The consensus, strictly increasing timepoint integers.
    """
    _p = print if verbose else (lambda *a, **k: None)

    # 1) Read & round each file's observation-hours
    rounded_lists = []
    lengths = []
    for fp in filepaths:
        with open(fp, "r", encoding="utf-8") as f:
            rsml = xmltodict.parse(f.read())
        hours_str = rsml["rsml"]["metadata"]["observation-hours"]
        hours = [float(x) for x in hours_str.split(",")]
        rounded = [int(round(x)) for x in hours]
        rounded_lists.append(rounded)
        lengths.append(len(rounded))

    # 2) Align by common length (use min length across files)
    L = min(lengths) if lengths else 0
    if L == 0:
        raise ValueError("No observation-hours found.")
    if len(set(lengths)) > 1:
        _p(f"[correct_timepoints] Warning: varying lengths across files {set(lengths)}; truncating to {L}.")

    meta = np.array([r[:L] for r in rounded_lists], dtype=int)  # shape: (n_files, L)

    # 3) Column-wise mode with robust tie-breaking
    corrected = []
    for j in range(L):
        col = meta[:, j]
        counts = Counter(col)
        # candidates with max frequency
        max_freq = max(counts.values())
        cands = [v for v, c in counts.items() if c == max_freq]

        if len(cands) == 1:
            pick = cands[0]
        else:
            # tie-break by closeness to median, then by smallest value
            med = int(round(np.median(col)))
            cands.sort(key=lambda v: (abs(v - med), v))
            pick = cands[0]
        corrected.append(int(pick))

    # 4) Optional strict-increase cleanup: ensure t[i] > t[i-1]
    if enforce_strict:
        for i in range(1, L):
            if corrected[i] <= corrected[i-1]:
                corrected[i] = corrected[i-1] + 1

    return corrected


# Wrapper: (roots, dataset) -> filepaths
def correct_timepoints(roots, dataset, base_dir="../rsml_files", *, verbose=False, enforce_strict=True):
    """
    Wrapper to preserve your earlier call style.

    Parameters
    ----------
    roots : iterable
        Your root listing; each element should expose the filename at index 0 (root[0]).
    dataset : str|int
        Dataset code appended to 'Hirros{dataset}'.
    base_dir : str
        Base directory containing 'Hirros{dataset}'.

    Returns
    -------
    list[int] : corrected timepoints
    """
    filepaths = [f"{base_dir}/Hirros{dataset}/{r[0]}" for r in roots]
    return correct_timepoints_from_files(filepaths, verbose=verbose, enforce_strict=enforce_strict)


def get_stagewise_length(roots, choice, max_time, include_primary, corrected_timepoints):
    tot_rootlengths = []
    for root in roots:
        print('Sample name:', root[0])
        print('------')
        left_all = get_branchlengths(root[1], include_primary)
        right_all = get_branchlengths(root[2], include_primary)

        l_stages =  get_lateral_stage(root[1], include_primary)
        r_stages =  get_lateral_stage(root[2], include_primary)

        left_len_t = stagewise_len_t(left_all, max_time, l_stages, choice, corrected_timepoints)
        right_len_t = stagewise_len_t(right_all, max_time, r_stages, choice, corrected_timepoints)

        tot_t = np.array([left_len_t, right_len_t])
        tot_rootlengths.append(tot_t)
        print()

    return tot_rootlengths 


def get_stagewise_num(roots, choice, max_time, include_primary, corrected_timepoints):
    tot_num_laterals = []
    for root in roots:
        print('Sample name:', root[0])
        print('------')

        l_stages =  get_lateral_stage(root[1], include_primary)
        r_stages =  get_lateral_stage(root[2], include_primary)

        left_num_t= stagewise_num_lat_t(root[1], max_time, l_stages, choice, corrected_timepoints)
        right_num_t = stagewise_num_lat_t(root[2], max_time, r_stages, choice, corrected_timepoints)

        tot_n = np.array([left_num_t, right_num_t])
        tot_num_laterals.append(tot_n)
        print()

    return tot_num_laterals 


def get_area_dfs(roots, labels, max_time, alphas, corrected_timepoints):
    areas = []
    indices = []
    for root in roots:
        label, left_root, right_root = root
        print(label)
        print()
        left_corrs = get_branch_coors(left_root)
        right_corrs = get_branch_coors(right_root)
        lr_points = get_pointcloud(left_corrs, max_time, corrected_timepoints)
        rr_points = get_pointcloud(right_corrs, max_time, corrected_timepoints)

        left_areas = area_over_time(lr_points, alphas)
        right_areas = area_over_time(rr_points, alphas)
        
        for i in range(4):
            left = ['L', alphas[i], labels[label]] + left_areas[:, i].tolist()
            areas.append(left)
            indices.append(label)
            right = ['R', alphas[i], labels[label]] + right_areas[:, i].tolist()
            areas.append(right)
            indices.append(label)
   
    areas = np.array(areas)

    timepoints = corrected_timepoints

    areas_df = pd.DataFrame(areas, columns=['side', 'alpha', 'label'] + timepoints, index=indices)

    conditions = [label_dict[int(label)] for label in areas_df["label"].tolist()]
    areas_df['condition'] = conditions
    areas_df['condition-side'] = areas_df['condition'] + '-' + areas_df['side']

    # Get the unique conditions in the current dataset
    unique_conditions = set(areas_df['condition-side'].tolist())
    
    # Filter side_dict to only include relevant entries
    filtered_side_dict = {k: v for k, v in full_side_dict.items() if k in unique_conditions}
    
    # Apply mapping
    labelsides = areas_df['condition-side'].tolist()
    areas_df['uniq-condition'] = [filtered_side_dict.get(l) for l in labelsides]
    
    # Define order based on available conditions
    available_conditions = set(areas_df['uniq-condition'].unique())

    condition_order = [cond for cond in full_condition_order if cond in available_conditions]   
    # Apply categorical ordering
    areas_df['uniq-condition'] = pd.Categorical(areas_df['uniq-condition'], 
                                                categories=condition_order, 
                                                ordered=True)
                                                
    for timepoint in timepoints:
        areas_df[timepoint] = pd.to_numeric(areas_df[timepoint])

    alpha_areas_dfs = []
    for alpha in alphas:
        alpha_areas_df = areas_df[areas_df['alpha'] == str(alpha)]
        alpha_areas_dfs.append(alpha_areas_df)
    
    return alpha_areas_dfs


def get_primary_df(roots, snapshots, label_df):
    """
    Create a combined dataframe for all roots with Ptotal values for both left and right sides.
    
    Parameters:
    -----------
    roots : list
        List of root data where each element is [root_index, left_root, right_root]
    snapshots : list
        List of time points to use as index
    label_df : DataFrame
        DataFrame that maps each root to a condition
        
    Returns:
    --------
    DataFrame
        A combined DataFrame with Ptotal values and condition labeling
    """
    # Lists to store data
    primary_data = []
    indices = []
    sides = []
    
    # Process each root
    for root in roots:
        root_idx = root[0]
        
        # Get PR lengths for left and right
        P_arr_left = get_PR_length(root[1], snapshots)
        P_arr_right = get_PR_length(root[2], snapshots)
        
        # Extract Ptotal values (assuming it's the last column)
        left_ptotal = P_arr_left[:, -1]  # Ptotal is the last column
        right_ptotal = P_arr_right[:, -1]
        
        # Add to data lists
        primary_data.append(left_ptotal)
        primary_data.append(right_ptotal)
        
        # Add indices and sides
        indices.extend([root_idx, root_idx])
        sides.extend(['L', 'R'])
    
    # Convert to numpy array and create DataFrame
    primary_data = np.array(primary_data)
    
    # Create DataFrame with snapshots as columns
    primary_df = pd.DataFrame(primary_data, columns=snapshots)
    
    # Add index and side information
    primary_df['root_idx'] = indices
    primary_df['side'] = sides
    
    # Merge with label_df to get condition information
    # Assuming label_df has root_idx as index
    primary_df = primary_df.set_index('root_idx')
    
    # Add label data
    primary_df['label'] = label_df.loc[primary_df.index, 'label'].values
    primary_df['condition'] = label_df.loc[primary_df.index, 'condition'].values
    
    # Reset index for easier manipulation
    primary_df = primary_df.reset_index()
    
    # Add condition-side
    primary_df['condition-side'] = primary_df['condition'] + '-' + primary_df['side']
    
    # Get unique condition-sides in the dataset
    unique_condition_sides = set(primary_df['condition-side'].unique())
    
    # Filter side_dict to include only relevant entries
    side_dict = {k: v for k, v in full_side_dict.items() if k in unique_condition_sides}
    
    # Apply mapping
    labelsides = primary_df['condition-side'].tolist()
    primary_df['uniq-condition'] = [side_dict.get(l) for l in labelsides]
    
    # Define order based on available conditions
    available_conditions = set(primary_df['uniq-condition'].unique())
    # full_condition_order = ['homo-high', 'hetero-high', 'hetero-low', 'homo-low']
    condition_order = [cond for cond in full_condition_order if cond in available_conditions]
    
    # Apply categorical ordering
    primary_df['uniq-condition'] = pd.Categorical(primary_df['uniq-condition'], 
                                              categories=condition_order, 
                                              ordered=True)
    
    return primary_df

# will add another one for Primary length at t0.

def organize_df(root_data, max_time, label_df, corrected_timepoints=None):
    root_data = np.array(root_data)
    data_sqze = np.concatenate((root_data[:,0, :], root_data[:,1, :]), axis=0)

    if corrected_timepoints is not None:
        colnames = corrected_timepoints
    else:
        print('Using default timepoints with default interval:', interval)
        print('If this is not the interval used in experiment, please correct it. Danger of failing given uneven intervals. ')
        colnames = list(range(0, max_time, interval))  # keep int. as it'd be easier to get different time

    data_t_df = pd.DataFrame(data_sqze, columns=colnames, index=label_df.index.tolist()*2)
    # add annotation columns
    data_t_df['label'] = label_df['label'].tolist()*2
    # condition: first two letters represent nitrate level (L for low or H for high), third letter for the side (L for left and R for right)
    data_t_df['condition'] = label_df['condition'].tolist()*2
    data_t_df['side'] = ['L']*len(label_df) + ['R']*len(label_df)
    # add annotation columns
    # label-side first two letters represent nitrate level (L for low or H for high), third letter for the side (L for left and R for right)
    data_t_df['condition-side'] = data_t_df['condition'] + '-' + data_t_df['side']
    
    
    # Get the unique condition-sides in the current dataset
    unique_condition_sides = set(data_t_df['condition-side'].unique())
    
    # Filter side_dict to only include relevant entries
    side_dict = {k: v for k, v in full_side_dict.items() if k in unique_condition_sides}
    
    # Apply mapping
    labelsides = data_t_df['condition-side'].tolist()
    data_t_df['uniq-condition'] = [side_dict.get(l) for l in labelsides]
    
    # Define order based on available conditions
    available_conditions = set(data_t_df['uniq-condition'].unique())
    # full_condition_order = ['homo-high', 'hetero-high', 'hetero-low', 'homo-low']
    condition_order = [cond for cond in full_condition_order if cond in available_conditions]
    
    # Apply categorical ordering
    data_t_df['uniq-condition'] = pd.Categorical(data_t_df['uniq-condition'], 
                                              categories=condition_order, 
                                              ordered=True)
    
    return data_t_df


def visualize_dataset(df, snapshot, measure, set_max=None, ylab=None, save_name=None):
    
    # Full order of all possible conditions
    full_order = ['homo-high', 'hetero-high', 'hetero-low', 'homo-low', 'homo-medium', 'hetero-medium', 'hetero-low(medium)']
    
    # Get only the conditions present in the dataframe
    available_conditions = sorted(df['uniq-condition'].unique(), 
                                 key=lambda x: full_order.index(x) if x in full_order else float('inf'))
    
    # Helper function to lighten a color (reduce saturation)
    def lighten_color(color, amount=0.3):
        """Lighten a color by reducing saturation"""
        if isinstance(color, str) and color.startswith('#'):
            # Convert hex to RGB
            rgb = mcolors.hex2color(color)
        else:
            # Assume it's already RGB or a named color
            rgb = mcolors.to_rgb(color)
        
        # Convert to HSV, reduce saturation, convert back
        hsv = mcolors.rgb_to_hsv(rgb)
        hsv[1] = hsv[1] * (1 - amount)  # Reduce saturation
        return mcolors.hsv_to_rgb(hsv)
    
    # Define colors for all possible conditions
    base_colors = {
        'len': [sns.color_palette("Set2")[-4], '#d1e231', sns.color_palette("Set2")[-3], sns.color_palette("Set2")[-2]],
        'num': [sns.color_palette("Paired")[4], sns.color_palette("Set2")[3], sns.color_palette("Paired")[8], sns.color_palette("Set2")[-1]],
        'area': ['royalblue', 'dodgerblue', 'cornflowerblue', 'lightsteelblue']
    }
    
    # Add lighter versions for medium conditions
    type2colors = {}
    for measure_type, colors in base_colors.items():
        extended_colors = colors.copy()
        # Add homo-medium (lighter version of homo-high)  
        extended_colors.append(lighten_color(colors[0], 0.4))  # homo-high is index 0
        # Add hetero-medium (lighter version of hetero-high)
        extended_colors.append(lighten_color(colors[1], 0.4))  # hetero-high is index 1
        # Add hetero-low(medium) (lighter version of hetero-low)
        extended_colors.append(lighten_color(colors[2], 0.4))  # hetero-low is index 2

        type2colors[measure_type] = extended_colors
    
    # Get colors for the specific measure
    all_colors = type2colors[measure]
    
    # Create a palette using only available conditions
    palette = {}
    for condition in available_conditions:
        idx = full_order.index(condition)
        palette[condition] = all_colors[idx]
    
    # Dynamic figure width: 2 * number of conditions, minimum of 6
    fig_width = max(6, 2 * len(available_conditions))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    # Use only available conditions for plotting
    sns.barplot(data=df, x='uniq-condition', y=snapshot, palette=palette, order=available_conditions,
                capsize=.1)
    sns.swarmplot(data=df, x='uniq-condition', y=snapshot, order=available_conditions,
                 color='black', size=8, alpha=.3)
    
    if set_max:
        ax.set_ylim([0, set_max])
    ax.set_ylabel(f'{ylab}', fontsize=16)
    ax.set_title(f'Day {snapshot//24} | Hour {snapshot}', fontsize=18)
    plt.show()
    
    if save_name:
        fig.savefig(f'{save_name}.pdf', transparent=True)