# Streamline analysis

from src.root import *
from src.temporal_macro import *
from src.spatial_macro import *

import pandas as pd
import numpy as np
import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

import xmltodict

import xmltodict
from collections import Counter

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


def make_label_df(pkl_file, *, order=None, coerce_int=True, validate=True):
    """
    Load a {filename -> label} mapping from a pickle file and build label_df.

    Parameters
    ----------
    pkl_file : str | Path
        Path to a pickle containing a dict-like mapping {filename: label}.
    order : list[str] | None
        Optional list of filenames to enforce index order (rows not present in
        the pickle are dropped; extra entries in the pickle are kept at the end).
    coerce_int : bool
        If True, cast labels to int before mapping conditions.
    validate : bool
        If True, error if any labels cannot be mapped by `label_dict`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
          - 'label' (int)
          - 'condition' (str, from `label_dict`)
        and index = filenames from the pickle (optionally reordered by `order`).
    """
    with open(pkl_file, "rb") as f:
        labels = pkl.load(f)

    if not hasattr(labels, "items"):
        raise TypeError("Pickle must contain a dict-like object of {filename: label}.")

    # Build base df
    df = pd.DataFrame(data=list(labels.values()),
                      index=list(labels.keys()),
                      columns=["label"])

    if coerce_int:
        df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    # Map to condition using the module-level label_dict
    # (expects label_dict to exist in format_output.py)
    missing = set(df["label"].unique()) - set(label_dict.keys())
    if validate and missing:
        raise KeyError(
            f"Found labels with no mapping in label_dict: {sorted(missing)}. "
            "Update `label_dict` or disable validation with validate=False."
        )

    df["condition"] = [label_dict.get(x, None) for x in df["label"]]

    return df




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


def get_stagewise_length(roots, choice, include_primary, corrected_timepoints):
    tot_rootlengths = []
    for root in roots:
        print('------')
        print('Sample name:', root[0])
        
        left_all = get_branchlengths(root[1], include_primary)
        right_all = get_branchlengths(root[2], include_primary)

        l_stages =  get_lateral_stage(root[1], include_primary)
        r_stages =  get_lateral_stage(root[2], include_primary)

        left_len_t = stagewise_len_t(left_all, l_stages, choice, corrected_timepoints)
        right_len_t = stagewise_len_t(right_all, r_stages, choice, corrected_timepoints)

        tot_t = np.array([left_len_t, right_len_t])
        tot_rootlengths.append(tot_t)
        print()

    return tot_rootlengths 


def get_stagewise_num(roots, choice, include_primary, corrected_timepoints):
    tot_num_laterals = []
    for root in roots:
        print('------')
        print('Sample name:', root[0])

        l_stages =  get_lateral_stage(root[1], include_primary)
        r_stages =  get_lateral_stage(root[2], include_primary)

        left_num_t= stagewise_num_lat_t(root[1], l_stages, choice, corrected_timepoints)
        right_num_t = stagewise_num_lat_t(root[2], r_stages, choice, corrected_timepoints)

        tot_n = np.array([left_num_t, right_num_t])
        tot_num_laterals.append(tot_n)
        print()

    return tot_num_laterals 


def get_area_dfs(roots, labels, alphas, corrected_timepoints):
    areas = []
    indices = []
    for root in roots:
        label, left_root, right_root = root
        print('------')
        print('Sample name:', label)
        print()
        left_corrs = get_branch_coors(left_root)
        right_corrs = get_branch_coors(right_root)
        lr_points = get_pointcloud(left_corrs, corrected_timepoints)
        rr_points = get_pointcloud(right_corrs, corrected_timepoints)

        left_areas = area_over_time(lr_points, alphas)
        right_areas = area_over_time(rr_points, alphas)
        
        for i in range(4):
            # left = ['L', alphas[i], labels[label]] + left_areas[:, i].tolist()
            left  = left_areas[:, i].tolist() +  ['L', alphas[i], labels[label]] 
            areas.append(left)
            indices.append(label)
            # right = ['R', alphas[i], labels[label]] + right_areas[:, i].tolist()
            right = right_areas[:, i].tolist() + ['R', alphas[i], labels[label]] 
            areas.append(right)
            indices.append(label)

    areas = np.array(areas)
    timepoints = corrected_timepoints

    areas_df = pd.DataFrame(areas, columns= timepoints + ['side', 'alpha', 'label'], index=indices)

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
        areas_df['alpha'] = pd.to_numeric(areas_df['alpha'], errors='raise').astype(int)

    alpha_areas_dfs = []
    for alpha in alphas:
        alpha_areas_df = areas_df[areas_df['alpha'] == alpha]
        alpha_areas_dfs.append(alpha_areas_df)
    
    return alpha_areas_dfs


def get_primary_over_time_df(roots, snapshots, label_df):
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

def get_primary_t0(roots, label_df):
    # Initialize lists to store data for the final DataFrame
    data_rows = []
    
    for root in roots:
            filename, left_root, right_root = root
            # Get all primary lengths for left and right sides, but only at t0:
            Primary_arr_left = get_PR_length(left_root, [0])[0, :]
            Primary_arr_right = get_PR_length(right_root, [0])[0, :]
            
            # Primary array contains [P0, P1, P2, P01, P12, Ptotal]
            primary_names = ['P0_length', 'P1_length', 'P2_length', 'P01_length', 'P12_length', 'Ptotal_length']
            
            # Get condition for this filename
            condition = label_df.loc[filename, 'condition']
            
            # Create row for left side
            left_row = {
                'condition': condition,
                'side': 'L',
                'condition-side': f'{condition}-L'
            }
            
            # Add all primary lengths for left side
            for i, name in enumerate(primary_names):
                left_row[name] = Primary_arr_left[i]
            
            # Create row for right side
            right_row = {
                'condition': condition,
                'side': 'R',
                'condition-side': f'{condition}-R'
            }
            
            # Add all primary lengths for right side
            for i, name in enumerate(primary_names):
                right_row[name] = Primary_arr_right[i]
            
            # Add filename as a column (since it will be the index)
            left_row['filename'] = filename
            right_row['filename'] = filename
            
            data_rows.append(left_row)
            data_rows.append(right_row)

    # Create DataFrame from collected data
    df = pd.DataFrame(data_rows)
    df.set_index('filename', inplace=True)
    df['uniq-condition'] = df['condition-side'].map(full_side_dict)

    return df


def organize_df(root_data, label_df, corrected_timepoints):
    root_data = np.array(root_data)
    data_sqze = np.concatenate((root_data[:,0, :], root_data[:,1, :]), axis=0)

    colnames = corrected_timepoints

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
    condition_order = [cond for cond in full_condition_order if cond in available_conditions]
    
    # Apply categorical ordering
    data_t_df['uniq-condition'] = pd.Categorical(data_t_df['uniq-condition'], 
                                              categories=condition_order, 
                                              ordered=True)
    
    return data_t_df



def normalize_df(len_df, roots, snapshots, i):
    """
    Normalize length-like DataFrame by a chosen PR series (index i in get_PR_length columns).
    - 0/0 -> 0
    - nonzero/0 -> 0 and PRINT a warning with times
    """
    import numpy as np
    normalized_df = len_df.copy()

    for root in roots:
        index = root[0]

        # Select rows for this sample
        left_mask  = (normalized_df.index == index) & (normalized_df['side'] == 'L')
        right_mask = (normalized_df.index == index) & (normalized_df['side'] == 'R')

        # Skip if the row isn't present
        if not left_mask.any() and not right_mask.any():
            continue

        # Extract numerator rows (may come out shape (1, T); flatten to (T,))
        if left_mask.any():
            row_left = normalized_df.loc[left_mask, snapshots].to_numpy(dtype=float)
            row_left = np.nan_to_num(row_left, nan=0.0)
            row_left = row_left.ravel()
        if right_mask.any():
            row_right = normalized_df.loc[right_mask, snapshots].to_numpy(dtype=float)
            row_right = np.nan_to_num(row_right, nan=0.0)
            row_right = row_right.ravel()

        # Denominators from PR arrays (shape (T,))
        P_arr_left  = np.nan_to_num(get_PR_length(root[1], snapshots)[:, i], nan=0.0)
        P_arr_right = np.nan_to_num(get_PR_length(root[2], snapshots)[:, i], nan=0.0)

        # Sanity check on lengths
        T = len(snapshots)
        if left_mask.any() and P_arr_left.shape[0] != T:
            raise ValueError(f"PR(left) length {P_arr_left.shape[0]} != snapshots length {T} for {index}")
        if right_mask.any() and P_arr_right.shape[0] != T:
            raise ValueError(f"PR(right) length {P_arr_right.shape[0]} != snapshots length {T} for {index}")

        # LEFT: detect nonzero/0, then safe divide
        if left_mask.any():
            bad_left = (P_arr_left == 0) & (row_left != 0)
            if np.any(bad_left):
                bad_times = [snapshots[j] for j in np.flatnonzero(bad_left)]
                print(f"[normalize_df] WARNING: nonzero/0 division for sample {index} side=L at times {bad_times}. Setting to 0.")
            norm_left = np.divide(row_left, P_arr_left,
                                  out=np.zeros_like(row_left, dtype=float),
                                  where=(P_arr_left != 0))
            normalized_df.loc[left_mask, snapshots] = norm_left

        # RIGHT: detect nonzero/0, then safe divide
        if right_mask.any():
            bad_right = (P_arr_right == 0) & (row_right != 0)
            if np.any(bad_right):
                bad_times = [snapshots[j] for j in np.flatnonzero(bad_right)]
                print(f"[normalize_df] WARNING: nonzero/0 division for sample {index} side=R at times {bad_times}. Setting to 0.")
            norm_right = np.divide(row_right, P_arr_right,
                                   out=np.zeros_like(row_right, dtype=float),
                                   where=(P_arr_right != 0))
            normalized_df.loc[right_mask, snapshots] = norm_right

    # Keep column order
    normalized_df = normalized_df[snapshots + ['label','condition','side','condition-side','uniq-condition']]
    return normalized_df


def get_avg_len_df(len_df, num_df, snapshots):
    """
    Compute average lateral length (len / num) from length and number DataFrames.

    Parameters
    ----------
    len_df : pd.DataFrame
        Output of organize_df with root lengths.
    num_df : pd.DataFrame
        Output of organize_df with lateral counts.
    snapshots : list
        List of snapshot timepoints (column names to use for division).

    Returns
    -------
    avg_len_df : pd.DataFrame
        DataFrame of same shape, with entries len/num (0 if num==0).
    """
    # Copy structure from len_df to preserve metadata columns
    avg_len_df = len_df.copy()

    # Safe division: where num==0, result=0
    for t in snapshots:
        L = pd.to_numeric(len_df[t], errors="coerce").fillna(0)
        N = pd.to_numeric(num_df[t], errors="coerce").fillna(0)
        avg = np.where((N > 0) & (L > 0), L / N, 0.0)
        avg_len_df[t] = avg

    # Reorder columns (snapshots first, then metadata)
    meta_cols = [c for c in len_df.columns if c not in snapshots]
    avg_len_df = avg_len_df[snapshots + meta_cols]

    return avg_len_df


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


def visualize_over_time(input_df,
                        corrected_timepoints,
                        measure: str,            # one of: "len", "num", "area"
                        ylim=None,
                        ylab=None,
                        title=None,
                        save_name=None):
    """
    Plot mean ± SEM over time for a DataFrame produced by `organize_df`
    (e.g., len_df, num_df, area_df). Colors and condition ordering match
    `visualize_dataset`.

    Parameters
    ----------
    input_df : pd.DataFrame
        Output of `organize_df` (must contain columns for each timepoint
        in `corrected_timepoints`, and a 'uniq-condition' column).
    corrected_timepoints : list
        The timepoint columns to plot (e.g., the consensus hours).
    measure : {"len","num","area"}
        Controls color palette selection and default y-label.
    ylim : float | (float, float) | None
        If float, interpreted as (0, ylim). If tuple, used directly.
    ylab : str | None
        Y-axis label; if None, a sensible default is chosen.
    title : str | None
        Figure title; if None, a generic title is used.
    save_name : str | None
        If provided, saves the figure to '{save_name}.pdf' (transparent).
    """
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # --- condition ordering & palette (matching visualize_dataset) ---
    full_order = ['homo-high', 'hetero-high', 'hetero-low', 'homo-low',
                  'homo-medium', 'hetero-medium', 'hetero-low(medium)']
    available_conditions = sorted(
        input_df['uniq-condition'].dropna().unique(),
        key=lambda x: full_order.index(x) if x in full_order else float('inf')
    )

    def lighten_color(color, amount=0.3):
        """Lighten a color by reducing saturation (used for *-medium entries)."""
        if isinstance(color, str) and color.startswith('#'):
            rgb = mcolors.hex2color(color)
        else:
            rgb = mcolors.to_rgb(color)
        hsv = mcolors.rgb_to_hsv(rgb)
        hsv[1] = hsv[1] * (1 - amount)  # reduce saturation
        return mcolors.hsv_to_rgb(hsv)

    base_colors = {
        'len':  [sns.color_palette("Set2")[-4], '#d1e231',
                 sns.color_palette("Set2")[-3], sns.color_palette("Set2")[-2]],
        'num':  [sns.color_palette("Paired")[4], sns.color_palette("Set2")[3],
                 sns.color_palette("Paired")[8], sns.color_palette("Set2")[-1]],
        'area': ['royalblue', 'dodgerblue', 'cornflowerblue', 'lightsteelblue']
    }
    # extend to include the three "*-medium" categories (lighter variants)
    extended = base_colors[measure].copy()
    extended.append(lighten_color(base_colors[measure][0], 0.4))  # homo-medium
    extended.append(lighten_color(base_colors[measure][1], 0.4))  # hetero-medium
    extended.append(lighten_color(base_colors[measure][2], 0.4))  # hetero-low(medium)

    palette = {cond: extended[full_order.index(cond)]
               for cond in available_conditions if cond in full_order}

    # --- melt and aggregate (mean ± SEM) ---
    melted_df = pd.melt(input_df,
                        id_vars=['uniq-condition'],
                        value_vars=corrected_timepoints,
                        var_name='time',
                        value_name=measure)

    melted_df['time'] = pd.to_numeric(melted_df['time'], errors='coerce')
    # guard against stray NaNs in time
    melted_df = melted_df.dropna(subset=['time'])

    # compute group stats
    stats_df = (melted_df
                .groupby(['uniq-condition', 'time'])[measure]
                .agg(['mean', 'sem'])
                .reset_index()
                .rename(columns={'uniq-condition': 'condition'}))

    # --- plot ---
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    # maintain the legend order as available_conditions
    for cond in available_conditions:
        cd = stats_df[stats_df['condition'] == cond]
        if cd.empty:
            continue
        color = palette.get(cond, None)
        # mean line
        plt.plot(cd['time'], cd['mean'], label=cond, marker='o', linewidth=2,
                 color=color)
        # mean ± SEM band
        ylo = cd['mean'] - cd['sem']
        yhi = cd['mean'] + cd['sem']
        plt.fill_between(cd['time'], ylo, yhi, alpha=0.2, edgecolor='none',
                         color=color)

    # labels / title
    default_ylab = {
        'len': 'Total lateral length',
        'num': '# laterals',
        'area': 'Alpha-complex area'
    }[measure]
    plt.xlabel('Time (hr)', fontsize=14)
    plt.ylabel(ylab or default_ylab, fontsize=14)
    plt.title(title or f'Growth curves by condition ({measure})', fontsize=18)

    # legend
    if available_conditions:
        plt.legend(title='Condition', fontsize=12, title_fontsize=12,
                   loc='best', frameon=True)

    # y-limits
    if ylim is not None:
        if isinstance(ylim, (list, tuple)) and len(ylim) == 2:
            plt.ylim(ylim)
        else:
            plt.ylim((0, float(ylim)))

    plt.tight_layout()
    plt.show()

    if save_name:
        # save after show so the on-screen image matches the file
        plt.gcf().savefig(f'{save_name}.pdf', transparent=True)

    # return stats_df  # handy for downstream_
