def plot_primary(ax, roots, labels, filename):
    # Plot the data on the given subplot ax
    left_root = roots[filename][0]
    right_root =roots[filename][1]
    rsa = left_root
    branch = rsa.primary
    tips = branch.tips
    lengths = [tip.length for tip in tips]
    times = [tip.time for tip in tips]
    ax.plot(times, lengths, color='palevioletred', alpha=1, label='left')

    rsa = right_root
    branch = rsa.primary
    tips = branch.tips
    lengths = [tip.length for tip in tips]
    times = [tip.time for tip in tips]
    ax.plot(times, lengths, color='slateblue', alpha=1, label='right')
    ax.set_title(f"(setup:{labels[filename]}-{filename[:3]})")
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Primary Length (mm)")
    ax.set_ylim([0, 95])
    ax.legend()


def visualize_cutoff(roots, max = 95):
    last_depths = []
    for root in roots.values():
        left_root, right_root = root
        last_depths.append(left_root.primary.tips[-1].y)
        last_depths.append(right_root.primary.tips[-1].y)

    plt.scatter(x = range(len(last_depths)), y = last_depths)
    plt.hlines(y = max, xmin =0, xmax = len(last_depths), ls = 'dashed', colors = ['grey'])
    plt.title('vertical cut-off')
    plt.show()


def fit_linear_function(x, y):
    # Fit a linear function to the data using np.polyfit()
    x = np.array(x)
    y = np.array(y)
    coefficients = np.polyfit(x, y, 1)
    slope = coefficients[0]
    intercept = coefficients[1]
    
    # Generate points for the fitted line
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_fit + intercept
    
    return x_fit, y_fit, slope, intercept

def primary_fit(rsa, max = 95):
    branch = rsa.primary
    tips = branch.tips
    times = [tip.time for tip in tips]
    lengths = [tip.length for tip in tips]
    depths = [tip.y for tip in tips]
    last_idx = next((i for i, num in enumerate(depths) if num > max), -1)
    lengths_lin = lengths[:last_idx]
    times_lin = times[:last_idx]
    time_fit, length_fit, slope, intercept = fit_linear_function(times_lin, lengths_lin)
    return times, lengths, time_fit, length_fit, slope, intercept


def plot_primary_fit(ax, roots, filename):
    # Plot the data on the given subplot ax
    left_root = roots[filename][0]
    right_root =roots[filename][1]
    times, lengths, time_fit, length_fit, slope_l, _ = primary_fit(left_root)
    # Plot the data and the fitted linear function
    ax.scatter(times, lengths, s = 10, alpha= .5, label='Left', color = 'palevioletred')
    ax.plot(time_fit, length_fit,  '-', label='Left Fit', color = 'crimson', lw = 2)

    times, lengths, time_fit, length_fit, slope_r, _ = primary_fit(right_root)
    ax.scatter(times, lengths, s = 20, alpha= .35, label='Right', color = 'slateblue')
    ax.plot(time_fit, length_fit, '-', label='Right Fit', color = 'mediumblue', lw = 2)
    ax.set_xlabel('Time (hr)')
    ax.set_ylabel('Length (mm)')
    ax.set_ylim([0, 95])
    ax.legend()
    return slope_l, slope_r


## Above are not used yet.
## Below are for fitting lateral roots.



import os, shutil
from pathlib import Path
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.format_output import (
    make_label_df, 
)

from src.temporal_macro import get_lateral_stage 
from scipy.optimize import curve_fit


cmap = plt.get_cmap('magma')
colors = cmap(np.linspace(0, 1, 4))[1:]

def plot_lateral_by_time(left_root, right_root, filename, label_df, out_figure_dir):
    fig, axes = plt.subplots(ncols = 2, figsize = (12, 5))
    rsa = left_root
    if rsa.laterals:   
        stages = get_lateral_stage(rsa)
        for i, branch in enumerate(rsa.laterals):
            tips = branch.tips
            lengths = [tip.length for tip in tips]
            times = [tip.time for tip in tips]
            axes[0].plot(times, lengths, color= colors[stages[i]], alpha=.8, lw = 2)
            
    axes[0].set_title("Left")
    axes[0].set_xlabel("Time (hr)")
    axes[0].set_ylabel("Lateral Length (mm)")
    axes[0].set_ylim([0, 40])

    rsa = right_root
    if rsa.laterals:   
        stages = get_lateral_stage(rsa)
        for i, branch in enumerate(rsa.laterals):
            tips = branch.tips
            lengths = [tip.length for tip in tips]
            times = [tip.time for tip in tips]
            axes[1].plot(times, lengths, color= colors[stages[i]], alpha=.8, lw = 2)
            
    axes[1].set_title("Right")
    axes[1].set_xlabel("Time (hr)")
    axes[1].set_ylim([0, 40])
    
    condition = label_df.loc[filename]['condition']

    plt.suptitle(f"Lateral Lengths by Time ({filename[:-5]}, setup:{condition})", fontsize = 16)
    plt.show()

    fig.savefig(f'{out_figure_dir}/samples/{condition}-{filename[:-5]}_lat_lengths_by_time.pdf')
    plt.close(fig)  


def plot_lateral_by_age(left_root, right_root, filename, label_df, out_figure_dir):

    fig, axes = plt.subplots(ncols = 2, figsize = (12, 5))
    rsa = left_root
    if rsa.laterals:   
        stages = get_lateral_stage(rsa)
        for i, branch in enumerate(rsa.laterals):
            tips = branch.tips
            lengths = [tip.length for tip in tips]
            times = [tip.time - branch.appiration for tip in tips]
            axes[0].plot(times, lengths, color= colors[stages[i]], alpha=.8, lw = 2)
            
    axes[0].set_title("Left")
    axes[0].set_xlabel("Time (hr)")
    axes[0].set_ylabel("Age Length (mm)")
    axes[0].set_ylim([0, 40])

    rsa = right_root
    if rsa.laterals:   
        stages = get_lateral_stage(rsa)
        for i, branch in enumerate(rsa.laterals):
            tips = branch.tips
            lengths = [tip.length for tip in tips]
            times = [tip.time - branch.appiration for tip in tips]
            axes[1].plot(times, lengths, color= colors[stages[i]], alpha=.8, lw = 2)
        
    axes[1].set_title("Right")
    axes[1].set_xlabel("Age (hr)")
    axes[1].set_ylim([0, 40])
    
    condition = label_df.loc[filename]['condition']

    plt.suptitle(f"Lateral Lengths by Age ({filename[:-5]}, setup:{condition})", fontsize = 16)
    fig.savefig(f'{out_figure_dir}/samples/{condition}-{filename[:-5]}_lat_lengths_by_age.pdf')
    plt.show()
    plt.close(fig)  

def sort_lists(times, lengths):
    # Combine the two lists into a list of tuples
    combined = list(zip(times, lengths))
    
    # Sort the combined list based on the values in the first list
    sorted_combined = sorted(combined, key=lambda x: x[0])
    
    # Unzip the sorted combined list back into two separate lists
    sorted_times, sorted_lengths = zip(*sorted_combined)
    
    return list(sorted_times), list(sorted_lengths)

# Helper func
def get_sublist_from_last_zero(lst):
    # Find the index of the last occurrence of zero in the list
    last_zero_index = -1
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] == 0:
            last_zero_index = i
            break
    
    # If no zero is found, return the entire list
    if last_zero_index == -1:
        return lst, 0
    
    # Return the sublist starting from the last zero
    return lst[last_zero_index:], last_zero_index

def gompertz_model_ti(x, A, Ti, kg):
    '''
    Gompertz model with Ti as a parameter
    '''
    return A * np.exp(-np.exp(-kg * (x- Ti)))

def gompertz_model_ti_derivative_num(x, A, Ti, kg, h=1e-6):
    '''
    Numerical derivative of Gompertz model with Ti
    '''
    return (gompertz_model_ti(x + h, A, Ti, kg) - gompertz_model_ti(x, A, Ti, kg)) / h

def gompertz_model_ti_derivative_ana(x, A, Ti, kg):
    '''
    Analytical derivative of Gompertz model with Ti
    '''
    return A * kg * np.exp(-kg * (x - Ti) - np.exp(-kg * (x - Ti)))

def fit_lateral_model(branch, plotting = False, rsa_path_name = None):
    '''
    Fit Gompertz model to lateral growth data

    Args:
    branch: root branch object
    plotting: boolean, if True, plot the model fit
    rsa_path_name: tuple, (setup, root_name)
    
    Returns:
    latency: float, time to start growth
    mean_growth: float, mean growth rate
    A: float, asymptote
    Ti: float, inflection point
    kg: float, growth rate
    max_growth: float, max growth rate
    '''
    tips = branch.tips
    lengths = [tip.length for tip in tips]
    times = [tip.time - branch.appiration for tip in tips]
    if len(lengths) > 4:
        
        sorted_times, sorted_lengths = sort_lists(times, lengths) 

        trimmed_lengths, last_zero_index = get_sublist_from_last_zero(sorted_lengths)
        trimmed_times = sorted_times[last_zero_index:]

        # get latency
        latency = sorted_times[last_zero_index]
        trimmed_times_shifted = [time - trimmed_times[0] for time in trimmed_times]

        # set guessed parameters to help model fit: 
        A_guess = max(lengths) # asymptote 
        Ti_guess = (trimmed_times_shifted[-1] + trimmed_times_shifted[0])/2 # inflection point guess, mid growth point
        # get mean growth
        mean_growth = (trimmed_lengths[-1]-trimmed_lengths[0])/(trimmed_times_shifted[-1] + trimmed_times_shifted[0])
        kg_guess = mean_growth/A_guess*np.e # use mean growth as a guess for max

        initial_guess = [A_guess, Ti_guess, kg_guess]

        # fit model with MSE:
        try:
            params, _ = curve_fit(gompertz_model_ti,  
                                    trimmed_times_shifted, 
                                    trimmed_lengths,
                                    p0 = initial_guess,
                                    maxfev=10000)
            
            # extract the fitted parameters
            A, Ti, kg = params

            # get max abs. growth:
            max_growth = A * kg/np.e

            if plotting == True:
                times_smooth_trim = np.linspace(trimmed_times_shifted[0], trimmed_times_shifted[-1], 100)
                lengths_smooth_trim = gompertz_model_ti(times_smooth_trim, A, Ti, kg)
                fig, ax = plt.subplots(figsize = (6, 4))
                ax.scatter(times, lengths)
                ax.plot(np.array(times_smooth_trim) + trimmed_times[0], lengths_smooth_trim, label = f'y = {A:.2f}*exp(-exp(-{kg:.2f}*(t - {Ti:.2f}) \nMax rate: {max_growth:.2f}')
                ax.set_title('Model fit diagnose')
                ax.set_xlabel('root age(hr)')
                ax.set_ylabel('root length (mm)')
                ax.legend()
                fig.savefig(f'{rsa_path_name[0]}/lr_{rsa_path_name[1]}_fit.pdf')
                # plt.show()
                plt.close(fig)  
            return latency, mean_growth, A, Ti, kg, max_growth
        
        except RuntimeError:
            print(f'failed to fit root {rsa_path_name[1]}\n')
            
            return latency, mean_growth, np.nan, np.nan, np.nan, np.nan

    else:
        print('root too short! \n')
        return np.nan

def produce_outputs(rsa, subdirectory, side = 'L'):

    if rsa.laterals:   

        lateral_stats = np.zeros((len(rsa.laterals), 6)) 

        for lr_idx, branch in enumerate(rsa.laterals):
            print(f'add data for branch {lr_idx}.. \n')
            lateral_stats[lr_idx] = fit_lateral_model(branch, plotting = True, rsa_path_name = [subdirectory + f'/{side}', lr_idx])

        lateral_stats_df = pd.DataFrame(lateral_stats, columns = ['hard_latency', 'mean_growth' ,'A', 'Ti', 'kg', 'max_growth'])

        appirations = [branch.appiration for branch in rsa.laterals]
        depths = [branch.depth for branch in rsa.laterals]

        lateral_stats_df['appiration'] = appirations
        lateral_stats_df['depth'] = depths
        
        stages = get_lateral_stage(rsa)
        lateral_stats_df['stage'] = stages

    # Return the empty DataFrame if there are no laterals:
    else:
        lateral_stats_df = pd.DataFrame(columns = ['hard_latency', 'mean_growth' ,'A', 'Ti', 'kg', 'max_growth', 'appiration', 'depth', 'stage'])
        print('no laterals in this root! \n')
    return lateral_stats_df


def plot_lateral_modeled(left_root, right_root, filename, left_stats_df, right_stats_df, label_df, out_figure_dir):
    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (12,10))

    if left_root.laterals:    
        for i, branch in enumerate(left_root.laterals):
            tips = branch.tips
            lengths = [tip.length for tip in tips]
            times = [tip.time for tip in tips]
            axes[0][0].plot(times, lengths, color= colors[left_stats_df.iloc[i]['stage']], alpha=.8, lw = 2)

        for i, branch in enumerate(left_root.laterals):
            times_smooth_trim = np.linspace(0, branch.tips[-1].time- left_stats_df.iloc[i]['appiration']- left_stats_df.iloc[i]['hard_latency'], 100)
            lengths_smooth_trim = gompertz_model_ti(times_smooth_trim, left_stats_df.iloc[i]['A'], left_stats_df.iloc[i]['Ti'], left_stats_df.iloc[i]['kg'])
            axes[1][0].plot(times_smooth_trim +  left_stats_df.iloc[i]['appiration'] + left_stats_df.iloc[i]['hard_latency'], lengths_smooth_trim, color= colors[left_stats_df.iloc[i]['stage']], alpha=.8, lw = 2)
    
    axes[0][0].set_title(f"Left Length by Time")
    axes[0][0].set_xlabel("Time (hr)")
    axes[0][0].set_ylabel("Lateral Length (mm)")
    axes[0][0].set_ylim([0, 40])

     
    axes[1][0].set_title(f"Left Length by Time (model)")
    axes[1][0].set_xlabel("Time (hr)")
    axes[1][0].set_ylabel("Lateral Length (mm)")
    axes[1][0].set_ylim([0, 40])

    if right_root.laterals:    

        for i, branch in enumerate(right_root.laterals):
            tips = branch.tips
            lengths = [tip.length for tip in tips]
            times = [tip.time for tip in tips]
            axes[0][1].plot(times, lengths, color= colors[right_stats_df.iloc[i]['stage']], alpha=.8, lw = 2)

        for i, branch in enumerate(right_root.laterals):
            times_smooth_trim = np.linspace(0, branch.tips[-1].time- right_stats_df.iloc[i]['appiration']- right_stats_df.iloc[i]['hard_latency'], 100)
            lengths_smooth_trim = gompertz_model_ti(times_smooth_trim, right_stats_df.iloc[i]['A'], right_stats_df.iloc[i]['Ti'], right_stats_df.iloc[i]['kg'])
            axes[1][1].plot(times_smooth_trim +  right_stats_df.iloc[i]['appiration'] + right_stats_df.iloc[i]['hard_latency'], lengths_smooth_trim, color= colors[right_stats_df.iloc[i]['stage']], alpha=.8, lw = 2)
        

    axes[0][1].set_title(f"Right Length by Time")
    axes[0][1].set_xlabel("Time (hr)")
    axes[0][1].set_ylabel("Lateral Length (mm)")
    axes[0][1].set_ylim([0, 40])

 
    axes[1][1].set_title(f"Right Length by Time (model)")
    axes[1][1].set_xlabel("Time (hr)")
    axes[1][1].set_ylabel("Lateral Length (mm)")
    axes[1][1].set_ylim([0, 40])

    condition = label_df.loc[filename]['condition']

    plt.suptitle(f'growth v.s. modeled ({filename[:-5]}, setup:{condition})', fontsize = 18)

    fig.savefig(f'{out_figure_dir}/samples/{condition}-{filename[:-5]}_lat_growth_n_model.pdf')

    plt.show()
    plt.close(fig)  
