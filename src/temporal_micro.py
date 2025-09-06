from root import *

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



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
