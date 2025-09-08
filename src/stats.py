
# move to a separate stats file later
def perform_anova_and_tukey(df, group_column, value_column):

    """
    Perform one-way ANOVA and Tukey's HSD test on the given dataframe.
    Parameters:
    -----------
    df : DataFrame
        The input dataframe containing the data for analysis.
    group_column : str  
        The name of the column containing group labels.
    value_column : str
        The name of the column containing the values to analyze.
    Returns:        
    --------
    dict
        A dictionary containing ANOVA results and Tukey's HSD results (if applicable).
        If Tukey's test is not performed (p >= 0.05), its value will be None.
    """

    # Get groups in the specified order
    ordered_groups = df[group_column].cat.categories
    
    # Prepare data for ANOVA, respecting the order
    groups = [df[df[group_column] == group][value_column] for group in ordered_groups]

    # Perform one-way ANOVA
    f_value, p_value = stats.f_oneway(*groups)
    
    # Calculate effect size (Eta-squared)
    all_data = df[value_column]
    # Ensure all values in groups are numeric and drop NaNs

    ss_between = sum(len(group) * (np.mean(group) - np.mean(all_data))**2 for group in groups)
    ss_total = sum((x - np.mean(all_data))**2 for group in groups for x in group)
    eta_squared = ss_between / ss_total
    
    results = {
        'ANOVA': {
            'F-value': f_value,
            'p-value': p_value,
            'eta_squared': eta_squared
        },
        'Tukey_HSD': None
    }
    
    # If p-value is significant, perform post-hoc Tukey's HSD test
    if p_value < 0.05:
        # Prepare data for Tukey's test
        data = df[value_column]
        groups = df[group_column]
        
        # Perform Tukey's test
        tukey_results = pairwise_tukeyhsd(data, groups)
        
        # Format Tukey results, respecting the original order
        tukey_dict = {}
        for group1 in ordered_groups:
            for group2 in ordered_groups:
                if group1 != group2:
                    # Find the corresponding row in tukey_results
                    row = next((r for r in tukey_results._results_table.data[1:] 
                                if set([r[0], r[1]]) == set([group1, group2])), None)
                    if row:
                        _, _, meandiff, p_adj, lower, upper, reject = row
                        tukey_dict[(group1, group2)] = {
                            'mean_difference': meandiff,
                            'lower_ci': lower,
                            'upper_ci': upper,
                            'reject_null': reject,
                            'p_value': p_adj
                        }
        
        results['Tukey_HSD'] = tukey_dict
    
    return results
