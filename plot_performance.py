"""
Created on Tue Aug 24 2021
Author: Maxwell Weil
"""

# Importing everything
import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind
from textwrap import wrap


# Arguments to parse for when submitting script
parser = argparse.ArgumentParser()
parser.add_argument("--dataframe_dir", type=str, default="/Users/maxwell/Desktop/dataframes/",
                    help="Directory for dataframes")
parser.add_argument("--save_dir", type=str, default="/Users/maxwell/Desktop/", help="Directory to save data to")
parser.add_argument("--net_names", type=str, nargs="+", required=True,
                    help="Network architecture name to load (ex. vgg11)")
parser.add_argument("--datasets", type=str, nargs="+", required=True, help="Dataset name to load (ex. cifar10)")
parser.add_argument("--epoch_num", type=str, default='150', help="Epoch number to load (ex. 150)")
parser.add_argument("--save_figs", dest='save_figs', action='store_true', help="Include argument to save figures")
parser.add_argument("--save_dfs", dest='save_dfs', action='store_true', help="Include argument to save dataframes")


# Primary functions to run through
def main(dataframe_dir, net_names, datasets, epoch_num, save_dir, save_figs, save_dfs):

    # Loading in all data into dictionary of dataframes
    dataframe_dict = load_data(dataframe_dir, net_names, datasets, epoch_num)

    # Removing bugged data from dataframes and renaming
    dataframe_dict = remove_bug_data(dataframe_dict)
    dataframe_dict = rename_configs(dataframe_dict)

    # Calculating statistics on dataframes and saving to dictionary of dictionaries
    all_stats_dict = create_stats_dict(dataframe_dict)

    # Creating dataframes for network statistics
    network_stats_df_dict = create_stats_df(all_stats_dict, dataframe_dict)

    # Saving dataframes to save_dir
    save_all(network_stats_df_dict, save_dir, save_dfs)

    # Plotting performance and saving to save_dir
    plot_performance(network_stats_df_dict, save_dir, save_figs)


# Function for loading in dataframes
def load_data(dataframe_dir, net_names, datasets, epoch_num):

    # Initializing dictionary for storing dataframes
    dataframe_dict = {}

    # Looping over all networks and datasets and adding dataframe to dictionary
    for network in net_names:
        for dataset in datasets:
            dataframe_dict[network, dataset] = pd.read_csv(
                os.path.join(dataframe_dir, f'{network}_{dataset}_epoch{epoch_num}_perfdf.csv'), index_col=0)

    # Returning dictionary
    return dataframe_dict


# Function for removing bugged data from dataframes
def remove_bug_data(dataframe_dict):

    # Looping over all network dataframes in dictionary
    for net in dataframe_dict:

        # Pulling out current dataframe
        working_df = dataframe_dict[net]

        # Removing any accuracies from df that are excessively low (due to possible bug)
        for config in working_df.columns.to_list():

            # This can be changed to any threshold, currently is 4 stdevs below mean
            threshold = working_df[config].mean() - 4 * working_df[config].std()

            # Checking for data less than threshold
            if any(working_df[config] < threshold):
                bad_idxs = working_df[working_df[config] < threshold].index
                for idx in bad_idxs:
                    print(f'Removed {idx} from {config} in {net[0]} on {net[1]}')

                # Changing bad data to NaN
                working_df.loc[bad_idxs, config] = None

        # Overwriting dataframe in dictionary
        dataframe_dict[net] = working_df

    # Returning dictionary without bad data
    return dataframe_dict


# Function to rename 'tanh' to 'ptanh'
def rename_configs(dataframe_dict):

    # Looping over all network dataframes
    for net in dataframe_dict:

        # Pulling out current dataframe
        working_df = dataframe_dict[net]

        # Creating dictionary of new names
        new_col_names = {}
        for config in working_df.columns:
            if 'tanh' in config:
                new_col_names[config] = config.replace('tanh', 'ptanh')
            else:
                new_col_names[config] = config

        # Renaming dataframe
        working_df = working_df.rename(new_col_names, axis=1)

        # Replacing with renamed dataframe
        dataframe_dict[net] = working_df

    # Returning dataframe dictiontary
    return dataframe_dict

# Function to create statistics dictionary from dataframes
def create_stats_dict(dataframe_dict):

    # Intializing dictionary to save statistic dictionaries in
    all_stats_dict = {}

    # Looping over all network dataframes in dictionary
    for net in dataframe_dict:

        # Pulling out current dataframe
        working_df = dataframe_dict[net]

        # Creating dictionary with average validation accuracy for each configuration
        acc_dict = working_df.mean(axis=0).to_dict()

        # Initializing new dictionary to store additional network information
        network_stats_dict = {}

        # For each configuration, adding the following values:
        # The validation accuracy from sample, the average validation accuracy across all samples,
        # The maximum accuracy of control networks, and the type of network.

        # Looping through all configurations stored in acc_dict
        for config in acc_dict.keys():

            # Checking if configuration matches cross-family mixed network name structure
            if 'swish' in config and 'ptanh' in config:
                max_comp_list = find_max_comp(acc_dict, config, 'CrossFamily')

                # Adding accuracies, average accuracy, max control network accuracy/name, and network type to dictionary
                network_stats_dict[config] = (working_df[config].to_list() + [acc_dict[config]] + max_comp_list)

            # Checking if configuration matches within-swish network name structure
            elif 'swish' in config and '-' in config:
                max_comp_list = find_max_comp(acc_dict, config, 'WithinSwish')
                network_stats_dict[config] = (working_df[config].to_list() + [acc_dict[config]] + max_comp_list)

            # Checking if configuration matches within-tanh network name structure
            elif 'ptanh' in config and '-' in config:
                max_comp_list = find_max_comp(acc_dict, config, 'WithinPTanh')
                network_stats_dict[config] = (working_df[config].to_list() + [acc_dict[config]] + max_comp_list)

            # If no other configuration, then the network is a control
            else:
                network_stats_dict[config] = (working_df[config].to_list() + [acc_dict[config], None, None, 'Control'])

        # Adding single network stats dictionary to larger dictionary
        all_stats_dict[net] = network_stats_dict

    # Returning dictionary with all network stats
    return all_stats_dict


# Function to find the maximum control network accuracy and name
def find_max_comp(acc_dict, config, net_type):

    # Extracting parameter values from configuration name
    param_vals = re.findall('(\d+(?:\.\d+)?)', config)

    # Creating a list of keys corresponding to control networks
    if net_type == 'CrossFamily':
        dict_keys = ['swish' + param_vals[0], 'ptanh' + param_vals[1]]
    elif net_type == 'WithinPTanh':
        dict_keys = ['ptanh' + param_vals[0], 'ptanh' + param_vals[1]]
    else:
        dict_keys = ['swish' + param_vals[0], 'swish' + param_vals[1]]

    # Subsetting dictionary to only look at control networks
    comp_nets = {key: acc_dict[key] for key in dict_keys}

    # Finding the maximum control network accuracy and the name of this control network
    max_comp_name = max(comp_nets, key=comp_nets.get)
    max_comp_val = comp_nets[max_comp_name]

    # Returning name, accuracy, and network type in list
    return [max_comp_name, max_comp_val, net_type]


# Function to create statistics dataframe from stats dictionary
def create_stats_df(all_stats_dict, dataframe_dict):

    # Initializing dictionary to return
    network_stats_df_dict = {}

    # Looping over all networks in dictionary
    for net in all_stats_dict:

        # Pulling out current dataframe
        working_df = dataframe_dict[net]

        # Creating list of sample names
        samples = working_df.index.to_list()

        # Creating new dataframe to include average accuracy, maximum control network accuracy, and network type
        network_stats_df = pd.DataFrame.from_dict(all_stats_dict[net], orient='index',
                                                  columns=samples + ['AvgAcc', 'MaxCompNet', 'MaxCompNetAcc', 'Type'])

        # Adding columns to find the percent difference between each sample and the maximum control network accuracy
        for column in network_stats_df:
            if column.startswith('sample'):
                network_stats_df[column + '_PercentChange'] = 100 * (
                            network_stats_df[column] - network_stats_df.MaxCompNetAcc) / network_stats_df.MaxCompNetAcc

        # Finding the average percent difference and the 95% confidence
        network_stats_df['AvgPercentChange'] = network_stats_df.loc[:, [i + '_PercentChange' for i in samples]].mean(
            axis=1)
        network_stats_df['AvgPercentChange95CI'] = network_stats_df.loc[:, [i + '_PercentChange' for i in samples]].std(
            axis=1) * 1.96 / np.sqrt(len(samples))

        # Sorting to help organize plotting later
        network_stats_df = network_stats_df.sort_values('AvgPercentChange')

        # Initializing p-value dictionary
        pval_dict = {}

        # Iterating over all configurations
        for row in network_stats_df.itertuples():

            try:

                # Conducting two-sided t-test between network of interest and control network accuracies
                t, p = ttest_ind(network_stats_df.loc[row.Index][0:20].dropna().to_list(),
                                 working_df[network_stats_df.loc[row.Index][21]].dropna().to_list())

                # Converting p-value to one-sided t-test results
                if t < 0:
                    p = 1. - p / 2.
                else:
                    p = p / 2.

                # Adding p-value to dictionary
                pval_dict[row.Index] = p

            except KeyError:
                # Printing error if pval cannot be calculated, can be ignored with control configurations
                if row.Type != 'Control':
                    print('Unable to calculate p-value for ' + str(row.Index))

        # Correcting p-values for multiple testing using Benjamini-Hochberg correction
        corr_pvals = multipletests(list(pval_dict.values()), alpha=0.05, method='fdr_bh')[1]

        # Adding corrected p-values to dictionary
        for key in range(len(pval_dict.keys())):
            pval_dict[list(pval_dict.keys())[key]] = [list(pval_dict.values())[key], corr_pvals[key]]

        # Adding column to dataframe with p-values
        pval_df = pd.DataFrame.from_dict(pval_dict, orient='index', columns=['Pval', 'CorrPval'])
        network_stats_df = network_stats_df.join(pval_df)

        # Adding network stats dataframe to dictionary of dataframes
        network_stats_df_dict[net] = network_stats_df

    # Returning dictionary of network stats dataframes
    return network_stats_df_dict


# Function to save out dataframes
def save_all(dataframe_dict, save_dir, save_dfs):
    # Only save if specified
    if save_dfs:
        for name in dataframe_dict:
            print('Saving dataframe to ' + os.path.join(save_dir, f'{name[0]}_{name[1]}_statsdf.csv'))
            dataframe_dict[name].to_csv(os.path.join(save_dir, f'{name[0]}_{name[1]}_statsdf.csv'))


# Function to create performance plot for Cross Family networks
def plot_performance(network_stats_df_dict, save_dir, save_figs):

    # Looping over all networks in dictionary
    for net in network_stats_df_dict:

        # Pulling out current dataframe
        working_df = network_stats_df_dict[net]

        # Only looking at mixed (Cross Family) networks
        mixed_only_df = working_df[working_df['Type'] == 'CrossFamily']

        # Setting up figure
        fig_height = 12
        fig_width = 10
        fig = plt.figure(figsize=(fig_width, fig_height))
        fig.set_facecolor('w')

        # Plotting vertical line at x=0
        plt.axvline(0, color='k', linestyle='--', alpha=0.5)

        # Plotting all average changes in performance for mixed network
        plt.scatter(mixed_only_df.AvgPercentChange, mixed_only_df.index, marker='o', s=2 * fig_height, color=[[0.5, 0.0, 1.0, 1.0]])

        # Plotting 95% CI for mixed networks
        plt.hlines(mixed_only_df.index, mixed_only_df.AvgPercentChange - mixed_only_df.AvgPercentChange95CI,
                   mixed_only_df.AvgPercentChange + mixed_only_df.AvgPercentChange95CI, colors=[[0.5, 0.0, 1.0, 1.0]],
                   alpha=0.5, linewidths=0.5 * fig_height)

        # Marking significant corrected p-values
        plt.scatter(np.repeat(10, len(mixed_only_df[mixed_only_df.CorrPval < 0.05])),
                    mixed_only_df[mixed_only_df.CorrPval < 0.05].index, marker='*', color='k')

        # Formatting plot with shaded bars
        for i in range(len(mixed_only_df.index)):
            if i % 2 == 0:
                plt.axhline(mixed_only_df.index[i], linewidth=1.25 * fig_height, alpha=0.1, color='grey')

        # Adding labels, setting margins, etc.
        plt.margins(y=0.01)
        plt.xlim([-11, 11])
        plt.tick_params(labeltop=True, top=True, labelsize=12)
        plt.title(
            '\n'.join(wrap(net[0].capitalize() + ' Cross-Family Mixed Network Performance on ' + net[1].upper(), 30)),
            fontsize=26)
        plt.ylabel('Network Architecture', fontsize=20)
        plt.xlabel('Validation Accuracy Relative to Control Network', fontsize=20)
        plt.tight_layout()

        # Only save if specified
        if save_figs:
            print('Saving figure to ' + os.path.join(save_dir, f'{net[0]}_{net[1]}_perf.png'))
            plt.savefig(os.path.join(save_dir, f'{net[0]}_{net[1]}_perf.png'), bbox_inches='tight')


# Run only if this script is the main script
if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))
