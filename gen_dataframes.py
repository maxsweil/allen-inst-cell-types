#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import torch
import os
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--net_name", type=str, required=True, help="Name of network (ex. vgg11, sticknet8)")
parser.add_argument("--dataset", type=str, required=True, help="Name of dataset (ex. cifar10, cifar100)")
parser.add_argument("--data_dir", default="/allen/programs/braintv/workgroups/nc-ophys/max.weil/data/data/nets/",
                    type=str, help="Directory to pull data from")
parser.add_argument("--save_dir", default="/allen/programs/braintv/workgroups/nc-ophys/max.weil/data/data/dataframes/",
                    type=str, help="Directory to save dataframe")
parser.add_argument("--epoch_num", default='150', type=str, help="Epoch number to draw from (typically the last epoch)")


def main(net_name, dataset, data_dir, save_dir, epoch_num):
    # Setting up paths to access data and save dataframe
    root = os.path.join(data_dir, dataset, net_name, 'adam/')
    save_path = save_dir

    # Establishing portions of filenames to search for
    start = net_name + '_case-'
    end = f'_epoch-{epoch_num}.pt'

    # Initializing dictionary of filepaths with cases and samples as values
    filepaths = dict()

    # Adding filepath of snapshot, case, and sample names to dictionary
    for roots, dirs, files in os.walk(root):
        for file in files:
            if end in file:
                name = remove_prefix(file, start).split('_')
                filepaths[os.path.join(roots, file)] = [name[0], name[1]]

    # Finding all unique cases and samples
    cases = set([item[0] for item in list(filepaths.values())])
    samples = set([item[1] for item in list(filepaths.values())])

    # Intializing dataframe to fill with performance data
    df = pd.DataFrame(columns=sorted(cases), index=sorted(samples))

    # Opening each snapshot and filling dataframe with validation accuracy metric
    for filepath in filepaths:
        df[filepaths[filepath][0]][filepaths[filepath][1]] = torch.load(filepath)['val_acc']

    # Saving dataframe to previously designated directory
    df.to_csv(os.path.join(save_path, f'{net_name}_{dataset}_perfdf.csv'))


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))