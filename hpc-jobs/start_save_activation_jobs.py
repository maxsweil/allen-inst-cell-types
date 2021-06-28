#!/allen/programs/braintv/workgroups/nc-ophys/max.weil/miniconda3/envs/MaxEnv/bin/ python
# -*- coding: utf-8 -*-

import argparse
import json
import sys
import re
import os
from itertools import chain

from simple_slurm import Slurm

# Setting paths for python environment to use and where to save output/error logs
python_executable = "/allen/programs/braintv/workgroups/nc-ophys/max.weil/miniconda3/envs/MaxEnv/bin/python"
job_dir = "/allen/programs/braintv/workgroups/nc-ophys/max.weil/slurm_logs/"

# args
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Set dataset")
parser.add_argument("--net_name", type=str, required=True, help="Set net_name")
parser.add_argument("--scheme", type=str, help="Set scheme", required=True)
parser.add_argument("--cases", default=[], nargs="+", type=str)
parser.add_argument("--final", dest="final", action="store_true")
parser.set_defaults(final=True)


def main(net_name, cases, scheme, dataset, final):
    job_title = "save_net_activations"

    # script, run_params and job_settings
    with open("job_params.json", "r") as json_file:
        job_params = json.load(json_file)

    with open("net_configs.json", "r") as json_file:
        net_configs = json.load(json_file)

    job_params = job_params[job_title]
    script = job_params["script"]

    # update any run params
    run_params = job_params["run_params"]
    run_params["net_name"] = net_name
    run_params["dataset"] = dataset

    job_settings = job_params["job_settings"]

    # set to avoid submitting jobs for the same net twice
    net_filepaths = set()

    # walk dir looking for nets to train
    net_dir = os.path.join(run_params["data_dir"], f"nets/{dataset}/{net_name}/{scheme}")
    for root, dirs, files in os.walk(net_dir):

        # only interested in locations files (nets) are saved
        if len(files) <= 0:
            continue

        slugs = root.split("/")

        # only interested in the given dataset
        if not dataset in slugs:
            continue

        # only interested in the given training scheme
        if not scheme in slugs:
            continue

        # only interested in the given cases (unless [])
        if len(cases) > 0 and not any(c in slugs for c in cases):
            continue

        # start from first or last epoch
        if final:
            net_filename = get_last_epoch(files)
            print(f"Submitting job to save activation output of {net_filename}.")
        else:
            net_filename = get_first_epoch(files)
            print(f"Job will save activations of initial snapshot {net_filename}.")

        # and add it to the training job set
        net_filepath = os.path.join(root, net_filename)
        net_filepaths.add(net_filepath)

    # loop over set, submitting jobs
    for net_filepath in net_filepaths:
        # update param
        run_params["net_filepath"] = net_filepath

        # prepare args
        params_list = list(chain.from_iterable((f"--{k}", str(run_params[k])) for k in run_params))
        params_string = " ".join(params_list)

        # Kicking off HPC job
        slurm = Slurm(
            job_name=f" {net_filepath}",  # Setting name for job
            output=job_dir + "output.out",  # Setting destination for output log
            error=job_dir + "error.err",  # Setting destination for error log
            **job_settings  # Importing all other jobs settings from job_params.json
        )
        # Executing slurm job using python path specified above and script/params from job_params.json
        slurm.sbatch(python_executable + ' ' + script + ' ' + params_string + ' ' + Slurm.SLURM_ARRAY_TASK_ID)


def get_epoch_from_filename(filename):
    epoch = re.search(r"\d+\.pt$", filename)
    epoch = int(epoch.group().split(".")[0]) if epoch else None

    return epoch


def get_first_epoch(net_filenames):
    for filename in net_filenames:

        epoch = get_epoch_from_filename(filename)
        if epoch == 0:
            return filename


def get_last_epoch(net_filenames):
    max_epoch = -1
    last_net_filename = None

    for filename in net_filenames:

        epoch = get_epoch_from_filename(filename)

        if epoch is None:
            continue

        if epoch > max_epoch:
            max_epoch = epoch
            last_net_filename = filename

    return last_net_filename


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))