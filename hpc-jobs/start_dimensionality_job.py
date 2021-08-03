#!/allen/programs/braintv/workgroups/nc-ophys/max.weil/miniconda3/envs/MaxEnv/bin/ python
# -*- coding: utf-8 -*-
import argparse
import json
import sys
from itertools import chain
import subprocess

from simple_slurm import Slurm

# Setting paths for python environment to use and where to save output/error logs
python_executable = "/allen/programs/braintv/workgroups/nc-ophys/max.weil/miniconda3/envs/MaxEnv/bin/python"
job_dir = "/allen/programs/braintv/workgroups/nc-ophys/max.weil/slurm_logs/"

# Arguments to pass with job submission
parser = argparse.ArgumentParser()
parser.add_argument("--net_name", default='sticknet8', type=str, help="Name of network (ex. vgg11, sticknet8)")
parser.add_argument("--dataset", default='cifar10', type=str, help="Name of dataset (ex. cifar10, cifar100)")
parser.add_argument("--config_groups", type=str, nargs="+", required=True,
                    help="Specific groups to analyze (ex. within-swish, control)")


def main(net_name, dataset, config_groups):
    # Open job_params.json file
    with open("job_params.json", "r") as json_file:
        job_params = json.load(json_file)

    # Pull out various parameters for chosen job
    job_title = 'create_dimensionality_dicts'
    job_params = job_params[job_title]
    script = job_params["script"]
    run_params = job_params["run_params"]
    job_settings = job_params["job_settings"]

    # Updating parameters for running chosen job
    run_params["net_name"] = net_name
    run_params["dataset"] = dataset

    with open(
            "/Users/maxwell/Desktop/merger/allen-inst-cell-types/hpc-jobs/net_configs.json",
            "r") as json_file:
        net_configs = json.load(json_file)

    for group in config_groups:
        cases = set(net_configs[group].keys())
        for case in cases:
            run_params["case"] = case

            # prepare args
            params_list = list(chain.from_iterable((f"--{k}", str(run_params[k])) for k in run_params))
            params_string = " ".join(params_list)

            # Kicking off HPC job
            slurm = Slurm(
                job_name=job_title,  # Setting name for job
                output=job_dir + "output.out",  # Setting destination for output log
                error=job_dir + "error.err",  # Setting destination for error log
                **job_settings  # Importing all other jobs settings from job_params.json
            )
            # Executing slurm job using python path specified above and script/params from job_params.json
            slurm.sbatch(python_executable + ' ' + script + ' ' + params_string + ' ' + Slurm.SLURM_ARRAY_TASK_ID)


if __name__ == "__main__":
    # Parsing arguments provided and printing them back
    args = parser.parse_args()
    print(args)

    # Submitting arguments to main to run Slurm job
    main(**vars(args))
