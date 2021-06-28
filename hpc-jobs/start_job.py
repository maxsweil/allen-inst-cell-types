#!/allen/programs/braintv/workgroups/nc-ophys/max.weil/miniconda3/envs/MaxEnv/bin/ python
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:01:34 2020

@author: briardoty
"""
import argparse
import json
import sys
from itertools import chain
import subprocess

from simple_slurm import Slurm

# Setting paths for python environment to use and where to save output/error logs
python_executable = "/allen/programs/braintv/workgroups/nc-ophys/max.weil/miniconda3/envs/MaxEnv/bin/python"
job_dir = "/allen/programs/braintv/workgroups/nc-ophys/max.weil/slurm_logs/"

# args
parser = argparse.ArgumentParser()
parser.add_argument("--job_title", type=str, help="Set value for job_title")
parser.add_argument("--case", default=None, type=str)
parser.add_argument("--schemes", default=[], nargs="+", type=str)
parser.add_argument("--net_name", default=None, type=str)
parser.add_argument("--cases", default=[], nargs="+", type=str)


def main(job_title, cases, case, net_name, schemes):
    # script, run_params and job_settings
    with open("job_params.json", "r") as json_file:
        job_params = json.load(json_file)

    job_params = job_params[job_title]
    script = job_params["script"]
    run_params = job_params["run_params"]
    job_settings = job_params["job_settings"]

    if len(cases) > 0:
        run_params["cases"] = param_arr_helper(cases)

    if len(schemes) > 0:
        run_params["schemes"] = param_arr_helper(schemes)

    if case is not None:
        run_params["case"] = case

    if net_name is not None:
        run_params["net_name"] = net_name

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
    slurm.sbatch(python_executable + ' ' + script + ' ' + params_string + ' ' + Slurm.SLURM_ARRAY_TASK_ID)
    # Executing slurm job using python path specified above and script/params from job_params.json


def param_arr_helper(param_arr):
    if param_arr is None or len(param_arr) == 0:
        return None

    return " ".join(str(p) for p in param_arr)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))