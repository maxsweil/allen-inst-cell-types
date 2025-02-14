#!/allen/programs/braintv/workgroups/nc-ophys/max.weil/miniconda3/envs/MaxEnv/bin/ python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 15:21:08 2020

@author: briardoty
"""
import argparse
import json
import sys
from itertools import chain

from simple_slurm import Slurm

# Setting paths for python environment to use and where to save output/error logs
python_executable = "/allen/programs/braintv/workgroups/nc-ophys/max.weil/miniconda3/envs/MaxEnv/bin/python"
job_dir = "/allen/programs/braintv/workgroups/nc-ophys/max.weil/slurm_logs/"

# args
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Set dataset")
parser.add_argument("--net_names", type=str, nargs="+", required=True, help="Set net_names")
parser.add_argument("--schemes", type=str, nargs="+", required=True, help="Set schemes")
parser.add_argument("--config_groups", type=str, nargs="+", required=True, help="Set config_groups")
parser.add_argument("--find_lr_avg", dest="find_lr_avg", action="store_true")
parser.set_defaults(find_lr_avg=False)


def main(dataset, net_names, schemes, config_groups, find_lr_avg):
    job_title = "gen_nets"

    # script, run_params and job_settings
    with open("job_params.json", "r") as json_file:
        job_params = json.load(json_file)

    with open("net_configs.json", "r") as json_file:
        net_configs = json.load(json_file)

    job_params = job_params[job_title]
    script = job_params["script"]
    run_params = job_params["run_params"]
    run_params["dataset"] = dataset
    job_settings = job_params["job_settings"]

    # kick off job for each net configuration
    for net_name in net_names:

        for scheme in schemes:

            for group in config_groups:

                # get net configs in group
                configs = net_configs[group]

                for case in configs.keys():

                    # get net config details
                    config = configs[case]

                    # update params for this net config
                    run_params["net_name"] = net_name
                    run_params["scheme"] = scheme
                    run_params["group"] = group
                    run_params["case"] = case

                    if config.get("conv_layers") is not None:
                        run_params["conv_layers"] = config["conv_layers"]

                    if config.get("fc_layers") is not None:
                        run_params["fc_layers"] = config["fc_layers"]

                    if config.get("default_fn") is not None:
                        run_params["default_fn"] = config["default_fn"]

                    act_fns = config["act_fns"]
                    run_params["act_fns"] = param_arr_helper(act_fns)

                    if config.get("act_fn_params") is not None:
                        run_params["act_fn_params"] = param_arr_helper(config.get("act_fn_params"))

                    # default repeating each fn once
                    if config.get("n_repeat") is None:
                        n_repeat = [1] * len(act_fns)
                    else:
                        n_repeat = config.get("n_repeat")
                    run_params["n_repeat"] = param_arr_helper(n_repeat)

                    # prepare args
                    params_list = list(chain.from_iterable((f"--{k}", str(run_params[k])) for k in run_params))
                    pretrained = config.get("pretrained")
                    if pretrained:
                        params_list.append("--pretrained")

                    spatial = config.get("spatial")
                    if spatial:
                        params_list.append("--spatial")

                    cfg_find_lr_avg = config.get("find_lr_avg")
                    if cfg_find_lr_avg or find_lr_avg:
                        params_list.append("--find_lr_avg")

                    params_string = " ".join(params_list)

                    # Kicking off HPC job
                    slurm = Slurm(
                        job_name = job_title + f"c-{case}",  # Setting name for job
                        output=job_dir + "output.out",  # Setting destination for output log
                        error=job_dir + "error.err",  # Setting destination for error log
                        **job_settings  # Importing all other jobs settings from job_params.json
                    )
                    # Executing slurm job using python path specified above and script/params from job_params.json
                    slurm.sbatch(
                        python_executable + ' ' + script + ' ' + params_string + ' ' + Slurm.SLURM_ARRAY_TASK_ID)


def param_arr_helper(param_arr):
    if param_arr is None or len(param_arr) == 0:
        return None

    return " ".join(str(p) for p in param_arr)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))