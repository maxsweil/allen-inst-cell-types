from modules.PredictionVisualizer import PredictionVisualizer
import os
import json

data_dir = "/home/briardoty/Source/allen-inst-cell-types/data_mountpoint"

# build group: case dict
group_dict = dict()
config_path = "/home/briardoty/Source/allen-inst-cell-types/hpc-jobs/net_configs.json"
with open(config_path, "r") as json_file:
    net_configs = json.load(json_file)

for g in net_configs.keys():
    cases = net_configs[g]
    case_names = cases.keys()
    group_dict[g] = list(case_names)

# init visualizer
vis = PredictionVisualizer(
    data_dir, 
    save_fig=True,
    save_png=True
    )

# plot layer groups
# layer_groups = ["layers-swish5-tanh0.5",
#     "layers-swish7.5-tanh0.1",
#     "layers-swish10-tanh0.5"]
# for lg in layer_groups:
#     cases = group_dict[lg] + [lg[len("layers-"):]]
#     vis.plot_predictions("cifar10",
#         ["sticknet8"],
#         ["adam"],
#         cases=cases,
#         excl_arr=["spatial", "test"],
#         pred_type="max",
#         cross_family=None,
#         pred_std=True,
#         small=False,
#         filename=f"{lg} prediction"
#         )

# all nets, all datasets
nets = ["vgg11", "sticknet8"]
datasets = ["cifar10"]
schemes = ["adam_lravg_nosplit"]
metric = "val_acc"
pred_type = "max"
for n in nets:
    for d in datasets:
        for s in schemes:
            vis.plot_predictions(d,
                [n],
                [s],
                cases=[],
                excl_arr=["spatial", "test", "ratio", "fc", "conv", "all"],
                pred_type=pred_type,
                metric=metric,
                cross_family=None,
                pred_std=True,
                small=False,
                filename=f"{n} {d} {s} {metric}"
                )

# plot ratio groups
ratio_groups = ["ratios-swish5-tanh0.5", "ratios-swish2-tanh2", "ratios-relu-tanh1"]
for rg in ratio_groups:
    cases = group_dict[rg] + [rg[len("ratios-"):]]
    vis.plot_predictions("cifar10",
        ["vgg11", "sticknet8"],
        ["adam"],
        cases=cases,
        excl_arr=["spatial", "test"],
        pred_type="max",
        cross_family=None,
        pred_std=True,
        small=False,
        filename=f"{rg} prediction"
        )