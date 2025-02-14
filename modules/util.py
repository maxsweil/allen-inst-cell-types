import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import re
import numpy as np
import json
import sys
from sklearn.model_selection import train_test_split


# function for getting an identifier for a given net state
def get_net_tag(net_name, case_id, sample, epoch):
    net_tag = f"{net_name}"

    if (case_id is not None):
        net_tag += f"_case-{case_id}"

    if (sample is not None):
        net_tag += f"_sample-{sample}"

    if (epoch is not None):
        net_tag += f"_epoch-{epoch}"

    return net_tag


def get_net_dir(data_dir, dataset, net_name, train_scheme, group, case, sample):
    """
    Builds and ensures the proper net directory exists, then returns
    its full path
    """

    net_dir = "nets/"

    if dataset is not None:
        net_dir += f"{dataset}/"

    if net_name is not None:
        net_dir += f"{net_name}/"

    if train_scheme is not None:
        net_dir += f"{train_scheme}/"

    if group is not None:
        net_dir += f"{group}/"

    if case is not None:
        net_dir += f"{case}/"

    if sample is not None:
        net_dir += f"sample-{sample}/"

    return ensure_sub_dir(data_dir, net_dir)


def ensure_sub_dir(data_dir, sub_dir):
    """
    Ensures existence of sub directory of data_dir and
    returns its absolute path.

    Args:
        sub_dir (TYPE): DESCRIPTION.

    Returns:
        sub_dir (TYPE): DESCRIPTION.

    """
    sub_dir = os.path.join(data_dir, sub_dir)

    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    return sub_dir


def refresh_seeds(data_dir):
    seeds_dict = dict()
    for i in range(10):
        s = os.urandom(4)
        seeds_dict[i] = s.hex()

    json_obj = json.dumps(seeds_dict)
    seeds_filepath = os.path.join(data_dir, "seeds.json")
    with open(seeds_filepath, "w") as json_file:
        json_file.write(json_obj)


def get_seed_for_sample(data_dir, sample):
    seeds_filepath = os.path.join(data_dir, "seeds.json")
    with open(seeds_filepath, "r") as json_file:
        seeds_json = json.load(json_file)

    return bytes.fromhex(seeds_json[str(sample)])


# standard normalization applied to all stimuli
normalize = transforms.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225])


def load_dataset(data_dir, name, batch_size):
    dataset_dir = os.path.join(data_dir, name)
    n_workers = 4

    if name == "cifar10":
        return load_cifar10(dataset_dir, batch_size, n_workers)
    if name == "cifar100":
        return load_cifar100(dataset_dir, batch_size, n_workers)
    elif name == "imagenette2":
        return load_imagenette(dataset_dir, batch_size, n_workers)
    elif name == "fashionmnist":
        return load_fashionMNIST(dataset_dir, batch_size, n_workers)
    else:
        print(f"Unrecognized dataset name {name}")
        sys.exit(-1)


def load_imagenette(dataset_dir, batch_size=4, n_workers=4):
    # standard transforms
    img_xy = 227
    train_xform = transforms.Compose([
        transforms.CenterCrop(img_xy),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    val_xform = transforms.Compose([
        transforms.CenterCrop(img_xy),
        transforms.ToTensor(),
        normalize
    ])

    # datasets
    train_set = datasets.ImageFolder(os.path.join(dataset_dir, "train"),
                                     transform=train_xform)
    val_set = datasets.ImageFolder(os.path.join(dataset_dir, "val"),
                                   transform=val_xform)

    # loaders
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size, shuffle=True, num_workers=n_workers)

    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size, shuffle=False, num_workers=n_workers)

    return (train_set, val_set, train_loader, val_loader)


def load_fashionMNIST(dataset_dir, batch_size=128, n_workers=4):
    # standard transforms
    img_xy = 56
    mnist_nrmlz = transforms.Normalize([0.5], [0.225])
    train_xform = transforms.Compose([
        transforms.Resize(img_xy),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_xy, 4),
        transforms.ToTensor(),
        mnist_nrmlz
    ])
    val_xform = transforms.Compose([
        transforms.Resize(img_xy),
        transforms.ToTensor(),
        mnist_nrmlz
    ])

    # datasets
    train_set = torchvision.datasets.FashionMNIST(root=dataset_dir, train=True,
                                                  download=True, transform=train_xform)

    val_set = torchvision.datasets.FashionMNIST(root=dataset_dir, train=False,
                                                download=True, transform=val_xform)

    # loaders
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size, shuffle=True, num_workers=n_workers)

    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size, shuffle=False, num_workers=n_workers)

    return (train_set, val_set, train_loader, val_loader)


def load_cifar10(dataset_dir, batch_size=128, n_workers=4,
                 val_frac=0.1):
    # standard transforms
    train_xform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize
    ])
    test_xform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # full dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=True,
        download=True, transform=train_xform,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=False,
        download=True, transform=test_xform)

    # loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=n_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=n_workers, shuffle=False)

    return (train_dataset, None, test_dataset,
            train_loader, None, test_loader)


def load_cifar10_3split(dataset_dir, batch_size=128, n_workers=4,
                        val_frac=0.1):
    # standard transforms
    train_xform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize
    ])
    test_xform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # full dataset
    full_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=True,
        download=True, transform=train_xform,
    )

    # train/val partition
    targets = full_dataset.targets
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)), test_size=val_frac,
        shuffle=True, stratify=targets)

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=False,
        download=True, transform=test_xform)

    # loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=n_workers, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        num_workers=n_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=n_workers, shuffle=False)

    return (train_dataset, val_dataset, test_dataset,
            train_loader, val_loader, test_loader)


def load_cifar10_activation(data_dir, batch_size=128, n_workers=4,
                            n_samples=500):
    dataset_dir = os.path.join(data_dir, "cifar10")

    # standard transforms
    test_xform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # full dataset
    full_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=False,
        download=True, transform=test_xform,
    )

    n_take = len(full_dataset) - n_samples
    frac_take = n_take / len(full_dataset)

    # partition
    targets = full_dataset.targets
    idx, _ = train_test_split(
        np.arange(len(targets)), test_size=frac_take,
        shuffle=True, stratify=targets)

    dataset = Subset(full_dataset, idx)

    # loaders
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        num_workers=n_workers, shuffle=False)

    _ = None
    return (dataset, _, _,
            loader, _, _)


def load_cifar100(dataset_dir, batch_size=128, n_workers=4,
                  val_frac=0.1):
    # standard transforms
    train_xform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize
    ])
    test_xform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # full dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=True,
        download=True, transform=train_xform,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=False,
        download=True, transform=test_xform)

    # loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=n_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=n_workers, shuffle=False)

    return (train_dataset, None, test_dataset,
            train_loader, None, test_loader)


def create_optimizer(name, manager, lr, momentum):
    if "sgd" in name:
        return optim.SGD(manager.net.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    elif "adam" in name:
        return optim.Adam(manager.net.parameters(), lr=lr, weight_decay=5e-4)
    else:
        print(f"Unknown optimizer configured: {name}")
        sys.exit(1)


def get_training_vars(name, manager, lr, lr_step_size=30, lr_gamma=0.5, momentum=0.9):
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(name, manager, lr, momentum)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size,
                                    gamma=lr_gamma)

    return (criterion, optimizer, scheduler)


def get_component_cases(case_dict, case):
    """
    Returns the names of cases that compose the given mixed case

    Args:
        case_dict (dict)
        case: the mixed case
    """

    # identify "component" cases...
    def param_to_float(p):
        return float(p) if p != "None" else p

    zipped_fn_name_and_param_arr = list(zip(case_dict[case]["act_fns"],
                                            [param_to_float(p) for p in case_dict[case]["act_fn_params"]]))
    component_cases = []

    for k1, v1 in zipped_fn_name_and_param_arr:
        for k2, v2 in case_dict.items():

            if (len(v2["act_fns"]) == 1
                    and v2["act_fns"][0] == k1
                    and param_to_float(v2["act_fn_params"][0]) == v1
                    and "_" not in k2):  # THIS IS A HACK TO GET RID OF OLD CASES
                component_cases.append(k2)
                break

    return component_cases


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
    data_dir = "/home/briardoty/Source/allen-inst-cell-types/data_mountpoint"
    # refresh_seeds(data_dir)

    s = get_seed_for_sample(data_dir, 2)
    x = 1