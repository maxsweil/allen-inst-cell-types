import os
import argparse
import pickle
import json
import numpy as np
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser()
parser.add_argument("--net_name", type=str, required=True, help="Network to analyze (ex. vgg11, sticknet8)")
parser.add_argument("--dataset", type=str, required=True, help="Dataset trained on (ex. cifar10, cifar100")
parser.add_argument("--config_group", type=str, required=True, help="Specific set of cases to analyze (ex. within-tanh, control)")
parser.add_argument("--data_dir", default="/allen/programs/braintv/workgroups/nc-ophys/max.weil/data/data/nets/",
                    type=str, help="Directory to pull data from")
parser.add_argument("--save_dir", default="/allen/programs/braintv/workgroups/nc-ophys/max.weil/data/data/dimensionality_data/",
                    type=str, help="Directory to save dataframe")


def main(net_name, dataset, config_group, data_dir, save_dir):
    cases = get_case_names(config_group)
    dicts = load_all(net_name, dataset, cases, data_dir)
    dicts = format_samples(dicts)
    dicts = compute_all(dicts)
    save_dicts(dicts, net_name, dataset, config_group, save_dir)
    return dicts


def get_case_names(config_group):

    # Opening network configuration json file
    with open("/allen/programs/braintv/workgroups/nc-ophys/max.weil/allen-inst-cell-types/hpc-jobs/net_configs.json", "r") as json_file:
        net_configs = json.load(json_file)

    # Finding all cases within single config group
    cases = set()
    configs = net_configs[config_group]
    for case in configs.keys():
        cases.add(case)
    cases = list(cases)
    return cases


def load_all(net_name, dataset, cases, data_dir):
    dicts = []
    for case in cases:
        try:
            dicts = dicts + load_samples(net_name, dataset, case, data_dir)
        except:
            print(case + ' crashed')
    return dicts


def load_samples(net_name, dataset, case, data_dir):
    dicts = []
    for sample in np.arange(0, 20):
        dicts.append(load_dict(case, sample=sample, dataset=dataset, net_name=net_name, data_dir=data_dir))
    return dicts


def load_dict(case, sample, dataset, net_name, data_dir, scheme='adam'):
    if ('swish' in case.lower()) & ('tanh' in case.lower()):
        group = 'cross-swish-tanh'
    elif ('-' in case.lower()) & ('swish' in case.lower()):
        group = 'within-swish'
    elif ('-' in case.lower()) & ('tanh' in case.lower()):
        group = 'within-tanh'
    elif 'swish' in case.lower():
        group = 'component-swish'
    elif 'tanh' in case.lower():
        group = 'component-tanh'
    elif 'relu' in case.lower():
        group = 'control'

    sample = 'sample-' + str(sample)
    filename = 'activation_dict.npy'
    filepath = os.path.join(data_dir, dataset, net_name, scheme, group, case, sample, filename)
    if sample == 'sample-0':
        print(filepath.split('nets/')[1])
    return np.load(filepath, allow_pickle=True)


def format_samples(dicts):
    for index, d in enumerate(dicts):
        dicts[index] = format_matrix(d.item())
    return dicts


def format_matrix(act_dict):
    '''
        Takes the activation function dictionary and creates a matrix of units x images for each layer

        act_dict, the activation dictionary briar saved out

        Returns the act_dict with a new key 'matrix_dict' which as an entry for each layer.
        The keys of matrix_dict are renamed to integers, preserving the order in activation_dict

        Each matrix has dimensions n_sample x n_features, where samples are image presentations, and features is units. So if we have a layer with 100 images and 200 units, the matrix is (100 x 200)
    '''

    # Create new dictionary to store matrix
    act_dict['matrix_dict'] = dict()

    for dex, layer in enumerate(act_dict['activation_dict']):
        key = str(dex) + '_' + layer.split('(')[0]
        act_dict['matrix_dict'][key] = format_matrix_inner(act_dict['activation_dict'][layer])
    return act_dict


def format_matrix_inner(activations):
    '''
        We get a list of activations. Each activation contains n-images x units or n-images x units x X-position x Y-position
    '''

    if activations[0].ndim == 2:
        # Fully Connected layer, just concatenate
        return np.concatenate(activations)
    elif activations[0].ndim == 4:
        # Convolutional layer, have to merge x/y coordinates
        activations = [np.reshape(x.values, [np.shape(x)[0], -1]) for x in activations]
        return np.concatenate(activations)
    else:
        raise Exception('Dimensions are weird')


def compute_all(dicts):
    for d in dicts:
        d = compute_category_dimensionality(d)
        d = compute_global_dimensionality(d)
    return dicts


def compute_category_dimensionality(act_dict):
    category_indexes = get_category_indexes(act_dict)
    categories = np.unique(category_indexes)

    act_dict['category_dimensions'] = dict()
    act_dict['category_dimensions_list'] = dict()
    act_dict['category_variance'] = dict()
    act_dict['category_variance_list'] = dict()
    for dex, layer in enumerate(act_dict['matrix_dict'].keys()):
        act_dict['category_dimensions_list'][layer + '_list'] = []
        act_dict['category_variance_list'][layer + '_list'] = []
        for category in categories:
            dimensionality, variance = layer_PCA(act_dict['matrix_dict'][layer][category_indexes == category, :])
            act_dict['category_dimensions_list'][layer + '_list'].append(dimensionality)
            act_dict['category_variance_list'][layer + '_list'].append(variance)
        act_dict['category_dimensions'][layer] = np.mean(act_dict['category_dimensions_list'][layer + '_list'])
        act_dict['category_variance'][layer] = np.mean(act_dict['category_variance_list'][layer + '_list'])
    return act_dict


def compute_global_dimensionality(act_dict):
    '''
        Computes the dimensionality using the formula:
        dimension = sum(eigenvalues)^2/sum(eigenvalues^2)

        skip is a list of layer indices to skip for practical reasons
    '''
    act_dict['global_dimensions'] = dict()
    act_dict['global_variance'] = dict()
    for dex, layer in enumerate(act_dict['matrix_dict'].keys()):
        dimensionality, variance = layer_PCA(act_dict['matrix_dict'][layer])
        act_dict['global_dimensions'][layer] = dimensionality
        act_dict['global_variance'][layer] = variance
    return act_dict


def get_category_indexes(act_dict):
    k = list(act_dict['activation_dict'].keys())[0]
    coords = [x.coords['img'].values for x in act_dict['activation_dict'][k]]
    return np.hstack(coords)


def layer_PCA(layer, eigenvalue=True):
    '''
        Does PCA on a single layer of activations that is size (samples x units)

    '''
    if eigenvalue:
        C = np.cov(layer)
        vals, vecs = np.linalg.eig(C)
        dimensionality = np.sum(vals) ** 2 / np.sum(vals ** 2)
        variance = np.sum(vals)
    else:
        pca = PCA(n_components=np.min(np.shape(layer))).fit(layer)
        dimensionality = np.sum(pca.singular_values_) ** 2 / np.sum(pca.singular_values_ ** 2)
        variance = np.sum(pca.singular_values_)
    return dimensionality, variance


def save_dicts(dicts, net_name, dataset, config_group, save_dir):
    label = net_name + '_' + dataset + '_' + config_group
    filename = os.path.join(save_dir, dataset, net_name) + label + '.pkl'
    save_dicts = make_save_dicts(dicts)
    with open(filename, 'wb') as handle:
        pickle.dump(save_dicts, handle)
    return


def make_save_dicts(dicts):
    save_dicts = []
    for d in dicts:
        save_d = {}
        save_d['net_name'] = d['net_name']
        save_d['dataset'] = d['dataset']
        save_d['train_scheme'] = d['train_scheme']
        save_d['group'] = d['group']
        save_d['case'] = d['case']
        save_d['sample'] = d['sample']
        save_d['category_dimensions'] = d['category_dimensions']
        save_d['category_variance'] = d['category_variance']
        save_dicts.append(save_d)
    return save_dicts

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))