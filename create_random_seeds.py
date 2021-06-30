import os
import binascii
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_samples", default=20, type=int, help="Number of samples to create seeds for")
parser.add_argument("--data_dir", default="/allen/programs/braintv/workgroups/nc-ophys/max.weil/data/data/", type=str, help="Location to save seed file")


def main(n_samples, data_dir):
    seeds_dict = {}

    for i in range(n_samples):
        seeds_dict[str(i)] = binascii.b2a_hex(os.urandom(4)).decode("utf-8") 

    with open(data_dir+'seeds.json', 'w') as file:
        json.dump(seeds_dict, file)
        
if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(**vars(args))