import pandas as pd
from tqdm import tqdm
import os
import argparse


def match_loc_probes(changed, locality):
    reordered_loc = []
    for i in tqdm(range(len(changed))):
        subj = changed.loc[i, 'subject']
        rel = changed.loc[i, 'relation']
        obj = changed.loc[i, 'object']

        loc = locality[(locality['subject_c'] == subj) & (locality['object_c'] == obj) & (locality['relation_c'] == rel)]
        if len(loc) != 1:
            print(f'No match for {subj}, {rel}, {obj} @ {i}')
            reordered_loc.append([subj, rel, obj, '', '', '', ''])
            continue
        loc = loc.iloc[0]
        reordered_loc.append([subj, rel, obj, loc['subject'], loc['relation'], loc['relation_description'], loc['object']])

    reordered_loc = pd.DataFrame(reordered_loc, columns=['changed_subject', 'changed_relation', 'changed_object',
        'subject', 'relation', 'relation_description', 'object'])
    return reordered_loc


def reorder_loc_dist(args, changed_dir, locality_dir):
    changed = pd.read_csv(changed_dir)
    # check if locality_dir is a directory or a file
    if os.path.isfile(locality_dir):
        locality = pd.read_csv(locality_dir)
    else:
        loc_file_list = os.listdir(locality_dir)
        loc_file_list = [f'{locality_dir}{file}' for file in loc_file_list]
        print(loc_file_list)
        locality = pd.DataFrame()
        for loc_file in loc_file_list:
            loc = pd.read_csv(loc_file)
            locality = pd.concat([locality, loc], ignore_index=True)

    reordered_loc = match_loc_probes(changed, locality)
    reordered_loc.to_csv(f'fact_triplets_processed/{args.old}_{args.new}/locality.csv', index=False)


def construct_generation_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--old', type=str, default='20240301')
    parser.add_argument('--new', type=str, default='20240320')
    arg = parser.parse_args()
    return arg


if __name__ == '__main__':
    args = construct_generation_args()
    changed_dir = f'fact_triplets_processed/{args.old}_{args.new}/changed.csv'
    locality_dir = f'fact_triplets_processed/{args.old}_{args.new}/loc/'
    reorder_loc_dist(args, changed_dir, locality_dir)
