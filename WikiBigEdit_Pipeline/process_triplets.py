import json
import os

import random
import sys

import pandas as pd
import dask.dataframe as dd
import numpy as np
from tqdm import tqdm
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import time
import gc
import dask.dataframe as dd
from dask import delayed, compute
warnings.filterwarnings("ignore")

SUPPORT_MODE = ['both', 'open', 'closed']


def filter_ambiguous(changed, unchanged):
    # classity probes as closed if there are multiple object values for the same subject+relation
    data = changed + unchanged
    df = pd.DataFrame(data, columns=['subject', 'relation', 'object', 'tag'])
    df = df.drop_duplicates()
    # combine all probes with the same subject + relation and count the number of objects
    classification = df.groupby(['subject', 'relation']).size().reset_index(name='count')
    classification['sub_rel'] = classification['subject'] + '-' + classification['relation']
    sub_rel_amb = classification[classification['count'] > 1]['sub_rel'].values
    sub_rel_amb = set(sub_rel_amb)

    unamb_changed, unamb_unchanged = [], []
    for i in tqdm(changed):
        if i[0]+'-'+i[1] not in sub_rel_amb:
            unamb_changed.append(i)
    for i in tqdm(unchanged):
        if i[0]+'-'+i[1] not in sub_rel_amb:
            unamb_unchanged.append(i)

    return unamb_changed, unamb_unchanged


def filter_circular_dependencies(data):
    processed_list = []
    dic = {}
    for i in tqdm(data):
        sentence = i[0] + i[1] + i[2]
        if sentence in dic:
            continue
        dic[sentence] = 1
        sub, rel, obj = i[0], i[1], i[2]
        sub = sub.lower()
        rel = rel.lower()
        obj = obj.lower()
        if sub in obj or obj in sub:
            continue
        if len(obj.split()) > 5:
            continue
        if not any(char in sub for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            continue
        if not any(char in obj for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            continue
        if 'category:' in sub or 'category:' in obj:
            continue
        if 'template:' in sub or 'template:' in obj:
            continue

        processed_list.append(i)
    return processed_list


def to_df(data, dask=False, n_partitions=8):
    if len(data[0]) == 3:
        processed_list = pd.DataFrame(data, columns=['subject', 'relation', 'object'])
    elif len(data[0]) == 4:
        processed_list = pd.DataFrame(data, columns=['subject', 'relation', 'object', 'tag'])
    elif len(data[0]) == 7:
        processed_list = pd.DataFrame(data, columns=['subject', 'relation', 'object', 'incorrect_choice_1', 'incorrect_choice_2', 'incorrect_choice_3', 'tag'])
    else:
        raise ValueError("Invalid data format")
    if dask:
        mem = processed_list.memory_usage(index=True).sum()
        print("Memory usage: ", mem/1024**2, "MB")
        processed_list = dd.from_pandas(processed_list, npartitions=n_partitions)
    return processed_list


def filter_nan_single_character(df):

    #filter all probes with "nan" in columns subject, relation, object
    df = df.dropna(subset=['subject', 'relation', 'object'])

    #filter all probes where subject or object consists of one or less characters
    df = df[df['subject'].str.len() > 1]
    df = df[df['object'].str.len() > 1]
    return df


def filter_unwanted_relations(data):
    #filter all relations from the list of ambiguous relations
    with open('Wikidata_relations/ambiguous_relations.json', 'r') as f:
        amb_relations = json.load(f)['ambiguous_relations']
    wikidata_relations = pd.read_csv('Wikidata_relations/wikidata_relations.csv', sep=';')
    del_relations = wikidata_relations[wikidata_relations['Datatype'] != 'WI']['Label'].values.tolist()
    del_relations += amb_relations
    del_relations_set = set(del_relations)
    processed_list = [i for i in tqdm(data) if i[1] not in del_relations_set]
    return processed_list


def compute_similarity_matrix(relations):
    vectorizer = TfidfVectorizer().fit_transform(relations)
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)


def process_chunk_dask(changed, unchanged, obj):
    changed_obj = changed[changed['object'] == obj]
    unchanged_obj = unchanged[unchanged['object'] != obj]

    locality_set = []
    relations = changed_obj['relation'].unique()
    unchanged_obj_relations = unchanged_obj['relation'].values

    for rel in relations:
        changed_obj_rel = changed_obj[changed_obj['relation'] == rel]
        if rel in unchanged_obj_relations:
            unchanged_obj_rel = unchanged_obj[unchanged_obj['relation'] == rel]
        else:
            similarity = np.array([fuzz.ratio(rel, r) for r in unchanged_obj_relations])
            sim_relation = unchanged_obj_relations[similarity.argmax()]
            unchanged_obj_rel = unchanged_obj[unchanged_obj['relation'] == sim_relation]

        subjects = changed_obj_rel['subject'].unique()
        unchanged_obj_rel_subjects = unchanged_obj_rel['subject'].values
        for sub in subjects:
            if sub in unchanged_obj_rel_subjects:
                unchanged_obj_rel_sub = unchanged_obj_rel[unchanged_obj_rel['subject'] == sub]
            else:
                try:
                    similarity = np.array([fuzz.ratio(sub, s) for s in unchanged_obj_rel_subjects])
                except:
                    print(sub)
                    for s in unchanged_obj_rel_subjects:
                        if not isinstance(s, str):
                            print(s)
                    raise
                sim_subject = unchanged_obj_rel_subjects[similarity.argmax()]
                unchanged_obj_rel_sub = unchanged_obj_rel[unchanged_obj_rel['subject'] == sim_subject]

            locality_set.append([
                sub, rel, obj,
                unchanged_obj_rel_sub['subject'].values[0],
                unchanged_obj_rel_sub['relation'].values[0],
                unchanged_obj_rel_sub['object'].values[0],
                unchanged_obj_rel_sub['relation_description'].values[0],
                changed_obj_rel[changed_obj_rel['subject'] == sub]['original_id'].values[0],
            ])
    return locality_set


def create_locality_probes(df_changed, df_unchanged, batch_size=50):
    df_changed['original_id'] = df_changed.index

    # Sort by object and relation
    df_changed = df_changed.sort_values(by=['object', 'relation'])

    objects = df_changed['object'].unique()

    locality_set = []
    print('Starting locality probe generation')
    for i in range(0, len(objects), batch_size):
        print(f'-- Batch {i // batch_size} of {len(objects) // batch_size}')
        batch_objects = objects[i:min(i + batch_size, len(objects))]
        delayed_tasks = []
        print(f'-- Processing {len(batch_objects)} objects')
        start_time = time.time()
        for obj in tqdm(batch_objects):
            delayed_tasks.append(delayed(process_chunk_dask)(df_changed, df_unchanged, obj))
        print(f'-- Delayed tasks created in {time.time() - start_time:.2f} seconds')
        start_time = time.time()
        results = compute(*delayed_tasks, scheduler='processes')
        print(f'-- Delayed tasks computed in {time.time() - start_time:.2f} seconds')
        start_time = time.time()
        for result in results:
            locality_set.extend(result)
        print(f'-- Results appended in {time.time() - start_time:.2f} seconds')
        start_time = time.time()
        # Explicitly trigger garbage collection
        gc.collect()
        print(f'-- Garbage collection completed in {time.time() - start_time:.2f} seconds')

    locality_df = pd.DataFrame(locality_set,
                               columns=['subject_c', 'relation_c', 'object_c', 'subject', 'relation', 'object', 'relation_description', 'original_id'])
    locality_df = locality_df.sort_values(by='original_id')
    locality_df = locality_df.reset_index(drop=True)
    return locality_df


def generate_mhop(changed):
    mhop_relations = []
    mhop_reldesc = []
    mhop_objects = []
    for i in tqdm(changed.index):
        object_ = changed.loc[i, 'object']
        tmp = changed[changed['subject'] == object_]
        if len(tmp) == 1:
            mhop_relations.append(tmp['relation'].values[0])
            mhop_reldesc.append(tmp['relation_description'].values[0])
            mhop_objects.append(tmp['object'].values[0])
        elif len(tmp) > 1:
            to_delete = ['mapping relation type', 'publisher', 'topic\'s main Wikimedia portal',
                         'Wikimedia outline', 'instance of', 'maintained by WikiProject', 'editor']
            tmp2 = tmp[~tmp['relation'].isin(to_delete)]
            if len(tmp2) >= 1:
                idx = np.random.choice(tmp2.index)
                mhop_relations.append(tmp2.loc[idx, 'relation'])
                mhop_reldesc.append(tmp2.loc[idx, 'relation_description'])
                mhop_objects.append(tmp2.loc[idx, 'object'])
            else:
                mhop_relations.append(np.nan)
                mhop_reldesc.append(np.nan)
                mhop_objects.append(np.nan)
        elif len(tmp) == 0:
            mhop_relations.append(np.nan)
            mhop_reldesc.append(np.nan)
            mhop_objects.append(np.nan)

    mhop = changed.copy()
    mhop['mhop_relation'] = mhop_relations
    mhop['mhop_relation_description'] = mhop_reldesc
    mhop['mhop_object'] = mhop_objects
    mhop['open'] = True
    mhop = clean_mhop(mhop)
    print(f'Number of not nan mhop probes: {len(mhop[~mhop["mhop_relation"].isna()])} '
          f'({round(len(mhop[~mhop["mhop_relation"].isna()]) / len(mhop) * 100, 2)})%')
    return mhop


def clean_mhop(mhop):
    mhop = mhop[mhop['open']]

    # no "instance of" relation
    mhop = mhop[(mhop['relation'] != 'instance of') & (mhop['mhop_relation'] != 'instance of')]

    # non useful relations
    to_delete = ['mapping relation type', 'publisher', 'topic\'s main Wikimedia portal', 'Wikimedia outline',
                 'maintained by WikiProject', 'editor']
    mhop = mhop[~mhop['mhop_relation'].isin(to_delete)]

    # circular relations
    mhop = mhop[mhop['subject'] != mhop['mhop_object']]

    # no wiki in object (account for nan values)
    mhop = mhop[~mhop['mhop_object'].str.contains('wiki', na=False)]
    mhop = mhop[~mhop['mhop_object'].str.contains('Wiki', na=False)]

    return mhop


def add_relation_description(df, relation_descriptions):
    df['relation_description'] = df['relation'].apply(lambda x: relation_descriptions[x])#, meta=('relation_description', 'object'))
    return df


def get_relation_description(changed, unchanged):
    data = changed + unchanged
    df = pd.DataFrame(data, columns=['subject', 'relation', 'object', 'tag'])
    # get unique relations
    relations = df['relation'].unique()
    wikidata_relations = pd.read_csv('Wikidata_relations/wikidata_relations.csv', sep=';')
    relation_descriptions = {}
    not_found = 0
    for relation in tqdm(relations):
        all_relations = wikidata_relations['Label'].values
        if relation in all_relations:
            description = wikidata_relations[wikidata_relations['Label'] == relation]['Description'].values[0]
            relation_descriptions[relation] = description
        else:
            # do a soft match to find the relation in the list of all relations
            similarity = [fuzz.ratio(relation, rel) for rel in all_relations]
            max_similarity = max(similarity)
            matched_relation = all_relations[similarity.index(max_similarity)]
            if max_similarity > 80 or relation in matched_relation:
                description = wikidata_relations[wikidata_relations['Label'] == matched_relation]['Description'].values[
                    0]
                relation_descriptions[relation] = description
            else:
                relation_descriptions[relation] = ''
                not_found += 1
    print("Number of relations not found: ", not_found)
    return relation_descriptions


def check_objects(changed, unchanged):
    changed_df = to_df(changed)
    unchanged_df = to_df(unchanged)

    changed_objects = set(changed_df['object'].values)
    unchanged_subjects = set(unchanged_df['subject'].values)

    print(f'Changed objects: {len(changed_objects)} | Unchanged subjects: {len(unchanged_subjects)}')

    common = changed_objects.intersection(unchanged_subjects)

    print(f'Common: {len(common)}')


def apply_filters(data, relation_descriptions):
    data = to_df(data)
    data = filter_nan_single_character(data)
    data = add_relation_description(data, relation_descriptions)
    return data


def loc_dist(args):
    os.makedirs(f'fact_triplets_processed/{args.old}_{args.new}/loc', exist_ok=True)

    assert args.split != -1, "Batch size must be specified"
    assert args.num_splits > 0, "Number of splits must be specified"

    changed = pd.read_csv(f'fact_triplets_processed/{args.old}_{args.new}/changed.csv')
    unchanged = pd.read_csv(f'fact_triplets_processed/{args.old}_{args.new}/unchanged.csv')

    print(f'Changed: {len(changed)} | Unchanged: {len(unchanged)}')
    changed, unchanged = filter_nan_single_character(changed), filter_nan_single_character(unchanged)
    print(f'Changed: {len(changed)} | Unchanged: {len(unchanged)}')

    objects = changed['object'].unique()
    np.random.seed(123)
    np.random.shuffle(objects)
    obj_split = objects[args.split*len(objects)//args.num_splits:min((args.split+1)*len(objects)//args.num_splits, len(objects))]
    changed_split = changed[changed['object'].isin(obj_split)]

    print(f'Processing split {args.split} of {args.num_splits}')
    print(f'Changed: {len(changed_split)} | Unchanged: {len(unchanged)}')
    locality = create_locality_probes(changed_split, unchanged, batch_size=args.batch_size)

    locality.to_csv(f'fact_triplets_processed/{args.old}_{args.new}/loc/locality_{args.split}.csv', index=False)


def main(args):
    os.makedirs(f'fact_triplets_processed/{args.old}_{args.new}', exist_ok=True)

    # Load the data
    with open(f'fact_triplets_raw/{args.old}_{args.new}/changed/total_changed_item.json') as f:
        changed = json.load(f)
    with open(f'fact_triplets_raw/{args.old}_{args.new}/unchanged/total_unchanged_item.json') as f:
        unchanged = json.load(f)
    print(f'Changed: {len(changed)} | Unchanged: {len(unchanged)}')

    print('----------- Changed -----------')
    changed_filtered = filter_circular_dependencies(changed)
    changed_filtered = filter_unwanted_relations(changed_filtered)

    print('----------- Unchanged -----------')
    unchanged_filtered = filter_circular_dependencies(unchanged)
    unchanged_filtered = filter_unwanted_relations(unchanged_filtered)

    print(f'Changed: {len(changed_filtered)} | Unchanged: {len(unchanged_filtered)}')

    print('----------- Filer Ambiguos Questions -----------')
    open_changed, open_unchanged = filter_ambiguous(changed_filtered, unchanged_filtered)
    print(f'Open Changed: {len(open_changed)} | Open Unchanged: {len(open_unchanged)}')

    print('----------- Get Relation Descriptions -----------')
    relation_descriptions = get_relation_description(changed_filtered, unchanged_filtered)

    print('----------- Filtering Proportions -----------')
    open_changed = apply_filters(open_changed, relation_descriptions)
    open_unchanged = apply_filters(open_unchanged, relation_descriptions)
    open_changed.to_csv(f'fact_triplets_processed/{args.old}_{args.new}/changed.csv', index=False)
    open_unchanged.to_csv(f'fact_triplets_processed/{args.old}_{args.new}/unchanged.csv', index=False)
    print(f'Open Changed: {len(open_changed)} | Open Unchanged: {len(open_unchanged)}')

    print('----------- Multi Hop -----------')
    mhop_open = generate_mhop(open_changed)
    mhop_open.to_csv(f'fact_triplets_processed/{args.old}_{args.new}/mhop.csv', index=False)


def construct_generation_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--old', type=str, default='20240301')
    parser.add_argument('--new', type=str, default='20240320')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--split', type=int, default=-1)
    parser.add_argument('--num_splits', type=int, default=10)
    parser.add_argument('--loc_dist', type=bool, default=False)
    arg = parser.parse_args()
    return arg


if __name__ == '__main__':
    args = construct_generation_args()

    if args.loc_dist:
        loc_dist(args)
    else:
        main(args)



