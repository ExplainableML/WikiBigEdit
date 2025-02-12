import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from extract_qa import generate_generality_open_qa_pairs
import argparse

def combine_qa(edit, generality, locality, mhop, genV2, args):
    assert len(edit) == len(generality) == len(locality)
    probes = pd.read_csv(f'twiki_probes/{args["old"]}_{args["new"]}/{args["mode"]}_changed.csv')
    qa_set = []
    for id in tqdm(range(len(edit))):
        subject = edit.loc[id, 'subject']
        if subject not in edit.loc[id, 'question']:
            s_words = subject.replace(',', '').split()
            q_words = edit.loc[id, 'question'].replace('?', '').replace(',', '').split()
            s_new = []
            # find the idx of the first word of subject in question
            id_start = 0
            for i in range(len(q_words)):
                if q_words[i] in s_words:
                    id_start = i
                    break
            # find the idx of the last word of subject in question going backwards from the end
            id_end = 0
            for i in range(len(q_words)-1, -1, -1):
                if q_words[i] in s_words:
                    id_end = i
                    break
            s_new = q_words[id_start:id_end+1]
            subject = ' '.join(s_new)

        if 'tag' not in edit.columns:
            if 'tag' not in probes.columns:
                tag = ''
            else:
                tmp = probes[(probes['subject'] == subject) & (probes['relation'] == edit.loc[id, 'relation']) & (probes['object'] == edit.loc[id, 'object'])]
                if len(tmp) == 0:
                    tag = ''
                elif len(tmp) == 1:
                    tag = tmp.iloc[0]['tag']
                else:
                    raise ValueError(f'Multiple probes found for {subject}, {edit.loc[id, "relation"]}, {edit.loc[id, "object"]}')
        else:
            tag = edit.loc[id, 'tag']

        gen = generality.loc[id, 'reformulated_question']
        #gen = get_reformulated_question(generality, edit.loc[id, 'question'], subject, edit.loc[id, 'relation'])

        genV2_q = {}
        for character in genV2.keys():
            tmp = genV2[character]
            if 'id' in tmp.columns:
                tmp = tmp[tmp['id'] == id]
            elif 'question' in tmp.columns:
                tmp = tmp[tmp['question'] == edit.loc[id, 'question']]
            else:
                raise ValueError
            if len(tmp) == 1:
                if 'reformulated_question' in tmp.columns:
                    genV2_q[f'genv2_{character}'] = tmp['reformulated_question'].values[0]
                else:
                    genV2_q[f'genv2_{character}'] = tmp['question'].values[0]
            else:
                print(f'No generality 2.0 found for {id} in {character} - generating question')
                mixed_genV2 = generate_generality_open_qa_pairs(edit.loc[[id]], model='gpt-4o-mini', character=character)
                assert len(mixed_genV2) == 1
                genV2_q[f'genv2_{character}'] = mixed_genV2['reformulated_question'].values[0]

        mhop_question, mhop_answer = None, None
        for m, row in mhop.iterrows():
            if row['subject'] == subject and row['relation'] == edit.loc[id, 'relation'] and row['object'] == edit.loc[id, 'object']:
                mhop_question = row['question']
                mhop_answer = row['answer']

        id_qa = {
            'tag': tag,
            'subject': subject,
            'src': edit.loc[id, 'question'],
            'rephrase': gen,
            'alt': edit.loc[id, 'answer'],
            'loc': locality.loc[id, 'question'],
            'loc_ans': locality.loc[id, 'answer'],
            'mhop': mhop_question,
            'mhop_ans': mhop_answer
        }
        id_qa.update(genV2_q)
        #if genV2_q['genv2_mixed'] is not None:
        #    qa_set.append(id_qa)
        qa_set.append(id_qa)

    return qa_set


def get_reformulated_question(generality, question, subject, relation):
    from fuzzywuzzy import fuzz
    ref_q = []
    for i, row in generality.iterrows():
        q = row['reformulated_question']
        if not isinstance(q, str):
            continue
        if subject in q:
            ref_q.append(q)

    if len(ref_q) == 0:
        ref_q = generality['reformulated_question'].values

    if len(ref_q) > 1:
        rel_score = []
        for q in ref_q:
            try:
                rel_score.append(fuzz.ratio(q, question))
            except:
                rel_score.append(0)
        ref_q = ref_q[np.argmax(rel_score)]
    elif len(ref_q) == 1:
        ref_q = ref_q[0]
    else:
        raise ValueError(f'No reformulated question found for {subject} in {relation}')

    return ref_q

def merge_loc_qa():
    loc_1 = pd.read_csv('QA_sets/20240501_20240601/batch_jobs/closed_gpt-3.5-turbo/loc_prev/loc_unfiltered_1.csv')
    loc_2 = pd.read_csv('QA_sets/20240501_20240601/batch_jobs/closed_gpt-3.5-turbo/loc_prev/loc_unfiltered_2.csv')

    edit = pd.read_csv('QA_sets/20240501_20240601/batch_jobs/closed_gpt-3.5-turbo/edit/edit_unfiltered.csv')

    edit_ids = edit['id'].tolist()
    loc_1_ids = loc_1['id'].tolist()
    loc_2_ids = loc_2['id'].tolist()

    for id in tqdm(edit_ids):
        if id not in loc_1_ids and id not in loc_2_ids:
            print(f'Missing {id} in loc_1 and loc_2')
            raise ValueError

    loc = pd.concat([loc_1, loc_2], ignore_index=True)
    loc_ids = loc['id'].tolist()
    for id in tqdm(edit_ids):
        if id not in loc_ids:
            print(f'Missing {id} in loc')
            raise ValueError
    loc.to_csv('QA_sets/20240501_20240601/batch_jobs/closed_gpt-3.5-turbo/loc/loc_unfiltered.csv', index=False)


def construct_generation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', type=str, default='20240220')
    parser.add_argument('--new', type=str, default='20240301')
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--pers_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--mhop_model', type=str, default='gpt-4o')
    arg = parser.parse_args()
    return arg


if __name__ == '__main__':
    args = construct_generation_args()
    old = args.old
    new = args.new
    model = args.model
    pers_model = args.pers_model
    mhop_model = args.mhop_model
    for mode in ['open']: #, 'closed']:
        edit = pd.read_csv(f"fact_qa_sets/{old}_{new}/edit_{model}.csv")
        generality = pd.read_csv(f"fact_qa_sets/{old}_{new}/generality_{model}.csv")
        locality = pd.read_csv(f"fact_qa_sets/{old}_{new}/locality_{model}.csv")
        mhop = pd.read_csv(f"fact_qa_sets/{old}_{new}/mhop_{mhop_model}.csv")
        characters = ['mixed']
        pers = {}
        for character in characters:
            pers[character] = pd.read_csv(f"fact_qa_sets/{old}_{new}/pers_{model}.csv")

        args = {'old': old, 'new': new, 'mode': mode, 'model': model}

        qa_set = combine_qa(edit, generality, locality, mhop, pers, args)

        with open(f"fact_qa_sets/{old}_{new}/{mode}_qa_{model}.json", 'w') as f:
            json.dump(qa_set, f)