"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""
import sys

import pandas as pd

from ..models.melo.melo import LORA

import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc, 
    test_batch_prediction_acc, 
    test_prediction_acc,
    test_generation_quality, 
    test_concept_gen,
    test_safety_gen,
    test_instance_change,
    PPL,
    kl_loc_loss,
    es,
    es_per_icl,
    per_generation,
    F1
)

def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    records: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False,
    few_shot_examples = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    if isinstance(model,LORA):
        model=model.model
    # First, unpack rewrite evaluation record.
    rewrite_prompts, rephrase_prompts, targets, modes, tags = [], [], [], [], []
    loc_prompts, portability_prompts = {}, {}
    if not isinstance(records, list):
        records = [records]
    for record in records:
        target_new, ground_truth = (
            record[x] for x in ["target_new", "ground_truth"]
        )
        fs_examples = {'open': "Which metro system does the Congress Nagar metro belong to? A: Nagpur Metro Rail \n "
                               #"Q: What is the color of Turnberry Lighthouse? A: white \n "
                               #"Q: What institution does Folsom Library belong to? A: Rensselaer Libraries \n "
                               #"Q: Who composed “Come and Get It”? A: Paul McCartney \n "
                               "Q: {}",
                       'closed': "Q: Is it correct to say that the color of the Turnberry Lighthouse is green? A: incorrect \n "
                                 "Q: Is it accurate to say that the Folsom Library is not a part of the Rensselaer Libraries? A: correct \n "
                                 "Q: Is Volt Netherlands a subsidiary of Volt Europa? A: correct \n "
                                 "Q: Is it correct that Zonia Baber was not educated at Chicago State University? A: incorrect \n "
                                 "Q: {} A:",
                       }
        mode = 'closed' if target_new in ['correct', 'incorrect'] else 'open'
        rewrite_prompt = record["prompt"]
        rephrase_prompt = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
        if few_shot_examples:
            rewrite_prompt = fs_examples[mode].format(rewrite_prompt)
            rephrase_prompt = fs_examples[mode].format(rephrase_prompt) if rephrase_prompt is not None else None
        rewrite_prompts.append(rewrite_prompt)
        if not rephrase_prompt is None or pd.isna(rephrase_prompt):
            rephrase_prompts.append(rephrase_prompt)
        targets.append(target_new)
        modes.append(mode)
        tags.append(record['tag'])

        if 'locality' in record.keys() and any(record['locality']):
            for locality_key in record['locality'].keys():
                if not locality_key in loc_prompts.keys():
                    loc_prompts[locality_key] = {'prompt': [], 'ground_truth': []}
                loc_prompt = record['locality'][locality_key]['prompt']
                if few_shot_examples:
                    loc_prompt = fs_examples[mode].format(loc_prompt)
                loc_prompts[locality_key]['prompt'].append(loc_prompt)
                loc_prompts[locality_key]['ground_truth'].append(record['locality'][locality_key]['ground_truth'])
        if 'portability' in record.keys() and any(record['portability']):
            for portability_key in record['portability'].keys():
                if not portability_key in portability_prompts.keys():
                    portability_prompts[portability_key] = {'prompt': [], 'ground_truth': []}
                if pd.isna(record['portability'][portability_key]['prompt']) or pd.isna(
                        record['portability'][portability_key]['ground_truth']):
                    continue
                portability_prompt = record['portability'][portability_key]['prompt']
                if few_shot_examples:
                    portability_prompt = fs_examples[mode].format(portability_prompt)
                portability_prompts[portability_key]['prompt'].append(portability_prompt)
                portability_prompts[portability_key]['ground_truth'].append(record['portability'][portability_key]['ground_truth'])

    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                              rewrite_prompts, targets, device=device, eval_metric=eval_metric, record_flops=True)

    if rephrase_prompts[0] is not None:
        reph = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                                rephrase_prompts, targets, device=device, test_rephrase=True, eval_metric=eval_metric)
    else:
        reph = None

    loc, port = {}, {}
    for locality_key in loc_prompts.keys():
        loc[locality_key] = compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                     loc_prompts[locality_key]['prompt'],
                                     loc_prompts[locality_key]['ground_truth'],
                                     device=device)
    for portability_key in portability_prompts.keys():
        if len(portability_prompts[portability_key]['prompt']) > 0:
            if hasattr(model, 'verbose') and portability_key == 'mhop':
                model.verbose = True
            port[portability_key] = compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            portability_prompts[portability_key]['prompt'],
                                            portability_prompts[portability_key]['ground_truth'],
                                            device=device)
            if hasattr(model, 'verbose'):
                model.verbose = False

    if test_generation:
        if hparams.alg_name == 'GRACE':
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=200, vanilla_generation=True)
        else:
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=200, vanilla_generation=False)
    for k in range(len(ret)):
        ret[k]['prompt'] = rewrite_prompts[k]
        ret[k]['target'] = targets[k]
        ret[k]['mode'] = modes[k]
        ret[k]['tag'] = tags[k]
        ret[k].update(reph[k]) if reph is not None else None
        ret[k].update({'locality': {}})
        for l_key in loc.keys():
            for l in loc[l_key]:
                ret[k]['locality'][l] = [loc[l_key][l][k]]
        ret[k].update({'portability': {}})
        for p_key in port.keys():
            for p in port[p_key]:
                try:
                    ret[k]['portability'][p] = {'performance': port[p_key][p]['performance'][k],
                                                'prompt': port[p_key][p]['prompt'][k],
                                                'ground_truth': port[p_key][p]['ground_truth'][k],
                                                'answer': port[p_key][p]['answer'][k]}
                except IndexError:
                    pass
    return ret


def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompts: list,
    targets: list,
    device,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em',
    record_flops: bool = False
) -> typing.Dict:
    
    if not test_rephrase:
        key = 'rewrite'
    else:
        key = 'rephrase'
    if eval_metric == 'ppl':
        ppl = PPL(model, tok, prompts, targets, device)
        res = [{
            f"{key}_ppl": p
        } for p in ppl]
    elif hparams.alg_name=="GRACE":
        # ppl = PPL(model, tok, prompt, target_new, device)
        if 't5' in model_name.lower():
            acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompts, targets, device)
            answ, forwar_time = None, None, None
        else:
            acc, answ, forward_time = test_prediction_acc(model, tok, hparams, prompts, targets, device, vanilla_generation=True)
        f1 = F1(model,tok,hparams,prompts,targets,device, vanilla_generation=True)
        res = [{
            f"{key}_acc": a,
            f"{key}_answ": ans,
            # f"{key}_PPL": p,
            f"{key}_F1":f
        } for a, ans, f in zip(acc, answ, f1)]
    else:
        if 't5' in model_name.lower():
            acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompts, targets, device)
            answ, forwad_time = None, None, None
        else:
            res, answ, forward_time = test_prediction_acc(model, tok, hparams, prompts, targets, device, record_flops=record_flops)
        res = [{
            f"{key}_acc": r['acc'],
            f"{key}_ppl": r['ppl'],
            f"{key}_f1": r['f1'],
            f"{key}_answ": a,
            f"{key}_forward_time": forward_time
        } for r, a in zip(res, answ)]
    return res

def compute_locality_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: typing.Union[str, List[str]],
    locality_ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:

    if 't5' in model_name.lower():
        loc_tokens = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True)
        loc_acc = 0
    else:
        loc_tokens, loc_acc = test_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True, vanilla_generation=hparams.alg_name=='GRACE')

    if type(loc_tokens) is not list:
        loc_tokens = [loc_tokens,]

    if type(loc_acc) is not list:
        loc_acc = [loc_acc,]

    ret = {
        f"{locality_key}_output": loc_tokens,
        f"{locality_key}_acc": loc_acc
    }
    return ret

def compute_portability_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    portability_key: str,
    prompt: typing.Union[str, List[str]],
    ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:
    #if portability_key == 'mhop':
    #    model.verbose = True

    if 't5' in model_name.lower():
        portability_correct = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, ground_truth, device)
        answer = None
    else:
        portability_correct, answer, _ = test_prediction_acc(model, tok, hparams, prompt, ground_truth, device, vanilla_generation=hparams.alg_name=='GRACE')

    ret = {
        f"{portability_key}": {'performance': portability_correct, 'prompt': prompt, 'ground_truth': ground_truth, 'answer': answer}
    }
    #model.verbose = False
    return ret

def compute_icl_edit_quality(
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoTokenizer,
        icl_examples,
        record: typing.Dict,
        device,
        pre_edit: bool = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )
    prompt = record["prompt"]
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    new_fact = f'New Fact: {prompt} {target_new}\nPrompt: {prompt}'

    if pre_edit:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new, prompt)
    else:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new, new_fact)
    ret = {
        f"rewrite_acc": edit_acc
    }
    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase is not None:
        rephrase_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                   target_new, f'New Fact: {prompt} {target_new}\nPrompt: {rephrase}')
        ret['rephrase_acc'] = rephrase_acc

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            if isinstance(record['locality'][locality_key]['ground_truth'], list):
                pre_neighbor = []
                post_neighbor = []
                for x_a, x_p in zip(record['locality'][locality_key]['ground_truth'],
                                    record['locality'][locality_key]['prompt']):
                    tmp_pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''], x_a,
                                                   f"New Fact: {prompt} {target_new}\nPrompt: {x_p}", neighborhood=True)
                    tmp_post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples, x_a,
                                                    f"New Fact: {prompt} {target_new}\nPrompt: {x_p}",
                                                    neighborhood=True)
                    if type(tmp_pre_neighbor) is not list:
                        tmp_pre_neighbor = [tmp_pre_neighbor, ]
                    if type(tmp_post_neighbor) is not list:
                        tmp_post_neighbor = [tmp_post_neighbor, ]
                    assert len(tmp_pre_neighbor) == len(tmp_post_neighbor)
                    pre_neighbor.append(tmp_pre_neighbor)
                    post_neighbor.append(tmp_post_neighbor)
                res = []
                for ans, label in zip(pre_neighbor, post_neighbor):
                    temp_acc = np.mean(np.equal(ans, label))
                    if np.isnan(temp_acc):
                        continue
                    res.append(temp_acc)
                ret['locality'][f'{locality_key}_acc'] = res
            else:
                pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''],
                                           record['locality'][locality_key]['ground_truth'],
                                           f"New Fact: {prompt} {target_new}\nPrompt: {record['locality'][locality_key]['prompt']}",
                                           neighborhood=True)
                post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                            record['locality'][locality_key]['ground_truth'],
                                            f"New Fact: {prompt} {target_new}\nPrompt: {record['locality'][locality_key]['prompt']}",
                                            neighborhood=True)
                if type(pre_neighbor) is not list:
                    pre_neighbor = [pre_neighbor, ]
                if type(post_neighbor) is not list:
                    post_neighbor = [post_neighbor, ]
                assert len(pre_neighbor) == len(post_neighbor)

                ret['locality'][f'{locality_key}_acc'] = np.mean(np.equal(pre_neighbor, post_neighbor))
    # Form a list of lists of prefixes to test.
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            if pre_edit:
                icl_input = ['']
                x_prefix = ""
            else:
                icl_input = icl_examples
                x_prefix = f"New Fact: {prompt} {target_new}\nPrompt: "
            if isinstance(record['portability'][portability_key]['ground_truth'], list):
                portability_acc = []
                for x_a, x_p in zip(record['portability'][portability_key]['ground_truth'],
                                    record['portability'][portability_key]['prompt']):
                    tmp_portability_acc = icl_lm_eval(model, model_name, hparams, tok, icl_input, x_a,
                                                      f"{x_prefix}{x_p}")
                portability_acc.append(tmp_portability_acc)
            else:
                portability_acc = icl_lm_eval(model, model_name, hparams, tok, [''],
                                              record['portability'][portability_key]['ground_truth'],
                                              record['portability'][portability_key]['prompt'])
                portability_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                              record['portability'][portability_key]['ground_truth'],
                                              f"New Fact: {prompt} {target_new}\nPrompt: {record['portability'][portability_key]['prompt']}")
            ret['portability'][f'{portability_key}_acc'] = portability_acc
    return ret

def icl_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        neighborhood=False
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    if 't5' in model_name.lower():
        target_len = len(tokenizer.encode(target))
        target_ids = tokenizer(f'{x} {target}', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples), return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).logits
            ans = torch.argmax(logits, dim=-1)[:,-target_len:-1].squeeze()
            target_ids = target_ids[:,-target_len:-1]
            if neighborhood:
                return ans.squeeze().detach().cpu().numpy().tolist()
            return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    elif 'llama' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,1:]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    else:
        target_ids = tokenizer(' ' + target + '\n', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,:-1]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()