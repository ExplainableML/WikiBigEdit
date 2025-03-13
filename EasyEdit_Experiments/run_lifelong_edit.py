import logging
import os.path
import sys

import numpy as np
import pandas as pd

import torch
import gc

sys.path.append('')
import json
import random
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf as oc
import wandb
from easyeditor import (
    FTHyperParams, 
    IKEHyperParams, 
    KNHyperParams, 
    MEMITHyperParams, 
    ROMEHyperParams, 
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams,
    GraceHyperParams,
    R_ROMEHyperParams,
    WISEHyperParams,
    RAGHyperParams,
    PTuningHyperParams
    )
from easyeditor import BaseEditor, LifelongEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import ZsreDataset

INCREMENTS = ['20240201_20240220', '20240220_20240301', '20240301_20240320', '20240320_20240401',
                 '20240401_20240501', '20240501_20240601', '20240601_20240620', '20240620_20240701']

@hydra.main(version_base=None, config_path="hydra/experiments")
def main(args: DictConfig) -> None:
    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'SERAC':
        editing_hparams = SERACHparams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'R-ROME':
        editing_hparams = R_ROMEHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'EVAL':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'RAG':
        editing_hparams = RAGHyperParams
    elif args.editing_method == 'PTuning':
        editing_hparams = PTuningHyperParams
    else:
        raise NotImplementedError

    #if len(args.qa.increments) == 1:
    #    args.qa.increments = [args.qa.increments]
    for i, increment in enumerate(args.qa.increments):
        if isinstance(increment, int):
            args.qa.increments[i] = INCREMENTS[increment]

    print(f'Args: {args}')

    random.seed(args.ds_seed)
    np.random.seed(args.ds_seed)

    hparams = editing_hparams.from_hparams(config=args.hparams)

    if args.editing_method == 'IKE':  # TODO create a training set
        train_data_path = os.path.join(args.data_dir, 'zsre_mend_train_10000.json')
        train_ds = ZsreDataset(train_data_path)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        encode_ike_facts(sentence_model, train_ds, hparams)
    else:
        train_ds = None

    editor = LifelongEditor.from_hparams(hparams)
    edited_model = None

    if args.checkpoint.load:
        mode = args.qa.modes[0] if len(args.qa.modes) == 1 else args.qa.modes_combine
        outstanding_increments, wandb_run_id = load_checkpoint(args, mode, None if args.editing_method == 'EVAL' else editor.model)
        #args.qa.increments = outstanding_increments
        if wandb_run_id is None:
            print('Cannot proceed wandb run - no checkpoint found')
            wandb_run_id = wandb.util.generate_id()
    else:
        wandb_run_id = wandb.util.generate_id()
        outstanding_increments = args.qa.increments
    print(f'wandb_run_id: {wandb_run_id}')

    if not args.debug:
        os.makedirs(os.path.join(args.metrics_save_dir, args.experiment, args.editing_method), exist_ok=True)
        wandb.init(project=args.wandb.project,
                   entity=args.wandb.entity,
                   name=args.editing_method,
                   id=wandb_run_id,
                   resume='allow',
                   notes=args.wandb.notes,
                   config=oc.to_container(args, resolve=True, throw_on_missing=True),
                   settings=wandb.Settings(_service_wait=300))

    print(f' Running {args.editing_method} on increments: {args.qa.increments}')
    all_data = {}
    for increment in args.qa.increments:
        print(f'++++++++++++ Increment: {increment} ++++++++++++')
        for mode in args.qa.modes:
            test_data = json.load(open(os.path.join(args.data_dir, f'wikibigedit/wiki_big_edit_{increment}.json'), 'r', encoding='utf-8'))

            if args.ds_size > 0:
                random.seed(args.ds_seed)
                test_data = random.sample(test_data, args.ds_size)
            all_data[increment] = test_data
            prompts, rephrase_prompts, target_new, tags, locality_inputs, portability_inputs, subject = extract_data(test_data)

        if len(args.qa.modes) > 1 or args.qa.modes_combine == 'mix':
            prompts, rephrase_prompts, target_new, tags, locality_inputs, portability_inputs, subject = shuffle_inputs(
                prompts, rephrase_prompts, target_new, tags, locality_inputs, portability_inputs, subject
            )
            mode = 'mix'
        elif len(args.qa.modes) > 1 and args.qa.modes_combine == 'concat':
            mode = 'concat'
        else:
            mode = args.qa.modes[0]

        if args.editing_method in ['WISE', 'GRACE']:
            loc_data = json.load(open('./data/zsre_mend_train.json', 'r', encoding='utf-8'))[:len(prompts)]
            loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]
        else:
            loc_prompts = None

        if not increment in outstanding_increments:
            print(f'Increment {increment} already processed')
            continue

        if args.editing_method == 'EVAL':
            summary_metrics, metrics = editor.eval_samples(
                prompts=prompts,
                rephrase_prompts=rephrase_prompts,
                target_new=target_new,
                tags=tags,
                ground_truth=target_new,
                few_shot_examples=args.qa.fs_examples,
                portability_inputs=portability_inputs,
                locality_inputs=locality_inputs,
                batch_size=args.batch_size,
            )
            if not args.debug:
                if args.checkpoint.save:
                    save_checkpoint(args, increment, mode, wandb_run_id)
                eval_metrics_to_wandb(summary_metrics)
            else:
                print(summary_metrics)
        elif args.editing_method == 'RAG' or args.editing_method == 'PTuning':
            metrics, edited_model = editor.rag(
                args=args,
                prompts=prompts,
                rephrase_prompts=rephrase_prompts,
                target_new=target_new,
                tags=tags,
                edited_model=edited_model,
                loc_prompts=loc_prompts,
                subject=subject,
                train_ds=train_ds,
                locality_inputs=locality_inputs,
                portability_inputs=portability_inputs,
                keep_original_weight=True,
                few_shot_examples=args.qa.fs_examples,
                past_eval_interval=args.qa.past_eval_interval,
                verbose=False,
            )

        else:
            metrics, edited_model, _ = editor.edit(
                args=args,
                prompts=prompts,
                rephrase_prompts=rephrase_prompts,
                target_new=target_new,
                tags=tags,
                loc_prompts=loc_prompts,
                subject=subject,
                train_ds=train_ds,
                locality_inputs=locality_inputs,
                portability_inputs=portability_inputs,
                keep_original_weight=False,
                few_shot_examples=args.qa.fs_examples,
                past_eval_interval=args.qa.past_eval_interval,
                verbose=False,
            )

            if args.checkpoint.save and not args.debug:
                save_checkpoint(args, increment, mode, wandb_run_id, model=edited_model)

        if args.qa.past_eval:
            past_metrics = {}
            for ts in all_data.keys():
                if ts == increment:
                    continue
                logging.info(f'Past eval on {ts}')
                #logging.info(f'Memory Size: {len(edited_model.memory)}')
                past_data = all_data[ts]
                past_data = random.sample(past_data, args.qa.past_eval.max_samples)
                prompts, rephrase_prompts, target_new, tags, locality_inputs, portability_inputs, subject = extract_data(past_data)

                summary_metrics, metrics = editor.eval_samples(
                    prompts=prompts,
                    rephrase_prompts=rephrase_prompts,
                    target_new=target_new,
                    tags=tags,
                    ground_truth=target_new,
                    few_shot_examples=False if args.editing_method == 'RAG' else args.qa.fs_examples,
                    locality_inputs=locality_inputs,
                    portability_inputs=portability_inputs,
                    edited_model=edited_model,
                    batch_size=args.batch_size,
                )
                past_metrics[f'past_{ts}'] = summary_metrics
            if not args.debug:
                eval_metrics_to_wandb(past_metrics)
            else:
                print(past_metrics)

            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

        #print(metrics)
        if not args.debug:
            os.makedirs(os.path.join(args.metrics_save_dir, args.experiment, args.editing_method,
                                        f'data_{args.ds_size}_{args.ds_seed}'), exist_ok=True)
            json.dump(metrics, open(os.path.join(args.metrics_save_dir, args.experiment, args.editing_method,
                                                 f'data_{args.ds_size}_{args.ds_seed}',
                                                 f'{increment}_{mode}_{args.qa.model}_results.json'), 'w'), indent=4)
        if hasattr(edited_model, 'out') and len(edited_model.out) > 0:
            data = pd.DataFrame(edited_model.out, columns=['context', 'question', 'answers'])
            data.to_csv(os.path.join(args.metrics_save_dir, args.experiment, args.editing_method,
                                     f'data_{args.ds_size}_{args.ds_seed}',
                                     f'{increment}_{mode}_{args.qa.model}_results.csv'), index=False)


def extract_data(test_data):
    prompts, rephrase_prompts, target_new, subject, tags = [], [], [], [], []
    locality_inputs = {
        'neighborhood': {
            'prompt': [],
            'ground_truth': [],
        },
    }
    portability_inputs = {
        'mhop': {
            'prompt': [],
            'ground_truth': [],
        },
    }
    prompts += [test_data_['update'] for test_data_ in test_data]
    rephrase_prompts += [edit_data_['rephrase'] for edit_data_ in test_data]
    locality_prompts = [edit_data_['loc'] for edit_data_ in test_data]
    target_new += [edit_data_['ans'] for edit_data_ in test_data]
    locality_ans = [edit_data_['loc_ans'] for edit_data_ in test_data]
    tags += [edit_data_['tag'] for edit_data_ in test_data]
    mhop_questions = [edit_data_['mhop'] for edit_data_ in test_data]
    mhop_answers = [edit_data_['mhop_ans'] for edit_data_ in test_data]
    personas_questions = {}
    for key in test_data[0].keys():
        if key.startswith('personas'):
            personas_questions[key] = [edit_data_[key] for edit_data_ in test_data]
            if key not in portability_inputs:
                portability_inputs[key] = {
                    'prompt': [],
                    'ground_truth': [],
                }

    locality_inputs['neighborhood']['prompt'] += locality_prompts
    locality_inputs['neighborhood']['ground_truth'] += locality_ans
    portability_inputs['mhop']['prompt'] += mhop_questions
    portability_inputs['mhop']['ground_truth'] += mhop_answers
    for key in personas_questions.keys():
        portability_inputs[key]['prompt'] += personas_questions[key]
        portability_inputs[key]['ground_truth'] += [edit_data_['alt'] for edit_data_ in test_data]

    subject += [edit_data_['subject'] for edit_data_ in test_data]
    return prompts, rephrase_prompts, target_new, tags, locality_inputs, portability_inputs, subject


def shuffle_inputs(prompts, rephrase_prompts, target_new, tags, locality_inputs, portability_inputs, subject):
    shuffle_idx = list(range(len(prompts)))
    random.shuffle(shuffle_idx)
    prompts = [prompts[i] for i in shuffle_idx]
    rephrase_prompts = [rephrase_prompts[i] for i in shuffle_idx]
    target_new = [target_new[i] for i in shuffle_idx]
    locality_inputs['neighborhood']['prompt'] = [locality_inputs['neighborhood']['prompt'][i] for i in shuffle_idx]
    locality_inputs['neighborhood']['ground_truth'] = [locality_inputs['neighborhood']['ground_truth'][i] for i in
                                                       shuffle_idx]
    subject = [subject[i] for i in shuffle_idx]
    tags = [tags[i] for i in shuffle_idx]
    for key in portability_inputs.keys():
        portability_inputs[key]['prompt'] = [portability_inputs[key]['prompt'][i] for i in shuffle_idx]
        portability_inputs[key]['ground_truth'] = [portability_inputs[key]['ground_truth'][i] for i in shuffle_idx]
    return prompts, rephrase_prompts, target_new, tags, locality_inputs, portability_inputs, subject


def save_checkpoint(args, increment, mode, wandb_run_id, model=None):
    checkpoint_dir = f'{args.checkpoint.save_dir}/{args.experiment}/{args.editing_method}/data_{args.ds_size}_{args.ds_seed}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    if args.checkpoint.save:
        # args to dict
        args_dict = oc.to_container(args, resolve=True, throw_on_missing=True)
        # add wandb_run_id to args_dict
        args_dict.update({'wandb_run_id': wandb_run_id})

        with open(os.path.join(checkpoint_dir, f'{increment}_{mode}_{args.qa.model}_args.json'), 'w') as f:
            json.dump(args_dict, f, indent=4)
        print(f'Saved args for increment {increment} to {checkpoint_dir}/{increment}_{mode}_{args.qa.model}_args.json')
        if model is not None:
            model.save_model(os.path.join(checkpoint_dir, f'{increment}_{mode}_{args.qa.model}'))


def load_checkpoint(args, mode, model=None):
    checkpoint_dir = f'{args.checkpoint.save_dir}/{args.experiment}/{args.editing_method}/data_{args.ds_size}_{args.ds_seed}'
    if args.checkpoint.load:
        wandb_run_id = None
        outstanding_increments = []
        for increment in args.qa.increments[::-1]:
            try:
                with open(os.path.join(checkpoint_dir, f'{increment}_{mode}_{args.qa.model}_args.json'), 'r') as f:
                    ckp_args = json.load(f)
                wandb_run_id = ckp_args['wandb_run_id']
                if model is not None:
                    model.load_model(
                        os.path.join(checkpoint_dir, f'{increment}_{mode}_{args.qa.model}'))
                print(f'Loaded checkpoint for increment {increment}')
                break
            except FileNotFoundError:
                outstanding_increments.append(increment)
                continue
        outstanding_increments = outstanding_increments[::-1]
        return outstanding_increments, wandb_run_id
    else:
        return None, None, None


def metrics_to_wandb(metrics):
    mean_log = {
        'pre_rewrite_acc_mean': [],
        'post_rewrite_acc_mean': [],
        'post_locality_mean': [],
        'pre_rephrase_acc_mean': [],
        'post_rephrase_acc_mean': [],
    }
    for m in metrics:
        mean_log['pre_rewrite_acc_mean'].append(m['pre']['rewrite_acc'][0])
        mean_log['post_rewrite_acc_mean'].append(m['post']['rewrite_acc'][0])
        mean_log['post_locality_mean'].append(m['post']['locality']['neighborhood_acc'][0])
        mean_log['pre_rephrase_acc_mean'].append(m['pre']['rephrase_acc'][0])
        mean_log['post_rephrase_acc_mean'].append(m['post']['rephrase_acc'][0])
        m_log = {
            'pre_rewrite_acc': m['pre']['rewrite_acc'][0],
            'post_rewrite_acc': m['post']['rewrite_acc'][0],
            'post_locality': m['post']['locality']['neighborhood_acc'][0],
            'pre_rephrase_acc': m['pre']['rephrase_acc'][0],
            'post_rephrase_acc': m['post']['rephrase_acc'][0],
        }
        wandb.log(m_log)
    mean_log = {k: sum(v) / len(v) for k, v in mean_log.items()}
    print(mean_log)
    wandb.log(mean_log)


def eval_metrics_to_wandb(metrics):
    wandb.log(metrics)


if __name__ == "__main__":
    main()