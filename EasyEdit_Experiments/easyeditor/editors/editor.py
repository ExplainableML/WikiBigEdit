import sys
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
import torch.nn as nn
import wandb
from tqdm import tqdm
import json
import torch
import numpy as np
import random
import math
import argparse
from ..models.melo.melo import LORA
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from accelerate import Accelerator
from ..util.globals import *
from .utils import _chunks, _prepare_requests, summary_metrics
from .batch_editor import BatchEditor
from ..evaluate import compute_edit_quality, compute_icl_edit_quality, compute_sent_metric, metrics_to_wandb
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *
from ..util.load_data import load_dataset
from ..evaluate.evaluate_utils import test_generation_quality, extract_metric
from ..models.rag.rag_main import RAG
from ..models.ptuning.ptuning_main import PromptTuning
from .utils import GPTWrapper

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

LOG = logging.getLogger(__name__)


def make_logs():
    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything(42)


class BaseEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):
        return cls(hparams)

    def __init__(self, hparams: HyperParams):
        assert hparams is not None, 'Error: hparams is None.'
        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name
        self.accelerator = None
        make_logs()
        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
            if 't5' in self.model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch_dtype,
                                                                        device_map=device_map)
                self.tok = T5Tokenizer.from_pretrained(self.model_name)
            elif 'gpt2' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype,
                                                                  device_map=device_map)
                self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'gpt' in self.model_name.lower():
                self.tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
                emb_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch_dtype,
                                                                  device_map=device_map)
                self.tok.pad_token_id = self.tok.eos_token_id
                self.model = GPTWrapper(self.model_name, self.tok, emb_model)
            elif 'llama' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype,
                                                                  device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, padding_side='left')
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'baichuan' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype,
                                                                  trust_remote_code=True, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'chatglm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype=torch_dtype,
                                                       device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.tok.unk_token_id = 64787
                # self.tok.pad_token_id = self.tok.eos_token_id
            elif 'internlm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype=torch_dtype,
                                                       device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'qwen2' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True,
                                                                  torch_dtype=torch_dtype if hparams.alg_name not in [
                                                                      'MEND'] else torch.bfloat16,
                                                                  device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>',
                                                         pad_token='<|endoftext|>', unk_token='<|endoftext|>',
                                                         trust_remote_code=True)
            elif 'qwen' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, fp32=False, trust_remote_code=True,
                                                                  device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>',
                                                         pad_token='<|endoftext|>', unk_token='<|endoftext|>',
                                                         trust_remote_code=True)
            elif 'mistral' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype,
                                                                  device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'gemma' in self.model_name.lower():
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=device_map,
                                                                  torch_dtype=torch_dtype)
            elif 'xgen' in self.model_name.lower():
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, padding_side='left')
                                                         #pad_token='<|endoftext|>')
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=device_map,
                                                                  torch_dtype=torch_dtype)
                self.tok.pad_token_id = 28
                #if self.tok.pad_token is None:
                #    print("Adding [PAD] token to the tokenizer...")
                #    self.tok.add_special_tokens({'pad_token': '[PAD]'})
                #    self.model.resize_token_embeddings(len(self.tok))

                #print(self.tok.pad_token)
                #print(self.tok.pad_token_id)
                #print(len(self.tok))
                #print(self.tok.encode('=≠*'))
                #print(f'Embedding size: {self.model.get_input_embeddings().weight.size()}')
                #assert self.tok.pad_token in self.tok.get_vocab(), "Padding token not added to the vocabulary!"
                #print(self.tok.decode([self.tok.pad_token_id]))
                #sys.exit()
            elif 'phi' in self.model_name.lower():
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=device_map,
                                                                  trust_remote_code=True, _attn_implementation='eager')
            else:
                raise NotImplementedError

            if self.tok is not None and (
                    isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(
                    self.tok, LlamaTokenizer)) and (hparams.alg_name not in ['ROME', 'MEMIT', 'EMMET', 'R-ROME']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'
            if self.tok is not None and (
                    'mistral' in self.model_name.lower() or 'llama' in self.model_name.lower() or 'qwen' in self.model_name.lower()) and (
                    hparams.alg_name in ['ROME', 'MEMIT', 'EMMET', 'R-ROME']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
                self.tok.padding_side = 'right'
        else:
            self.model, self.tok = self.model_name

        if hparams.model_parallel:
            hparams.device = str(self.model.device).split(":")[1]
            #print(f"CUDA:0 Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
            #print(f"CUDA:1 Memory Allocated: {torch.cuda.memory_allocated(1) / 1024 ** 3:.2f} GB")
            #self.accelerator = Accelerator()
            #self.model.to(self.accelerator.device)
            #self.model = self.accelerator.prepare(self.model)
        if not hparams.model_parallel and hasattr(hparams, 'device'):
            self.model.to(f'cuda:{hparams.device}')

        # Enable Data Parallel if specified
        #if hasattr(hparams, 'data_parallel') and hparams.data_parallel:
        #    LOG.info("Wrapping model in DataParallel for multi-GPU training...")
        #    self.model = torch.nn.DataParallel(self.model)

        print(
            f"Model {self.model_name} instantiated on device {self.model.device} (model parallel: {hparams.model_parallel})")
        # check the dtype the llm is using
        if hasattr(self.model, 'dtype'):
            LOG.info(f"Model dtype: {self.model.dtype}")

        self.hparams = hparams

    def edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             locality_inputs: Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             sequential_edit=False,
             verbose=True,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        test_generation = kwargs.pop('test_generation', False)

        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts, ], [target_new, ]

        if hasattr(self.hparams, 'batch_size') and not BatchEditor.is_batchable_method(
                self.alg_name):  # For Singleton Editing, bs=1
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'

        if ground_truth is not None:
            ground_truth = [ground_truth, ] if isinstance(ground_truth, str) else ground_truth
        else:  # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>'] * (len(prompts))

        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = _prepare_requests(prompts, target_new, ground_truth, rephrase_prompts, locality_inputs,
                                         portability_inputs, **kwargs)

        return self.edit_requests(requests, sequential_edit, verbose, test_generation=test_generation, **kwargs)

    def batch_edit(self,
                   prompts: List[str],
                   target_new: List[str],
                   ground_truth: Optional[List[str]] = None,
                   rephrase_prompts: Optional[List[str]] = None,
                   locality_prompts: Optional[List[str]] = None,
                   locality_ground_truth: Optional[List[str]] = None,
                   keep_original_weight=False,
                   verbose=True,
                   **kwargs
                   ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        few_shot_examples = kwargs['few_shot_examples'] if 'few_shot_examples' in kwargs.keys() else None
        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth, ]
            else:
                assert len(ground_truth) == len(prompts)
        else:  # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        assert BatchEditor.is_batchable_method(
            self.alg_name), f'The Method {self.alg_name} can not batch edit examples.'

        requests = _prepare_requests(prompts, target_new, ground_truth, rephrase_prompts,
                                     locality_prompts, locality_ground_truth, **kwargs)

        assert hasattr(self.hparams, 'batch_size'), f'Method {self.alg_name} found, pls specify the batch_size....'
        all_metrics = []
        for record_chunks in _chunks(requests, self.hparams.batch_size):
            start = time()

            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                record_chunks,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight,
            )
            exec_time = time() - start
            LOG.info(f"Execution editing took {exec_time}")

            start = time()
            chunk_metrics = []
            for i, request in enumerate(record_chunks):
                metrics = {
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request,
                                                 self.hparams.device, test_generation=test_generation,
                                                 few_shot_examples=few_shot_examples)
                }

                chunk_metrics.append(metrics)

            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

            for i, request in enumerate(record_chunks):
                chunk_metrics[i]["pre"] = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok,
                                                               request, self.hparams.device,
                                                               test_generation=test_generation,
                                                               few_shot_examples=few_shot_examples)

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {chunk_metrics[i]}"
                    )

            LOG.info(f"Evaluation took {time() - start}")
            all_metrics.extend(chunk_metrics)
        return all_metrics, edited_model, weights_copy

    def edit_requests(self,
                      requests,
                      sequential_edit=False,
                      verbose=True,
                      test_generation=False,
                      **kwargs
                      ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        eval_metric = kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        few_shot_examples = kwargs['few_shot_examples'] if 'few_shot_examples' in kwargs.keys() else None
        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'
        all_metrics = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        else:
            for i, request in enumerate(tqdm(requests)):
                if self.alg_name == 'IKE':
                    assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                    metrics = {
                        "pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                        request, self.hparams.device, pre_edit=True)}
                else:
                    metrics = {"pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                                                           self.hparams.device, eval_metric=eval_metric,
                                                           test_generation=test_generation,
                                                           few_shot_examples=few_shot_examples)}
                all_metrics.append(metrics)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)

        def edit_func(request):
            if self.alg_name == 'IKE':
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                icl_examples = None
            return edited_model, weights_copy, icl_examples

        def edit_evaluation(all_metrics, request, edited_model, idx, test_generation, icl_examples, **kwargs):
            eval_metric = kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
            few_shot_examples = kwargs['few_shot_examples'] if 'few_shot_examples' in kwargs.keys() else None
            if self.alg_name == 'IKE':
                all_metrics[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                     request, self.hparams.device),
                })
            else:
                all_metrics[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request,
                                                 self.hparams.device, eval_metric=eval_metric,
                                                 test_generation=test_generation, few_shot_examples=few_shot_examples)
                })
                if "metric_kwargs" in kwargs:
                    all_metrics[idx].update(
                        compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok,
                                            metric_kwargs=kwargs["metric_kwargs"][idx], device=self.hparams.device))
                if 'locality' in all_metrics[idx]['post'].keys():
                    for locality_key in request['locality'].keys():
                        locality_result = []
                        for ans, label in zip(all_metrics[idx]['post']['locality'][f'{locality_key}_output'],
                                              all_metrics[idx]['pre']['locality'][f'{locality_key}_output']):
                            locality_result.append(np.mean(np.equal(ans, label)))
                        all_metrics[idx]['post']['locality'][f'{locality_key}_acc'] = locality_result
                        all_metrics[idx]['post']['locality'].pop(f'{locality_key}_output')
                    all_metrics[idx]['pre'].pop('locality')

            if verbose:
                LOG.info(f"{idx} editing: {request['prompt']} -> {request['target_new']}  \n\n {all_metrics[idx]}")

        if sequential_edit:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
            for i, request in enumerate(requests):
                edit_evaluation(all_metrics, request, edited_model, i, test_generation, icl_examples, **kwargs)
        else:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
                edit_evaluation(all_metrics, request, edited_model, i, test_generation, icl_examples, **kwargs)
                if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                    with torch.no_grad():
                        weights_copy()
                elif self.alg_name == 'LoRA':
                    edited_model.unload()
                    del self.model.peft_config
                elif self.alg_name == 'MELO':
                    self.model = edited_model
                elif self.alg_name == 'LoRA':
                    self.model = edited_model
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        if isinstance(edited_model, LORA):
            edited_model = edited_model.model
        if len(all_metrics) != 0:
            summary_metrics(all_metrics)

        return all_metrics, edited_model, weights_copy

    def normal_edit(
            self,
            prompts: List[str],
            target_new: List[str],
            sequential_edit=False,
    ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        """
        assert len(prompts) == len(target_new)
        ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        assert BatchEditor.is_batchable_method(
            self.alg_name), f'The Method {self.alg_name} can not batch edit examples.'

        requests = _prepare_requests(prompts, target_new, ground_truth)

        assert hasattr(self.hparams, 'batch_size'), f'Method {self.alg_name} found, pls specify the batch_size....'

        # print(f"[editor.py][batch_edit] `batch_size`={self.hparams.batch_size}")
        # for epc in range(epoch):
        #     print(f"[editor.py][batch_edit] `Epoch` = {epc+1}")
        #     for record_chunks in self._chunks(requests, self.hparams.batch_size):
        start = time()

        edited_model, weights_copy = self.apply_algo(
            self.model,
            self.tok,
            requests,  # record_chunks -> requests
            self.hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=False,
        )
        exec_time = time() - start
        LOG.info(f"Execution editing took {exec_time}")

        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        return None, edited_model, weights_copy


class LifelongEditor(BaseEditor):
    def __init__(self,
                 hparams: HyperParams,
                 ):
        super().__init__(hparams)
        self.qa_benchmarks = {}

    def eval_samples(
            self,
            prompts: List[str],
            target_new: List[str],
            tags: Optional[List[str]] = None,
            locality_inputs: Optional[Dict] = None,
            portability_inputs: Optional[Dict] = None,
            ground_truth: Optional[List[str]] = None,
            rephrase_prompts: Optional[List[str]] = None,
            edited_model: Optional[nn.Module] = None,
            batch_size: int = 1,
            **kwargs
    ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        assert len(prompts) == len(target_new)
        few_shot_examples = kwargs['few_shot_examples'] if 'few_shot_examples' in kwargs.keys() else None
        avg_edit_decay = kwargs['avg_edit_decay'] if 'avg_edit_decay' in kwargs.keys() else None
        indices = kwargs['indices'] if 'indices' in kwargs.keys() else None
        if not edited_model:
            print("No edited model found, using the original model for evaluation.")
            edited_model = self.model

        tags = tags if tags is not None else [''] * len(prompts)
        requests = _prepare_requests(prompts, target_new, ground_truth, tags, rephrase_prompts,
                                     locality_inputs, portability_inputs, **kwargs)

        batched_requests = [requests[i * batch_size: min((i + 1) * batch_size, len(requests))]
                            for i in range(math.ceil(len(requests) / batch_size))]
        all_metrics = []
        for requests in tqdm(batched_requests):
            past_metrics = compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, requests,
                                                self.hparams.device, test_generation=False,
                                                few_shot_examples=few_shot_examples)

            all_metrics.extend([{'past': pm} for pm in past_metrics])

        metric_names = [['past', 'rewrite_acc'],
                        ['past', 'locality', 'neighborhood_acc'],
                        ['past', 'portability', 'mhop', 'performance', 'acc'],
                        ['past', 'portability', 'genv2_mixed', 'performance', 'acc'],
                        ['past', 'rephrase_acc']]

        mean_log = {f'{"_".join(k)}_mean': [] for k in metric_names}

        for j, m in enumerate(all_metrics):
            for k in metric_names:
                try:
                    mean_log[f'{"_".join(k)}_mean'].append(extract_metric(m, k))
                except KeyError:
                    pass

        out_metrics = {k: np.mean(v) for k, v in mean_log.items()}

        return out_metrics, all_metrics

    def eval_general_capabilities(self, args: argparse.Namespace):
        start_time = time()
        datasets = args.qa.base_eval.datasets
        metrics = {}
        for dataset in datasets:
            if self.qa_benchmarks is not None and dataset in self.qa_benchmarks.keys():
                questions, answers = self.qa_benchmarks[dataset]
            else:
                questions, answers = load_dataset(dataset, ds_size=args.qa.base_eval.ds_size,
                                                  ds_seed=args.qa.base_eval.ds_seed,
                                                  fs_ex=args.qa.base_eval.fs_examples,
                                                  data_dir=os.path.join(args.data_dir, 'qa_benchmarks'))
                self.qa_benchmarks[dataset] = (questions, answers)
            tags = [''] * len(questions)
            requests = _prepare_requests(questions, answers, ground_truth=answers, tags=tags)
            batched_requests = [requests[i * args.batch_size: min((i + 1) * args.batch_size, len(requests))]
                                for i in range(math.ceil(len(requests) / args.batch_size))]

            dataset_metrics = []
            for requests in tqdm(batched_requests):
                dataset_metrics.extend(
                    compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, requests,
                                         self.hparams.device, test_generation=False))

            metrics[dataset] = np.mean([metric['rewrite_acc'] for metric in dataset_metrics])

        metrics['mean'] = np.mean(list(metrics.values()))
        print(f"General Capabilities Evaluation took {time() - start_time} seconds")
        return metrics

    def edit(self,
             args: argparse.Namespace,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             tags: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             locality_inputs: Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             keep_original_weight=False,
             verbose=True,
             summary_metrics=False,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        seed_everything(args.seed)
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        few_shot_examples = kwargs['few_shot_examples'] if 'few_shot_examples' in kwargs.keys() else None
        past_eval_interval = kwargs['past_eval_interval'] if 'past_eval_interval' in kwargs.keys() else 1
        avg_edit_decay = {}
        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts, ], [target_new, ]

        #if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
        #    self.hparams.batch_size = 1

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth, ]
            else:
                assert len(ground_truth) == len(prompts)
        else:  # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        # assert (locality_prompts is None and locality_ground_truth is None) or \
        #        (isinstance(locality_prompts, str) and isinstance(locality_ground_truth, str)) or \
        #        len(locality_prompts) == len(locality_ground_truth) or print('Error in locality Input.')
        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            tags = tags if tags is not None else [''] * len(prompts)
            requests = _prepare_requests(prompts, target_new, ground_truth, tags, rephrase_prompts,
                                         locality_inputs, portability_inputs, **kwargs)

        if args.qa.eval_max_samples:
            eval_requests = subset_requests(requests, args.qa.eval_max_samples, seed=args.ds_seed)
        else:
            eval_requests = requests
            random.seed(args.ds_seed)
            random.shuffle(eval_requests)

        batched_requests = [requests[i * args.batch_size: min((i + 1) * args.batch_size, len(requests))]
                            for i in range(math.ceil(len(requests) / args.batch_size))]
        if hasattr(args, 'eval_batch_size'):
            batched_eval_requests = [eval_requests[i * args.eval_batch_size: min((i + 1) * args.eval_batch_size, len(eval_requests))]
                                        for i in range(math.ceil(len(eval_requests) / args.eval_batch_size))]
        else:
            batched_eval_requests = batched_requests

        # if not os.path.exists(RESULTS_DIR):
        #     os.mkdir(RESULTS_DIR)
        # base_case_path = RESULTS_DIR / self.hparams_fname.rsplit('.', 1)[0]
        # if not os.path.exists(base_case_path):
        #     os.mkdir(base_case_path)
        # print(f"Results will be stored at {base_case_path}")

        if self.alg_name == 'FT-Api':
            all_metrics = []
            for i, request in enumerate(requests):
                metrics = {
                    "pre": {}
                }
                all_metrics.append(metrics)

            start = time()
            edited_model, weights_copy = self.apply_algo(
                requests,
                self.hparams
            )
            exec_time = time() - start

            LOG.info(f"Execution editing took {exec_time}")

            for i, request in enumerate(requests):
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": {}
                })

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )
            return all_metrics, edited_model, weights_copy

        all_metrics = []
        logging.info(f"Pre Edit Evaluation")
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        else:
            print(f'Evaluate {len(batched_eval_requests)} batches')
            for i, batch in enumerate(tqdm(batched_eval_requests)):
                if self.alg_name == 'IKE':
                    assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                    pre_metrics = compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                           batch, self.hparams.device, pre_edit=True)
                else:
                    #pre_metrics = [{} for _ in range(len(batch))]
                    pre_metrics = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, batch,
                                                       self.hparams.device, test_generation=test_generation,
                                                       few_shot_examples=few_shot_examples)

                for b, p_m in enumerate(pre_metrics):
                    all_metrics.append({
                        "pre": p_m
                    })

        logging.info(f"Editing")
        if self.alg_name == 'LoRA':
            start = time()
            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                requests,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight,
                train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
            )
            self.model = edited_model
            exec_time = time() - start
        for i, request in enumerate(tqdm(batched_requests)):
            start = time()
            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds']
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
            elif self.alg_name == 'LoRA':
                pass
            else:
                if not (args.qa.filter_edits and all_metrics[i]['pre']['rewrite_acc'][0] == 1.0):
                    edited_model, weights_copy = self.apply_algo(
                        self.model,
                        self.tok,
                        request,
                        self.hparams,
                        copy=False,
                        return_orig_weights=True,
                        keep_original_weight=keep_original_weight,
                        train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                    )
                    exec_time = time() - start
                    LOG.info(f"Execution {i} editing took {exec_time}")
                    self.model = edited_model.model if self.hparams.alg_name in ['WISE', 'GRACE'] else self.model
            if verbose:
                LOG.info(
                    f"{i} editing: {[r['prompt'] for r in request]} -> {[r['target_new'] for r in request]}"
                )

            if hasattr(args, 'eval_batch_size'):
                batched_eval_requests = [
                    request[i * args.eval_batch_size: min((i + 1) * args.eval_batch_size, len(request))]
                    for i in range(math.ceil(len(request) / args.eval_batch_size))]
            else:
                batched_eval_requests = [request, ]

            post_metrics = []
            for eval_batch in batched_eval_requests:
                if self.alg_name == 'IKE':
                    post_metrics.extend(compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok,
                                                            icl_examples, eval_batch, self.hparams.device))
                else:
                    post_metrics.extend(compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, eval_batch,
                                                        self.hparams.device, test_generation=test_generation,
                                                        few_shot_examples=few_shot_examples))

            for b, p_m in enumerate(post_metrics):
                idx = i * args.batch_size + b
                all_metrics[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request[b],
                    "time": exec_time,
                    "post": p_m
                })

                if 'locality' in all_metrics[idx]['post'].keys():
                    for locality_key in request[b]['locality'].keys():
                        assert len(all_metrics[idx]['post']['locality'][f'{locality_key}_output']) == \
                               len(all_metrics[idx]['pre']['locality'][f'{locality_key}_output'])
                        locality_result = []
                        for ans, label in zip(all_metrics[idx]['post']['locality'][f'{locality_key}_output'],
                                              all_metrics[idx]['pre']['locality'][f'{locality_key}_output']):
                            try:
                                locality_result.append(np.mean(np.equal(ans, label)))
                            except ValueError as e:
                                print(f'Error in {idx} - {locality_key} - {ans} - {label}')
                                print(all_metrics[idx])
                                raise e
                        all_metrics[idx]['post']['locality'][f'{locality_key}_score'] = locality_result
                        all_metrics[idx]['post']['locality'].pop(f'{locality_key}_output')
                        all_metrics[idx]['pre']['locality'].pop(f'{locality_key}_output')

                if idx == len(requests) - 1:
                    if args.qa.base_eval:
                        all_metrics[idx]['general'] = self.eval_general_capabilities(args)
                    compute_mean = True
                else:
                    compute_mean = False
                metrics_to_wandb(all_metrics, idx, avg_edit_decay=None, compute_mean=compute_mean)

        if isinstance(edited_model, LORA):
            edited_model = edited_model.model
        # for melo

        if summary_metrics and len(all_metrics) != 0:
            if isinstance(all_metrics, dict):
                all_metrics = [all_metrics, ]
            logs_dir = './logs'
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
            output_file = os.path.join(logs_dir, 'results.json')
            with open(output_file, 'w') as f:
                json.dump(all_metrics, f, ensure_ascii=False, indent=4)

            mean_metrics = dict()
            for eval in ["pre", "post"]:
                mean_metrics[eval] = dict()
                for key in ["rewrite_acc", "rephrase_acc"]:
                    if key in all_metrics[0][eval].keys():
                        mean_metrics[eval][key] = np.mean([metric[eval][key] for metric in all_metrics])
                for key in ["locality", "portability"]:
                    if key in all_metrics[0][eval].keys() and all_metrics[0][eval][key] != {}:
                        mean_metrics[eval][key] = dict()
                        for lkey in all_metrics[0][eval][key].keys():
                            if lkey.endswith("acc"):
                                mean_metrics[eval][key][lkey] = np.mean(
                                    [metric[eval][key][lkey] for metric in all_metrics])
            mean_metrics["time"] = np.mean([metric["time"] for metric in all_metrics])

            print("Metrics Summary: ", mean_metrics)

        return all_metrics, edited_model, weights_copy

    def rag(self,
            args: argparse.Namespace,
            prompts: Union[str, List[str]],
            target_new: Union[str, List[str]],
            ground_truth: Optional[Union[str, List[str]]] = None,
            tags: Optional[Union[str, List[str]]] = None,
            rephrase_prompts: Optional[Union[str, List[str]]] = None,
            locality_inputs: Optional[Dict] = None,
            portability_inputs: Optional[Dict] = None,
            edited_model=None,
            keep_original_weight=False,
            verbose=True,
            summary_metrics=False,
            **kwargs
            ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        seed_everything(args.seed)
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        few_shot_examples = kwargs['few_shot_examples'] if 'few_shot_examples' in kwargs.keys() else None
        past_eval_interval = kwargs['past_eval_interval'] if 'past_eval_interval' in kwargs.keys() else 1
        if edited_model is None:
            if self.alg_name == 'RAG':
                edited_model = RAG(self.hparams, self.model, self.tok, self.hparams.device)
            elif self.alg_name == 'PTuning':
                edited_model = PromptTuning(self.hparams, self.model, self.tok, self.hparams.device)

        avg_edit_decay = {}
        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts, ], [target_new, ]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth, ]
            else:
                assert len(ground_truth) == len(prompts)
        else:  # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        # assert (locality_prompts is None and locality_ground_truth is None) or \
        #        (isinstance(locality_prompts, str) and isinstance(locality_ground_truth, str)) or \
        #        len(locality_prompts) == len(locality_ground_truth) or print('Error in locality Input.')
        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            tags = tags if tags is not None else [''] * len(prompts)
            requests = _prepare_requests(prompts, target_new, ground_truth, tags, rephrase_prompts,
                                         locality_inputs, portability_inputs, **kwargs)

        if args.qa.eval_max_samples:
            eval_requests = subset_requests(requests, args.qa.eval_max_samples, seed=args.ds_seed)
        else:
            eval_requests = requests
            random.seed(args.ds_seed)
            random.shuffle(eval_requests)

        batched_requests = [requests[i * args.batch_size: min((i + 1) * args.batch_size, len(requests))]
                            for i in range(math.ceil(len(requests) / args.batch_size))]
        batched_eval_requests = [eval_requests[i * args.batch_size: min((i + 1) * args.batch_size, len(eval_requests))]
                                    for i in range(math.ceil(len(eval_requests) / args.batch_size))]

        all_metrics = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        else:
            for i, batch in enumerate(tqdm(batched_eval_requests)):
                if self.alg_name == 'IKE':
                    assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                    pre_metrics = compute_icl_edit_quality(edited_model, self.model_name, self.hparams, self.tok, [''],
                                                           batch, self.hparams.device, pre_edit=True)
                else:
                    start = time()
                    pre_metrics = compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, batch,
                                                       self.hparams.device, test_generation=test_generation,
                                                       few_shot_examples=few_shot_examples)
                    #pre_metrics = [{} for j in range(len(batch))]
                    #print(f'Batch: {batch}')
                    #print(f'Pre-Edit Metrics: {pre_metrics}')
                    #print(f"Pre-editing eval took {time() - start}")
                    #if i == 19:
                    #sys.exit()

                for b, p_m in enumerate(pre_metrics):
                    all_metrics.append({
                        "pre": p_m
                    })

            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                ### Store the pre_edit metric to refrain computing repeatedly
                json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)

        LOG.info(f'Adapt')
        exec_times = edited_model.adapt(batched_requests)

        LOG.info(f'Evaluation')
        for i, request in enumerate(tqdm(batched_eval_requests)):
            post_metrics = compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request,
                                                self.hparams.device, test_generation=test_generation,
                                                few_shot_examples=False)

            for b, p_m in enumerate(post_metrics):
                idx = i * args.batch_size + b
                all_metrics[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request[b],
                    "time": exec_times[i],
                    "post": p_m
                })

                if "metric_kwargs" in kwargs:
                    all_metrics[idx].update(
                        compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok,
                                            metric_kwargs=kwargs["metric_kwargs"][i], device=self.hparams.device))
                #print(f'id: {idx} - {len(all_metrics)}')
                #avg_edit_decay[idx] = [all_metrics[idx]['post']['rewrite_acc']]

                if 'locality' in all_metrics[idx]['post'].keys():
                    for locality_key in request[b]['locality'].keys():
                        assert len(all_metrics[idx]['post']['locality'][f'{locality_key}_output']) == \
                               len(all_metrics[idx]['pre']['locality'][f'{locality_key}_output'])
                        locality_result = []
                        for ans, label in zip(all_metrics[idx]['post']['locality'][f'{locality_key}_output'],
                                              all_metrics[idx]['pre']['locality'][f'{locality_key}_output']):
                            try:
                                locality_result.append(np.mean(np.equal(ans, label)))
                            except ValueError as e:
                                print(f'Error in {idx} - {locality_key} - {ans} - {label}')
                                locality_result.append(0)
                        all_metrics[idx]['post']['locality'][f'{locality_key}_score'] = locality_result
                        all_metrics[idx]['post']['locality'].pop(f'{locality_key}_output')
                        all_metrics[idx]['pre']['locality'].pop(f'{locality_key}_output')

            self.model = edited_model.model if self.hparams.alg_name in ['WISE', 'GRACE'] else self.model

            """
            if idx % args.qa.past_eval.interval == 0 or idx == len(prompts)-1:
                past_prompts = prompts[:idx + 1]
                past_target_new = target_new[:idx + 1]
                past_ground_truth = ground_truth[:idx + 1]
                past_rephrase_prompts = rephrase_prompts[:idx + 1]
                past_portability_inputs = {'mhop': {k: v[:idx + 1] for k, v in portability_inputs['mhop'].items()}} if portability_inputs is not None else None
                past_subject = kwargs['subject'][:idx + 1] if 'subject' in kwargs.keys() else None
                indices = range(idx + 1)
                if len(past_prompts) > args.qa.past_eval.max_samples and args.qa.past_eval.max_samples > 0 and not idx == len(prompts) - 1:
                    # Randomly sample `max_samples` from the past
                    indices = np.random.choice(len(past_prompts), args.qa.past_eval.max_samples, replace=False)
                    past_prompts = [past_prompts[i] for i in indices]
                    past_target_new = [past_target_new[i] for i in indices]
                    past_ground_truth = [past_ground_truth[i] for i in indices]
                    past_rephrase_prompts = [past_rephrase_prompts[i] for i in indices]
                    past_portability_inputs = [past_portability_inputs[i] for i in indices] if past_portability_inputs is not None else None
                    if past_subject is not None:
                        past_subject = [past_subject[i] for i in indices]

                all_metrics[i]['past'], avg_edit_decay = self.eval_samples(past_prompts, past_target_new,
                                                                           None, None,
                                                                           past_portability_inputs,
                                                                           past_ground_truth,
                                                                           past_rephrase_prompts,
                                                                           subject=past_subject,
                                                                           few_shot_examples=few_shot_examples,
                                                                           indices=indices,
                                                                           batch_size=args.batch_size)
            """

        if args.qa.base_eval:
            all_metrics[-1]['general'] = self.eval_general_capabilities(args)

        LOG.info(f'Evaluation complete, logging to wandb...')
        for idx in tqdm(range(len(all_metrics))):
            metrics_to_wandb(all_metrics, idx, avg_edit_decay=None, compute_mean=idx == len(all_metrics) - 1)

            if verbose:
                LOG.info(
                    f"{idx} editing: {requests[idx]['prompt']} -> {requests[idx]['target_new']}"#  \n {all_metrics[idx]}"
                )
        if summary_metrics and len(all_metrics) != 0:
            if isinstance(all_metrics, dict):
                all_metrics = [all_metrics, ]
            logs_dir = './logs'
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
            output_file = os.path.join(logs_dir, 'results.json')
            with open(output_file, 'w') as f:
                json.dump(all_metrics, f, ensure_ascii=False, indent=4)

            mean_metrics = dict()
            for eval in ["pre", "post"]:
                mean_metrics[eval] = dict()
                for key in ["rewrite_acc", "rephrase_acc"]:
                    if key in all_metrics[0][eval].keys():
                        mean_metrics[eval][key] = np.mean([metric[eval][key] for metric in all_metrics])
                for key in ["locality", "portability"]:
                    if key in all_metrics[0][eval].keys() and all_metrics[0][eval][key] != {}:
                        mean_metrics[eval][key] = dict()
                        for lkey in all_metrics[0][eval][key].keys():
                            if lkey.endswith("acc"):
                                mean_metrics[eval][key][lkey] = np.mean(
                                    [metric[eval][key][lkey] for metric in all_metrics])
            mean_metrics["time"] = np.mean([metric["time"] for metric in all_metrics])

            #print("Metrics Summary: ", mean_metrics)

        return all_metrics, edited_model


    def generate_edit(
            self,
            prompts: Union[str, List[str]],
            target_new: Union[str, List[str]],
            ground_truth: Optional[Union[str, List[str]]] = None,
            rephrase_prompts: Optional[Union[str, List[str]]] = None,
            locality_inputs: Optional[Dict] = None,
            portability_inputs: Optional[Dict] = None,
            sequential_edit=False,
            verbose=True,
            **kwargs
    ):
        eval_metric = kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        test_generation = kwargs.pop('test_generation', False)

        assert len(prompts) == len(target_new)

        if hasattr(self.hparams, 'batch_size'):
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'

        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = _prepare_requests(prompts, target_new, ground_truth, rephrase_prompts, locality_inputs,
                                         portability_inputs, **kwargs)

        def text_generate(
                model,
                model_name,
                hparams: HyperParams,
                tok: AutoTokenizer,
                query,
                device,
                eval_metric: str = 'token_em',
                test_generation=False
        ):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
            text = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tok.encode(text, return_tensors="pt").to(f"cuda:{device}")
            template_length = len(model_inputs[0])
            generated_ids = model.generate(
                input_ids=model_inputs,
                max_new_tokens=512
            )
            trimmed_generated_ids = generated_ids[0][template_length:]
            response = tok.decode(trimmed_generated_ids, skip_special_tokens=True)
            return response

        all_results = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            results = kwargs['pre_edit']
            all_results = results
        else:
            for i, request in enumerate(tqdm(requests)):
                results = {}
                results['pre'] = {}
                results['pre']['rewrite_ans'] = text_generate(self.model, self.model_name, self.hparams, self.tok,
                                                              request['prompt'], self.hparams.device,
                                                              eval_metric=eval_metric, test_generation=test_generation)
                results['pre']['rephrase_ans'] = text_generate(self.model, self.model_name, self.hparams, self.tok,
                                                               request['rephrase_prompt'], self.hparams.device,
                                                               eval_metric=eval_metric, test_generation=test_generation)
                por_results = []
                for pr in request['portability']['por_hop']['prompt']:
                    por_results.append(
                        text_generate(self.model, self.model_name, self.hparams, self.tok, pr, self.hparams.device,
                                      eval_metric=eval_metric, test_generation=test_generation))
                if 'locality' in request.keys() and 'loc_hop' in request['locality'].keys():
                    loc_results = []
                    for pr in request['locality']['loc_hop']['prompt']:
                        loc_results.append(
                            text_generate(self.model, self.model_name, self.hparams, self.tok, pr, self.hparams.device,
                                          eval_metric=eval_metric, test_generation=test_generation))
                    results['pre']['locality_ans'] = loc_results
                results['pre']['portability_ans'] = por_results
                all_results.append(results)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                json.dump(all_results, open(kwargs['pre_file'], 'w'), indent=4)

        def edit_func(request):
            if self.alg_name == 'IKE':
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                icl_examples = None
            return edited_model, weights_copy, icl_examples

        def post_edit_results(all_results, request, edited_model, idx, eval_metric, test_generation, icl_examples,
                              **kwargs):
            if self.alg_name == 'IKE':
                all_results[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                     request, self.hparams.device),
                })
            else:
                results_post = {}
                results_post['rewrite_ans'] = text_generate(edited_model, self.model_name, self.hparams, self.tok,
                                                            request['prompt'], self.hparams.device,
                                                            eval_metric=eval_metric, test_generation=test_generation)
                results_post['rephrase_ans'] = text_generate(edited_model, self.model_name, self.hparams, self.tok,
                                                             request['rephrase_prompt'], self.hparams.device,
                                                             eval_metric=eval_metric, test_generation=test_generation)
                por_results = []
                for pr in request['portability']['por_hop']['prompt']:
                    por_results.append(
                        text_generate(edited_model, self.model_name, self.hparams, self.tok, pr, self.hparams.device,
                                      eval_metric=eval_metric, test_generation=test_generation))
                if 'locality' in request.keys() and 'loc_hop' in request['locality'].keys():
                    loc_results = []
                    for pr in request['locality']['loc_hop']['prompt']:
                        loc_results.append(text_generate(edited_model, self.model_name, self.hparams, self.tok, pr,
                                                         self.hparams.device, eval_metric=eval_metric,
                                                         test_generation=test_generation))
                    results_post['locality_ans'] = loc_results
                results_post['portability_ans'] = por_results
                if test_generation:
                    if self.hparams.alg_name == 'GRACE':
                        results_post['fluency'] = test_generation_quality(model=edited_model, tok=self.tok,
                                                                          prefixes=request['prompt'] if isinstance(
                                                                              request['prompt'], list) else [
                                                                              request['prompt'], ], max_out_len=100,
                                                                          vanilla_generation=True)
                    else:
                        results_post['fluency'] = test_generation_quality(model=edited_model, tok=self.tok,
                                                                          prefixes=request['prompt'] if isinstance(
                                                                              request['prompt'], list) else [
                                                                              request['prompt'], ], max_out_len=100,
                                                                          vanilla_generation=False)
                all_results[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": results_post
                })
            if verbose:
                LOG.info(f"{idx} editing: {request['prompt']} -> {request['target_new']}")

        if sequential_edit:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
            for i, request in enumerate(requests):
                post_edit_results(all_results, request, edited_model, i, eval_metric, test_generation, icl_examples,
                                  **kwargs)
        else:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
                post_edit_results(all_results, request, edited_model, i, eval_metric, test_generation, icl_examples,
                                  **kwargs)
                if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                    with torch.no_grad():
                        weights_copy()
                elif self.alg_name == 'LoRA':
                    edited_model.unload()
                    del self.model.peft_config
                elif self.alg_name == 'MELO':
                    self.model = edited_model
                elif self.alg_name == 'LoRA':
                    self.model = edited_model
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        if isinstance(edited_model, LORA):
            edited_model = edited_model.model
        if len(all_results) != 0:
            summary_metrics(all_results)

        return all_results, edited_model, weights_copy


def subset_requests(requests, max_samples, seed=42):
    # fin all indices where porability mhop prompts are not nan
    mhop_indices, non_mhop_indices = [], []
    for i, r in enumerate(requests):
        if 'mhop' in r['portability'] and r['portability']['mhop']['prompt'] is not None:
            mhop_indices.append(i)
        else:
            non_mhop_indices.append(i)
    #indices = []  # TODO: remove this line

    np.random.seed(seed)
    if len(mhop_indices) > max_samples:
        indices = np.random.choice(mhop_indices, max_samples, replace=False)
    elif len(mhop_indices) < max_samples:
        indices = mhop_indices
        indices += np.random.choice(non_mhop_indices, max_samples - len(mhop_indices), replace=False).tolist()
    else:
        indices = mhop_indices
    #print(indices)
    return [requests[i] for i in indices]


def subset_requests_old(requests, max_samples, seed=42):
    subset = requests[:max_samples]
    # shuffle
    #np.random.seed(seed)
    #np.random.shuffle(subset)
    subset = [requests[i] for i in [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]]
    return subset

