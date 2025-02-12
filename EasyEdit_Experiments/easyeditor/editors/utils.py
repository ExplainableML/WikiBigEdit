from typing import Optional, Union, List, Tuple, Dict
import os
import json
import numpy as np
import torch
import openai
from transformers.modeling_outputs import CausalLMOutput
import time

def _chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i: i + n]
        
def get_all_acc_keys(dict_list):
    all_keys = set()

    def recursive_keys(d):
        for k, v in d.items():
            if k.endswith('acc'):
                all_keys.add(k)
            if isinstance(v, dict):
                recursive_keys(v)
                
    for dictionary in dict_list:
        recursive_keys(dictionary)

    return all_keys
    
def summary_metrics(all_metrics):
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
        for key in ["rewrite_acc", "rephrase_acc", 'rewrite_ppl']:
            if key in all_metrics[0][eval].keys():
                mean_metrics[eval][key] = np.mean([metric[eval][key] for metric in all_metrics])
        for key in ["locality", "portability"]:
            if key in all_metrics[0][eval].keys() and all_metrics[0][eval][key] != {}:
                mean_metrics[eval][key] = dict()
                for lkey in get_all_acc_keys(all_metrics):
                    metrics = [metric[eval][key][lkey] for metric in all_metrics if lkey in metric[eval][key].keys()]
                    if len(metrics) > 0:
                        mean_metrics[eval][key][lkey] = np.mean(metrics)
                    # mean_metrics[eval][key][lkey] = np.mean(
                    #     [metric[eval][key][lkey] for metric in all_metrics])
    # mean_metrics["time"] = np.mean([metric["time"] for metric in all_metrics])

    print("Metrics Summary: ", mean_metrics)

def _prepare_requests(prompts: Union[str, List[str]],
                      target_new: Union[str, List[str]],
                      ground_truth: Union[str, List[str]],
                      tags: Union[str, List[str]],
                      rephrase_prompts: Optional[Union[str, List[str]]] = None,
                      locality_inputs: Optional[Dict] = None,
                      portability_inputs: Optional[Dict] = None,
                      **kwargs
                      ):

    requests = [{
        'prompt': prompt,
        'target_new': target_new_,
        'ground_truth': ground_truth_,
        'tag': tag,
        'portability': {},
        'locality': {}
    }
    for prompt, ground_truth_, target_new_, tag in zip(prompts, ground_truth, target_new, tags)
    ]

    if 'subject' in kwargs:
        if isinstance(kwargs['subject'], str):
            kwargs['subject'] = [kwargs['subject'],]
        else:
            assert len(kwargs['subject']) == len(prompts)
        for prompt_, subject_ in zip(prompts, kwargs['subject']):
            #strip all the punctuations
            s = ''.join(e for e in subject_ if e.isalnum())
            p = ''.join(e for e in prompt_ if e.isalnum())
            assert s in p, print(f'Subject:{s} do not exist in prompt: {p}')

        for i, request in enumerate(requests):
            request.update(
                {
                    'subject': kwargs['subject'][i]
                }
            )
    if 'loc_prompts' in kwargs and kwargs['loc_prompts'] is not None:
        if isinstance(kwargs['loc_prompts'], str):
            kwargs['loc_prompts'] = [kwargs['loc_prompts'],]
        else:
            assert len(kwargs['loc_prompts']) == len(prompts)

        for i, request in enumerate(requests):
            request.update(
                {
                    'loc_prompt': kwargs['loc_prompts'][i]
                }
            )

    if rephrase_prompts is not None:
        if isinstance(rephrase_prompts, str):
            rephrase_prompts = [rephrase_prompts,]

        for i, request in enumerate(requests):
            request.update(
                {
                    'rephrase_prompt': rephrase_prompts[i],
                }
            )
    if locality_inputs is not None:
        for locality_key in locality_inputs.keys():
            if isinstance(locality_inputs[locality_key]['prompt'], str):
                locality_inputs[locality_key]['prompt'] = [locality_inputs[locality_key]['prompt'],]
                locality_inputs[locality_key]['ground_truth'] = [locality_inputs[locality_key]['ground_truth'], ]
            assert len(locality_inputs[locality_key]['prompt']) == len(locality_inputs[locality_key]['ground_truth']) \
            == len(requests), print('One Edit instance needs one locality input.....')

            for i, request in enumerate(requests):
                if locality_inputs[locality_key]['prompt'][i] is not None:
                    request['locality'].update(
                        {
                            locality_key: {
                                f'prompt': locality_inputs[locality_key]['prompt'][i],
                                f'ground_truth': locality_inputs[locality_key]['ground_truth'][i]
                            }
                        }
                    )

    if portability_inputs is not None:
        for portability_key in portability_inputs.keys():
            if isinstance(portability_inputs[portability_key]['prompt'], str):
                portability_inputs[portability_key]['prompt'] = [portability_inputs[portability_key]['prompt'],]
                portability_inputs[portability_key]['ground_truth'] = [portability_inputs[portability_key]['ground_truth'], ]
            assert len(portability_inputs[portability_key]['prompt']) == len(portability_inputs[portability_key]['ground_truth']) \
            == len(requests), 'One Edit instance needs one portability input.....'

            for i, request in enumerate(requests):
                if portability_inputs[portability_key]['prompt'][i] is not None:
                    request['portability'].update(
                        {
                            portability_key: {
                                'prompt': portability_inputs[portability_key]['prompt'][i],
                                'ground_truth': portability_inputs[portability_key]['ground_truth'][i]
                            }
                        }
                    )
    return requests


class GPTWrapper(torch.nn.Module):
    def __init__(self, model, tokenizer, emb_model):
        super(GPTWrapper, self).__init__()
        self.model = model
        self.client = openai.OpenAI(api_key='sk-proj-8uHWB2qvDeYf6HBakGrIT3BlbkFJiLDHEOpD09V9qAPTs3Ss')
        self.tokenizer = tokenizer
        self.emb_model = emb_model
        self.device = 'cpu'

    def get_input_embeddings(self):
        return self.emb_model.get_input_embeddings()

    def forward(self, input_ids, **kwargs):
        responses = []
        for input_id in input_ids:
            instructions = 'You are given a factual qa example and a question. Answer the question solely with the name of the factual entity following the answer format of the example. \n\n'
            prompt = self.tokenizer.decode(input_id)
            prompt = prompt[:prompt.rfind('A:') + 2]
            try_ = 0
            while try_ <= 3:
                try:
                    try_ += 1
                    stream = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": instructions + prompt}],
                        stream=True,
                    )
                    response = "".join(
                        [chunk.choices[0].delta.content if chunk.choices[0].delta.content is not None else '' for chunk
                         in stream])
                    break
                except Exception as e:
                    time.sleep(5)
                    continue

            responses.append(prompt + ' ' + response + ' A')
            #print(f'Response: {responses[-1]}')

        # Encode the responses back into token IDs using the tokenizer
        response_token_ids = [self.tokenizer.encode(response) for response in responses]

        # Pad and create a tensor for the response token IDs
        max_length = max(len(ids) for ids in input_ids)
        padded_responses = []
        for ids in response_token_ids:
            if len(ids) < max_length:
                padded_responses.append(ids + [self.tokenizer.pad_token_id] * (max_length - len(ids)))
            else:
                padded_responses.append(ids[-max_length:])
        response_tensor = torch.tensor(padded_responses, dtype=torch.long)

        # Create logits tensor with dummy values (since logits aren't available from GPT-4 API)
        # Shape: (batch_size, sequence_length, vocab_size)
        vocab_size = self.tokenizer.vocab_size
        logits = torch.zeros(
            response_tensor.size(0),  # batch_size
            response_tensor.size(1),  # sequence_length
            vocab_size,               # vocab_size
            dtype=torch.float,
        )

        # Scatter the token IDs to simulate logits (place high probabilities for the actual tokens)
        for i, response in enumerate(padded_responses):
            for j, token_id in enumerate(response):
                logits[i, j, token_id] = 10.0  # Assign high logit value to the correct token ID

        # Create and return a CausalLMOutput object
        return CausalLMOutput(
            logits=logits,  # Logits tensor
            hidden_states=None,    # Optional, leave None unless implementing hidden states
            attentions=None         # Optional, leave None unless implementing attention outputs
        )
