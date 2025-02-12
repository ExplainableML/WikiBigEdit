from copy import deepcopy
from typing import Any, Dict, List, Tuple
from peft import get_peft_model, AdaLoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig, PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import math

from .lora_hparams import LoRAHyperParams


def apply_lora_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: LoRAHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    if hparams.lora_distributed:
        #edited_model = execute_lora_distributed(
        edited_model = execute_lora_accelerate(
            model,
            tok,
            requests,
            hparams,
            **kwargs,
        )
    else:
        edited_model = execute_lora(model, tok, requests, hparams, keep_original_weight)

    return edited_model, weights_copy


def init_distributed_training(local_rank: int):
    """Initialize the distributed environment."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)


def cleanup_distributed_training():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def merge_lora_weights_with_interpolation(
        base_model: AutoModelForCausalLM,
        base_state_dict: Dict[str, torch.Tensor],
        peft_model: PeftModel,
        alpha: float = 0.5,
) -> AutoModelForCausalLM:
    """
    Merges the LoRA weights with the base model through interpolation.

    Args:
        base_model (AutoModelForCausalLM): The original base model without LoRA weights.
        peft_model (PeftModel): The model with LoRA weights applied.
        alpha (float): The interpolation factor.
                       0.0 keeps the base model weights,
                       1.0 fully applies the LoRA-adapted weights.
        save_path (Optional[str]): Path to save the merged model (optional).

    Returns:
        AutoModelForCausalLM: The merged model with interpolated weights.
    """
    # Validate alpha
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("Alpha must be between 0.0 and 1.0")

    # Merge the LoRA weights into the PEFT model
    peft_model.merge_and_unload()  # This will integrate the LoRA weights into the underlying base model

    # Perform interpolation of weights
    #base_state_dict = {k: v.to('cpu') for k, v in base_model.state_dict().items()}
    adapted_state_dict = {k: v.to('cpu') for k, v in peft_model.state_dict().items()}

    interpolated_state_dict = {}
    print(f'Interpolating with alpha={alpha}')
    for key in base_state_dict:
        if 'q_proj' in key or 'v_proj' in key:
            if f'base_model.model.{key}' in adapted_state_dict:
                base_param = base_state_dict[key]
                adapted_param = adapted_state_dict[f'base_model.model.{key}']
                interpolated_param = (1 - alpha) * base_param + alpha * adapted_param
                interpolated_state_dict[key] = interpolated_param
            else:
                #interpolated_state_dict[key] = base_state_dict[key]
                print(f"Key {key} not found in adapted_state_dict")

    del peft_model

    base_model.load_state_dict(interpolated_state_dict, strict=False)

    return base_model


def execute_lora(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: LoRAHyperParams,
        keep_original_weight=False,
        **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the Lora update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    model.config.use_cache = False
    model.supports_gradient_checkpointing = True  #
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    device = torch.device(f'cuda:{hparams.device}')
    if hparams.lora_type == "lora":
        Config = LoraConfig
    elif hparams.lora_type == "adalora":
        Config = AdaLoraConfig
    else:
        raise NotImplementedError

    if hparams.merge:
        base_state_dict = {k: v.to('cpu') for k, v in model.state_dict().items()}
    else:
        base_state_dict = None

    if not keep_original_weight and hasattr(model, 'peft_config') and not hparams.merge:
        peft_model = model
    else:
        peft_config = Config(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hparams.rank,
            lora_alpha=hparams.lora_alpha, lora_dropout=hparams.lora_dropout,
            layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
            target_modules=hparams.target_modules
        )
        model.to(device)
        peft_model = get_peft_model(model, peft_config)

    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    peft_model.print_trainable_parameters()
    requests = deepcopy(requests)
    #for request in requests:
    #    print(
    #        f"Executing LoRA algo for: "
    #        f"[{request['prompt']}] -> [{request['target_new']}]"
    #    )

    # Define inputs
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]

    # Configure optimizer / gradients
    lr = hparams.lr * hparams.batch_size / 256
    opt = torch.optim.Adam(
        peft_model.parameters(),
        lr=lr,
        weight_decay=hparams.weight_decay,
    )

    # Configure scheduler (cosine decay with 10% warmup)
    warmup_steps = int(0.1 * hparams.num_steps)  # 10% of the total steps

    # Define the lambda function for learning rate scheduling
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup phase
            return current_step / warmup_steps
        else:
            # Cosine decay phase
            progress = (current_step - warmup_steps) / (hparams.num_steps - warmup_steps)
            return 0.5 * (1 + math.cos(torch.pi * progress)) + 1e-6

    # Initialize the scheduler
    scheduler = LambdaLR(opt, lr_lambda)

    if hparams.fp16:
        # Initialize GradScaler for mixed precision
        scaler = GradScaler()

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        #print(20 * "=")
        #print(f"Epoch: {it}")
        #print(20 * "=")
        loss_meter.reset()

        b = 0
        for txt, tgt in zip(
                chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            #print('Batch', b)
            b += 1
            mask_token = -100
            opt.zero_grad()
            if 't5' in hparams.model_name.lower():
                inputs = tok(txt, return_tensors="pt", padding=True).to(device)
                bs = inputs["input_ids"].shape[0]
                target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                    device
                )
                inputs['labels'] = target_ids
                logits = peft_model(**inputs).logits
                unmasked_log_probs = logits.log_softmax(-1).gather(-1, inputs['labels'].unsqueeze(-1)).squeeze(-1)
                mask = inputs['labels'] != -100
                n_tokens = mask.float().sum()
                avg_log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                nll = -avg_log_prob
                loss = nll
            else:
                # src_trg_inputs = tok(txt + tgt, return_tensors="pt", padding=True).to(device)
                # bs = src_trg_inputs["input_ids"].shape[0]
                # targ = deepcopy(src_trg_inputs['input_ids'])
                # pred = peft_model(**src_trg_inputs).logits
                # pred = pred[:, :-1]
                # targ = targ[:, 1:]
                # mask = targ != -100
                # n_tokens = mask.float().sum()
                # unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)
                # log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                # loss = -log_prob
                # eos_token = tok.decode(tok.eos_token_id)
                full_prompt = [f"{p} {l}" for p, l in zip(txt, tgt)]
                prompt_ids = tok(list(txt), return_tensors="pt", padding=True, truncation=True, max_length=hparams.max_length)["input_ids"]
                num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_ids]
                tokens = tok(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=hparams.max_length)
                bs = tokens["input_ids"].shape[0]
                tokens["labels"] = tokens["input_ids"].clone()
                num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in tokens["labels"]]
                for i in range(len(txt)):
                    tokens["labels"][i][num_pad_toks[i]:num_pad_toks[i]+num_prompt_toks[i]] = mask_token
                tokens["labels"][tokens["input_ids"] == tok.pad_token_id] = mask_token
                tokens = tokens.to(device)
                # Forward pass with autocast
                if hparams.fp16:
                    with autocast():
                        pred = peft_model(**tokens)
                        loss = pred.loss
                else:
                    loss = peft_model(**tokens).loss
                # pred = peft_model(**tokens)
                # loss = pred.loss
                # targ = target_ids
                # pred = peft_model(**src_trg_inputs).logits
                # pred = pred[:, :-1]
                # pred = pred[:, -targ.size(1):]

                # mask = targ != -100
                # n_tokens = mask.float().sum()
                # unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)
                # log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                # loss = -log_prob
            #print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=bs)

            # if loss.item() >= 1e-3:
            if hparams.fp16:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
        scheduler.step()
        current_lr = opt.param_groups[0]['lr']
        print(f"Step {it + 1}/{hparams.num_steps}, Learning Rate: {current_lr:.6f}, Total loss {loss_meter.avg}")

    if hparams.merge:
        model_out = merge_lora_weights_with_interpolation(model, base_state_dict, peft_model, alpha=hparams.merge_alpha)
    else:
        model_out = peft_model
    return model_out


def execute_lora_distributed(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: LoRAHyperParams,
        **kwargs: Any
):
    """Apply LoRA with distributed training."""
    init_distributed_training(hparams.local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"{rank} Memory Allocated: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"{rank} Memory Cached: {torch.cuda.memory_reserved() / 1e6} MB")
    # clear cache
    torch.cuda.empty_cache()
    print(f"{rank} Memory Allocated: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"{rank} Memory Cached: {torch.cuda.memory_reserved() / 1e6} MB")

    device = torch.device(f'cuda:{rank}')
    model.to(device)

    if hparams.lora_type == "lora":
        Config = LoraConfig
    elif hparams.lora_type == "adalora":
        Config = AdaLoraConfig
    else:
        raise NotImplementedError

    peft_config = Config(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=hparams.rank,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        layers_to_transform=hparams.layers if hparams.layers else None,
        target_modules=hparams.target_modules
    )
    peft_model = get_peft_model(model, peft_config)
    print(f"{rank} Memory Allocated: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"{rank} Memory Cached: {torch.cuda.memory_reserved() / 1e6} MB")
    # Wrap model in DistributedDataParallel
    peft_model = DDP(peft_model, device_ids=[rank], output_device=rank)

    # Optimizer and scheduler
    opt = torch.optim.Adam(
        peft_model.parameters(),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    warmup_steps = int(0.1 * hparams.num_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            progress = (current_step - warmup_steps) / (hparams.num_steps - warmup_steps)
            return 0.5 * (1 + math.cos(torch.pi * progress))

    scheduler = LambdaLR(opt, lr_lambda)

    # Dataset and DataLoader with DistributedSampler
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]

    dataset = [(txt, tgt) for txt, tgt in zip(texts, targets)]
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=hparams.batch_size)

    # Training loop
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        loss_meter.reset()
        for txt, tgt in dataloader:
            txt, tgt = list(txt), list(tgt)

            mask_token = -100
            opt.zero_grad()
            if 't5' in hparams.model_name.lower():
                inputs = tok(txt, return_tensors="pt", padding=True).to(device)
                target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(device)
                inputs['labels'] = target_ids
                loss = peft_model(**inputs).loss
            else:
                full_prompt = [f"{p} {l}" for p, l in zip(txt, tgt)]
                prompt_ids = tok(txt, return_tensors="pt", padding=True, truncation=True, max_length=hparams.max_length)["input_ids"]
                num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_ids]
                tokens = tok(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=hparams.max_length)
                tokens["labels"] = tokens["input_ids"].clone()
                num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in tokens["labels"]]
                for i in range(len(txt)):
                    tokens["labels"][i][num_pad_toks[i]:num_pad_toks[i]+num_prompt_toks[i]] = mask_token
                tokens["labels"][tokens["input_ids"] == tok.pad_token_id] = mask_token
                tokens = tokens.to(device)
                loss = peft_model(**tokens).loss

            loss_meter.update(loss.item(), n=len(txt))
            loss.backward()
            opt.step()

        scheduler.step()

        # Log progress (only on rank 0)
        if rank == 0:
            current_lr = opt.param_groups[0]['lr']
            print(f"Step {it + 1}/{hparams.num_steps}, Learning Rate: {current_lr:.6f}, Total loss {loss_meter.avg}")

    cleanup_distributed_training()
    return peft_model


from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
import torch

def execute_lora_accelerate(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: LoRAHyperParams,
        keep_original_weight=False,
        **kwargs: Any,
):
    accelerator = Accelerator()
    device = accelerator.device

    # Prepare LoRA config
    if not keep_original_weight and hasattr(model, 'peft_config'):
        peft_model = model
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=hparams.rank,
            lora_alpha=hparams.lora_alpha, lora_dropout=hparams.lora_dropout,
            layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
            target_modules=hparams.target_modules
        )
        peft_model = get_peft_model(model, peft_config)

    # Configure optimizer and scheduler
    opt = torch.optim.Adam(peft_model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)
    scheduler = get_scheduler(
        "cosine",
        optimizer=opt,
        num_warmup_steps=int(0.1 * hparams.num_steps),
        num_training_steps=hparams.num_steps,
    )

    # Wrap model, optimizer, and scheduler with Accelerate
    peft_model, opt, scheduler = accelerator.prepare(peft_model, opt, scheduler)

    # Prepare data
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]

    # Training loop
    for it in range(hparams.num_steps):
        accelerator.print(f"Step {it + 1}/{hparams.num_steps}")
        for txt, tgt in zip(chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)):
            # Tokenize and prepare inputs
            tokens = tok(txt + tgt, return_tensors="pt", padding=True, truncation=True, max_length=hparams.max_length)
            tokens["labels"] = tokens["input_ids"].clone()
            tokens = accelerator.prepare(tokens)

            # Forward pass
            pred = peft_model(**tokens)
            loss = pred.loss

            # Backpropagation
            accelerator.backward(loss)
            opt.step()
            scheduler.step()
            opt.zero_grad()

        accelerator.print(f"Loss: {loss.item()}")

    return peft_model


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk