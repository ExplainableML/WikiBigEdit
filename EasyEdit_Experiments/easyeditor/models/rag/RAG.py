from typing import Any, Dict, List, Tuple
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .RAG import RAG
from .utils import tokenize, get_context_templates
from .rag_hparams import RAGHyperParams


def apply_rag_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: RAGHyperParams,
        copy=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    request = requests[0]
    if copy:
        model = deepcopy(model)
    editor = RAG(model=model, config=hparams, device=hparams.device)
    tokens, act_mask, deact_mask = tokenize(request, tokenizer=tok, device=hparams.device, context_templates=context_templates, hparams=hparams)
    editor.edit(config=hparams, tokens=tokens, act_mask=act_mask, deact_mask=deact_mask)

    weights_copy = editor.reset_layer

    return editor, weights_copy