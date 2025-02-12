from dataclasses import dataclass
from typing import List
from ...util.hparams import HyperParams
import yaml


@dataclass
class LoRAHyperParams(HyperParams):
    # Method
    lora_type: str
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float
    target_modules: List[str]
    rank: int
    lora_alpha: float
    lora_dropout: float

    # Module templates

    device: int
    alg_name: str
    model_name: str
    fp16: bool

    merge: bool = False
    merge_alpha: float = 0.5

    # Defaults
    batch_size: int = 128
    max_length: int = 40
    model_parallel: bool = False
    data_parallel: bool = False
    lora_distributed: bool = False
    local_rank: int = 0

    @classmethod
    def from_hparams(cls, hparams_name_or_path=None, config=None):
        assert hparams_name_or_path or config, print('RAGHyperParams requires either hparams_name_or_path or config')
        if hparams_name_or_path:
            if '.yaml' not in hparams_name_or_path:
                hparams_name_or_path = hparams_name_or_path + '.yaml'

            with open(hparams_name_or_path, "r") as stream:
                config = yaml.safe_load(stream)
                config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'LoRA') or print(
            f'LoRAHyperParams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]} ')
        return cls(**config)
