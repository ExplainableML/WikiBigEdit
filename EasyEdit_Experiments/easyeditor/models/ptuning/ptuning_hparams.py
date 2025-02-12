from dataclasses import dataclass
from ...util.hparams import HyperParams
from typing import Optional, Any, List
import yaml


@dataclass
class PTuningHyperParams(HyperParams):

    model_name: str

    prompt_length: int
    lr: float
    epochs: int

    alg_name: str
    device: int

    batch_size: int = 1
    max_length: int = 100
    model_parallel: bool = False


    @classmethod
    def from_hparams(cls, hparams_name_or_path=None, config=None):
        assert hparams_name_or_path or config, print('RAGHyperParams requires either hparams_name_or_path or config')
        if hparams_name_or_path:
            if '.yaml' not in hparams_name_or_path:
                hparams_name_or_path = hparams_name_or_path + '.yaml'

            with open(hparams_name_or_path, "r") as stream:
                config = yaml.safe_load(stream)
                config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'PTuning') or print(f'RAGTrainingHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        print(config)
        return cls(**config)
