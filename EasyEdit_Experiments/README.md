# Lifelong Knowledge Editing with EasyEdit

## Overview
This codebase is an adapted version of the **EasyEdit** framework, modified for lifelong knowledge editing experiments on the **WikiBigEdit** benchmark.

For reference, the original EasyEdit repository can be found [here](https://github.com/thunlp/EasyEdit).

## Downloading the WikiBigEdit Dataset
The WikiBigEdit dataset can be downloaded from the [WikiBigEdit repository](https://huggingface.co/datasets/lukasthede/WikiBigEdit) and should be saved in the `data/wikibigedit` directory.

## Running Experiments
To run an editing experiment, use the following command:
```bash
python run_lifelong_edit.py --config-name=<config>
```
The experiment configurations are managed using Hydra, allowing for flexible and structured configuration management.
All experiment configs are save in the `hydra/experiments` directory.

## Logging

Experiments are logged using Weights & Biases (wandb) for tracking performance metrics and model behavior over time.
To use wandb add your username to the config wile in `hydra/experiments/wandb/wandb.yaml`.

## Installation

Ensure you have the necessary dependencies installed:
```bash
pip install -r requirements.txt
```

Additionally, set up wandb for logging by logging in with:
```bash
wandb login
```

