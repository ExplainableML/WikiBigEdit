# WikiBigEdit: Lifelong Knowledge Editing Benchmark

## Abstract

Keeping large language models factually up-to-date is crucial for deployment, yet costly retraining remains a challenge. Knowledge editing offers a promising alternative, but methods are only tested on small-scale or synthetic edit benchmarks.
In this work, we aim to bridge research into lifelong knowledge editing to real-world edits at practically relevant scale.
We first introduce __WikiBigEdit__; a large-scale benchmark of real-world Wikidata edits, built to automatically extend lifelong for future-proof benchmarking. In its first instance, it includes over 500K question-answer pairs for knowledge editing alongside a comprehensive evaluation pipeline.
Finally, we use __WikiBigEdit__ to study existing knowledge editing techniques' ability to incorporate large volumes of real-world facts and contrast their capabilities to generic modification techniques such as retrieval augmentation and continual finetuning to acquire a complete picture of the practical extent of current lifelong knowledge editing.
## Codebase Structure

This repository contains two main components:

### [`benchmark_pipeline/`](benchmark_pipeline/)

Contains the WikiBigEdit extraction pipeline for generating the benchmark dataset. It automatically extracts, formats, and evaluates real-world factual updates from Wikidata.

For setup and usage instructions, see the README.

### [`editing_experiments/`](editing_experiments/)

Contains the adapted EasyEdit codebase for running lifelong knowledge editing experiments on WikiBigEdit. It supports structured experiment configurations via Hydra and logs results with Weights & Biases (wandb).

For details on running experiments, see the README.