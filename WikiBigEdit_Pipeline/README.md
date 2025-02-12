# WikiBigEdit Benchmark Extraction Pipeline

## Overview

This codebase contains the extraction pipeline for __WikiBigEdit__, the benchmark described in the paper _Understanding the Limits of Lifelong Knowledge Editing in LLMs_. __WikiBigEdit__ is a large-scale dataset that facilitates the study of knowledge editing in language models by automatically extracting, formatting, and evaluating real-world factual updates from Wikidata.

![Pipeline Overview](images/benchmark_extraction_pipeline.svg)

## Features

* __Automated Wikidata Change Extraction__: The pipeline detects factual modifications in Wikidata snapshots.

* __Locality and Multi-Hop Probes__: Evaluates whether edited facts remain localized and support complex reasoning.

* __Comprehensive Evaluation__: Supports evaluation of factual consistency, generalization, and reasoning abilities.

* __Scalability__: Designed to handle large-scale factual updates efficiently.

## Installation

### Requirements

Ensure you have the following dependencies installed:

* Python 3.8+

* Required Python packages (install via pip): 
```
pip install -r requirements.txt
```

## Usage

### Step 1: Download Wikidata Snapshots

Download the wikidata snapshots from https://dumps.wikimedia.org/wikidatawiki/ by executing the following script:
```
bash download_wikidata.sh <date>
```
provide the date in the format `YYYYMMDD`.

### Step 2: Generate Benchmark Timestep
One-Click Benchmark Generation

To generate the benchmark with a single command, run:

```
bash generate_benchmark.sh <old_dump_date> <new_dump_date>
```

This script automates the entire pipeline, including:
* Extracting Wikidata Changes
* Formatting Extracted Changes
* Running Locality and Multi-Hop Probes in Parallel
* Combining Locality Results
* Generating and Combining QA Samples

