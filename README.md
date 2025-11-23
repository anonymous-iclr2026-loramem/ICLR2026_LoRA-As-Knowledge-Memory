# Understanding LoRA As Knowledge Memory

This repository contains the anonymized implementation for the "Understanding LoRA As Knowledge Memory". This work presents a systematic and comprehensive empirical study of using Low-Rank Adaptation (LoRA) as a parametric knowledge memory component for Large Language Models (LLMs).

## Overview

This research explores LoRA's potential beyond task adaptation, investigating its use as a dedicated module for knowledge storage. The study addresses fundamental questions about LoRA's memory characteristics, optimization strategies, multi-module systems, and integration with existing methods like In-Context Learning (ICL) and Retrieval-Augmented Generation (RAG).

## Repository Structure

```
.
├── 0_prepare_base_model.py          # Script to download and prepare base models
├── 1_GRID_SEARCH/                    # Grid search experiments on LoRA rank and data size
│   ├── train.py                      # Main training script
│   ├── run_experiment.py             # Experiment runner
│   ├── analysis/                     # Evaluation and analysis scripts
│   ├── configs/                      # Experiment configurations
│   └── data/                         # PhoneBook and CounterFact datasets
│
├── 2_PAPERQA/                        # PaperQA dataset experiments
│   ├── scripts/                      # Data preparation, training, and evaluation scripts
│   ├── logic/                        # Core training and utility logic
│   ├── configs/                      # Experiment configurations
│   └── data/                         # Processed PaperQA data
│
├── 3_NQA/                            # NarrativeQA dataset experiments
│   ├── scripts/                      # End-to-end experiment pipeline
│   ├── logic/                        # Training, evaluation, and utility modules
│   └── configs/                      # Experiment configurations
│
├── src/                              # Shared source code
│   ├── chunking/                     # Text chunking utilities
│   ├── datasets/                     # Dataset loaders (PhoneBook, CounterFact, etc.)
│   ├── evaluation/                   # Evaluation metrics and evaluators
│   ├── synthesis/                    # Synthetic data generation (QA, summaries)
│   └── trainers/                     # Training methods (NTP, DCD, ablation)
│
├── configs/                          # Global configuration files
├── data/                             # Shared datasets
├── requirements.txt                  # Python dependencies
└── environment.yml                   # Conda environment specification
```

## Installation

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate <env_name>
```

### Using pip

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Base Model

First, download and prepare the base model:

```bash
python 0_prepare_base_model.py
```

Edit the script to select your desired model (e.g., Llama-3.1-8B-Instruct, Qwen3-14B).

### 2. Run Grid Search Experiments

For fundamental memory capacity analysis:

```bash
cd 1_GRID_SEARCH
python run_experiment.py --config configs/<config_file>.yaml
```

### 3. Run PaperQA Experiments

For long-context knowledge memorization:

```bash
cd 2_PAPERQA
# Prepare data
python scripts/01a_prepare_introductions.py
python scripts/01b_prepare_qa.py

# Generate configs
python scripts/02a_generate_singlelora_configs.py
python scripts/02b_generate_multilora_configs.py

# Train LoRA modules
python scripts/03_train_lora.py --config configs/<config_file>.yaml

# Evaluate
python scripts/04a_evaluate_singlelora.py
python scripts/04b_evaluate_multilora.py
python scripts/04d_evaluate_multilora_merging.py
```

### 4. Run NarrativeQA Experiments

For real-world document understanding:

```bash
cd 3_NQA
# Prepare data
python scripts/01_prepare_source_doc.py
python scripts/02_prepare_all_qa.py

# Run experiments
python scripts/02_run_experiment.py --config configs/<config_file>.yaml

# Evaluate
python scripts/10_evaluate_multilora.py
```

## Experimental Setup

### Datasets

- **PhoneBook (PB)**: Synthetic symbolic associations (names and phone numbers)
- **CounterFact (CF)**: Counterfactual statements contradicting pre-trained knowledge
- **PaperQA**: Long-form academic paper introductions
- **NarrativeQA**: Real-world narrative question answering

### Evaluation Metrics

- Exact match (PhoneBook)
- Efficacy score (CounterFact)
- ROUGE scores (PaperQA, NarrativeQA)
- Comparison with ICL and RAG baselines
