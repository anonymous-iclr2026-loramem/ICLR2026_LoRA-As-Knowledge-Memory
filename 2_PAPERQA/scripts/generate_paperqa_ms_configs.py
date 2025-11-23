#!/usr/bin/env python3
# scripts/generate_combined_paperqa_configs.py

import os
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Experiment Basic Settings ---
# Qwen3 Models (0.6B, 1.7B, 4B, 8B, 14B)
QWEN3_MODELS = {
    "qwen3_0.6b": "models/Qwen3-0.6B_base",
    "qwen3_1.7b": "models/Qwen3-1.7B_base", 
    "qwen3_4b": "models/Qwen3-4B_base",
    "qwen3_8b": "models/Qwen3-8B_base",
    "qwen3_14b": "models/Qwen3-14B_base"
}

# --- Common Hyperparameter Grid Settings ---
# Define grid for num_train_steps and learning_rate applied to all models
HYPERPARAMETER_GRID = {
    "num_train_steps": [250, 500, 1000, 2000],
    "learning_rate": [5.0e-04, 1.0e-04, 5.0e-05, 1.0e-05]
}

# Data Path Settings
QA40_DATA_ROOT_DIR = "outputs/paperqa_augsc_gpt_data/paperqa"
ORIGINAL_DATA_ROOT_DIR = "data/b8aug_original_intro_data"

# Configuration Save Path (Anonymized)
CONFIG_DIR = "configs/paperqa_ms2"

# Original Data Paper IDs (0 to 14)
ORIGINAL_PAPER_IDS = range(15)

# --- YAML Template (Fixed Experiment Parameters) ---
CONFIG_TEMPLATE = """
experiment:
  name: "{exp_name}"
  seed: 42

resources:
  gpus: [0]

model:
  base_model_id: "{model_path}"
  lora_config:
    r: 16
    lora_alpha: 32
    target_modules: "all-linear"
    lora_dropout: 0.1
    bias: "none"
    task_type: "CAUSAL_LM"

dataset:
  name: "paperqa"
  data_file_path: "{data_file_path}"
  use_chat_template: true

training:
  method: "ntp"
  num_train_steps: {num_train_steps}
  warmup_ratio: 0.1
  learning_rate: {learning_rate}
  batch_size: 8
  checkpoint_strategy:
    policy: "none"

evaluation:
  script: "analysis/evaluate_paperqa.py"
"""

def get_available_qa40_tasks():
    """Generates a list of tasks by finding only QA40 data files in QA40_DATA_ROOT_DIR."""
    tasks = []
    paper_ids = set()

    if not os.path.exists(QA40_DATA_ROOT_DIR):
        logging.error(f"QA40_DATA_ROOT_DIR '{QA40_DATA_ROOT_DIR}' does not exist.")
        return [], []

    # Filename pattern matching including only QA40
    # e.g., paper_0_QA40.jsonl, paper_0_QA40_Summary2.jsonl, etc.
    pattern = r'paper_(\d+)_(.*QA40.*)\.jsonl$'

    for filename in os.listdir(QA40_DATA_ROOT_DIR):
        match = re.match(pattern, filename)
        if match:
            paper_idx = int(match.group(1))
            task_info = match.group(2)  # Task info including QA40

            paper_ids.add(paper_idx)
            tasks.append(task_info)

    # Remove duplicates and sort
    unique_tasks = sorted(list(set(tasks)))
    sorted_paper_ids = sorted(list(paper_ids))

    logging.info(f"Found QA40 paper IDs: {sorted_paper_ids}")
    logging.info(f"Found QA40 augmentation tasks: {unique_tasks}")

    return sorted_paper_ids, unique_tasks

def generate_qa40_configs(model_name, model_path):
    """Generates QA40 data configuration files for a specific model."""
    qa40_paper_ids, qa40_tasks = get_available_qa40_tasks()
    
    if not qa40_paper_ids or not qa40_tasks:
        logging.warning(f"Skipping QA40 config generation for model {model_name} as QA40 data could not be found.")
        return 0
    
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    count = 0
    total_configs = len(qa40_paper_ids) * len(qa40_tasks) * len(HYPERPARAMETER_GRID["num_train_steps"]) * len(HYPERPARAMETER_GRID["learning_rate"])
    logging.info(f"Generating {total_configs} QA40 config files for model {model_name} in '{CONFIG_DIR}'...")
    logging.info(f"  - Hyperparameter grid: num_train_steps={HYPERPARAMETER_GRID['num_train_steps']}, learning_rate={HYPERPARAMETER_GRID['learning_rate']}")

    for paper_idx in qa40_paper_ids:
        for task_info in qa40_tasks:
            for num_steps in HYPERPARAMETER_GRID["num_train_steps"]:
                for lr in HYPERPARAMETER_GRID["learning_rate"]:
                    doc_id = f"paper_{paper_idx}"
                    
                    # Apply experiment naming convention (including hyperparameter info)
                    lr_str = f"{lr:.0e}".replace("e-0", "e-").replace("e+", "e")
                    exp_name = f"paperqa_qa40_{model_name}_{doc_id}_{task_info}_steps{num_steps}_lr{lr_str}"
                    
                    # Generate data file path
                    data_filename = f"{doc_id}_{task_info}.jsonl"
                    data_file_path = os.path.join(QA40_DATA_ROOT_DIR, data_filename)
                    
                    # Check if the actual file exists
                    if not os.path.exists(data_file_path):
                        logging.warning(f"QA40 data file does not exist: {data_file_path}")
                        continue
                    
                    # Template formatting
                    config_str = CONFIG_TEMPLATE.format(
                        model_path=model_path,
                        exp_name=exp_name,
                        data_file_path=os.path.abspath(data_file_path),
                        num_train_steps=num_steps,
                        learning_rate=lr
                    )
                    
                    # Save to file
                    filepath = os.path.join(CONFIG_DIR, f"{exp_name}.yaml")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(config_str)
                    
                    count += 1
    
    logging.info(f"✅ Model {model_name}: {count} QA40 configuration files generated.")
    return count

def generate_original_configs(model_name, model_path):
    """Generates Original data configuration files for a specific model."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    count = 0
    total_configs = len(ORIGINAL_PAPER_IDS) * len(HYPERPARAMETER_GRID["num_train_steps"]) * len(HYPERPARAMETER_GRID["learning_rate"])
    logging.info(f"Generating {total_configs} Original config files for model {model_name} in '{CONFIG_DIR}'...")
    logging.info(f"  - Hyperparameter grid: num_train_steps={HYPERPARAMETER_GRID['num_train_steps']}, learning_rate={HYPERPARAMETER_GRID['learning_rate']}")

    for paper_idx in ORIGINAL_PAPER_IDS:
        for num_steps in HYPERPARAMETER_GRID["num_train_steps"]:
            for lr in HYPERPARAMETER_GRID["learning_rate"]:
                doc_id = f"paper_{paper_idx}"
                
                # Apply experiment naming convention (specify original data, include hyperparameter info)
                lr_str = f"{lr:.0e}".replace("e-0", "e-").replace("e+", "e")
                exp_name = f"paperqa_original_{model_name}_{doc_id}_Original_R16_steps{num_steps}_lr{lr_str}"
                
                # Generate data file path
                data_filename = f"{doc_id}_OriginalIntro.jsonl"
                data_file_path = os.path.join(ORIGINAL_DATA_ROOT_DIR, data_filename)
                
                # Template formatting
                config_str = CONFIG_TEMPLATE.format(
                    exp_name=exp_name,
                    model_path=model_path,
                    data_file_path=os.path.abspath(data_file_path),
                    num_train_steps=num_steps,
                    learning_rate=lr
                )
                
                # Save to file
                filepath = os.path.join(CONFIG_DIR, f"{exp_name}.yaml")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(config_str)
                
                count += 1
    
    logging.info(f"✅ Model {model_name}: {count} Original configuration files generated.")
    return count

def generate_all_configs():
    """Generates configuration files for all models and data types."""
    total_configs = 0
    
    logging.info("=== Starting generation of combined PaperQA config files ===")
    logging.info(f"Supported models: {list(QWEN3_MODELS.keys())}")
    logging.info(f"Supported data: QA40, Original")
    
    for model_name, model_path in QWEN3_MODELS.items():
        logging.info(f"\n--- Generating configs for model {model_name} ---")
        
        # Generate QA40 data configs
        qa40_count = generate_qa40_configs(model_name, model_path)
        
        # Generate Original data configs
        original_count = generate_original_configs(model_name, model_path)
        
        model_total = qa40_count + original_count
        total_configs += model_total
        
        logging.info(f"Model {model_name} complete: Total {model_total} config files generated")
    
    logging.info(f"\n=== All Complete ===")
    logging.info(f"Total {total_configs} configuration files generated.")
    logging.info(f"All generated config files are saved in directory '{CONFIG_DIR}'.")

if __name__ == "__main__":
    generate_all_configs()