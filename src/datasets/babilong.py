# src/datasets/babilong.py

import logging
from datasets import load_dataset, concatenate_datasets
from typing import List, Dict, Any

class BabilongLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_config = config['dataset']
        # List of available lengths (names) on Hugging Face Hub (based on 1k-samples dataset)
        self.available_lengths = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Loads Babilong data according to the length and task_types specified in the config
        and converts it to the standard format.
        """
        task_types = self.dataset_config.get('task_types')
        if not task_types:
            # Specify that task_types is mandatory
            raise ValueError("'dataset.task_types' must be specified in the config to use the 'babilong' dataset (e.g., ['qa1']).")

        target_length = self.dataset_config.get('length')
        
        # If length is not specified, iterate through all available lengths
        lengths_to_load = [target_length] if target_length else self.available_lengths

        logging.info(f"Starting Babilong data load... Target lengths: {lengths_to_load}, Target tasks: {task_types}")

        all_samples = []
        
        # Iterate through all specified lengths
        for length_str in lengths_to_load:
            # Iterate through all specified tasks
            for task in task_types:
                try:
                    # Load using 'name' for length (subset) and 'split' for task
                    dset = load_dataset("RMT-team/babilong-1k-samples", name=length_str, split=task)
                    
                    for i, sample in enumerate(dset):
                        if sample: # If data is not empty
                            all_samples.append({
                                "dataset": "babilong",
                                "id": f"{length_str}_{task}_item_{i}",
                                "context": sample['input'],
                                "question": sample['question'],
                                "target_answer": sample['target'],
                                "metadata": {
                                    "task_type": task,
                                    "context_length_category": length_str
                                }
                            })
                except Exception as e:
                    logging.error(f"Error loading length '{length_str}', task '{task}': {e}")
        
        logging.info(f"Loaded a total of {len(all_samples)} Babilong samples.")
        return all_samples