# src/datasets/phonebook.py

import pandas as pd
import logging
import re
from typing import List, Dict, Any

class PhoneBookLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_config = config.get('dataset', {})
        self.num_entries = self.dataset_config.get('num_entries', 400)

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Default data loading method called by run_experiment.py.
        Since it's for training, it calls load_data_for_training().
        """
        return self.load_data_for_training()

    def _get_train_csv_path(self) -> str:
        """Helper function to consistently generate the CSV file path used for training."""
        
        # 

        explicit_path = self.dataset_config.get('data_file_path')
        if explicit_path:
            logging.info(f"Using the path specified in the config: {explicit_path}")
            return explicit_path
        
        # Fallback path generation for compatibility (only runs if no explicit path is provided)
        logging.warning("'dataset.data_file_path' is not specified. Guessing filename using legacy logic.")
        prefix = self.dataset_config.get('dataset_prefix', 'phonebook')
        train_type = self.dataset_config.get('train_type', 'qa')
        num_entries = self.dataset_config.get('num_entries', 0)
        size_str = f"{num_entries // 1000}K" if num_entries >= 1000 else str(num_entries)

        # Handle LowData experiment naming format (e.g., 50items)
        if 'items' in self.config.get('experiment', {}).get('name', ''):
            size_str = f"{num_entries}items"

        return f"data/{prefix}_{train_type}_{size_str}.csv"

    def load_data_for_training(self) -> List[Dict[str, Any]]:
        """
        Loads the pre-formatted CSV file for training.
        """
        csv_path = self._get_train_csv_path()
        logging.info(f"Loading training data: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            # NTPTrainer expects a list of dictionaries in {'text': ...} format
            return df.to_dict('records')
        except FileNotFoundError:
            logging.error(f"Training data file not found: {csv_path}")
            return []

    def load_data_for_eval(self) -> List[Dict[str, Any]]:
        """
        Loads and reconstructs the data 'used for training' for evaluation purposes.
        """
        train_csv_path = self._get_train_csv_path()
        logging.info(f"Loading evaluation data (same as training data): {train_csv_path}")

        try:
            df_train = pd.read_csv(train_csv_path)
        except FileNotFoundError:
            logging.error(f"Original training file for evaluation ({train_csv_path}) not found.")
            return []
            
        eval_samples = []
        
        # Reverse parse Name and Phone from the 'text' column of the training data (CSV)
        for i, row in df_train.iterrows():
            text = row['text']
            name, phone = None, None

            # Parse 'qa' format (Question: What is the phone number of {Name}? Answer: {Phone})
            qa_match = re.search(r"Question: What is the phone number of (.*?)\? Answer: (.*)", text)
            # Parse 'raw' format ({Name}: {Phone})
            raw_match = re.search(r"(.*?): ([\d-]+)", text)

            if qa_match:
                name, phone = qa_match.groups()
            elif raw_match:
                name, phone = raw_match.groups()

            if name and phone:
                sample = {
                    "id": f"eval_item_{i}",
                    "target_answer": str(phone).strip(),
                    "metadata": {"name": str(name).strip()}
                }
                eval_samples.append(sample)

        if not eval_samples:
            logging.error(f"Failed to parse evaluation samples from '{train_csv_path}'. Please check the file format.")

        return eval_samples