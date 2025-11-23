# src/evaluation/__init__.py

import logging
from .evaluators import evaluate_babilong, evaluate_quality
from .phonebook_evaluator import evaluate_phonebook

def get_evaluator(config):
    """
    Returns the appropriate evaluation function based on the evaluation.metric 
    defined in the configuration file.
    """
    dataset_name = config.get('dataset', {}).get('name')
    
    if dataset_name == "phonebook":
        return evaluate_phonebook
    
    metric_name = config['evaluation']['metric']
    logging.info(f"Retrieving evaluation function for metric: '{metric_name}'.")

    if metric_name == "exact_match":
        return evaluate_babilong
    elif metric_name == "accuracy":
        return evaluate_quality
    else:
        raise ValueError(f"Unsupported evaluation metric: {metric_name}")