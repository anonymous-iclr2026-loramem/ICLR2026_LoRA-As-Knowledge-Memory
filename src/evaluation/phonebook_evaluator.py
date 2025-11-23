# src/evaluation/phonebook_evaluator.py

import torch
import logging
from typing import Dict, Any, List
from src.models.model_loader import apply_chat_template

def evaluate_phonebook(model, tokenizer, data_point: Dict[str, Any], prompt_template: str) -> Dict[str, Any]:
    """
    Evaluates a single sample from the PhoneBook dataset using the provided prompt template.
    """
    device = model.device
    
    # Generate prompt
    user_message = prompt_template.format(
        name=data_point.get('metadata', {}).get('name', ''),
        phone=data_point.get('target_answer', '')
    )
    
    prompt = apply_chat_template(tokenizer, user_message)
    
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=8192
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=40, do_sample=False, temperature=1.0,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    generated_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Scoring
    generated_clean = generated_answer.strip()
    target_clean = data_point['target_answer'].strip()
    is_correct = (generated_clean == target_clean)
    
    return {
        "id": data_point['id'], 
        "dataset": "phonebook", 
        "metadata": data_point['metadata'],
        "inputs": {"question": user_message, "target_answer": target_clean},
        "prediction": {"generated_answer": generated_answer, "generated_clean": generated_clean},
        "result": {"is_correct": bool(is_correct), "metric_score": 1.0 if is_correct else 0.0}
    }

def evaluate_phonebook_batch(model, tokenizer, data_points: List[Dict[str, Any]], prompt_template: str) -> List[Dict[str, Any]]:
    """Evaluates multiple samples in a batch."""
    results = []
    for data_point in data_points:
        result = evaluate_phonebook(model, tokenizer, data_point, prompt_template)
        results.append(result)
    
    correct_count = sum(1 for r in results if r['result']['is_correct'])
    accuracy = correct_count / len(data_points) if data_points else 0.0
    logging.info(f"Evaluation complete: {correct_count}/{len(data_points)} = {accuracy:.2%}")
    return results