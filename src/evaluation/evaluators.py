# src/evaluation/evaluators.py

import torch
import torch.nn.functional as F
import logging
from typing import Dict, Any

def evaluate_babilong(model, tokenizer, data_point: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates a single sample from the Babilong dataset using generation.
    Metric: Exact Match (EM)
    """
    device = model.device
    context = data_point['context']
    question = data_point['question']
    target_answer = data_point['target_answer']

    # Simplified prompt format based on the Babilong paper
    prompt = (
        "You are an expert at reading comprehension.\n"
        "Based on the context below, answer the following question.\n\n"
        "--- CONTEXT ---\n"
        f"{context}\n\n"
        "--- QUESTION ---\n"
        f"{question}\n\n"
        "--- ANSWER ---\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50, # Limit to 50 tokens as answers are usually short
            do_sample=False,   # Greedy decoding
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Extract only the generated portion
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    generated_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Normalize and compare using Exact Match
    normalized_pred = generated_answer.lower().strip()
    normalized_target = target_answer.lower().strip()
    is_correct = (normalized_pred == normalized_target)

    # Generate log entry for raw_predictions.jsonl
    raw_log = {
        "id": data_point['id'],
        "metadata": data_point['metadata'],
        "inputs": {
            "context": context,
            "question": question,
            "target_answer": target_answer
        },
        "prediction": {
            "generated_answer": generated_answer
        },
        "result": {
            "is_correct": bool(is_correct),
            "metric_score": 1.0 if is_correct else 0.0
        }
    }
    return raw_log


def evaluate_quality(model, tokenizer, data_point: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates a single sample from the QUALITY dataset using multiple-choice scoring.
    Metric: Accuracy
    """
    device = model.device
    context = data_point['context']
    question = data_point['question']
    options = data_point['options']
    gold_label_index = data_point['gold_label_index']
    
    # Define template
    prompt_template = (
        "<|user|>\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Choose one of the following options:\n"
        "{options_str}\n"
        "Answer:<|end|>\n"
        "<|assistant|>"
    )
    options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    
    base_prompt = prompt_template.format(
        context=context,
        question=question,
        options_str=options_str
    )
    
    option_scores = []
    with torch.no_grad():
        for i, option_text in enumerate(options):
            # Create full input for each option (e.g., <prompt> A. {option_text})
            answer_part = f" {chr(65+i)}. {option_text}"
            full_input_text = base_prompt + answer_part
            
            inputs = tokenizer(full_input_text, return_tensors="pt", truncation=True, max_length=4096).to(device)
            outputs = model(**inputs)
            
            # Tokenize the answer part separately to define the loss calculation range
            answer_tokens = tokenizer(answer_part, add_special_tokens=False).input_ids
            answer_len = len(answer_tokens)
            
            # Extract logits corresponding to the answer part from the full logits
            answer_logits = outputs.logits[:, -answer_len-1:-1, :]
            
            # Sum log probabilities of the answer tokens
            log_probs = F.log_softmax(answer_logits, dim=-1)
            answer_tokens_tensor = torch.tensor(answer_tokens, device=device).unsqueeze(0).unsqueeze(-1)
            gathered_log_probs = torch.gather(log_probs, 2, answer_tokens_tensor).squeeze(-1)
            
            # Use the sum of log probabilities as the score for the option
            score = gathered_log_probs.sum().item()
            option_scores.append(score)

    predicted_index = torch.argmax(torch.tensor(option_scores)).item()
    is_correct = (predicted_index == gold_label_index)

    raw_log = {
        "id": data_point['question_id'],
        "metadata": {
            "is_difficult": data_point['is_difficult'],
            "context_token_length": len(tokenizer.encode(context))
        },
        "inputs": {
            "question": question,
            "options": options,
            "gold_label_index": gold_label_index
        },
        "prediction": {
            "predicted_label_index": predicted_index,
            "option_scores": option_scores
        },
        "result": {
            "is_correct": bool(is_correct),
            "metric_score": 1.0 if is_correct else 0.0
        }
    }
    return raw_log