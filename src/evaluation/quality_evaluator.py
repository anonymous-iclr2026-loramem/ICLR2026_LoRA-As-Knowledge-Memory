# src/evaluation/quality_evaluator.py

import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from tqdm import tqdm
from ..datasets.quality_loader import QualityDatasetLoader

logger = logging.getLogger(__name__)

class QualityEvaluator:
    """
    QuALITY dataset evaluator supporting Log-likelihood and Generation methods.
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda:0"):
        """
        Args:
            model: The model to evaluate.
            tokenizer: The tokenizer associated with the model.
            device: The device to use for evaluation (e.g., 'cuda:0', 'cpu').
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
    def get_evaluation_prompt(self, question: str, options: List[str], 
                            with_context: bool = False, context: str = "") -> str:
        """Generates the evaluation prompt based on configuration."""
        letters = ["A", "B", "C", "D"]
        options_str = "\n".join([f"{letters[i]}: {option}" for i, option in enumerate(options)])
        
        if with_context and context:
            prompt_template = """Context: {context}

Question: {question}

Options:
{options}

Please answer using the following format:
1. Begin your answer with the phrase "The correct answer is".
2. State the letter of the correct option (e.g., A, B, C, D).
3. Follow the letter with a colon and the exact text of the option you chose.

Answer:"""
            prompt = prompt_template.format(
                context=context,
                question=question,
                options=options_str
            )
        else:
            prompt_template = """Question: {question}

Options:
{options}

Please answer using the following format:
1. Begin your answer with the phrase "The correct answer is".
2. State the letter of the correct option (e.g., A, B, C, D).
3. Follow the letter with a colon and the exact text of the option you chose.

Answer:"""
            prompt = prompt_template.format(
                question=question,
                options=options_str
            )
        
        return prompt
    
    def compute_log_likelihood(self, prompt: str, option: str) -> float:
        """Computes the length-normalized log-likelihood of a specific option."""
        full_text = prompt + " The correct answer is " + option
        
        # Tokenization
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        prompt_inputs = self.tokenizer(prompt + " The correct answer is", return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Extract logits corresponding to the option
            prompt_length = prompt_inputs.input_ids.shape[1]
            # Align logits to input tokens (next token prediction)
            option_logits = logits[0, prompt_length-1:-1]
            option_labels = inputs.input_ids[0, prompt_length:]
            
            # Apply Log softmax
            log_probs = F.log_softmax(option_logits, dim=-1)
            
            # Sum log-likelihoods for option tokens
            option_log_likelihood = 0.0
            for i, label in enumerate(option_labels):
                if i < option_logits.shape[0]:
                    option_log_likelihood += log_probs[i, label].item()
            
            # Length normalization
            normalized_ll = option_log_likelihood / len(option_labels)
            
        return normalized_ll
    
    def evaluate_loglikelihood(self, eval_data: List[Dict[str, Any]], 
                             with_context: bool = False) -> Dict[str, Any]:
        """Evaluates accuracy using the Log-likelihood method."""
        correct_predictions = 0
        total_questions = len(eval_data)
        
        results_by_difficulty = {"easy": {"correct": 0, "total": 0}, 
                               "hard": {"correct": 0, "total": 0}}
        detailed_results = []
        
        logger.info(f"Starting log-likelihood evaluation on {total_questions} questions")
        
        for item in tqdm(eval_data, desc="Evaluating (log-likelihood)"):
            question = item["question"]
            options = item["options"]
            correct_answer = item["answer_index"]
            is_hard = item.get("hard", False)
            context = item.get("article_content", "") if with_context else ""
            
            # Calculate log-likelihood for each option
            option_letters = ["A", "B", "C", "D"]
            option_likelihoods = {}
            
            prompt = self.get_evaluation_prompt(question, options, with_context, context)
            
            for i, option in enumerate(options):
                likelihood = self.compute_log_likelihood(prompt, f"{option_letters[i]}: {option}")
                option_likelihoods[i] = likelihood
            
            # Select option with highest likelihood
            predicted_answer = max(option_likelihoods.keys(), key=lambda x: option_likelihoods[x])
            is_correct = (predicted_answer == correct_answer)
            
            if is_correct:
                correct_predictions += 1
            
            # Aggregate by difficulty
            difficulty = "hard" if is_hard else "easy"
            results_by_difficulty[difficulty]["total"] += 1
            if is_correct:
                results_by_difficulty[difficulty]["correct"] += 1
            
            detailed_results.append({
                "question": question,
                "predicted_answer": predicted_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "is_hard": is_hard,
                "option_likelihoods": option_likelihoods
            })
        
        # Calculate final metrics
        accuracy = correct_predictions / total_questions if total_questions > 0 else 0
        easy_acc = (results_by_difficulty["easy"]["correct"] / 
                   results_by_difficulty["easy"]["total"] 
                   if results_by_difficulty["easy"]["total"] > 0 else 0)
        hard_acc = (results_by_difficulty["hard"]["correct"] / 
                   results_by_difficulty["hard"]["total"] 
                   if results_by_difficulty["hard"]["total"] > 0 else 0)
        
        return {
            "method": "log_likelihood",
            "with_context": with_context,
            "accuracy": accuracy,
            "correct": correct_predictions,
            "total": total_questions,
            "easy_accuracy": easy_acc,
            "hard_accuracy": hard_acc,
            "easy_questions": results_by_difficulty["easy"]["total"],
            "hard_questions": results_by_difficulty["hard"]["total"],
            "detailed_results": detailed_results
        }
    
    def generate_answer(self, prompt: str, max_new_tokens: int = 10) -> str:
        """Generates an answer using the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        return generated_text
    
    def extract_answer_letter(self, generated_text: str) -> Optional[str]:
        """Extracts the answer letter (A, B, C, D) from the generated text."""
        import re
        
        # Pattern match: "The correct answer is A"
        match = re.search(r"The correct answer is ([A-D])", generated_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Fallback: Find the first A, B, C, or D char
        for char in generated_text:
            if char.upper() in ['A', 'B', 'C', 'D']:
                return char.upper()
        
        return None
    
    def evaluate_generation(self, eval_data: List[Dict[str, Any]], 
                          with_context: bool = False) -> Dict[str, Any]:
        """Evaluates accuracy using the Generation method."""
        correct_predictions = 0
        total_questions = len(eval_data)
        invalid_responses = 0
        
        results_by_difficulty = {"easy": {"correct": 0, "total": 0}, 
                               "hard": {"correct": 0, "total": 0}}
        detailed_results = []
        
        logger.info(f"Starting generation evaluation on {total_questions} questions")
        
        for item in tqdm(eval_data, desc="Evaluating (generation)"):
            question = item["question"]
            options = item["options"]
            correct_answer_letter = item["answer_letter"]
            is_hard = item.get("hard", False)
            context = item.get("article_content", "") if with_context else ""
            
            prompt = self.get_evaluation_prompt(question, options, with_context, context)
            prompt += " The correct answer is"
            
            generated_text = self.generate_answer(prompt, max_new_tokens=10)
            predicted_letter = self.extract_answer_letter(generated_text)
            
            is_correct = False
            if predicted_letter is None:
                invalid_responses += 1
            else:
                is_correct = (predicted_letter == correct_answer_letter)
                
            if is_correct:
                correct_predictions += 1
            
            # Aggregate by difficulty
            difficulty = "hard" if is_hard else "easy"
            results_by_difficulty[difficulty]["total"] += 1
            if is_correct:
                results_by_difficulty[difficulty]["correct"] += 1
            
            detailed_results.append({
                "question": question,
                "predicted_letter": predicted_letter,
                "correct_letter": correct_answer_letter,
                "generated_text": generated_text,
                "is_correct": is_correct,
                "is_hard": is_hard,
                "is_valid": predicted_letter is not None
            })
        
        # Calculate final metrics
        accuracy = correct_predictions / total_questions if total_questions > 0 else 0
        easy_acc = (results_by_difficulty["easy"]["correct"] / 
                   results_by_difficulty["easy"]["total"] 
                   if results_by_difficulty["easy"]["total"] > 0 else 0)
        hard_acc = (results_by_difficulty["hard"]["correct"] / 
                   results_by_difficulty["hard"]["total"] 
                   if results_by_difficulty["hard"]["total"] > 0 else 0)
        
        return {
            "method": "generation",
            "with_context": with_context,
            "accuracy": accuracy,
            "correct": correct_predictions,
            "total": total_questions,
            "invalid_responses": invalid_responses,
            "invalid_rate": invalid_responses / total_questions if total_questions > 0 else 0,
            "easy_accuracy": easy_acc,
            "hard_accuracy": hard_acc,
            "easy_questions": results_by_difficulty["easy"]["total"],
            "hard_questions": results_by_difficulty["hard"]["total"],
            "detailed_results": detailed_results
        }
    
    def evaluate_article(self, article_id: str, quality_loader: QualityDatasetLoader,
                       methods: List[str] = ["log_likelihood", "generation"],
                       with_context: bool = False) -> Dict[str, Any]:
        """Runs evaluation for a specific article ID."""
        article_info = quality_loader.get_article_by_id(article_id)
        if not article_info:
            raise ValueError(f"Article not found: {article_id}")
        
        eval_data = quality_loader.create_evaluation_data(
            article_info['article_content'], article_id
        )
        
        results = {
            "article_id": article_id,
            "num_questions": len(eval_data),
            "evaluations": {}
        }
        
        for method in methods:
            if method == "log_likelihood":
                result = self.evaluate_loglikelihood(eval_data, with_context)
            elif method == "generation":
                result = self.evaluate_generation(eval_data, with_context)
            else:
                logger.warning(f"Unknown evaluation method: {method}")
                continue
            
            results["evaluations"][method] = result
        
        return results

def create_quality_evaluator(model_path: str, device: str = "cuda:0") -> QualityEvaluator:
    """Factory function to create a QualityEvaluator instance."""
    from ..models.model_loader import load_model_and_tokenizer
    
    logger.info(f"Loading model {model_path} for evaluation")
    model, tokenizer = load_model_and_tokenizer(model_path, device, use_chat_template=True)
    
    return QualityEvaluator(model, tokenizer, device)

if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)
    
    # Load QuALITY dataset
    from ..datasets.quality_loader import load_quality_dataset
    quality_loader = load_quality_dataset()
    
    # Create evaluator
    evaluator = create_quality_evaluator("Qwen/Qwen3-8B")
    
    # Test with the first article
    articles = quality_loader.get_article_list()
    if articles:
        first_article_id = articles[0]['article_id']
        print(f"Testing evaluation on {first_article_id}")
        
        results = evaluator.evaluate_article(
            first_article_id, 
            quality_loader, 
            methods=["generation"],  # Use generation only for quick testing
            with_context=False
        )
        
        print(f"Results: {results['evaluations']['generation']['accuracy']:.4f}")
        print(f"Easy: {results['evaluations']['generation']['easy_accuracy']:.4f}")
        print(f"Hard: {results['evaluations']['generation']['hard_accuracy']:.4f}")