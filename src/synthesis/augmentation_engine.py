# src/synthesis/augmentation_engine.py

import os
import gc
import json
import logging
import re
import pandas as pd
import torch
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .prompt_templates import (
    get_summary_prompt, get_qa_prompt, get_paraphrase_prompt, get_rewrite_prompt,
    get_easy_proposition_prompt, get_medium_relation_prop_prompt,
    get_medium_relation_qa_prompt, get_hard_implication_prompt
)

logger = logging.getLogger(__name__)

class AugmentationEngine:
    """Engine that performs data augmentation using 11 different methods."""
    
    # Define 11 augmentation methods
    AUGMENTATION_METHODS = {
        "org_1": {"org": 1},
        "rewrite_1": {"rewrite": 1},
        "paraphrase_1": {"paraphrase": 1},
        "summary_5": {"summary": 5},
        "qa_20": {"qa": 20},
        "hs_10_10_10_10": {"easy_proposition": 10, "medium_relation_prop": 10, 
                           "medium_relation_qa": 10, "hard_implication": 10},
        "org_1_summary_5": {"org": 1, "summary": 5},
        "org_1_qa_10": {"org": 1, "qa": 10},
        "paraphrase_1_qa_10": {"paraphrase": 1, "qa": 10},
        "rewrite_1_qa_10": {"rewrite": 1, "qa": 10},
        "summary_5_qa_10": {"summary": 5, "qa": 10}
    }
    
    def __init__(self, model, tokenizer, device: str = "cuda:0"):
        """
        Args:
            model: Generative model (e.g., Qwen3-8B)
            tokenizer: Tokenizer
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
    def parse_json_from_response(self, text: str) -> List[Dict]:
        """Safely extracts a JSON list block from the LLM response."""
        match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
        if not match:
            logger.warning("Response does not contain a valid JSON list block.")
            return []
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from the extracted block.")
            return []
    
    def parse_qa_xml_response(self, text: str) -> List[Dict[str, str]]:
        """Parses XML-formatted Q&A response."""
        qa_pairs = []
        questions = re.findall(r"<question id='(\d+)'>(.*?)</question>", text, re.DOTALL)
        answers = re.findall(r"<answer id='(\d+)'>(.*?)</answer>", text, re.DOTALL)
        
        for (q_id, question), (a_id, answer) in zip(questions, answers):
            if q_id == a_id:
                qa_pairs.append({
                    "question": question.strip(),
                    "answer": answer.strip()
                })
        return qa_pairs
    
    def generate_text(self, prompt: str, max_tokens: int = 1024, 
                      temperature: float = 0.6, do_sample: bool = True) -> str:
        """Generates text."""
        # Apply Chat template
        from ..models.model_loader import apply_chat_template
        formatted_prompt = apply_chat_template(self.tokenizer, prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        return response
    
    def generate_original(self, text: str, count: int = 1) -> List[str]:
        """Returns original text (no augmentation)."""
        return [text] * count
    
    def generate_rewrite(self, text: str, count: int = 1) -> List[str]:
        """Generates rewritten text."""
        results = []
        prompt = get_rewrite_prompt(text)
        
        for _ in range(count):
            try:
                response = self.generate_text(prompt, max_tokens=1024)
                if response:
                    results.append(response)
            except Exception as e:
                logger.error(f"Failed to generate rewrite: {e}")
                
        return results
    
    def generate_paraphrase(self, text: str, count: int = 1) -> List[str]:
        """Generates paraphrased text."""
        results = []
        prompt = get_paraphrase_prompt(text)
        
        for _ in range(count):
            try:
                response = self.generate_text(prompt, max_tokens=1024)
                if response:
                    results.append(response)
            except Exception as e:
                logger.error(f"Failed to generate paraphrase: {e}")
                
        return results
    
    def generate_summary(self, text: str, count: int = 5) -> List[str]:
        """Generates summary."""
        results = []
        prompt = get_summary_prompt(text)
        
        for _ in range(count):
            try:
                response = self.generate_text(prompt, max_tokens=512)
                if response:
                    results.append(response)
            except Exception as e:
                logger.error(f"Failed to generate summary: {e}")
                
        return results
    
    def generate_qa(self, text: str, count: int = 20) -> List[str]:
        """Generates Q&A (XML parsing method)."""
        results = []
        num_calls = (count + 3) // 4  # Generate 4 at a time
        
        for _ in range(num_calls):
            try:
                prompt = get_qa_prompt(text, num_questions=4)
                response = self.generate_text(prompt, max_tokens=2048)
                
                qa_pairs = self.parse_qa_xml_response(response)
                for qa in qa_pairs[:count - len(results)]:
                    formatted_qa = f"Question: {qa['question']} Answer: {qa['answer']}"
                    results.append(formatted_qa)
                    
                if len(results) >= count:
                    break
            except Exception as e:
                logger.error(f"Failed to generate QA: {e}")
                
        return results[:count]
    
    def generate_easy_proposition(self, text: str, count: int = 10) -> List[str]:
        """Extracts Key Facts."""
        results = []
        prompt = get_easy_proposition_prompt(text, num_facts=count)
        
        try:
            response = self.generate_text(prompt, max_tokens=3072)
            parsed_items = self.parse_json_from_response(response)
            
            for item in parsed_items[:count]:
                formatted_text = item.get("proposition", "")
                if formatted_text:
                    results.append(formatted_text)
        except Exception as e:
            logger.error(f"Failed to generate easy proposition: {e}")
            
        return results
    
    def generate_medium_relation_prop(self, text: str, count: int = 10) -> List[str]:
        """Generates relational propositions."""
        results = []
        prompt = get_medium_relation_prop_prompt(text, num_props=count)
        
        try:
            response = self.generate_text(prompt, max_tokens=4096)
            parsed_items = self.parse_json_from_response(response)
            
            for item in parsed_items[:count]:
                prop = item.get("proposition", "")
                reason = item.get("reasoning", "")
                if prop:
                    formatted_text = f"Proposition: {prop} This is because: {reason}"
                    results.append(formatted_text)
        except Exception as e:
            logger.error(f"Failed to generate medium relation prop: {e}")
            
        return results
    
    def generate_medium_relation_qa(self, text: str, count: int = 10) -> List[str]:
        """Generates relational Q&A."""
        results = []
        prompt = get_medium_relation_qa_prompt(text, num_questions=count)
        
        try:
            response = self.generate_text(prompt, max_tokens=4096)
            parsed_items = self.parse_json_from_response(response)
            
            for item in parsed_items[:count]:
                q = item.get("question", "")
                a = item.get("answer", "")
                reason = item.get("reasoning", "")
                if q and a:
                    formatted_text = f"Question: {q} Reasoning and Answer: {reason}"
                    results.append(formatted_text)
        except Exception as e:
            logger.error(f"Failed to generate medium relation QA: {e}")
            
        return results
    
    def generate_hard_implication(self, text: str, count: int = 10) -> List[str]:
        """Generates comprehensive implications."""
        results = []
        prompt = get_hard_implication_prompt(text, num_implications=count)
        
        try:
            response = self.generate_text(prompt, max_tokens=4096)
            parsed_items = self.parse_json_from_response(response)
            
            for item in parsed_items[:count]:
                imp = item.get("implication", "")
                reason = item.get("reasoning", "")
                if imp:
                    formatted_text = f"A key implication is that {imp}. This can be understood because {reason}"
                    results.append(formatted_text)
        except Exception as e:
            logger.error(f"Failed to generate hard implication: {e}")
            
        return results
    
    def generate_augmentation(self, text: str, method: str) -> List[str]:
        """Performs data augmentation using a specific method."""
        if method not in self.AUGMENTATION_METHODS:
            raise ValueError(f"Unknown augmentation method: {method}")
        
        method_config = self.AUGMENTATION_METHODS[method]
        all_results = []
        
        logger.info(f"Generating augmentation with method: {method}")
        
        # Generate for each type
        for aug_type, count in method_config.items():
            if aug_type == "org":
                results = self.generate_original(text, count)
            elif aug_type == "rewrite":
                results = self.generate_rewrite(text, count)
            elif aug_type == "paraphrase":
                results = self.generate_paraphrase(text, count)
            elif aug_type == "summary":
                results = self.generate_summary(text, count)
            elif aug_type == "qa":
                results = self.generate_qa(text, count)
            elif aug_type == "easy_proposition":
                results = self.generate_easy_proposition(text, count)
            elif aug_type == "medium_relation_prop":
                results = self.generate_medium_relation_prop(text, count)
            elif aug_type == "medium_relation_qa":
                results = self.generate_medium_relation_qa(text, count)
            elif aug_type == "hard_implication":
                results = self.generate_hard_implication(text, count)
            else:
                logger.warning(f"Unknown augmentation type: {aug_type}")
                continue
                
            logger.info(f"Generated {len(results)}/{count} items for {aug_type}")
            all_results.extend(results)
        
        return all_results
    
    def create_training_dataset(self, text: str, method: str) -> pd.DataFrame:
        """Creates a training dataset."""
        augmented_texts = self.generate_augmentation(text, method)
        
        df = pd.DataFrame([{"text": text} for text in augmented_texts if text.strip()])
        logger.info(f"Created training dataset with {len(df)} samples using {method}")
        
        return df
    
    def save_training_dataset(self, text: str, method: str, output_path: str, 
                              item_id: str = None) -> Dict[str, Any]:
        """Creates and saves the training dataset."""
        df = self.create_training_dataset(text, method)
        
        # Create directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Generate metadata
        metadata = {
            "item_id": item_id,
            "augmentation_method": method,
            "original_text_length": len(text),
            "original_token_count": len(text.split()),
            "augmented_samples": len(df),
            "total_augmented_tokens": sum(len(row['text'].split()) for _, row in df.iterrows()),
            "output_path": output_path,
            "method_config": self.AUGMENTATION_METHODS[method]
        }
        
        logger.info(f"Saved {len(df)} augmented samples to {output_path}")
        return metadata

def create_augmentation_engine(model_id: str = "Qwen/Qwen3-8B", 
                             device: str = "cuda:0") -> AugmentationEngine:
    """Initializes and returns the augmentation engine."""
    from ..models.model_loader import load_model_and_tokenizer
    
    logger.info(f"Loading model {model_id} on {device}")
    model, tokenizer = load_model_and_tokenizer(model_id, device, use_chat_template=True)
    
    return AugmentationEngine(model, tokenizer, device)

if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)
    
    # Sample text
    sample_text = """
    Artificial intelligence (AI) refers to the simulation of human intelligence in machines 
    that are programmed to think like humans and mimic their actions. The term may also be 
    applied to any machine that exhibits traits associated with a human mind such as learning 
    and problem-solving. AI systems are designed to perform tasks that typically require 
    human intelligence, such as visual perception, speech recognition, decision-making, 
    and translation between languages.
    """
    
    # Initialize engine
    engine = create_augmentation_engine()
    
    # Test each method
    for method in list(AugmentationEngine.AUGMENTATION_METHODS.keys())[:3]:  # Test first 3 only
        print(f"\n=== Testing {method} ===")
        results = engine.generate_augmentation(sample_text.strip(), method)
        print(f"Generated {len(results)} samples")
        for i, result in enumerate(results[:2], 1):  # Print first 2 only
            print(f"{i}. {result[:100]}...")