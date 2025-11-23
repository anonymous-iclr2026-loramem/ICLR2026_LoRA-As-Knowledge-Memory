# src/synthesis/synthesizer.py

import os
import json
import logging
import hashlib
from typing import List, Dict
from vllm import SamplingParams

from .generators import SummaryGenerator, QAGenerator
from .generation_models import VLLMLocal

class Synthesizer:
    def __init__(self, config):
        self.config = config
        self.synthesis_config = config['synthesis']
        self.cache_enabled = self.synthesis_config.get('use_cache', False)
        self.cache_dir = "data/synthesis_cache"

        if self.cache_enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Generation model factory
        gen_model_name = self.synthesis_config['generator_model']
        if gen_model_name == "self-dcd":
            # "self-dcd" implies using vLLM as the base model
            model_id = config['model']['base_model_id']
            self.generation_model = VLLMLocal(model_id)
        # TODO: Implement support for other models (e.g., "gpt4o")
        else:
            raise ValueError(f"Unsupported generation model: {gen_model_name}")

        # Generator factory
        self.generators = {
            "summary": SummaryGenerator(),
            "qa": QAGenerator(),
        }

    def _get_cache_path(self, chunk: str, recipe_str: str) -> str:
        """Generates a unique cache file path based on chunk content and recipe."""
        unique_string = chunk + recipe_str
        hash_id = hashlib.md5(unique_string.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_id}.jsonl")

    def synthesize_for_chunk(self, chunk: str) -> List[Dict]:
        """Executes the full synthetic data generation pipeline for a single chunk."""
        recipe = self.synthesis_config['recipe']
        recipe_str = json.dumps(recipe, sort_keys=True)
        cache_path = self._get_cache_path(chunk, recipe_str)

        # 1. Check cache
        if self.cache_enabled and os.path.exists(cache_path):
            logging.info(f"Loading cached synthetic data: {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f]

        # 2. Generate data if no cache
        logging.info("No cache found. Starting new synthetic data generation...")
        all_new_data = []
        
        # Parse recipe and execute prompt generation
        for item in recipe:
            item_type, count = item.split(':')
            count = int(count)
            
            if item_type not in self.generators:
                logging.warning(f"Unsupported generation type: {item_type}")
                continue
            
            generator = self.generators[item_type]
            prompts = generator.create_prompts(chunk, count)
            
            # Configure vLLM SamplingParams
            sampling_params = SamplingParams(
                n=count if item_type == "summary" else 1,
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024
            )

            generation_outputs = self.generation_model.generate(prompts, sampling_params)
            processed_data = generator.postprocess(generation_outputs)
            
            # Append original chunk (context) information to generated data
            for data_point in processed_data:
                data_point['context'] = chunk
            
            all_new_data.extend(processed_data)

        # 3. Save cache
        if self.cache_enabled:
            logging.info(f"Saving generated data to cache: {cache_path}")
            with open(cache_path, 'w', encoding='utf-8') as f:
                for data_point in all_new_data:
                    f.write(json.dumps(data_point, ensure_ascii=False) + '\n')
        
        return all_new_data