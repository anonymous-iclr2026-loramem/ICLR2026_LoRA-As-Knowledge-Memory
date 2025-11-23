# src/synthesis/generation_models.py

import logging
from vllm import LLM, SamplingParams
from typing import List

class BaseGenerationModel:
    """Base class serving as a common interface for generation models."""
    def generate(self, prompts: List[str], sampling_params: SamplingParams):
        raise NotImplementedError

class VLLMLocal(BaseGenerationModel):
    """
    Model wrapper for text generation using vLLM on local GPUs.
    """
    def __init__(self, model_id: str):
        logging.info(f"Initializing vLLM model: {model_id}")
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=1,  # Usually 1 for single experiments
            trust_remote_code=True,
            dtype="float16",
        )
        logging.info("vLLM model initialization complete.")

    def generate(self, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
        """
        Performs text generation using vLLM for the given list of prompts.
        """
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

