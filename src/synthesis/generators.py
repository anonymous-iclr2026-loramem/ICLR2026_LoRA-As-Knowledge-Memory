# src/synthesis/generators.py

import re
from typing import List, Dict
from .prompt_templates import get_summary_prompt, get_qa_prompt

class BaseGenerator:
    """Common interface for generators."""
    def create_prompts(self, chunk: str, count: int) -> List[str]:
        raise NotImplementedError
    
    def postprocess(self, generation_outputs: List[str]) -> List[Dict]:
        raise NotImplementedError

class SummaryGenerator(BaseGenerator):
    def create_prompts(self, chunk: str, count: int) -> List[str]:
        # Summarization uses the same prompt multiple times to ensure diversity via sampling.
        return [get_summary_prompt(chunk)] * count

    def postprocess(self, generation_outputs: List[str]) -> List[Dict]:
        results = []
        for text in generation_outputs:
            if text.strip():
                results.append({
                    "type": "summary",
                    "synthetic_sample": f"Summary: {text.strip()}"
                })
        return results

class QAGenerator(BaseGenerator):
    def create_prompts(self, chunk: str, count: int) -> List[str]:
        # Q&A generates multiple pairs in a single request.
        return [get_qa_prompt(chunk, count)]

    def postprocess(self, generation_outputs: List[str]) -> List[Dict]:
        results = []
        # Process the single result corresponding to the single prompt
        full_text = generation_outputs[0]
        
        questions = re.findall(r"<question id='\d+'>(.*?)</question>", full_text, re.DOTALL)
        answers = re.findall(r"<answer id='\d+'>(.*?)</answer>", full_text, re.DOTALL)

        for q, a in zip(questions, answers):
            if q.strip() and a.strip():
                results.append({
                    "type": "qa",
                    "synthetic_sample": f"Question: {q.strip()} Answer: {a.strip()}"
                })
        return results