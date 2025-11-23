# src/synthesis/prompt_templates.py

def get_summary_prompt(text_chunk: str) -> str:
    """Returns a prompt for summary generation."""
    # Request summary length proportional to the word count of the chunk
    target_words = max(50, len(text_chunk.split()) // 4)
    return (
        f"Summarize the following text in around {target_words} words without omitting any important details.\n"
        "The summary should be grammatically correct and summarize all the different sections in the text.\n\n"
        "********** Text **********\n"
        f"{text_chunk}\n"
        "********************"
    )

def get_qa_prompt(text_chunk: str, num_questions: int = 4) -> str:
    """Returns a prompt for generating Q&A pairs."""
    return (
        f"Create {num_questions} questions that can be answerable from the following text, along with their answers.\n"
        "Strive to generate challenging questions that require aggregating information across the provided text.\n"
        "Format your answer as follows:\n\n"
        "<question id='1'>QUESTION 1 HERE</question>\n"
        "<answer id='1'>ANSWER 1 HERE</answer>\n"
        "<question id='2'>QUESTION 2 HERE</question>\n"
        "<answer id='2'>ANSWER 2 HERE</answer>\n\n"
        "********** Text **********\n"
        f"{text_chunk}\n"
        "********************"
    )

def get_paraphrase_prompt(text_chunk: str) -> str:
    """Returns a prompt for paraphrase generation."""
    return (
        "Paraphrase the following text while preserving its exact meaning and all key information. "
        "Use different sentence structures, vocabulary, and phrasing, but ensure that all facts, "
        "entities, numbers, and technical terms remain accurate. The paraphrase should be natural "
        "and coherent while being distinctly different from the original.\n\n"
        "********** Original Text **********\n"
        f"{text_chunk}\n"
        "********************\n\n"
        "Paraphrased version:"
    )

def get_rewrite_prompt(text_chunk: str) -> str:
    """Returns a prompt for text rewriting (PRAG style)."""
    return (
        "Rewrite the following passage. While keeping the entities, proper nouns, and key details "
        "such as names, locations, and terminology intact, create a new version of the text that "
        "expresses the same ideas in a different way. Make sure the revised passage is distinct "
        "from the original one, but preserves the core meaning and relevant information.\n\n"
        "********** Original Text **********\n"
        f"{text_chunk}\n"
        "********************\n\n"
        "Rewritten version:"
    )

def get_json_qa_prompt(text_chunk: str, num_questions: int = 3) -> str:
    """Returns a prompt for JSON-formatted Q&A generation (PRAG style)."""
    return f"""I will provide a passage of text, and you need to generate {num_questions} different questions based on the content of this passage. Each question should be answerable using the information provided in the passage. Additionally, please provide an appropriate answer for each question derived from the passage.

You need to generate the question and answer in the following format:
[
    {{
        "question": "What is the capital of France?",
        "answer": "Paris"
    }},
    {{
        "question": "Who wrote 'Hamlet'?",
        "answer": "William Shakespeare"
    }}
]

This list should have at least {num_questions} elements. You only need to output this list in the above format.

Passage:
{text_chunk}"""

# Advanced augmentation prompt templates
def get_easy_proposition_prompt(text_chunk: str, num_facts: int = 20) -> str:
    """Returns a prompt for extracting key facts (propositions)."""
    return f"""Extract exactly {num_facts} of the most important and atomic Key Facts from the following text.

[REQUIREMENTS]
1. Focus on definitions, core claims, and quantitative data.
2. Each fact must be a single, self-contained piece of information.
3. For each fact, provide the original sentence(s) from the text as 'evidence'.
4. Your entire output must be ONLY a single JSON list of objects.

[OUTPUT FORMAT]
[
  {{
    "proposition": "The first extracted key fact sentence.",
    "evidence": "The original sentence from the text that supports this fact."
  }},
  ...
]

[FULL DOCUMENT]
{text_chunk}"""

def get_medium_relation_prop_prompt(text_chunk: str, num_props: int = 20) -> str:
    """Returns a prompt for generating relational propositions."""
    return f"""Read the entire text below and generate {num_props} new Relational Propositions by logically connecting information from different parts of the document.

[REQUIREMENTS]
1. The proposition must describe a relationship (e.g., cause-and-effect, comparison) that is not explicitly stated but can be inferred.
2. For each proposition, you MUST explain the step-by-step reasoning based on evidence from the original text.
3. Your entire output must be ONLY a single JSON list of objects.

[OUTPUT FORMAT EXAMPLE]
[
  {{
    "proposition": "The first generated relational proposition.",
    "reasoning": "1. [Evidence 1 from text]... 2. [Evidence 2 from text]... 3. Therefore, this proposition can be inferred."
  }},
  ...
]

[FULL DOCUMENT]
{text_chunk}"""

def get_medium_relation_qa_prompt(text_chunk: str, num_questions: int = 10) -> str:
    """Returns a prompt for generating relational Q&A."""
    return f"""Read the entire text below and generate {num_questions} relational questions that REQUIRE combining information from different parts of the document to be answered.

[REQUIREMENTS]
1. The question must not be answerable with a single, isolated piece of information.
2. Along with the question, provide a definitive 'answer' and the step-by-step 'reasoning' to derive it, based on evidence from the original text.
3. Your entire output must be ONLY a single JSON list of objects.

[OUTPUT FORMAT EXAMPLE]
[
  {{
    "question": "The first generated relational question?",
    "answer": "The final answer to the question.",
    "reasoning": "1. [Evidence 1 from text]... 2. [Evidence 2 from text]... 3. Therefore, the answer is..."
  }},
  ...
]

[FULL DOCUMENT]
{text_chunk}"""

def get_hard_implication_prompt(text_chunk: str, num_implications: int = 10) -> str:
    """Returns a prompt for generating synthesized implications."""
    return f"""After understanding the core argument of the entire text, generate {num_implications} of the most important and profound Synthesized Implications.

[REQUIREMENTS]
1. The implication should be a high-level insight, not a simple summary. It must be a deep understanding synthesized from multiple facts throughout the document.
2. For each implication, provide detailed 'reasoning' from a holistic perspective.
3. Your entire output must be ONLY a single JSON list of objects.

[OUTPUT FORMAT EXAMPLE]
[
  {{
    "implication": "The first generated synthesized implication.",
    "reasoning": "The document argues that... This implies... when viewed in the broader context of... because..."
  }},
  ...
]

[FULL DOCUMENT]
{text_chunk}"""

# TODO: Add other prompt templates such as entity_graph, implication, etc., according to future research plans.