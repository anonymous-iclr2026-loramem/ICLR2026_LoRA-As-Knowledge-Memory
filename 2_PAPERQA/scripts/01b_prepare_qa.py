# paperqa/scripts/01b_prepare_qa.py

import os
import sys
import json
import logging
from pathlib import Path

# Setup Paths
# Adjust paths relative to the project root or environment variables
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPERQA_DIR = PROJECT_ROOT / "data" / "paper_dataset"
SAVE_DIR = PROJECT_ROOT / "paperqa" / "data"
EXISTING_QA_DIR = PROJECT_ROOT / "outputs" / "paperqa_augsc_gpt_data" / "paperqa"

sys.path.append(str(PAPERQA_DIR))

def load_existing_qa_data(paper_id: int) -> list[dict]:
    """Loads and parses existing QA40 data."""
    qa_file = EXISTING_QA_DIR / f"paper_{paper_id}_QA40.jsonl"
    
    if not qa_file.exists():
        logging.warning(f"QA40 file for paper {paper_id} not found: {qa_file}")
        return []
    
    qa_pairs = []
    try:
        with open(qa_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                text = data.get('text', '')
                
                # Extract question and answer from "Question: ... Answer: ..." format
                if text.startswith("Question: ") and " Answer: " in text:
                    parts = text.split(" Answer: ", 1)
                    if len(parts) == 2:
                        question = parts[0].replace("Question: ", "").strip()
                        answer = parts[1].strip()
                        qa_pairs.append({
                            "question": question,
                            "answer": answer
                        })
        
        logging.info(f"Loaded {len(qa_pairs)} QA pairs from paper {paper_id}.")
        return qa_pairs
        
    except Exception as e:
        logging.error(f"Failed to read QA40 file for paper {paper_id}: {e}")
        return []

def process_qa_data_with_formatting(paper_id: int, format_type: str = "bracket") -> list[dict]:
    """Loads existing QA40 data and applies formatting."""
    qa_pairs = load_existing_qa_data(paper_id)
    
    if not qa_pairs:
        logging.warning(f"Could not load QA data for paper {paper_id}.")
        return []
    
    formatted_qa_pairs = []
    for qa in qa_pairs:
        formatted_qa = qa.copy()
        if format_type == "bracket":
            formatted_qa["question"] = f"[Paper ID: {paper_id}] {qa['question']}"
        elif format_type == "natural":
            formatted_qa["question"] = f"In the Paper ID {paper_id}, {qa['question']}"
        else:
            formatted_qa["question"] = f"[Paper ID: {paper_id}] {qa['question']}"  # Default
        
        formatted_qa_pairs.append(formatted_qa)
    
    logging.info(f"Processed {len(formatted_qa_pairs)} QA pairs for paper {paper_id} in {format_type} format.")
    return formatted_qa_pairs

def prepare_individual_qa(papers):
    """
    Generates QA data for each paper.
    Saves in two formatting styles:
    1. [Paper ID: {paper_id}] format
    2. "In the Paper ID {paper_id}" format
    """
    output_dir_bracket = SAVE_DIR / "individual_qa_bracket"
    output_dir_natural = SAVE_DIR / "individual_qa_natural"
    output_dir_bracket.mkdir(parents=True, exist_ok=True)
    output_dir_natural.mkdir(parents=True, exist_ok=True)
    
    logging.info("Processing individual QA data in two formats.")
    
    all_qa_pairs_bracket = []
    all_qa_pairs_natural = []
    
    for paper in papers:
        paper_id = paper['id']
        title = paper['title']
        
        logging.info(f"Processing paper {paper_id}: {title}...")
        
        try:
            # 1. Process QA in Bracket format
            qa_pairs_bracket = process_qa_data_with_formatting(paper_id, format_type="bracket")
            
            if qa_pairs_bracket:
                output_file_bracket = output_dir_bracket / f"paper_{paper_id}_qa.jsonl"
                with open(output_file_bracket, 'w', encoding='utf-8') as f:
                    for qa in qa_pairs_bracket:
                        f.write(json.dumps(qa, ensure_ascii=False) + '\n')
            
            # 2. Process QA in Natural format
            qa_pairs_natural = process_qa_data_with_formatting(paper_id, format_type="natural")
            
            if qa_pairs_natural:
                output_file_natural = output_dir_natural / f"paper_{paper_id}_qa.jsonl"
                with open(output_file_natural, 'w', encoding='utf-8') as f:
                    for qa in qa_pairs_natural:
                        f.write(json.dumps(qa, ensure_ascii=False) + '\n')
            
            if qa_pairs_bracket and qa_pairs_natural:
                logging.info(f"Saved QA data for paper {paper_id} in two formats.")
                logging.info(f"  - Bracket format: '{output_file_bracket}' ({len(qa_pairs_bracket)} items)")
                logging.info(f"  - Natural format: '{output_file_natural}' ({len(qa_pairs_natural)} items)")
                
                all_qa_pairs_bracket.extend(qa_pairs_bracket)
                all_qa_pairs_natural.extend(qa_pairs_natural)
            else:
                logging.warning(f"Could not process QA data for paper {paper_id}.")
            
        except Exception as e:
            logging.error(f"Error processing paper {paper_id}: {e}")
            continue
    
    logging.info(f"Completed generating QA data for {len(papers)} papers in two formats.")
    return all_qa_pairs_bracket, all_qa_pairs_natural

def prepare_concatenated_qa(all_qa_pairs_bracket, all_qa_pairs_natural):
    """
    Concatenates QA data generated from individual papers into two single files.
    """
    output_file_bracket = SAVE_DIR / "concatenated_qa_bracket.jsonl"
    output_file_natural = SAVE_DIR / "concatenated_qa_natural.jsonl"
    
    logging.info("Generating concatenated QA files from individual paper QA data.")
    
    # Save all QA pairs in Bracket format
    with open(output_file_bracket, 'w', encoding='utf-8') as f:
        for qa in all_qa_pairs_bracket:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    # Save all QA pairs in Natural format
    with open(output_file_natural, 'w', encoding='utf-8') as f:
        for qa in all_qa_pairs_natural:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    logging.info(f"Saved {len(all_qa_pairs_bracket)} QA pairs in two formats.")
    logging.info(f"  - Bracket format: '{output_file_bracket}'")
    logging.info(f"  - Natural format: '{output_file_natural}'")

def main():
    """
    Generates both individual and concatenated QA data.
    """
    input_file = PAPERQA_DIR / "paperQA.json"
    
    if not input_file.exists():
        raise FileNotFoundError(f"File not found: '{input_file}'")
    
    logging.info(f"Loading paper data from '{input_file}'.")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    logging.info("=== QA Data Preparation Started ===")
    
    # 1. Generate individual QA data
    all_qa_pairs_bracket, all_qa_pairs_natural = prepare_individual_qa(papers)
    
    # 2. Generate concatenated QA data
    prepare_concatenated_qa(all_qa_pairs_bracket, all_qa_pairs_natural)
    
    logging.info("=== QA Data Preparation Completed ===")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()