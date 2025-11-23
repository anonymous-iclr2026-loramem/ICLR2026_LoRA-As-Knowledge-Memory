# scripts/prepare_base_model.py
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model Selection ---
# Uncomment the model you wish to download

# # Qwen3-0.6B
# MODEL_ID = "Qwen/Qwen3-0.6B"
# OUTPUT_PATH = "models/Qwen3-0.6B_base"

# # Qwen3-1.7B
# MODEL_ID = "Qwen/Qwen3-1.7B"
# OUTPUT_PATH = "models/Qwen3-1.7B_base"

# # Qwen3-4B
# MODEL_ID = "Qwen/Qwen3-4B"
# OUTPUT_PATH = "models/Qwen3-4B_base"

# # Qwen3-8B
# MODEL_ID = "Qwen/Qwen3-8B"
# OUTPUT_PATH = "models/Qwen3-8B_base"

# Qwen3-14B (Currently Active)
MODEL_ID = "Qwen/Qwen3-14B"
OUTPUT_PATH = "models/Qwen3-14B_base"

# # Llama-3.1-8B-Instruct
# MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
# OUTPUT_PATH = "models/Llama-3.1-8B-Instruct_base"

# # Gemma-3-12B
# MODEL_ID = "google/gemma-3-12b-it"
# OUTPUT_PATH = "models/gemma-3-12b-it_base"

def main():
    # Check if model already exists to avoid re-downloading
    if os.path.exists(os.path.join(OUTPUT_PATH, 'config.json')):
        logging.info(f"Model already exists at '{OUTPUT_PATH}'. Skipping download.")
        return

    logging.info(f"Downloading model '{MODEL_ID}' to '{OUTPUT_PATH}'...")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Save to local path
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    logging.info("✅ Model download and save complete!")

if __name__ == "__main__":
    main()