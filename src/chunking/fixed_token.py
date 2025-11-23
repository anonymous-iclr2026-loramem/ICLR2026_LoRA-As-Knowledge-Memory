# src/chunking/fixed_token.py

import logging
from transformers import AutoTokenizer
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

class FixedTokenChunker:
    """
    Chunker that splits text based on a fixed token length using the LangChain library.
    """
    def __init__(self, config):
        self.config = config
        
        # TODO: Currently hardcoded, but update to read these values from config in the future.
        self.chunk_size = 2048
        self.chunk_overlap = 200
        
        logging.info(f"FixedTokenChunker (LangChain) initialized: size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        model_id = config['model']['base_model_id']
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Create LangChain TextSplitter based on the tokenizer
        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk(self, text: str) -> List[str]:
        """
        Generates and returns a list of chunks from the given text based on token count.
        
        Args:
            text (str): The original text to split.
            
        Returns:
            List[str]: A list of split text chunks.
        """
        if not text:
            return []
            
        chunks = self.text_splitter.split_text(text)
            
        logging.info(f"Split original text into {len(chunks)} chunks.")
        return chunks