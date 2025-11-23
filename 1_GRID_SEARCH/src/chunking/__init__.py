# src/chunking/__init__.py

import logging
from .fixed_token import FixedTokenChunker

def get_chunker(config):
    """
    Returns the appropriate Chunker instance based on the configuration.
    """
    method = config.get('chunking', {}).get('method', 'fixed_token')
    logging.info(f"Using '{method}' chunking strategy.")

    if method == "fixed_token":
        return FixedTokenChunker(config)
    else:
        raise ValueError(f"Unsupported chunking method: {method}")