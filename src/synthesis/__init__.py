# src/synthesis/__init__.py

from .synthesizer import Synthesizer
from .phonebook_generator import PhoneBookSynthesizer

def get_synthesizer(config):
    """
    Returns the appropriate Synthesizer based on the configuration.
    """
    generator = config.get('synthesis', {}).get('generator', 'default')
    
    if generator == 'phonebook':
        return PhoneBookSynthesizer(config)
    else:
        return Synthesizer(config)