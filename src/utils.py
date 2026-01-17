"""
Utility functions for the AI Research Assistant.
"""

import os
import json
import logging
from typing import List, Dict, Any, Callable
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ResearchAssistant")

# Rich console for prettier output
console = Console()

def load_dotenv_if_exists():
    """Load environment variables from .env file if it exists."""
    try:
        from dotenv import load_dotenv
        if os.path.exists('.env'):
            load_dotenv()
            logger.info("Loaded environment variables from .env file")
    except ImportError:
        logger.warning("python-dotenv not installed. Skipping .env loading.")

def get_tokenizer(model_name: str = "NousResearch/Llama-2-7b-hf"):
    """Get a tokenizer for the specified model."""
    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Failed to load tokenizer {model_name}: {e}")
        # Fallback to a simpler tokenizer
        return AutoTokenizer.from_pretrained("bert-base-uncased")

def token_length_function(text: str, tokenizer=None) -> int:
    """Calculate the token length of a text using a tokenizer."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    
    try:
        if not isinstance(text, str):
            text = str(text)
        return len(tokenizer.encode(text))
    except Exception as e:
        logger.error(f"Tokenization error for text: {text[:100]} - {e}")
        return 0  # or raise, depending on how you want to handle it

def display_markdown(content: str, title: str = None):
    """Display markdown content in a panel."""
    console.print(Panel(
        Markdown(content),
        title=title if title else "",
        border_style="cyan"
    ))

def ensure_dir(directory: str) -> str:
    """Ensure a directory exists and return its path."""
    os.makedirs(directory, exist_ok=True)
    return directory

def safe_json_loads(json_str: str, default=None):
    """Safely load a JSON string, returning default if parsing fails."""
    try:
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}")
        return default if default is not None else {}