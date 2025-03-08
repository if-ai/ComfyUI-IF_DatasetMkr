# utils.py
import os
import io
import re
import yaml
import logging
import requests
from io import BytesIO
from aiohttp import web
from dotenv import load_dotenv
from typing import Tuple, Optional, Dict, Union, List, Any
import node_helpers
import folder_paths


from typing import Union, List, Tuple

logger = logging.getLogger(__name__)


def get_api_key(api_key_name, engine):
    local_engines = ["ollama", "llamacpp", "kobold", "lmstudio", "textgen", "sentence_transformers"]
    
    if engine.lower() in local_engines:
        print(f"You are using {engine} as the engine, no API key is required.")
        return "1234"
    
    # Try to get the key from .env first
    load_dotenv()
    api_key = os.getenv(api_key_name)
    
    if api_key:
        print(f"API key for {api_key_name} found in .env file")
        return api_key
    
    # If .env is empty, get the key from os.environ
    api_key = os.getenv(api_key_name)
    
    if api_key:
        print(f"API key for {api_key_name} found in environment variables")
        return api_key
    
    print(f"API key for {api_key_name} not found in .env file or environment variables")
    raise ValueError(f"{api_key_name} not found. Please set it in your .env file or as an environment variable.")


def format_response(self, response):
        """
        Format the response by adding appropriate line breaks and paragraph separations.
        """
        paragraphs = re.split(r"\n{2,}", response)

        formatted_paragraphs = []
        for para in paragraphs:
            if "```" in para:
                parts = para.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 1:  # This is a code block
                        parts[i] = f"\n```\n{part.strip()}\n```\n"
                para = "".join(parts)
            else:
                para = para.replace(". ", ".\n")

            formatted_paragraphs.append(para.strip())

        return "\n\n".join(formatted_paragraphs)
