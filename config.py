"""
Configuration - load settings from .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = "gpt-4o-mini"   # Fast and cheap

# Embedding model (local, free)
EMBED_MODEL = "all-MiniLM-L6-v2"

# Paths
DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Retrieval
TOP_K = 5

# (No hard error — openai_client.py will fallback to local scoring if key absent/invalid)
