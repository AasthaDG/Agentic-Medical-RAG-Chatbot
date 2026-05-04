"""
config.py
---------
Central configuration for the Medical RAG Chatbot.
All modules import from here — change values here, affects the whole system.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# ── Embedding model (HuggingFace sentence-transformers) ──
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, strong semantic quality

# ── Dataset ──────────────────────────────────────────
DATASET_NAME    = "lavita/ChatDoctor-HealthCareMagic-100k"
DATASET_SIZE    = int(os.getenv("DATASET_SIZE", 10000))  # matches resume: 10K+
RANDOM_SEED     = 42

# ── FAISS index paths ─────────────────────────────────
INDEX_PATH     = "data/medical_rag.index"
DOCUMENTS_PATH = "data/documents.pkl"

# ── Retrieval ─────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", 5))  # top-k docs retrieved per query

# ── Generation ────────────────────────────────────────
MAX_TOKENS_PLANNER   = 150
MAX_TOKENS_GENERATOR = 500
TEMPERATURE_PLANNER  = 0.1   # low temp = deterministic intent classification
TEMPERATURE_GENERATOR = 0.3  # slightly creative but grounded

# ── Evaluation ────────────────────────────────────────
EVAL_RESULTS_PATH = "evaluation/benchmark_results.csv"
EVAL_CHART_PATH   = "evaluation/benchmark_chart.png"
