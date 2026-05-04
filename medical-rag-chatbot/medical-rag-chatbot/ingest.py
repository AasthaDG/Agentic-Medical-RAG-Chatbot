"""
ingest.py
---------
Scalable data ingestion and embedding pipeline.

What this does:
  1. Downloads 10K+ real doctor-patient conversations from HuggingFace
  2. Cleans and formats documents
  3. Embeds them using sentence-transformers
  4. Stores embeddings in a FAISS vector index for fast semantic search

Run this ONCE before launching the chatbot:
  python ingest.py

Outputs:
  data/medical_rag.index   ← FAISS index (the vector database)
  data/documents.pkl       ← raw documents (for retrieval display)
"""

import os
import time
import pickle
import argparse

import faiss
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

import pandas as pd

from config import (
    EMBEDDING_MODEL,
    DATASET_NAME,
    DATASET_SIZE,
    RANDOM_SEED,
    INDEX_PATH,
    DOCUMENTS_PATH,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_healthcare_data(n: int = DATASET_SIZE) -> pd.DataFrame:
    """
    Downloads the ChatDoctor HealthCareMagic dataset from HuggingFace.
    Samples n rows and returns a cleaned DataFrame.
    """
    print(f"\n📥  Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    df = (
        pd.DataFrame(dataset)
        .dropna(subset=["input", "output"])
        .query("input.str.len() > 20 and output.str.len() > 50")
        .sample(n=n, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )
    print(f"✅  Loaded {len(df):,} healthcare conversations")
    return df


def format_documents(df: pd.DataFrame) -> list[str]:
    """
    Converts each row into a single document string.
    Format: 'Patient: <question>\nDoctor: <answer>'
    This is what gets embedded and stored in the vector index.
    """
    docs = []
    for _, row in df.iterrows():
        doc = f"Patient: {row['input'].strip()}\nDoctor: {row['output'].strip()}"
        docs.append(doc)
    return docs


def embed_documents(docs: list[str], model_name: str, batch_size: int = 256) -> np.ndarray:
    """
    Encodes all documents into dense vector embeddings using sentence-transformers.
    Processes in batches for memory efficiency.

    Returns:
        np.ndarray of shape (n_docs, embedding_dim)
    """
    print(f"\n🧠  Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)

    print(f"🔄  Embedding {len(docs):,} documents in batches of {batch_size}...")
    t0 = time.time()
    all_embeddings = []

    for i in tqdm(range(0, len(docs), batch_size), desc="Embedding"):
        batch = docs[i : i + batch_size]
        batch_emb = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(batch_emb)

    embeddings = np.vstack(all_embeddings).astype("float32")
    elapsed = time.time() - t0
    print(f"✅  Embeddings shape: {embeddings.shape} — done in {elapsed:.1f}s")
    return embedder, embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds a FAISS flat index using Inner Product (cosine similarity after L2 norm).
    IndexFlatIP is exact — no approximation. Best for ≤ 100K vectors.
    """
    print(f"\n📦  Building FAISS index...")
    faiss.normalize_L2(embeddings)           # normalize → cosine similarity
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"✅  FAISS index built — {index.ntotal:,} vectors, dim={dim}")
    return index


def save_artifacts(index: faiss.Index, documents: list[str]) -> None:
    """Persists the FAISS index and raw documents to disk."""
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(DOCUMENTS_PATH, "wb") as f:
        pickle.dump(documents, f)
    print(f"\n💾  Saved index  → {INDEX_PATH}")
    print(f"💾  Saved docs   → {DOCUMENTS_PATH}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(n: int = DATASET_SIZE):
    print("=" * 60)
    print("  Medical RAG — Data Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load data
    df = load_healthcare_data(n)

    # Step 2: Format documents
    documents = format_documents(df)
    print(f"\n📄  Sample document:\n{documents[0][:300]}...\n")

    # Step 3: Embed
    embedder, embeddings = embed_documents(documents, EMBEDDING_MODEL)

    # Step 4: Build FAISS index
    index = build_faiss_index(embeddings)

    # Step 5: Save
    save_artifacts(index, documents)

    print("\n🎉  Ingestion complete! You can now run: python app.py")
    return index, documents, embedder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical RAG ingestion pipeline")
    parser.add_argument("--n", type=int, default=DATASET_SIZE, help="Number of docs to embed")
    args = parser.parse_args()
    main(n=args.n)
