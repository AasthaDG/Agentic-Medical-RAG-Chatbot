"""
orchestrator.py
---------------
MedicalRAGOrchestrator — the central agentic coordinator.

Wires together: PlannerAgent → RetrieverAgent → GeneratorAgent

This is the entry point for all query handling. The orchestrator:
  1. Accepts a raw user query
  2. Passes it through the three-stage pipeline
  3. Returns a structured result with response + metadata

Usage:
    from orchestrator import MedicalRAGOrchestrator
    rag = MedicalRAGOrchestrator.from_disk()
    result = rag.answer("What are symptoms of high blood pressure?")
    print(result["response"])
"""

import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

from agents import PlannerAgent, RetrieverAgent, GeneratorAgent
from config import (
    INDEX_PATH,
    DOCUMENTS_PATH,
    EMBEDDING_MODEL,
    TOP_K,
)


class MedicalRAGOrchestrator:
    """
    End-to-end agentic RAG orchestrator for medical query handling.

    Architecture:
        User Query
            ↓
        PlannerAgent     ← intent classification + query rewriting
            ↓
        RetrieverAgent   ← semantic search over FAISS index
            ↓
        GeneratorAgent   ← grounded response generation via GPT
            ↓
        Structured Result (response + sources + metadata)
    """

    def __init__(
        self,
        index: faiss.Index,
        documents: list,
        top_k: int = TOP_K,
    ):
        print("\n🔧  Initializing MedicalRAGOrchestrator...")
        self.planner   = PlannerAgent()
        self.retriever = RetrieverAgent(index, documents, top_k=top_k)
        self.generator = GeneratorAgent()
        print("✅  Orchestrator ready — all agents initialized\n")

    @classmethod
    def from_disk(cls, index_path: str = INDEX_PATH, docs_path: str = DOCUMENTS_PATH) -> "MedicalRAGOrchestrator":
        """
        Loads a pre-built FAISS index and document store from disk.
        Run ingest.py first if these files don't exist.
        """
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            raise FileNotFoundError(
                f"Index not found at '{index_path}'. "
                "Run `python ingest.py` first to build the index."
            )

        print(f"📂  Loading FAISS index from {index_path}...")
        index = faiss.read_index(index_path)

        print(f"📂  Loading documents from {docs_path}...")
        with open(docs_path, "rb") as f:
            documents = pickle.load(f)

        print(f"✅  Loaded {index.ntotal:,} vectors and {len(documents):,} documents")
        return cls(index=index, documents=documents)

    def answer(self, user_query: str, use_mmr: bool = False) -> dict:
        """
        Full agentic pipeline: plan → retrieve → generate.

        Args:
            user_query: Raw question from the user
            use_mmr: If True, uses Maximal Marginal Relevance for diverse retrieval

        Returns:
            dict with:
              - query: original user query
              - plan: PlannerAgent output (intent, search_query, is_emergency)
              - retrieved_docs: list of retrieved document dicts
              - response: final generated answer
              - sources: citation-ready source excerpts
              - is_emergency: bool
              - intent: classified query intent
        """
        print(f'\n{"─"*55}')
        print(f'  Query: "{user_query[:70]}..."')
        print(f'{"─"*55}')

        # Stage 1 — Plan
        plan = self.planner.plan(user_query)

        # Stage 2 — Retrieve
        if use_mmr:
            retrieved = self.retriever.retrieve_with_mmr(plan["search_query"])
        else:
            retrieved = self.retriever.retrieve(plan["search_query"])

        # Stage 3 — Generate
        generation = self.generator.generate(plan, retrieved)

        print(f'{"─"*55}\n')

        return {
            "query":          user_query,
            "plan":           plan,
            "retrieved_docs": retrieved,
            "response":       generation["response"],
            "sources":        generation["sources"],
            "is_emergency":   generation["is_emergency"],
            "intent":         generation["intent"],
        }
