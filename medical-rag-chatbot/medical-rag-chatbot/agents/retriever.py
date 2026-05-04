"""
agents/retriever.py
-------------------
RetrieverAgent — semantic search over the FAISS vector index.

Responsibilities:
  - Embed the planner's search query using sentence-transformers
  - Query the FAISS index for top-k nearest neighbors
  - Return ranked, scored document chunks as context for the generator

This is the core RAG component — grounding responses in real
doctor-patient conversations rather than model hallucination.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, TOP_K


class RetrieverAgent:
    """
    Performs semantic search over a pre-built FAISS index.

    Uses cosine similarity (via L2-normalized inner product) to find
    the most semantically relevant medical conversations for a given query.

    Args:
        index: FAISS index loaded from disk
        documents: list of raw document strings corresponding to index vectors
        top_k: number of documents to retrieve per query
    """

    def __init__(self, index: faiss.Index, documents: list, top_k: int = TOP_K):
        self.name = "RetrieverAgent"
        self.index = index
        self.documents = documents
        self.top_k = top_k

        print(f"  [RetrieverAgent] Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        print(f"  [RetrieverAgent] Ready — index has {index.ntotal:,} vectors")

    def retrieve(self, search_query: str) -> list[dict]:
        """
        Encodes the query and retrieves top-k most similar documents.

        Args:
            search_query: Planner-rewritten query string

        Returns:
            List of dicts, each with:
              - document: full text of the retrieved conversation
              - similarity_score: cosine similarity (0–1, higher = more relevant)
              - rank: 1-indexed position in results
        """
        # Embed and normalize the query
        query_vec = self.embedder.encode(
            [search_query], convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(query_vec)

        # Search the index
        scores, indices = self.index.search(query_vec, self.top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            results.append({
                "document": self.documents[idx],
                "similarity_score": float(score),
                "rank": rank,
            })

        top_score = results[0]["similarity_score"] if results else 0.0
        print(
            f"  [RetrieverAgent] Retrieved {len(results)} docs "
            f"(top similarity: {top_score:.3f})"
        )
        return results

    def retrieve_with_mmr(self, search_query: str, lambda_param: float = 0.7) -> list[dict]:
        """
        Maximal Marginal Relevance retrieval — reduces redundancy in results.
        Balances relevance to query vs. diversity among retrieved docs.

        Args:
            search_query: query string
            lambda_param: 1.0 = pure relevance, 0.0 = pure diversity

        Returns:
            Diversified list of top-k documents
        """
        # Get a larger candidate pool first
        candidate_pool = self.top_k * 3
        query_vec = self.embedder.encode(
            [search_query], convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, candidate_pool)

        candidates = []
        candidate_vecs = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            candidates.append({
                "document": self.documents[idx],
                "similarity_score": float(score),
            })
            # Re-embed candidate for MMR diversity calculation
            vec = self.embedder.encode(
                [self.documents[idx]], convert_to_numpy=True
            ).astype("float32")
            faiss.normalize_L2(vec)
            candidate_vecs.append(vec[0])

        # MMR selection
        selected = []
        selected_vecs = []
        remaining = list(range(len(candidates)))

        for _ in range(min(self.top_k, len(candidates))):
            best_idx = None
            best_score = float("-inf")
            for i in remaining:
                relevance = candidates[i]["similarity_score"]
                if selected_vecs:
                    redundancy = max(
                        float(np.dot(candidate_vecs[i], sv))
                        for sv in selected_vecs
                    )
                else:
                    redundancy = 0.0
                mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            selected.append({**candidates[best_idx], "rank": len(selected) + 1})
            selected_vecs.append(candidate_vecs[best_idx])
            remaining.remove(best_idx)

        print(f"  [RetrieverAgent] MMR retrieved {len(selected)} diverse docs")
        return selected
