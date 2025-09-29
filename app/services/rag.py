"""
Enhanced RAG Service with Hybrid Search, Re-ranking, and Query Rewriting
"""

import os
from typing import List, Dict, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import CrossEncoder

from .embeddings import get_embedding_model, embed_texts

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class EnhancedRAG:
    def __init__(self, qdrant_host: str, qdrant_port: int, collection: str):
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection = collection
        self.embedder = get_embedding_model()
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize cross-encoder for re-ranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Cache for BM25
        self._bm25_index = None
        self._corpus_docs = None
        self._corpus_metadata = None
        
    def query_rewrite(self, query: str, mode: str = "expand") -> str:
        """
        Rewrite user query for better retrieval
        Modes: expand, clarify, correct
        """
        if mode == "expand":
            # Expand abbreviations and add context
            prompt = f"""Expand this data science interview question for better search results.
            Add relevant technical terms and synonyms without changing the meaning.
            
            Original: {query}
            Expanded (keep it concise):"""
            
        elif mode == "clarify":
            # Clarify ambiguous queries
            prompt = f"""Clarify this potentially ambiguous data science question.
            Make it more specific while preserving the intent.
            
            Original: {query}
            Clarified:"""
            
        else:  # correct
            # Fix common misspellings and terms
            prompt = f"""Correct any technical term misspellings in this query.
            Fix only obvious errors, preserve the original intent.
            
            Original: {query}
            Corrected:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a search query optimizer for data science content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            rewritten = response.choices[0].message.content.strip()
            # Don't make it too long
            if len(rewritten) > len(query) * 3:
                return query
            return rewritten
            
        except Exception as e:
            print(f"Query rewrite failed: {e}")
            return query
    
    def _load_bm25_index(self):
        """Load or create BM25 index"""
        if self._bm25_index is not None:
            return
        
        # Load all documents from Qdrant
        limit = 1000
        offset = 0
        all_docs = []
        all_metadata = []
        
        while True:
            results = self.qdrant.scroll(
                collection_name=self.collection,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )[0]
            
            if not results:
                break
            
            for point in results:
                text = point.payload.get("text", "")
                all_docs.append(text)
                metadata = point.payload.copy()
                metadata["id"] = point.id
                all_metadata.append(metadata)
            
            offset += limit
        
        # Tokenize and create BM25 index
        tokenized_docs = [word_tokenize(doc.lower()) for doc in all_docs]
        self._bm25_index = BM25Okapi(tokenized_docs)
        self._corpus_docs = all_docs
        self._corpus_metadata = all_metadata
    
    def dense_search(self, query: str, top_k: int = 10, filter: Optional[Filter] = None) -> List[Dict]:
        """Vector-based semantic search"""
        query_vector = embed_texts(self.embedder, [query])[0]
        
        hits = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filter
        )
        
        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append({
                "id": hit.id,
                "score": hit.score,
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "category": payload.get("category", "unknown"),
                "method": "dense"
            })
        
        return results
    
    def sparse_search(self, query: str, top_k: int = 10, category_filter: Optional[str] = None) -> List[Dict]:
        """BM25 keyword-based search"""
        self._load_bm25_index()
        
        query_tokens = word_tokenize(query.lower())
        doc_scores = self._bm25_index.get_scores(query_tokens)
        
        # Apply category filter if specified
        if category_filter:
            for i, metadata in enumerate(self._corpus_metadata):
                if metadata.get("category") != category_filter:
                    doc_scores[i] = -1
        
        # Get top-k indices
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if doc_scores[idx] <= 0:
                continue
                
            metadata = self._corpus_metadata[idx]
            results.append({
                "id": metadata.get("id"),
                "score": float(doc_scores[idx]),
                "text": self._corpus_docs[idx],
                "source": metadata.get("source", ""),
                "category": metadata.get("category", "unknown"),
                "method": "sparse"
            })
        
        return results
    
    def hybrid_search(
        self, 
        query: str, 
        top_k: int = 10, 
        alpha: float = 0.5,
        mode: str = "all"
    ) -> List[Dict]:
        """
        Hybrid search combining dense and sparse retrieval
        alpha: weight for dense search (1-alpha for sparse)
        """
        # Create filter for mode
        filter = None
        category_filter = None
        if mode != "all":
            filter = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=mode))]
            )
            category_filter = mode
        
        # Get results from both methods
        dense_results = self.dense_search(query, top_k=top_k*2, filter=filter)
        sparse_results = self.sparse_search(query, top_k=top_k*2, category_filter=category_filter)
        
        # Combine scores using reciprocal rank fusion
        combined_scores = {}
        
        # Process dense results
        for i, result in enumerate(dense_results):
            text = result["text"]
            # Use reciprocal rank for score normalization
            score = alpha * (1.0 / (i + 1))
            combined_scores[text] = {
                "score": score,
                "data": result
            }
        
        # Process sparse results
        for i, result in enumerate(sparse_results):
            text = result["text"]
            score = (1 - alpha) * (1.0 / (i + 1))
            
            if text in combined_scores:
                combined_scores[text]["score"] += score
            else:
                combined_scores[text] = {
                    "score": score,
                    "data": result
                }
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(), 
            key=lambda x: x[1]["score"], 
            reverse=True
        )[:top_k]
        
        # Format output
        results = []
        for text, info in sorted_results:
            result = info["data"].copy()
            result["score"] = info["score"]
            result["method"] = "hybrid"
            results.append(result)
        
        return results
    
    def rerank_results(self, query: str, results: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        """
        Re-rank results using cross-encoder
        """
        if not results:
            return results
        
        # Prepare pairs for re-ranking
        pairs = [[query, r["text"]] for r in results]
        
        # Get re-ranking scores
        try:
            scores = self.reranker.predict(pairs)
            
            # Add rerank scores to results
            for result, score in zip(results, scores):
                result["rerank_score"] = float(score)
            
            # Sort by rerank score
            reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
            
            if top_k:
                reranked = reranked[:top_k]
            
            return reranked
            
        except Exception as e:
            print(f"Re-ranking failed: {e}")
            return results
    
    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        mode: str = "all",
        use_rewrite: bool = True,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        hybrid_alpha: float = 0.5
    ) -> Tuple[List[Dict], Dict]:
        """
        Complete retrieval pipeline with all enhancements
        Returns: (results, metadata)
        """
        metadata = {
            "original_query": query,
            "rewritten_query": None,
            "search_method": None,
            "reranking_applied": False
        }
        
        # 1. Query rewriting
        if use_rewrite:
            rewritten_query = self.query_rewrite(query, mode="expand")
            metadata["rewritten_query"] = rewritten_query
            search_query = rewritten_query
        else:
            search_query = query
        
        # 2. Search
        if use_hybrid:
            results = self.hybrid_search(
                search_query,
                top_k=top_k*2 if use_rerank else top_k,
                alpha=hybrid_alpha,
                mode=mode
            )
            metadata["search_method"] = "hybrid"
        else:
            # Fallback to dense search
            filter = None
            if mode != "all":
                filter = Filter(
                    must=[FieldCondition(key="category", match=MatchValue(value=mode))]
                )
            results = self.dense_search(
                search_query,
                top_k=top_k*2 if use_rerank else top_k,
                filter=filter
            )
            metadata["search_method"] = "dense"
        
        # 3. Re-ranking
        if use_rerank and results:
            results = self.rerank_results(search_query, results, top_k=top_k)
            metadata["reranking_applied"] = True
        else:
            results = results[:top_k]
        
        return results, metadata
    
    def generate_answer(
        self,
        query: str,
        context: List[Dict],
        mode: str = "all",
        temperature: float = 0.7
    ) -> str:
        """
        Generate answer using retrieved context
        """
        # Prepare context text
        context_text = "\n\n".join([
            f"[Source: {doc.get('source', 'Unknown')}]\n{doc['text']}"
            for doc in context
        ])
        
        # Select appropriate prompt based on mode
        if mode == "behavioral":
            system_prompt = """You are an expert behavioral interview coach. 
            Help candidates structure their responses using the STAR method (Situation, Task, Action, Result).
            Provide guidance that is specific, actionable, and professional."""
            
            user_prompt = f"""Based on the following reference materials, help answer this behavioral interview question.
            
            Reference Materials:
            {context_text}
            
            Question: {query}
            
            Provide a structured response that a candidate could adapt to their own experience."""
            
        else:  # technical or all
            system_prompt = """You are an expert Data Science interview coach with deep knowledge of ML/DL concepts.
            Provide clear, accurate, and comprehensive answers that demonstrate understanding."""
            
            user_prompt = f"""Based on the following technical references, answer this data science interview question.
            
            Technical Context:
            {context_text}
            
            Question: {query}
            
            Provide a clear, structured answer with examples where appropriate."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {e}"