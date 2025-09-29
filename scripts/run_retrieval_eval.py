"""
Retrieval Evaluation Script
Compares different retrieval approaches and selects the best one
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from app.services.rag import RagOrchestrator
from app.services.embeddings import get_embedding_model, embed_texts
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

# Evaluation questions with expected relevant topics
EVAL_QUESTIONS = [
    {
        "query": "What is gradient descent and how does it work?",
        "relevant_topics": ["gradient descent", "optimization", "learning rate", "backpropagation"],
        "category": "technical"
    },
    {
        "query": "Explain the difference between bagging and boosting",
        "relevant_topics": ["bagging", "boosting", "ensemble", "random forest", "adaboost"],
        "category": "technical"
    },
    {
        "query": "What is overfitting and how to prevent it?",
        "relevant_topics": ["overfitting", "regularization", "dropout", "cross-validation"],
        "category": "technical"
    },
    {
        "query": "Tell me about a time you handled a difficult stakeholder",
        "relevant_topics": ["stakeholder", "communication", "conflict", "collaboration"],
        "category": "behavioral"
    },
    {
        "query": "How do CNNs work for image classification?",
        "relevant_topics": ["CNN", "convolution", "pooling", "image", "neural network"],
        "category": "technical"
    }
]

class RetrievalEvaluator:
    def __init__(self):
        self.qdrant = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333"))
        )
        self.collection = os.getenv("QDRANT_COLLECTION", "interview_chunks")
        self.embedder = get_embedding_model()
        self.rag = RagOrchestrator(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            collection=self.collection
        )
        
    def load_corpus(self) -> Tuple[List[str], List[Dict]]:
        """Load all documents from Qdrant for BM25"""
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
                all_metadata.append(point.payload)
            
            offset += limit
            
        return all_docs, all_metadata
    
    def evaluate_dense_retrieval(self, questions: List[Dict], top_k: int = 5) -> Dict:
        """Evaluate vector-based retrieval"""
        scores = []
        latencies = []
        
        for q in tqdm(questions, desc="Dense Retrieval"):
            import time
            start = time.time()
            results = self.rag.retrieve(q["query"], top_k=top_k)
            latency = time.time() - start
            latencies.append(latency)
            
            # Calculate relevance score
            score = self._calculate_relevance_score(results, q["relevant_topics"])
            scores.append(score)
        
        return {
            "method": "dense_retrieval",
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "mean_latency": np.mean(latencies),
            "scores": scores
        }
    
    def evaluate_bm25(self, questions: List[Dict], top_k: int = 5) -> Dict:
        """Evaluate BM25 keyword-based retrieval"""
        # Load corpus
        docs, metadata = self.load_corpus()
        tokenized_docs = [word_tokenize(doc.lower()) for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)
        
        scores = []
        latencies = []
        
        for q in tqdm(questions, desc="BM25 Retrieval"):
            import time
            start = time.time()
            
            # BM25 search
            query_tokens = word_tokenize(q["query"].lower())
            doc_scores = bm25.get_scores(query_tokens)
            top_indices = np.argsort(doc_scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    "text": docs[idx],
                    "score": doc_scores[idx],
                    **metadata[idx]
                })
            
            latency = time.time() - start
            latencies.append(latency)
            
            # Calculate relevance score
            score = self._calculate_relevance_score(results, q["relevant_topics"])
            scores.append(score)
        
        return {
            "method": "bm25",
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "mean_latency": np.mean(latencies),
            "scores": scores
        }
    
    def evaluate_hybrid(self, questions: List[Dict], top_k: int = 5, alpha: float = 0.5) -> Dict:
        """Evaluate hybrid retrieval (dense + BM25)"""
        # Load corpus for BM25
        docs, metadata = self.load_corpus()
        tokenized_docs = [word_tokenize(doc.lower()) for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)
        
        scores = []
        latencies = []
        
        for q in tqdm(questions, desc="Hybrid Retrieval"):
            import time
            start = time.time()
            
            # Dense retrieval
            dense_results = self.rag.retrieve(q["query"], top_k=top_k*2)
            
            # BM25 retrieval
            query_tokens = word_tokenize(q["query"].lower())
            doc_scores = bm25.get_scores(query_tokens)
            top_indices = np.argsort(doc_scores)[::-1][:top_k*2]
            
            # Combine scores
            combined_scores = {}
            
            # Add dense scores
            for i, res in enumerate(dense_results):
                text = res["text"]
                combined_scores[text] = alpha * (1.0 / (i + 1))  # Reciprocal rank
            
            # Add BM25 scores
            for idx in top_indices:
                text = docs[idx]
                if text in combined_scores:
                    combined_scores[text] += (1 - alpha) * (1.0 / (list(top_indices).index(idx) + 1))
                else:
                    combined_scores[text] = (1 - alpha) * (1.0 / (list(top_indices).index(idx) + 1))
            
            # Sort by combined score
            sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            results = []
            for text, score in sorted_results:
                results.append({"text": text, "score": score})
            
            latency = time.time() - start
            latencies.append(latency)
            
            # Calculate relevance score
            score = self._calculate_relevance_score(results, q["relevant_topics"])
            scores.append(score)
        
        return {
            "method": f"hybrid_alpha_{alpha}",
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "mean_latency": np.mean(latencies),
            "scores": scores,
            "alpha": alpha
        }
    
    def _calculate_relevance_score(self, results: List[Dict], relevant_topics: List[str]) -> float:
        """Calculate relevance score based on keyword presence"""
        if not results:
            return 0.0
        
        scores = []
        for i, result in enumerate(results):
            text = result.get("text", "").lower()
            matches = sum(1 for topic in relevant_topics if topic.lower() in text)
            position_weight = 1.0 / (i + 1)  # Higher weight for top results
            scores.append((matches / len(relevant_topics)) * position_weight)
        
        return sum(scores) / len(results)
    
    def run_evaluation(self) -> Dict:
        """Run complete evaluation"""
        print("Starting Retrieval Evaluation...")
        print(f"Using {len(EVAL_QUESTIONS)} evaluation questions")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(EVAL_QUESTIONS),
            "methods": []
        }
        
        # Evaluate dense retrieval
        dense_results = self.evaluate_dense_retrieval(EVAL_QUESTIONS)
        results["methods"].append(dense_results)
        
        # Evaluate BM25
        bm25_results = self.evaluate_bm25(EVAL_QUESTIONS)
        results["methods"].append(bm25_results)
        
        # Evaluate hybrid with different alpha values
        for alpha in [0.3, 0.5, 0.7]:
            hybrid_results = self.evaluate_hybrid(EVAL_QUESTIONS, alpha=alpha)
            results["methods"].append(hybrid_results)
        
        # Find best method
        best_method = max(results["methods"], key=lambda x: x["mean_score"])
        results["best_method"] = best_method["method"]
        results["best_score"] = best_method["mean_score"]
        
        # Save results
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"retrieval_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("RETRIEVAL EVALUATION RESULTS")
        print("="*50)
        
        df_results = pd.DataFrame(results["methods"])
        df_results = df_results.sort_values("mean_score", ascending=False)
        
        print("\nMethod Comparison:")
        print(df_results[["method", "mean_score", "std_score", "mean_latency"]].to_string(index=False))
        
        print(f"\n🏆 Best Method: {results['best_method']}")
        print(f"   Score: {results['best_score']:.3f}")
        
        print(f"\nFull results saved to: {output_file}")
        
        return results


if __name__ == "__main__":
    evaluator = RetrievalEvaluator()
    evaluator.run_evaluation()