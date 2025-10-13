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

# Import from correct location
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.services.rag import EnhancedRAG
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)

# Evaluation questions with expected relevant topics
EVAL_QUESTIONS = [
    {
        "query": "What is gradient descent and how does it work?",
        "relevant_topics": ["gradient", "descent", "optimization", "learning rate"],
        "category": "machine_learning"
    },
    {
        "query": "Explain the difference between bagging and boosting",
        "relevant_topics": ["bagging", "boosting", "ensemble", "random forest"],
        "category": "machine_learning"
    },
    {
        "query": "What is overfitting and how to prevent it?",
        "relevant_topics": ["overfitting", "regularization", "dropout", "validation"],
        "category": "machine_learning"
    },
    {
        "query": "Tell me about a time you handled a difficult stakeholder",
        "relevant_topics": ["stakeholder", "communication", "conflict", "resolution"],
        "category": "behavioral"
    },
    {
        "query": "How do CNNs work for image classification?",
        "relevant_topics": ["cnn", "convolution", "pooling", "image", "neural"],
        "category": "deep_learning"
    },
    {
        "query": "What is backpropagation?",
        "relevant_topics": ["backpropagation", "gradient", "chain rule", "neural"],
        "category": "deep_learning"
    },
    {
        "query": "Why should I hire you?",
        "relevant_topics": ["hire", "qualifications", "strengths", "match"],
        "category": "behavioral"
    }
]

class RetrievalEvaluator:
    def __init__(self):
        self.rag = EnhancedRAG(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            collection=os.getenv("QDRANT_COLLECTION", "interview_chunks")
        )
        
    def evaluate_retrieval_config(
        self,
        questions: List[Dict],
        top_k: int = 5,
        use_hybrid: bool = False,
        use_rerank: bool = False,
        use_rewrite: bool = False
    ) -> Dict:
        """Evaluate a specific retrieval configuration"""
        scores = []
        latencies = []
        
        config_name = f"top{top_k}_hybrid{use_hybrid}_rerank{use_rerank}_rewrite{use_rewrite}"
        
        for q in tqdm(questions, desc=f"Eval {config_name}"):
            import time
            start = time.time()
            
            try:
                results, metadata = self.rag.retrieve(
                    query=q["query"],
                    top_k=top_k,
                    mode=q["category"],
                    use_hybrid=use_hybrid,
                    use_rerank=use_rerank,
                    use_rewrite=use_rewrite
                )
                
                latency = time.time() - start
                latencies.append(latency)
                
                # Calculate relevance score
                score = self._calculate_relevance_score(results, q["relevant_topics"])
                scores.append(score)
                
            except Exception as e:
                print(f"Error: {e}")
                scores.append(0)
                latencies.append(0)
        
        return {
            "config_name": config_name,
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "use_rerank": use_rerank,
            "use_rewrite": use_rewrite,
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "mean_latency": np.mean(latencies),
            "scores": scores
        }
    
    def _calculate_relevance_score(self, results: List[Dict], relevant_topics: List[str]) -> float:
        """Calculate relevance score based on keyword presence"""
        if not results:
            return 0.0
        
        scores = []
        for i, result in enumerate(results):
            text = result.get("text", "").lower()
            question = result.get("question", "").lower()
            combined_text = text + " " + question
            
            matches = sum(1 for topic in relevant_topics if topic.lower() in combined_text)
            position_weight = 1.0 / (i + 1)  # Higher weight for top results
            scores.append((matches / len(relevant_topics)) * position_weight)
        
        return sum(scores) / len(results)
    
    def run_evaluation(self) -> Dict:
        """Run complete evaluation"""
        print("="*60)
        print("STARTING RETRIEVAL EVALUATION")
        print("="*60)
        print(f"Using {len(EVAL_QUESTIONS)} evaluation questions\n")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(EVAL_QUESTIONS),
            "configurations": []
        }
        
        # Test different configurations
        configs = [
            {"top_k": 5, "use_hybrid": False, "use_rerank": False, "use_rewrite": False},
            {"top_k": 5, "use_hybrid": False, "use_rerank": True, "use_rewrite": False},
            {"top_k": 5, "use_hybrid": False, "use_rerank": True, "use_rewrite": True},
            {"top_k": 5, "use_hybrid": True, "use_rerank": True, "use_rewrite": True},
            {"top_k": 3, "use_hybrid": False, "use_rerank": True, "use_rewrite": True},
            {"top_k": 7, "use_hybrid": False, "use_rerank": True, "use_rewrite": True},
        ]
        
        for config in configs:
            result = self.evaluate_retrieval_config(EVAL_QUESTIONS, **config)
            results["configurations"].append(result)
        
        # Find best configuration
        best_config = max(results["configurations"], key=lambda x: x["mean_score"])
        results["best_configuration"] = best_config
        
        # Save results
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"retrieval_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        self._print_summary(results)
        
        print(f"\nFull results saved to: {output_file}")
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("RETRIEVAL EVALUATION SUMMARY")
        print("="*60)
        
        df_results = pd.DataFrame(results["configurations"])
        df_results = df_results.sort_values("mean_score", ascending=False)
        
        print("\nConfiguration Comparison:")
        print(df_results[["config_name", "mean_score", "std_score", "mean_latency"]].to_string(index=False))
        
        best = results["best_configuration"]
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   Config: {best['config_name']}")
        print(f"   Top-K: {best['top_k']}")
        print(f"   Hybrid: {best['use_hybrid']}")
        print(f"   Rerank: {best['use_rerank']}")
        print(f"   Rewrite: {best['use_rewrite']}")
        print(f"   Score: {best['mean_score']:.3f}")
        print(f"   Latency: {best['mean_latency']:.2f}s")


if __name__ == "__main__":
    evaluator = RetrievalEvaluator()
    evaluator.run_evaluation()