"""
Q&A Retrieval Evaluation Script
Focused on single correct answer retrieval (not multi-document)
Uses MRR, Hit@K, and Precision metrics appropriate for Q&A systems
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.services.rag import EnhancedRAG

# Test questions with their expected answer identifiers
# Each question has multiple query variations to test robustness
EVAL_QUESTIONS = [
    # Machine Learning
    {
        "query": "What is gradient descent and how does it work?",
        "query_variations": [
            "explain gradient descent",
            "how does gradient descent optimize models",
            "what's the gradient descent algorithm"
        ],
        "category": "machine_learning",
        "expected_answer_keywords": ["gradient descent", "optimization", "loss function", "learning rate", "iterative"]
    },
    {
        "query": "Explain the difference between bagging and boosting",
        "query_variations": [
            "bagging vs boosting",
            "what's the difference between ensemble methods",
            "compare bagging and boosting algorithms"
        ],
        "category": "machine_learning",
        "expected_answer_keywords": ["bagging", "boosting", "ensemble", "parallel", "sequential"]
    },
    {
        "query": "What is overfitting and how to prevent it?",
        "query_variations": [
            "how to avoid overfitting",
            "explain overfitting in machine learning",
            "techniques to prevent overfitting"
        ],
        "category": "machine_learning",
        "expected_answer_keywords": ["overfitting", "regularization", "validation", "cross-validation"]
    },
    {
        "query": "How would you handle imbalanced datasets?",
        "query_variations": [
            "techniques for imbalanced data",
            "dealing with class imbalance",
            "what to do with imbalanced classes"
        ],
        "category": "machine_learning",
        "expected_answer_keywords": ["imbalanced", "resampling", "smote", "class weights"]
    },
    {
        "query": "What's the difference between L1 and L2 regularization?",
        "query_variations": [
            "L1 vs L2 regularization",
            "lasso vs ridge regression",
            "compare L1 and L2 penalties"
        ],
        "category": "machine_learning",
        "expected_answer_keywords": ["L1", "L2", "lasso", "ridge", "sparsity"]
    },
    
    # Deep Learning
    {
        "query": "How do CNNs work for image classification?",
        "query_variations": [
            "explain convolutional neural networks",
            "how do CNNs process images",
            "what makes CNNs good for computer vision"
        ],
        "category": "deep_learning",
        "expected_answer_keywords": ["cnn", "convolution", "pooling", "filters", "feature maps"]
    },
    {
        "query": "What is backpropagation?",
        "query_variations": [
            "explain backpropagation algorithm",
            "how does backprop work",
            "what is backward pass in neural networks"
        ],
        "category": "deep_learning",
        "expected_answer_keywords": ["backpropagation", "gradient", "chain rule", "weights"]
    },
    {
        "query": "What is the vanishing gradient problem?",
        "query_variations": [
            "explain vanishing gradients",
            "why do gradients vanish in deep networks",
            "how to solve vanishing gradient problem"
        ],
        "category": "deep_learning",
        "expected_answer_keywords": ["vanishing gradient", "deep networks", "activation", "relu"]
    },
    {
        "query": "Explain attention mechanism in transformers",
        "query_variations": [
            "how does attention work",
            "what is self-attention in transformers",
            "explain transformer attention"
        ],
        "category": "deep_learning",
        "expected_answer_keywords": ["attention", "transformer", "query", "key", "value"]
    },
    {
        "query": "What's the purpose of dropout in neural networks?",
        "query_variations": [
            "explain dropout regularization",
            "how does dropout prevent overfitting",
            "why use dropout in neural networks"
        ],
        "category": "deep_learning",
        "expected_answer_keywords": ["dropout", "regularization", "overfitting", "neurons"]
    },
    
    # Behavioral
    {
        "query": "Tell me about a time you handled a difficult stakeholder",
        "query_variations": [
            "describe dealing with a challenging stakeholder",
            "how do you manage difficult stakeholders",
            "example of stakeholder conflict resolution"
        ],
        "category": "behavioral",
        "expected_answer_keywords": ["stakeholder", "communication", "conflict", "collaboration", "STAR"]
    },
    {
        "query": "Why should I hire you?",
        "query_variations": [
            "what makes you a good fit for this role",
            "why are you the best candidate",
            "what unique value do you bring"
        ],
        "category": "behavioral",
        "expected_answer_keywords": ["skills", "qualifications", "value", "fit", "achievements"]
    },
    {
        "query": "Describe a challenging project you worked on",
        "query_variations": [
            "tell me about a difficult project",
            "what was your most challenging project",
            "describe a complex project experience"
        ],
        "category": "behavioral",
        "expected_answer_keywords": ["project", "challenge", "problem-solving", "result", "STAR"]
    },
    {
        "query": "How do you prioritize tasks when you have multiple deadlines?",
        "query_variations": [
            "describe your prioritization process",
            "how do you manage multiple deadlines",
            "what's your approach to task prioritization"
        ],
        "category": "behavioral",
        "expected_answer_keywords": ["prioritize", "deadline", "time management", "urgent", "important"]
    },
    {
        "query": "Tell me about a time you failed and what you learned",
        "query_variations": [
            "describe a failure and your learnings",
            "what's a mistake you made and how you grew",
            "tell me about a time things didn't go as planned"
        ],
        "category": "behavioral",
        "expected_answer_keywords": ["failure", "learning", "growth", "accountability", "STAR"]
    }
]


class QARetrievalEvaluator:
    def __init__(self):
        self.rag = EnhancedRAG(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            collection=os.getenv("QDRANT_COLLECTION", "interview_chunks")
        )
        
    def calculate_mrr(self, results: List[Dict], expected_keywords: List[str]) -> float:
        """
        Mean Reciprocal Rank - position of first relevant answer
        Returns 1/rank if found, 0 if not found
        """
        for i, result in enumerate(results):
            if self._is_relevant(result, expected_keywords):
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_hit_at_k(self, results: List[Dict], expected_keywords: List[str], k: int) -> float:
        """
        Hit@K - is there a relevant answer in top-K results?
        Returns 1 if yes, 0 if no
        """
        top_k_results = results[:min(k, len(results))]
        for result in top_k_results:
            if self._is_relevant(result, expected_keywords):
                return 1.0
        return 0.0
    
    def calculate_precision_at_k(self, results: List[Dict], expected_keywords: List[str], k: int) -> float:
        """
        Precision@K - proportion of relevant results in top-K
        """
        top_k_results = results[:min(k, len(results))]
        if not top_k_results:
            return 0.0
        
        relevant_count = sum(1 for r in top_k_results if self._is_relevant(r, expected_keywords))
        return relevant_count / len(top_k_results)
    
    def _is_relevant(self, result: Dict, expected_keywords: List[str], threshold: float = 0.5) -> bool:
        """
        Check if a result is relevant based on keyword matching
        """
        text = result.get("text", "").lower()
        question = result.get("question", "").lower()
        answer = result.get("answer", "").lower()
        combined = f"{text} {question} {answer}"
        
        matches = sum(1 for kw in expected_keywords if kw.lower() in combined)
        coverage = matches / len(expected_keywords) if expected_keywords else 0
        
        return coverage >= threshold
    
    def evaluate_single_query(
        self,
        query: str,
        category: str,
        expected_keywords: List[str],
        top_k: int,
        use_rerank: bool,
        use_rewrite: bool
    ) -> Dict:
        """Evaluate retrieval for a single query"""
        try:
            results, metadata = self.rag.retrieve(
                query=query,
                top_k=top_k,
                mode=category,
                use_hybrid=False,
                use_rerank=use_rerank,
                use_rewrite=use_rewrite
            )
            
            if not results:
                return {
                    "success": False,
                    "mrr": 0.0,
                    "hit@1": 0.0,
                    "hit@3": 0.0,
                    "hit@5": 0.0,
                    "precision@3": 0.0,
                    "precision@5": 0.0,
                    "num_results": 0
                }
            
            return {
                "success": True,
                "mrr": self.calculate_mrr(results, expected_keywords),
                "hit@1": self.calculate_hit_at_k(results, expected_keywords, 1),
                "hit@3": self.calculate_hit_at_k(results, expected_keywords, 3),
                "hit@5": self.calculate_hit_at_k(results, expected_keywords, 5),
                "precision@3": self.calculate_precision_at_k(results, expected_keywords, 3),
                "precision@5": self.calculate_precision_at_k(results, expected_keywords, 5),
                "num_results": len(results)
            }
            
        except Exception as e:
            print(f"Error: {e}")
            return {
                "success": False,
                "error": str(e),
                "mrr": 0.0,
                "hit@1": 0.0,
                "hit@3": 0.0,
                "hit@5": 0.0,
                "precision@3": 0.0,
                "precision@5": 0.0,
                "num_results": 0
            }
    
    def evaluate_configuration(
        self,
        questions: List[Dict],
        top_k: int = 5,
        use_rerank: bool = False,
        use_rewrite: bool = False
    ) -> Dict:
        """Evaluate a configuration with all queries including variations"""
        config_name = f"top{top_k}_rerank{use_rerank}_rewrite{use_rewrite}"
        
        all_results = []
        
        print(f"\nEvaluating: {config_name}")
        print("="*60)
        
        for q in tqdm(questions, desc=config_name):
            # Test main query
            main_result = self.evaluate_single_query(
                query=q["query"],
                category=q["category"],
                expected_keywords=q["expected_answer_keywords"],
                top_k=top_k,
                use_rerank=use_rerank,
                use_rewrite=use_rewrite
            )
            
            all_results.append({
                "query": q["query"],
                "query_type": "main",
                "category": q["category"],
                **main_result
            })
            
            # Test query variations
            for variation in q.get("query_variations", []):
                var_result = self.evaluate_single_query(
                    query=variation,
                    category=q["category"],
                    expected_keywords=q["expected_answer_keywords"],
                    top_k=top_k,
                    use_rerank=use_rerank,
                    use_rewrite=use_rewrite
                )
                
                all_results.append({
                    "query": variation,
                    "query_type": "variation",
                    "category": q["category"],
                    **var_result
                })
        
        # Calculate aggregate metrics
        successful = [r for r in all_results if r["success"]]
        
        if successful:
            metrics = {
                "mean_mrr": np.mean([r["mrr"] for r in successful]),
                "mean_hit@1": np.mean([r["hit@1"] for r in successful]),
                "mean_hit@3": np.mean([r["hit@3"] for r in successful]),
                "mean_hit@5": np.mean([r["hit@5"] for r in successful]),
                "mean_precision@3": np.mean([r["precision@3"] for r in successful]),
                "mean_precision@5": np.mean([r["precision@5"] for r in successful]),
            }
            
            # Metrics by category
            categories = set(r["category"] for r in successful)
            category_metrics = {}
            for cat in categories:
                cat_results = [r for r in successful if r["category"] == cat]
                category_metrics[cat] = {
                    "mrr": np.mean([r["mrr"] for r in cat_results]),
                    "hit@1": np.mean([r["hit@1"] for r in cat_results]),
                    "count": len(cat_results)
                }
            
            # Separate metrics for main queries vs variations
            main_queries = [r for r in successful if r["query_type"] == "main"]
            variations = [r for r in successful if r["query_type"] == "variation"]
            
            query_type_metrics = {
                "main": {
                    "mrr": np.mean([r["mrr"] for r in main_queries]) if main_queries else 0,
                    "hit@1": np.mean([r["hit@1"] for r in main_queries]) if main_queries else 0,
                },
                "variations": {
                    "mrr": np.mean([r["mrr"] for r in variations]) if variations else 0,
                    "hit@1": np.mean([r["hit@1"] for r in variations]) if variations else 0,
                }
            }
        else:
            metrics = {k: 0 for k in ["mean_mrr", "mean_hit@1", "mean_hit@3", 
                                       "mean_hit@5", "mean_precision@3", "mean_precision@5"]}
            category_metrics = {}
            query_type_metrics = {}
        
        return {
            "config_name": config_name,
            "top_k": top_k,
            "use_rerank": use_rerank,
            "use_rewrite": use_rewrite,
            "metrics": metrics,
            "category_metrics": category_metrics,
            "query_type_metrics": query_type_metrics,
            "success_rate": len(successful) / len(all_results) if all_results else 0,
            "total_queries": len(all_results),
            "detailed_results": all_results
        }
    
    def run_evaluation(self) -> Dict:
        """Run complete Q&A retrieval evaluation"""
        print("="*60)
        print("Q&A RETRIEVAL EVALUATION")
        print("="*60)
        print(f"Base Questions: {len(EVAL_QUESTIONS)}")
        
        total_variations = sum(len(q.get("query_variations", [])) for q in EVAL_QUESTIONS)
        print(f"Query Variations: {total_variations}")
        print(f"Total Test Queries: {len(EVAL_QUESTIONS) + total_variations}")
        
        categories = {}
        for q in EVAL_QUESTIONS:
            cat = q["category"]
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\nQuestions by Category:")
        for cat, count in categories.items():
            print(f"  {cat}: {count}")
        print("="*60)
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "num_base_questions": len(EVAL_QUESTIONS),
            "num_total_queries": len(EVAL_QUESTIONS) + total_variations,
            "questions_by_category": categories,
            "configurations": []
        }
        
        # Test configurations
        configs = [
            {"top_k": 1, "use_rerank": False, "use_rewrite": False},
            {"top_k": 3, "use_rerank": False, "use_rewrite": False},
            {"top_k": 5, "use_rerank": False, "use_rewrite": False},
            {"top_k": 3, "use_rerank": True, "use_rewrite": False},
            {"top_k": 5, "use_rerank": True, "use_rewrite": False},
            {"top_k": 5, "use_rerank": True, "use_rewrite": True},
        ]
        
        for config in configs:
            result = self.evaluate_configuration(EVAL_QUESTIONS, **config)
            all_results["configurations"].append(result)
        
        # Find best configuration based on MRR (most important for Q&A)
        best_config = max(all_results["configurations"], key=lambda x: x["metrics"]["mean_mrr"])
        all_results["best_configuration"] = best_config
        
        # Save results
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"qa_retrieval_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        self._print_summary(all_results)
        
        print(f"\nFull results saved to: {output_file}")
        
        return all_results
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("Q&A RETRIEVAL EVALUATION SUMMARY")
        print("="*60)
        
        # Configuration comparison
        print("\nConfiguration Performance:")
        summary_data = []
        for config in results["configurations"]:
            m = config["metrics"]
            summary_data.append({
                "config": config["config_name"],
                "MRR": m["mean_mrr"],
                "Hit@1": m["mean_hit@1"],
                "Hit@3": m["mean_hit@3"],
                "P@3": m["mean_precision@3"],
                "success": config["success_rate"]
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values("MRR", ascending=False)
        print(df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
        
        # Best configuration
        print("\n" + "="*60)
        print("BEST CONFIGURATION (Highest MRR)")
        print("="*60)
        best = results["best_configuration"]
        print(f"\nConfig: {best['config_name']}")
        print(f"Top-K: {best['top_k']}")
        print(f"Rerank: {best['use_rerank']}")
        print(f"Rewrite: {best['use_rewrite']}")
        
        print("\nMetrics:")
        for metric, value in best['metrics'].items():
            display_name = metric.replace('mean_', '').replace('@', ' @ ').upper()
            print(f"  {display_name}: {value:.3f}")
        
        # Performance by category
        if best.get("category_metrics"):
            print("\nPerformance by Category:")
            for cat, metrics in best["category_metrics"].items():
                print(f"  {cat}:")
                print(f"    MRR: {metrics['mrr']:.3f}")
                print(f"    Hit@1: {metrics['hit@1']:.3f}")
                print(f"    Queries: {metrics['count']}")
        
        # Performance: main vs variations
        if best.get("query_type_metrics"):
            print("\nPerformance by Query Type:")
            for qtype, metrics in best["query_type_metrics"].items():
                print(f"  {qtype}:")
                print(f"    MRR: {metrics['mrr']:.3f}")
                print(f"    Hit@1: {metrics['hit@1']:.3f}")


if __name__ == "__main__":
    evaluator = QARetrievalEvaluator()
    evaluator.run_evaluation()