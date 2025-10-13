"""
LLM Evaluation Script
Tests different prompts and configurations to find the best approach
"""

import os
import json
import time
from typing import List, Dict
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Import from correct location
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.services.rag import EnhancedRAG

# Test questions for evaluation
TEST_QUESTIONS = [
    {
        "query": "Explain gradient descent in simple terms",
        "expected_elements": ["optimization", "loss", "learning rate", "iterative"],
        "category": "machine_learning"
    },
    {
        "query": "How would you handle missing data in a dataset?",
        "expected_elements": ["imputation", "deletion", "analysis", "domain knowledge"],
        "category": "machine_learning"
    },
    {
        "query": "Tell me about a challenging project you worked on",
        "expected_elements": ["situation", "task", "action", "result"],
        "category": "behavioral"
    },
    {
        "query": "What's the difference between L1 and L2 regularization?",
        "expected_elements": ["lasso", "ridge", "sparsity", "penalty"],
        "category": "machine_learning"
    },
    {
        "query": "How do you stay updated with DS trends?",
        "expected_elements": ["learning", "resources", "community", "practice"],
        "category": "behavioral"
    },
    {
        "query": "Explain backpropagation in neural networks",
        "expected_elements": ["gradient", "chain rule", "weights", "backward pass"],
        "category": "deep_learning"
    },
    {
        "query": "What are CNNs and how do they work?",
        "expected_elements": ["convolution", "pooling", "filters", "image"],
        "category": "deep_learning"
    }
]

class LLMEvaluator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.rag = EnhancedRAG(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            collection=os.getenv("QDRANT_COLLECTION", "interview_chunks")
        )
        
    def evaluate_configuration(
        self,
        questions: List[Dict],
        temperature: float = 0.7,
        top_k: int = 5,
        use_hybrid: bool = True,
        use_rerank: bool = True
    ) -> Dict:
        """Evaluate a specific RAG configuration"""
        results = []
        latencies = []
        
        config_name = f"temp_{temperature}_top{top_k}_hybrid{use_hybrid}_rerank{use_rerank}"
        
        for q in tqdm(questions, desc=f"Testing {config_name}"):
            start_time = time.time()
            
            try:
                # Retrieve context
                context_docs, metadata = self.rag.retrieve(
                    query=q["query"],
                    top_k=top_k,
                    mode=q["category"],
                    use_hybrid=use_hybrid,
                    use_rerank=use_rerank,
                    use_rewrite=True
                )
                
                # Generate answer
                answer = self.rag.generate_answer(
                    query=q["query"],
                    context=context_docs,
                    mode=q["category"],
                    temperature=temperature
                )
                
                latency = time.time() - start_time
                
                # Evaluate response quality
                quality_score = self._evaluate_response_quality(
                    answer, 
                    q["expected_elements"]
                )
                
                results.append({
                    "question": q["query"],
                    "category": q["category"],
                    "answer": answer,
                    "quality_score": quality_score,
                    "latency": latency,
                    "answer_length": len(answer),
                    "num_sources": len(context_docs),
                    "search_method": metadata.get("search_method")
                })
                latencies.append(latency)
                
            except Exception as e:
                print(f"Error with question '{q['query']}': {e}")
                results.append({
                    "question": q["query"],
                    "category": q["category"],
                    "answer": None,
                    "quality_score": 0,
                    "latency": 0,
                    "error": str(e)
                })
        
        # Calculate metrics
        valid_results = [r for r in results if r["answer"] is not None]
        
        return {
            "config_name": config_name,
            "temperature": temperature,
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "use_rerank": use_rerank,
            "mean_quality": sum(r["quality_score"] for r in valid_results) / len(valid_results) if valid_results else 0,
            "mean_latency": sum(r["latency"] for r in valid_results) / len(valid_results) if valid_results else 0,
            "mean_length": sum(r["answer_length"] for r in valid_results) / len(valid_results) if valid_results else 0,
            "success_rate": len(valid_results) / len(results) if results else 0,
            "results": results
        }
    
    def _evaluate_response_quality(self, answer: str, expected_elements: List[str]) -> float:
        """Evaluate response quality based on expected elements"""
        if not answer:
            return 0.0
        
        answer_lower = answer.lower()
        score = 0.0
        
        # Check for expected elements
        elements_found = sum(1 for element in expected_elements if element.lower() in answer_lower)
        element_score = elements_found / len(expected_elements) if expected_elements else 0
        
        # Check response structure
        structure_score = 0.0
        if len(answer) > 100:  # Adequate length
            structure_score += 0.25
        if "\n" in answer or "." in answer:  # Has structure
            structure_score += 0.25
        if any(marker in answer for marker in ["1.", "2.", "-", "•", ":", "example", "Example"]):  # Has formatting
            structure_score += 0.25
        if len(answer.split()) > 50:  # Comprehensive
            structure_score += 0.25
        
        # Combine scores
        total_score = (element_score * 0.6) + (structure_score * 0.4)
        return min(1.0, total_score)
    
    def run_complete_evaluation(self) -> Dict:
        """Run comprehensive LLM evaluation"""
        print("="*60)
        print("STARTING LLM EVALUATION")
        print("="*60)
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "num_test_questions": len(TEST_QUESTIONS),
            "evaluations": []
        }
        
        # Test different configurations
        configs = [
            {"temperature": 0.5, "top_k": 5, "use_hybrid": False, "use_rerank": True},
            {"temperature": 0.7, "top_k": 5, "use_hybrid": False, "use_rerank": True},
            {"temperature": 0.7, "top_k": 5, "use_hybrid": True, "use_rerank": True},
            {"temperature": 0.7, "top_k": 3, "use_hybrid": False, "use_rerank": True},
            {"temperature": 0.7, "top_k": 7, "use_hybrid": False, "use_rerank": True},
        ]
        
        for config in configs:
            print(f"\nTesting configuration: {config}")
            result = self.evaluate_configuration(TEST_QUESTIONS, **config)
            all_results["evaluations"].append(result)
        
        # Find best configuration
        best_config = max(all_results["evaluations"], key=lambda x: x["mean_quality"])
        
        all_results["best_configuration"] = {
            "config_name": best_config["config_name"],
            "temperature": best_config["temperature"],
            "top_k": best_config["top_k"],
            "use_hybrid": best_config["use_hybrid"],
            "use_rerank": best_config["use_rerank"],
            "mean_quality": best_config["mean_quality"],
            "mean_latency": best_config["mean_latency"]
        }
        
        # Save detailed results
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"llm_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Print summary
        self._print_summary(all_results)
        
        print(f"\nFull results saved to: {output_file}")
        
        return all_results
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("LLM EVALUATION SUMMARY")
        print("="*60)
        
        # All configurations summary
        print("\nConfigurations Performance:")
        df = pd.DataFrame(results["evaluations"])
        df_summary = df[["config_name", "mean_quality", "mean_latency", "success_rate"]].copy()
        df_summary = df_summary.sort_values("mean_quality", ascending=False)
        print(df_summary.to_string(index=False))
        
        # Best configuration
        print("\n🏆 BEST CONFIGURATION:")
        best = results["best_configuration"]
        print(f"   Config: {best['config_name']}")
        print(f"   Temperature: {best['temperature']}")
        print(f"   Top-K: {best['top_k']}")
        print(f"   Hybrid: {best['use_hybrid']}")
        print(f"   Rerank: {best['use_rerank']}")
        print(f"   Quality Score: {best['mean_quality']:.3f}")
        print(f"   Avg Latency: {best['mean_latency']:.2f}s")


if __name__ == "__main__":
    evaluator = LLMEvaluator()
    evaluator.run_complete_evaluation()