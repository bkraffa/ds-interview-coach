import os
import json
import time
from typing import List, Dict, Tuple
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

# Test questions with reference answers for comparison
TEST_QUESTIONS = [
    # ============== MACHINE LEARNING (5 questions) ==============
    {
        "query": "Explain gradient descent in simple terms",
        "category": "machine_learning",
        "reference_answer": "Gradient descent is an optimization algorithm used to minimize a loss function. It works by iteratively moving in the direction of steepest descent (negative gradient) to find the minimum. The learning rate controls how big each step is.",
        "key_concepts": ["optimization", "loss function", "gradient", "learning rate", "iterative"]
    },
    {
        "query": "How would you handle missing data in a dataset?",
        "category": "machine_learning",
        "reference_answer": "Strategies include: 1) Deletion (listwise or pairwise), 2) Imputation (mean, median, mode, or model-based), 3) Using algorithms that handle missing values, 4) Creating indicator variables. The choice depends on the amount and pattern of missing data.",
        "key_concepts": ["deletion", "imputation", "missing data patterns", "domain knowledge"]
    },
    {
        "query": "What's the difference between L1 and L2 regularization?",
        "category": "machine_learning",
        "reference_answer": "L1 (Lasso) adds absolute value of coefficients as penalty, leads to sparse solutions. L2 (Ridge) adds squared coefficients, shrinks weights but doesn't zero them out. L1 for feature selection, L2 for general overfitting prevention.",
        "key_concepts": ["L1", "L2", "sparsity", "feature selection", "overfitting"]
    },
    {
        "query": "Explain the bias-variance tradeoff",
        "category": "machine_learning",
        "reference_answer": "Bias is error from overly simplistic assumptions, leading to underfitting. Variance is error from sensitivity to training data fluctuations, leading to overfitting. The goal is to balance both - reducing one often increases the other. Optimal model complexity minimizes total error.",
        "key_concepts": ["bias", "variance", "underfitting", "overfitting", "model complexity"]
    },
    {
        "query": "What's the difference between bagging and boosting?",
        "category": "machine_learning",
        "reference_answer": "Bagging trains models independently in parallel on bootstrap samples and averages predictions (e.g., Random Forest). Boosting trains models sequentially, each correcting errors of previous ones, giving more weight to misclassified examples (e.g., AdaBoost, XGBoost). Bagging reduces variance, boosting reduces bias.",
        "key_concepts": ["bagging", "boosting", "ensemble", "parallel", "sequential", "variance reduction"]
    },
    
    # ============== DEEP LEARNING (5 questions) ==============
    {
        "query": "Explain backpropagation in neural networks",
        "category": "deep_learning",
        "reference_answer": "Backpropagation calculates gradients of the loss with respect to weights using the chain rule. It propagates errors backward through the network, computing partial derivatives layer by layer, which are then used to update weights via gradient descent.",
        "key_concepts": ["chain rule", "gradient", "backward pass", "weight updates"]
    },
    {
        "query": "What are CNNs and how do they work?",
        "category": "deep_learning",
        "reference_answer": "CNNs use convolutional layers to automatically learn spatial hierarchies of features. They apply filters/kernels across the input, followed by pooling for dimensionality reduction. This makes them effective for image processing tasks.",
        "key_concepts": ["convolution", "filters", "pooling", "spatial hierarchy", "feature learning"]
    },
    {
        "query": "What is the vanishing gradient problem?",
        "category": "deep_learning",
        "reference_answer": "Vanishing gradients occur when gradients become extremely small as they propagate backward through deep networks, making early layers learn very slowly. This is common with sigmoid/tanh activations. Solutions include ReLU activations, batch normalization, residual connections, and careful weight initialization.",
        "key_concepts": ["vanishing gradient", "deep networks", "activation functions", "ReLU", "batch normalization"]
    },
    {
        "query": "How does attention mechanism work in transformers?",
        "category": "deep_learning",
        "reference_answer": "Attention computes a weighted sum of values based on similarity between queries and keys. Self-attention allows each position to attend to all positions in the sequence. Multi-head attention learns different representation subspaces. This enables capturing long-range dependencies without recurrence.",
        "key_concepts": ["attention", "query-key-value", "self-attention", "multi-head", "transformers"]
    },
    {
        "query": "What's the purpose of dropout in neural networks?",
        "category": "deep_learning",
        "reference_answer": "Dropout randomly deactivates neurons during training, forcing the network to learn redundant representations and preventing co-adaptation of features. This acts as regularization, reducing overfitting. At inference, all neurons are active but outputs are scaled by the dropout rate.",
        "key_concepts": ["dropout", "regularization", "overfitting", "co-adaptation", "ensemble effect"]
    },
    
    # ============== BEHAVIORAL (5 questions) ==============
    {
        "query": "Tell me about a challenging project you worked on",
        "category": "behavioral",
        "reference_answer": "Use STAR method: Situation (context), Task (what needed to be done), Action (steps taken), Result (outcome and learnings). Be specific, quantify impact, and show problem-solving skills.",
        "key_concepts": ["STAR method", "specific example", "quantifiable results", "learnings"]
    },
    {
        "query": "Why should I hire you?",
        "category": "behavioral",
        "reference_answer": "Match your skills to job requirements, provide specific examples of achievements, show cultural fit, demonstrate passion for the role, and explain unique value you bring. Be confident but not arrogant.",
        "key_concepts": ["skills match", "achievements", "cultural fit", "unique value"]
    },
    {
        "query": "Describe a time when you had to work with a difficult team member",
        "category": "behavioral",
        "reference_answer": "Use STAR: Describe the situation professionally, explain the challenge without badmouthing, detail your approach (communication, empathy, finding common ground), and highlight positive outcome. Emphasize collaboration, conflict resolution, and emotional intelligence.",
        "key_concepts": ["conflict resolution", "communication", "empathy", "collaboration", "professionalism"]
    },
    {
        "query": "How do you handle tight deadlines and multiple priorities?",
        "category": "behavioral",
        "reference_answer": "Discuss prioritization frameworks (urgent/important matrix), time management techniques, communication with stakeholders about tradeoffs, breaking work into manageable chunks, and knowing when to ask for help. Give specific examples of successfully managing competing demands.",
        "key_concepts": ["prioritization", "time management", "communication", "stakeholder management", "adaptability"]
    },
    {
        "query": "Tell me about a time you failed and what you learned",
        "category": "behavioral",
        "reference_answer": "Choose a genuine failure that wasn't catastrophic, explain context without making excuses, describe what went wrong and why, detail corrective actions taken, emphasize lessons learned and how you've applied them since. Show growth mindset and accountability.",
        "key_concepts": ["failure", "accountability", "learning", "growth mindset", "self-awareness", "improvement"]
    }
]

class ImprovedLLMEvaluator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.rag = EnhancedRAG(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            collection=os.getenv("QDRANT_COLLECTION", "interview_chunks")
        )
        
        # Model configuration
        self.generation_model = "gpt-4o-mini"  # Model that generates answers
        self.judge_model = "gpt-4o"  # Model that evaluates answers
        
    def evaluate_with_llm_judge(
        self,
        query: str,
        generated_answer: str,
        reference_answer: str,
        key_concepts: List[str],
        category: str
    ) -> Dict:
        """Use GPT-4 as a judge to evaluate the generated answer"""
        
        judge_prompt = f"""You are an expert evaluator assessing the quality of interview coaching responses.

Question: {query}
Category: {category}

Reference Answer (Gold Standard):
{reference_answer}

Generated Answer (To Evaluate):
{generated_answer}

Key Concepts that should be covered: {', '.join(key_concepts)}

Please evaluate the generated answer on the following criteria (rate each 0-10):

1. **Accuracy**: Is the information correct and aligned with the reference?
2. **Completeness**: Does it cover all key concepts and important points?
3. **Clarity**: Is it well-structured and easy to understand?
4. **Relevance**: Does it directly address the question asked?
5. **Actionability**: For interview prep, is the advice practical and actionable?

Provide your evaluation in JSON format:
{{
    "accuracy": <score>,
    "completeness": <score>,
    "clarity": <score>,
    "relevance": <score>,
    "actionability": <score>,
    "overall_score": <average of above>,
    "strengths": "<brief description>",
    "weaknesses": "<brief description>",
    "missing_concepts": ["<concept1>", "<concept2>"]
}}

Be objective and critical in your assessment."""

        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of educational content."},
                    {"role": "user", "content": judge_prompt}
                ],
                temperature=0.3,  # Lower temperature for consistent evaluation
                response_format={"type": "json_object"}
            )
            
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation
            
        except Exception as e:
            print(f"Error in LLM judge evaluation: {e}")
            return {
                "accuracy": 0,
                "completeness": 0,
                "clarity": 0,
                "relevance": 0,
                "actionability": 0,
                "overall_score": 0,
                "error": str(e)
            }
    
    def calculate_concept_coverage(self, answer: str, key_concepts: List[str]) -> float:
        """Simple keyword-based concept coverage"""
        answer_lower = answer.lower()
        covered = sum(1 for concept in key_concepts if concept.lower() in answer_lower)
        return covered / len(key_concepts) if key_concepts else 0
    
    def evaluate_configuration(
        self,
        questions: List[Dict],
        temperature: float = 0.7,
        top_k: int = 5,
        use_rerank: bool = True,
        use_rewrite: bool = True
    ) -> Dict:
        """Evaluate a specific RAG configuration using LLM judge"""
        results = []
        
        config_name = f"temp_{temperature}_top{top_k}_rerank{use_rerank}_rewrite{use_rewrite}"
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {config_name}")
        print(f"{'='*60}")
        
        for q in tqdm(questions, desc=f"Testing {config_name}"):
            start_time = time.time()
            
            try:
                # Retrieve context
                context_docs, metadata = self.rag.retrieve(
                    query=q["query"],
                    top_k=top_k,
                    mode=q["category"],
                    use_hybrid=False,
                    use_rerank=use_rerank,
                    use_rewrite=use_rewrite
                )
                
                # Generate answer using generation model
                answer = self.rag.generate_answer(
                    query=q["query"],
                    context=context_docs,
                    mode=q["category"],
                    temperature=temperature
                )
                
                latency = time.time() - start_time
                
                # Evaluate using LLM judge (GPT-4)
                llm_eval = self.evaluate_with_llm_judge(
                    query=q["query"],
                    generated_answer=answer,
                    reference_answer=q["reference_answer"],
                    key_concepts=q["key_concepts"],
                    category=q["category"]
                )
                
                # Calculate concept coverage
                concept_coverage = self.calculate_concept_coverage(answer, q["key_concepts"])
                
                results.append({
                    "question": q["query"],
                    "category": q["category"],
                    "answer": answer,
                    "reference_answer": q["reference_answer"],
                    "llm_judge_scores": llm_eval,
                    "concept_coverage": concept_coverage,
                    "latency": latency,
                    "answer_length": len(answer),
                    "num_sources": len(context_docs),
                    "search_method": metadata.get("search_method")
                })
                
                # Print detailed results for this question
                print(f"\n  Q: {q['query'][:60]}...")
                print(f"  Overall Score: {llm_eval.get('overall_score', 0):.1f}/10")
                print(f"  Concept Coverage: {concept_coverage:.1%}")
                
            except Exception as e:
                print(f"\n  ❌ Error with question '{q['query']}': {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "question": q["query"],
                    "category": q["category"],
                    "error": str(e)
                })
        
        # Calculate aggregate metrics
        valid_results = [r for r in results if "llm_judge_scores" in r]
        
        if valid_results:
            avg_scores = {
                "accuracy": np.mean([r["llm_judge_scores"]["accuracy"] for r in valid_results]),
                "completeness": np.mean([r["llm_judge_scores"]["completeness"] for r in valid_results]),
                "clarity": np.mean([r["llm_judge_scores"]["clarity"] for r in valid_results]),
                "relevance": np.mean([r["llm_judge_scores"]["relevance"] for r in valid_results]),
                "actionability": np.mean([r["llm_judge_scores"]["actionability"] for r in valid_results]),
                "overall": np.mean([r["llm_judge_scores"]["overall_score"] for r in valid_results])
            }
        else:
            avg_scores = {k: 0 for k in ["accuracy", "completeness", "clarity", "relevance", "actionability", "overall"]}
        
        return {
            "config_name": config_name,
            "temperature": temperature,
            "top_k": top_k,
            "use_rerank": use_rerank,
            "use_rewrite": use_rewrite,
            "generation_model": self.generation_model,
            "judge_model": self.judge_model,
            "avg_llm_scores": avg_scores,
            "mean_concept_coverage": np.mean([r["concept_coverage"] for r in valid_results]) if valid_results else 0,
            "mean_latency": np.mean([r["latency"] for r in valid_results]) if valid_results else 0,
            "success_rate": len(valid_results) / len(results) if results else 0,
            "detailed_results": results
        }
    
    def run_complete_evaluation(self) -> Dict:
        """Run comprehensive evaluation with LLM judge"""
        print("="*60)
        print("LLM EVALUATION WITH GPT-4 AS A JUDGE")
        print("="*60)
        print(f"Generation Model: {self.generation_model}")
        print(f"Judge Model: {self.judge_model}")
        print(f"Test Questions: {len(TEST_QUESTIONS)}")
        print("="*60)
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "generation_model": self.generation_model,
            "judge_model": self.judge_model,
            "num_test_questions": len(TEST_QUESTIONS),
            "evaluations": []
        }
        
        # Test different configurations
        configs = [
            {"temperature": 0.5, "top_k": 5, "use_rerank": True, "use_rewrite": True},
            {"temperature": 0.7, "top_k": 5, "use_rerank": True, "use_rewrite": True},
            {"temperature": 0.7, "top_k": 5, "use_rerank": False, "use_rewrite": True},
            {"temperature": 0.7, "top_k": 3, "use_rerank": True, "use_rewrite": True},
            {"temperature": 0.7, "top_k": 7, "use_rerank": True, "use_rewrite": True},
        ]
        
        for config in configs:
            result = self.evaluate_configuration(TEST_QUESTIONS, **config)
            all_results["evaluations"].append(result)
            
            # Add delay to avoid rate limits
            time.sleep(2)
        
        # Find best configuration
        best_config = max(
            all_results["evaluations"], 
            key=lambda x: x["avg_llm_scores"]["overall"]
        )
        
        all_results["best_configuration"] = {
            "config_name": best_config["config_name"],
            "temperature": best_config["temperature"],
            "top_k": best_config["top_k"],
            "use_rerank": best_config["use_rerank"],
            "use_rewrite": best_config["use_rewrite"],
            "avg_scores": best_config["avg_llm_scores"],
            "concept_coverage": best_config["mean_concept_coverage"],
            "latency": best_config["mean_latency"]
        }
        
        # Save detailed results
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"llm_eval_judge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Print summary
        self._print_summary(all_results)
        
        print(f"\n📄 Full results saved to: {output_file}")
        
        return all_results
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("LLM EVALUATION SUMMARY (GPT-4 Judge)")
        print("="*60)
        
        # Configuration comparison
        print("\nConfiguration Performance:")
        summary_data = []
        for eval_result in results["evaluations"]:
            summary_data.append({
                "config": eval_result["config_name"],
                "overall": eval_result["avg_llm_scores"]["overall"],
                "accuracy": eval_result["avg_llm_scores"]["accuracy"],
                "completeness": eval_result["avg_llm_scores"]["completeness"],
                "clarity": eval_result["avg_llm_scores"]["clarity"],
                "concept_cov": eval_result["mean_concept_coverage"],
                "latency": eval_result["mean_latency"]
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values("overall", ascending=False)
        print(df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
        
        # Best configuration details
        print("\n" + "🏆"*30)
        print("BEST CONFIGURATION")
        print("🏆"*30)
        best = results["best_configuration"]
        print(f"\n  Config: {best['config_name']}")
        print(f"  Temperature: {best['temperature']}")
        print(f"  Top-K: {best['top_k']}")
        print(f"  Rerank: {best['use_rerank']}")
        print(f"  Rewrite: {best['use_rewrite']}")
        print(f"\n  Scores (0-10):")
        for metric, score in best['avg_scores'].items():
            print(f"    {metric.capitalize()}: {score:.2f}")
        print(f"\n  Concept Coverage: {best['concept_coverage']:.1%}")
        print(f"  Avg Latency: {best['latency']:.2f}s")


if __name__ == "__main__":
    import numpy as np
    evaluator = ImprovedLLMEvaluator()
    evaluator.run_complete_evaluation()