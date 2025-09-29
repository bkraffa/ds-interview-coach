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

from app.services.rag import RagOrchestrator

# Test questions for evaluation
TEST_QUESTIONS = [
    {
        "query": "Explain gradient descent in simple terms",
        "expected_elements": ["optimization", "loss", "learning rate", "iterative"],
        "category": "technical"
    },
    {
        "query": "How would you handle missing data in a dataset?",
        "expected_elements": ["imputation", "deletion", "analysis", "domain knowledge"],
        "category": "technical"
    },
    {
        "query": "Tell me about a challenging project you worked on",
        "expected_elements": ["situation", "task", "action", "result"],
        "category": "behavioral"
    },
    {
        "query": "What's the difference between L1 and L2 regularization?",
        "expected_elements": ["lasso", "ridge", "sparsity", "penalty"],
        "category": "technical"
    },
    {
        "query": "How do you stay updated with DS trends?",
        "expected_elements": ["learning", "resources", "community", "practice"],
        "category": "behavioral"
    }
]

# Different prompt templates to test
PROMPT_TEMPLATES = {
    "concise": """You are a Data Science interview coach. Answer the following question concisely and clearly.

Context:
{context}

Question: {question}

Provide a clear, structured answer focusing on key points.""",
    
    "detailed": """You are an expert Data Science interview coach with years of experience preparing candidates.

Based on the following context, provide a comprehensive answer to the interview question.

Context Information:
{context}

Interview Question: {question}

Your response should:
1. Directly answer the question
2. Include relevant examples where appropriate
3. Use technical terminology accurately
4. Be structured and easy to follow

Answer:""",
    
    "guided": """You are helping a candidate prepare for a data science interview.

Use this reference material:
{context}

Question: {question}

Structure your response as follows:
- Main Concept: Brief explanation
- Key Points: Bullet points of essential information
- Example: If applicable
- Interview Tip: What interviewers look for

Response:""",
    
    "adaptive": """Role: Senior Data Science Interview Coach

Context: {context}

Question Type: {category}
Question: {question}

For technical questions: Focus on clarity, accuracy, and demonstrating deep understanding.
For behavioral questions: Use STAR method and emphasize relevant skills.

Provide a response that would impress an interviewer:"""
}

class LLMEvaluator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.rag = RagOrchestrator(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            collection=os.getenv("QDRANT_COLLECTION", "interview_chunks")
        )
        
    def evaluate_prompt_template(
        self, 
        template_name: str, 
        template: str, 
        questions: List[Dict],
        temperature: float = 0.7,
        top_k: int = 5
    ) -> Dict:
        """Evaluate a specific prompt template"""
        results = []
        latencies = []
        
        for q in tqdm(questions, desc=f"Testing {template_name}"):
            # Retrieve context
            context_docs = self.rag.retrieve(q["query"], top_k=top_k)
            context = "\n\n".join([doc["text"] for doc in context_docs])
            
            # Format prompt
            prompt = template.format(
                context=context,
                question=q["query"],
                category=q.get("category", "general")
            )
            
            # Get LLM response
            start_time = time.time()
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful Data Science interview coach."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=500
                )
                
                answer = response.choices[0].message.content
                latency = time.time() - start_time
                
                # Evaluate response quality
                quality_score = self._evaluate_response_quality(
                    answer, 
                    q["expected_elements"]
                )
                
                results.append({
                    "question": q["query"],
                    "answer": answer,
                    "quality_score": quality_score,
                    "latency": latency,
                    "answer_length": len(answer)
                })
                latencies.append(latency)
                
            except Exception as e:
                print(f"Error with question '{q['query']}': {e}")
                results.append({
                    "question": q["query"],
                    "answer": None,
                    "quality_score": 0,
                    "latency": 0,
                    "error": str(e)
                })
        
        # Calculate metrics
        valid_results = [r for r in results if r["answer"] is not None]
        
        return {
            "template_name": template_name,
            "temperature": temperature,
            "top_k": top_k,
            "mean_quality": sum(r["quality_score"] for r in valid_results) / len(valid_results) if valid_results else 0,
            "mean_latency": sum(r["latency"] for r in valid_results) / len(valid_results) if valid_results else 0,
            "mean_length": sum(r["answer_length"] for r in valid_results) / len(valid_results) if valid_results else 0,
            "success_rate": len(valid_results) / len(results) if results else 0,
            "results": results
        }
    
    def evaluate_temperature_settings(self, questions: List[Dict]) -> List[Dict]:
        """Test different temperature settings"""
        temperatures = [0.3, 0.5, 0.7, 0.9]
        results = []
        
        # Use best prompt template (or default)
        template = PROMPT_TEMPLATES["detailed"]
        
        for temp in temperatures:
            print(f"\nTesting temperature: {temp}")
            result = self.evaluate_prompt_template(
                template_name=f"detailed_temp_{temp}",
                template=template,
                questions=questions,
                temperature=temp
            )
            results.append(result)
        
        return results
    
    def evaluate_context_sizes(self, questions: List[Dict]) -> List[Dict]:
        """Test different context sizes (top_k values)"""
        top_k_values = [3, 5, 7, 10]
        results = []
        
        template = PROMPT_TEMPLATES["detailed"]
        
        for top_k in top_k_values:
            print(f"\nTesting top_k: {top_k}")
            result = self.evaluate_prompt_template(
                template_name=f"detailed_top_k_{top_k}",
                template=template,
                questions=questions,
                temperature=0.7,
                top_k=top_k
            )
            results.append(result)
        
        return results
    
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
        if any(marker in answer for marker in ["1.", "2.", "-", "•", ":", "example"]):  # Has formatting
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
            "evaluations": {}
        }
        
        # 1. Evaluate all prompt templates
        print("\n1. Evaluating Prompt Templates...")
        prompt_results = []
        for name, template in PROMPT_TEMPLATES.items():
            print(f"\n   Testing: {name}")
            result = self.evaluate_prompt_template(
                template_name=name,
                template=template,
                questions=TEST_QUESTIONS,
                temperature=0.7,
                top_k=5
            )
            prompt_results.append(result)
        
        all_results["evaluations"]["prompt_templates"] = prompt_results
        
        # 2. Evaluate temperature settings with best prompt
        print("\n2. Evaluating Temperature Settings...")
        best_prompt = max(prompt_results, key=lambda x: x["mean_quality"])
        best_template_name = best_prompt["template_name"]
        
        temp_results = self.evaluate_temperature_settings(TEST_QUESTIONS)
        all_results["evaluations"]["temperature_settings"] = temp_results
        
        # 3. Evaluate context sizes
        print("\n3. Evaluating Context Sizes (top_k)...")
        context_results = self.evaluate_context_sizes(TEST_QUESTIONS)
        all_results["evaluations"]["context_sizes"] = context_results
        
        # Find best configuration
        all_configs = prompt_results + temp_results + context_results
        best_config = max(all_configs, key=lambda x: x["mean_quality"])
        
        all_results["best_configuration"] = {
            "name": best_config["template_name"],
            "temperature": best_config["temperature"],
            "top_k": best_config["top_k"],
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
        
        # Prompt templates summary
        print("\nPrompt Templates Performance:")
        prompt_df = pd.DataFrame(results["evaluations"]["prompt_templates"])
        print(prompt_df[["template_name", "mean_quality", "mean_latency", "success_rate"]].to_string(index=False))
        
        # Temperature settings summary
        print("\nTemperature Settings Performance:")
        temp_df = pd.DataFrame(results["evaluations"]["temperature_settings"])
        print(temp_df[["template_name", "temperature", "mean_quality"]].to_string(index=False))
        
        # Context size summary
        print("\nContext Size Performance:")
        context_df = pd.DataFrame(results["evaluations"]["context_sizes"])
        print(context_df[["template_name", "top_k", "mean_quality"]].to_string(index=False))
        
        # Best configuration
        print("\nBEST CONFIGURATION:")
        best = results["best_configuration"]
        print(f"   Template: {best['name']}")
        print(f"   Temperature: {best['temperature']}")
        print(f"   Top-K: {best['top_k']}")
        print(f"   Quality Score: {best['mean_quality']:.3f}")
        print(f"   Avg Latency: {best['mean_latency']:.2f}s")


if __name__ == "__main__":
    evaluator = LLMEvaluator()
    evaluator.run_complete_evaluation()