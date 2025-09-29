"""
RAG Evaluation System using DeepEval with Gemini
Comprehensive evaluation of RAG pipeline performance with rate limiting
"""

import os
import json
import pandas as pd
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from deepeval import evaluate
from deepeval.models import GeminiModel
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

# Import for custom templates
try:
    from deepeval.metrics.contextual_relevancy import ContextualRelevancyTemplate
except ImportError:
    ContextualRelevancyTemplate = None

try:
    from deepeval.metrics.contextual_recall import ContextualRecallTemplate
except ImportError:
    ContextualRecallTemplate = None
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from rate_limiter import GeminiRateLimiter, RateLimitedEvaluator

class RAGEvaluator:
    """
    Comprehensive RAG evaluation using DeepEval with Gemini models
    """
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash-lite", enable_rate_limiting: bool = True):
        """
        Initialize RAG evaluator with Gemini model and rate limiting
        
        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model to use for evaluation
            enable_rate_limiting: Whether to enable rate limiting
        """
        # Get API key from environment if not provided
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        self.model_name = model_name
        self.enable_rate_limiting = enable_rate_limiting
        
        # Initialize rate limiter for Gemini 2.0 Flash-Lite
        if enable_rate_limiting:
            self.rate_limiter = GeminiRateLimiter(
                requests_per_minute=200,
                tokens_per_minute=1_000_000,
                requests_per_day=301_000_000,
                buffer_factor=0.85  # Use 85% of limits for extra safety
            )
        else:
            self.rate_limiter = None
        
        # Initialize Gemini model for evaluation
        self.gemini_model = GeminiModel(
            model_name=self.model_name,
            api_key=self.api_key,
            temperature=0  # Deterministic evaluation
        )
        
        # Initialize evaluation metrics
        self.metrics = self._initialize_metrics()
        
        # Create rate-limited wrapper if needed
        if self.rate_limiter:
            self.rate_limited_evaluator = RateLimitedEvaluator(self, self.rate_limiter)
        
    def _initialize_metrics(self) -> Dict:
        """Initialize all RAG evaluation metrics"""
        # Create custom templates to fix compatibility issues
        custom_relevancy_template = None
        custom_recall_template = None
        
        if ContextualRelevancyTemplate:
            class FixedContextualRelevancyTemplate(ContextualRelevancyTemplate):
                @staticmethod
                def generate_verdicts(input: str, context: str):
                    return f"""Based on the input and context, please generate a JSON object to indicate whether each statement found in the context is relevant to the provided input.

Example JSON:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "statement": "The company offers a 30-day return policy."
        }},
        {{
            "verdict": "no", 
            "statement": "The weather is sunny today."
        }}
    ]
}}

Input: {input}

Context: {context}

Please analyze each statement in the context and determine if it's relevant to the input. Return a JSON object with the verdicts array:"""
            
            custom_relevancy_template = FixedContextualRelevancyTemplate
        
        if ContextualRecallTemplate:
            class FixedContextualRecallTemplate(ContextualRecallTemplate):
                @staticmethod
                def generate_verdicts(expected_output: str, retrieval_context: List[str]):
                    context_text = "\n".join(retrieval_context)
                    return f"""For EACH sentence in the given expected output below, determine whether the sentence can be attributed to the retrieval context.

Example JSON:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "This information is directly stated in the retrieval context."
        }},
        {{
            "verdict": "no",
            "reason": "This information is not found in the retrieval context."
        }}
    ]
}}

Expected Output: {expected_output}

Retrieval Context: {context_text}

Please analyze each sentence in the expected output and determine if it can be attributed to the retrieval context. Return a JSON object with the verdicts array:"""
            
            custom_recall_template = FixedContextualRecallTemplate
        
        metrics = {
            'answer_relevancy': AnswerRelevancyMetric(
                threshold=0.7,
                model=self.gemini_model,
                include_reason=True,
                verbose_mode=False
            ),
            'faithfulness': FaithfulnessMetric(
                threshold=0.7,
                model=self.gemini_model,
                include_reason=True,
                verbose_mode=False
            ),
            'contextual_recall': ContextualRecallMetric(
                threshold=0.7,
                model=self.gemini_model,
                include_reason=True,
                verbose_mode=False
            )
        }
        
        # Add contextual relevancy with custom template if available
        if custom_relevancy_template:
            metrics['contextual_relevancy'] = ContextualRelevancyMetric(
                threshold=0.7,
                model=self.gemini_model,
                include_reason=True,
                verbose_mode=False,
                evaluation_template=custom_relevancy_template
            )
        else:
            # Fallback to default if template import failed
            metrics['contextual_relevancy'] = ContextualRelevancyMetric(
                threshold=0.7,
                model=self.gemini_model,
                include_reason=True,
                verbose_mode=False
            )
        
        # Add contextual recall with custom template if available
        if custom_recall_template:
            metrics['contextual_recall'] = ContextualRecallMetric(
                threshold=0.7,
                model=self.gemini_model,
                include_reason=True,
                verbose_mode=False,
                evaluation_template=custom_recall_template
            )
        else:
            # Fallback to default if template import failed
            metrics['contextual_recall'] = ContextualRecallMetric(
                threshold=0.7,
                model=self.gemini_model,
                include_reason=True,
                verbose_mode=False
            )
        
        return metrics
    
    def create_test_case(
        self,
        input_query: str,
        actual_output: str,
        retrieval_context: List[str],
        expected_output: str = None
    ) -> LLMTestCase:
        """
        Create a test case for evaluation
        
        Args:
            input_query: User's question
            actual_output: RAG system's response
            retrieval_context: Retrieved document chunks
            expected_output: Expected ideal response (optional)
        
        Returns:
            LLMTestCase for evaluation
        """
        return LLMTestCase(
            input=input_query,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
            expected_output=expected_output
        )
    
    def evaluate_single_response(
        self,
        input_query: str,
        actual_output: str,
        retrieval_context: List[str],
        expected_output: str = None,
        metrics_to_use: List[str] = None,
        use_rate_limiting: bool = None
    ) -> Dict:
        """
        Evaluate a single RAG response
        
        Args:
            input_query: User's question
            actual_output: RAG system's response
            retrieval_context: Retrieved document chunks
            expected_output: Expected ideal response (optional)
            metrics_to_use: List of metric names to use (default: all)
            use_rate_limiting: Override rate limiting setting
        
        Returns:
            Dictionary with evaluation results
        """
        # Determine if rate limiting should be used
        if use_rate_limiting is None:
            use_rate_limiting = self.enable_rate_limiting
        
        # Use rate-limited evaluation if enabled
        if use_rate_limiting and self.rate_limiter:
            return self.rate_limited_evaluator.evaluate_single_response_with_rate_limit(
                input_query=input_query,
                actual_output=actual_output,
                retrieval_context=retrieval_context,
                expected_output=expected_output,
                metrics_to_use=metrics_to_use
            )
        
        # Original evaluation logic (without rate limiting)
        return self._evaluate_single_response_direct(
            input_query, actual_output, retrieval_context, expected_output, metrics_to_use
        )
    
    def _evaluate_single_response_direct(
        self,
        input_query: str,
        actual_output: str,
        retrieval_context: List[str],
        expected_output: str = None,
        metrics_to_use: List[str] = None
    ) -> Dict:
        """
        Direct evaluation without rate limiting (internal method)
        """
        # Create test case
        test_case = self.create_test_case(
            input_query, actual_output, retrieval_context, expected_output
        )
        
        # Select metrics to use
        if metrics_to_use is None:
            metrics_to_use = list(self.metrics.keys())
        
        # Filter metrics based on available data
        available_metrics = []
        metric_names = []
        
        for metric_name in metrics_to_use:
            if metric_name == 'contextual_recall' and expected_output is None:
                continue  # Skip contextual recall if no expected output
            
            if metric_name in self.metrics:
                available_metrics.append(self.metrics[metric_name])
                metric_names.append(metric_name)
        
        # Run evaluation
        results = {}
        for i, metric in enumerate(available_metrics):
            metric_name = metric_names[i]
            
            # Special handling for contextual_relevancy
            if metric_name == 'contextual_relevancy':
                try:
                    # Try with shorter context if the original fails
                    metric.measure(test_case)
                    results[metric_name] = {
                        'score': metric.score,
                        'reason': metric.reason if hasattr(metric, 'reason') else None,
                        'threshold': metric.threshold,
                        'passed': metric.score >= metric.threshold
                    }
                except Exception as e:
                    error_msg = str(e)
                    if "'NoneType' object has no attribute 'verdicts'" in error_msg:
                        # Try with simplified context
                        try:
                            # Create a simplified test case with shorter context
                            simplified_context = []
                            for ctx in retrieval_context:
                                # Limit each context chunk to 200 characters
                                simplified_context.append(ctx[:200] + "..." if len(ctx) > 200 else ctx)
                            
                            simplified_test_case = LLMTestCase(
                                input=input_query,
                                actual_output=actual_output,
                                retrieval_context=simplified_context[:3],  # Limit to 3 chunks
                                expected_output=expected_output
                            )
                            
                            metric.measure(simplified_test_case)
                            results[metric_name] = {
                                'score': metric.score,
                                'reason': f"(Simplified context) {metric.reason if hasattr(metric, 'reason') else 'Evaluated with shortened context'}",
                                'threshold': metric.threshold,
                                'passed': metric.score >= metric.threshold
                            }
                        except Exception as e2:
                            results[metric_name] = {
                                'score': 0.0,
                                'reason': f"Failed even with simplified context: {str(e2)}",
                                'threshold': getattr(metric, 'threshold', 0.7),
                                'passed': False
                            }
                    else:
                        results[metric_name] = {
                            'score': 0.0,
                            'reason': f"Error: {error_msg}",
                            'threshold': getattr(metric, 'threshold', 0.7),
                            'passed': False
                        }
            else:
                # Standard handling for other metrics
                try:
                    metric.measure(test_case)
                    results[metric_name] = {
                        'score': metric.score,
                        'reason': metric.reason if hasattr(metric, 'reason') else None,
                        'threshold': metric.threshold,
                        'passed': metric.score >= metric.threshold
                    }
                except Exception as e:
                    error_msg = str(e)
                    if "API key" in error_msg.lower():
                        reason = "API key error - please check your Gemini API key"
                    elif "503" in error_msg or "overloaded" in error_msg.lower():
                        reason = "Gemini API is currently overloaded - please try again later"
                    else:
                        reason = f"Error: {error_msg}"
                    
                    results[metric_name] = {
                        'score': 0.0,
                        'reason': reason,
                        'threshold': getattr(metric, 'threshold', 0.7),
                        'passed': False
                    }
        
        return results
    
    def evaluate_batch(
        self,
        test_cases: List[Dict],
        metrics_to_use: List[str] = None,
        progress_callback=None,
        use_rate_limiting: bool = None
    ) -> pd.DataFrame:
        """
        Evaluate multiple RAG responses in batch
        
        Args:
            test_cases: List of test case dictionaries
            metrics_to_use: List of metric names to use
            progress_callback: Function to call with progress updates
            use_rate_limiting: Override rate limiting setting
        
        Returns:
            DataFrame with evaluation results
        """
        # Determine if rate limiting should be used
        if use_rate_limiting is None:
            use_rate_limiting = self.enable_rate_limiting
        
        # Use rate-limited batch evaluation if enabled
        if use_rate_limiting and self.rate_limiter:
            def progress_wrapper(current, total):
                if progress_callback:
                    progress_callback(current, total)
                else:
                    print(f"Evaluating case {current+1}/{total}...")
                    if self.rate_limiter:
                        stats = self.rate_limiter.get_usage_stats()
                        print(f"  Rate limit usage: {stats['requests_per_minute']['current']}/{stats['requests_per_minute']['limit']} RPM")
            
            results = self.rate_limited_evaluator.evaluate_batch_with_rate_limit(
                test_cases=test_cases,
                metrics_to_use=metrics_to_use,
                progress_callback=progress_wrapper
            )
            
            return pd.DataFrame(results)
        
        # Original batch evaluation (without rate limiting)
        results = []
        
        for i, case in enumerate(test_cases):
            if progress_callback:
                progress_callback(i, len(test_cases))
            else:
                print(f"Evaluating case {i+1}/{len(test_cases)}...")
            
            result = self._evaluate_single_response_direct(
                input_query=case['input'],
                actual_output=case['actual_output'],
                retrieval_context=case['retrieval_context'],
                expected_output=case.get('expected_output'),
                metrics_to_use=metrics_to_use
            )
            
            # Add case metadata
            result['case_id'] = i
            result['input'] = case['input']
            result['actual_output'] = case['actual_output']
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def evaluate_from_chat_history(
        self,
        chat_history: List[Dict],
        metrics_to_use: List[str] = None
    ) -> pd.DataFrame:
        """
        Evaluate RAG responses from chat history
        
        Args:
            chat_history: List of chat interactions from session state
            metrics_to_use: List of metric names to use
        
        Returns:
            DataFrame with evaluation results
        """
        test_cases = []
        
        for chat in chat_history:
            # Extract retrieval context from sources
            retrieval_context = []
            if 'sources' in chat and chat['sources']:
                retrieval_context = [source['text'] for source in chat['sources']]
            
            test_cases.append({
                'input': chat['query'],
                'actual_output': chat['response'],
                'retrieval_context': retrieval_context,
                'timestamp': chat.get('timestamp')
            })
        
        return self.evaluate_batch(test_cases, metrics_to_use)
    
    def generate_evaluation_report(
        self,
        results_df: pd.DataFrame,
        save_path: str = None
    ) -> Dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            results_df: DataFrame with evaluation results
            save_path: Path to save the report (optional)
        
        Returns:
            Dictionary with report summary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_used': self.model_name,
            'total_cases': len(results_df),
            'metrics_summary': {},
            'overall_performance': {}
        }
        
        # Calculate metrics summary
        metric_columns = [col for col in results_df.columns if col.endswith('_score') or col in self.metrics.keys()]
        
        for metric_name in self.metrics.keys():
            if f"{metric_name}_score" in results_df.columns:
                scores = results_df[f"{metric_name}_score"].dropna()
                if len(scores) > 0:
                    report['metrics_summary'][metric_name] = {
                        'mean_score': float(scores.mean()),
                        'median_score': float(scores.median()),
                        'min_score': float(scores.min()),
                        'max_score': float(scores.max()),
                        'std_score': float(scores.std()),
                        'pass_rate': float((scores >= self.metrics[metric_name].threshold).mean())
                    }
        
        # Overall performance
        all_scores = []
        for metric_name in self.metrics.keys():
            if f"{metric_name}_score" in results_df.columns:
                scores = results_df[f"{metric_name}_score"].dropna()
                all_scores.extend(scores.tolist())
        
        if all_scores:
            report['overall_performance'] = {
                'average_score': sum(all_scores) / len(all_scores),
                'total_evaluations': len(all_scores)
            }
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {save_path}")
        
        return report
    
    def quick_evaluate(
        self,
        input_query: str,
        actual_output: str,
        retrieval_context: List[str]
    ) -> Dict:
        """
        Quick evaluation for real-time feedback
        
        Args:
            input_query: User's question
            actual_output: RAG system's response
            retrieval_context: Retrieved document chunks
        
        Returns:
            Dictionary with quick evaluation results
        """
        # Use only fast metrics for quick evaluation
        quick_metrics = ['answer_relevancy', 'contextual_relevancy']
        
        return self.evaluate_single_response(
            input_query=input_query,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
            metrics_to_use=quick_metrics,
            use_rate_limiting=True  # Always use rate limiting for real-time
        )
    
    def get_rate_limit_stats(self) -> Dict:
        """
        Get current rate limit usage statistics
        
        Returns:
            Dictionary with rate limit statistics
        """
        if self.rate_limiter:
            return self.rate_limiter.get_usage_stats()
        else:
            return {'rate_limiting': 'disabled'}

def create_sample_test_cases() -> List[Dict]:
    """Create sample test cases for demonstration"""
    return [
        {
            'input': "What is the refund policy?",
            'actual_output': "We offer a 30-day full refund at no extra cost.",
            'retrieval_context': ["All customers are eligible for a 30 day full refund at no extra cost."],
            'expected_output': "You are eligible for a 30 day full refund at no extra cost."
        },
        {
            'input': "How do I contact support?",
            'actual_output': "You can contact our support team via email at support@company.com or call us at 1-800-123-4567.",
            'retrieval_context': [
                "Contact support: Email support@company.com",
                "Phone support available at 1-800-123-4567 during business hours"
            ],
            'expected_output': "Contact support via email at support@company.com or phone at 1-800-123-4567."
        }
    ]

if __name__ == "__main__":
    # Example usage
    evaluator = RAGEvaluator()
    
    # Test with sample cases
    sample_cases = create_sample_test_cases()
    results = evaluator.evaluate_batch(sample_cases)
    
    print("Evaluation Results:")
    print(results)
    
    # Generate report
    report = evaluator.generate_evaluation_report(results, "evaluation_report.json")
    print("\nReport Summary:")
    print(json.dumps(report, indent=2))