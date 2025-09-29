"""
Fallback evaluation system for Streamlit Cloud compatibility
Provides basic evaluation metrics without async dependencies
"""

import time
import google.generativeai as genai
from typing import Dict, List, Optional

class StreamlitCompatibleEvaluator:
    """Simplified evaluator that works in Streamlit Cloud environment"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-lite"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup Gemini client with error handling"""
        try:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
        except Exception as e:
            print(f"Error setting up Gemini client: {e}")
            self.client = None
    
    def evaluate_response_simple(self, query: str, response: str, context: List[str]) -> Dict:
        """
        Simple evaluation using direct Gemini API calls
        Returns basic metrics without complex async operations
        """
        if not self.client:
            return {"error": "Gemini client not configured"}
        
        try:
            # Simple relevancy check
            relevancy_score = self._evaluate_relevancy(query, response)
            
            # Simple faithfulness check
            faithfulness_score = self._evaluate_faithfulness(response, context)
            
            return {
                "answer_relevancy": {
                    "score": relevancy_score,
                    "reason": f"Response relevancy to query assessed as {relevancy_score:.3f}"
                },
                "faithfulness": {
                    "score": faithfulness_score,
                    "reason": f"Response faithfulness to context assessed as {faithfulness_score:.3f}"
                },
                "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "success"
            }
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle specific API errors
            if "429" in error_msg or "quota" in error_msg.lower():
                return {
                    "error": "API quota exceeded. Please wait before trying again.",
                    "error_type": "quota_exceeded",
                    "retry_after": 60
                }
            elif "503" in error_msg or "overloaded" in error_msg.lower():
                return {
                    "error": "API temporarily overloaded. Please try again later.",
                    "error_type": "api_overloaded",
                    "retry_after": 30
                }
            else:
                return {
                    "error": f"Evaluation failed: {error_msg}",
                    "error_type": "general_error"
                }
    
    def _evaluate_relevancy(self, query: str, response: str) -> float:
        """Evaluate how relevant the response is to the query"""
        try:
            prompt = f"""
            Rate the relevancy of this response to the given query on a scale of 0.0 to 1.0.
            
            Query: {query}
            Response: {response}
            
            Consider:
            - Does the response directly address the query?
            - Is the information provided relevant?
            - How well does it answer what was asked?
            
            Respond with only a number between 0.0 and 1.0 (e.g., 0.85)
            """
            
            result = self.client.generate_content(prompt)
            score_text = result.text.strip()
            
            # Extract numeric score
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            except ValueError:
                # Fallback: simple keyword matching
                return self._simple_relevancy_score(query, response)
                
        except Exception:
            return self._simple_relevancy_score(query, response)
    
    def _evaluate_faithfulness(self, response: str, context: List[str]) -> float:
        """Evaluate how faithful the response is to the provided context"""
        if not context:
            return 0.5  # Neutral score if no context
        
        try:
            context_text = "\n".join(context[:3])  # Use first 3 context chunks
            
            prompt = f"""
            Rate how faithful this response is to the provided context on a scale of 0.0 to 1.0.
            
            Context: {context_text}
            Response: {response}
            
            Consider:
            - Does the response contradict the context?
            - Is the information in the response supported by the context?
            - Are there any factual inconsistencies?
            
            Respond with only a number between 0.0 and 1.0 (e.g., 0.92)
            """
            
            result = self.client.generate_content(prompt)
            score_text = result.text.strip()
            
            try:
                score = float(score_text)
                return max(0.0, min(1.0, score))
            except ValueError:
                return self._simple_faithfulness_score(response, context)
                
        except Exception:
            return self._simple_faithfulness_score(response, context)
    
    def _simple_relevancy_score(self, query: str, response: str) -> float:
        """Fallback relevancy scoring using keyword overlap"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(response_words))
        return min(1.0, overlap / len(query_words))
    
    def _simple_faithfulness_score(self, response: str, context: List[str]) -> float:
        """Fallback faithfulness scoring using context overlap"""
        if not context:
            return 0.5
        
        context_text = " ".join(context).lower()
        response_words = set(response.lower().split())
        context_words = set(context_text.split())
        
        if not response_words:
            return 0.0
        
        overlap = len(response_words.intersection(context_words))
        return min(1.0, overlap / len(response_words))

def get_fallback_evaluator(api_key: str) -> StreamlitCompatibleEvaluator:
    """Get a Streamlit-compatible evaluator instance"""
    return StreamlitCompatibleEvaluator(api_key)