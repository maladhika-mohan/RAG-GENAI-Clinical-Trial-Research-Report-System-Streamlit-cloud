"""
Custom RAG Evaluation Metrics
Built from scratch using Python and Gemini API for maximum compatibility
"""

import time
import re
import json
from typing import Dict, List, Optional, Tuple
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CustomRAGEvaluator:
    """Custom RAG evaluator with built-in metrics using Gemini API"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-lite"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        self.embedding_model = None
        self._setup_client()
        self._setup_embeddings()
    
    def _setup_client(self):
        """Setup Gemini client"""
        try:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
        except Exception as e:
            print(f"Error setting up Gemini client: {e}")
            self.client = None
    
    def _setup_embeddings(self):
        """Setup sentence transformer for semantic similarity"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.embedding_model = None
    
    def evaluate_response(self, query: str, response: str, context: List[str], 
                         metrics: List[str] = None) -> Dict:
        """
        Evaluate response using custom metrics
        
        Args:
            query: User's question
            response: Generated response
            context: Retrieved document chunks
            metrics: List of metrics to evaluate
        
        Returns:
            Dictionary with evaluation results
        """
        if not self.client:
            return {"error": "Gemini client not configured"}
        
        if metrics is None:
            metrics = ['answer_relevancy', 'faithfulness', 'contextual_relevancy', 'contextual_recall']
        
        results = {}
        
        try:
            # Answer Relevancy
            if 'answer_relevancy' in metrics:
                results['answer_relevancy'] = self._evaluate_answer_relevancy(query, response)
            
            # Faithfulness
            if 'faithfulness' in metrics:
                results['faithfulness'] = self._evaluate_faithfulness(response, context)
            
            # Contextual Relevancy
            if 'contextual_relevancy' in metrics:
                results['contextual_relevancy'] = self._evaluate_contextual_relevancy(query, context)
            
            # Contextual Recall
            if 'contextual_recall' in metrics:
                results['contextual_recall'] = self._evaluate_contextual_recall(query, response, context)
            
            results['evaluation_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            results['status'] = 'success'
            
        except Exception as e:
            error_msg = str(e)
            
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
        
        return results
    
    def _evaluate_answer_relevancy(self, query: str, response: str) -> Dict:
        """
        Evaluate how relevant the response is to the query
        Uses both semantic similarity and LLM judgment
        """
        try:
            # Method 1: Semantic similarity using embeddings
            semantic_score = self._semantic_similarity(query, response)
            
            # Method 2: LLM-based evaluation
            llm_score = self._llm_answer_relevancy(query, response)
            
            # Combine scores (weighted average)
            final_score = (semantic_score * 0.4) + (llm_score * 0.6)
            
            return {
                'score': round(final_score, 3),
                'reason': f"Answer relevancy assessed through semantic similarity ({semantic_score:.3f}) and LLM judgment ({llm_score:.3f}). The response {'directly addresses' if final_score > 0.7 else 'partially addresses' if final_score > 0.4 else 'poorly addresses'} the query.",
                'semantic_score': semantic_score,
                'llm_score': llm_score
            }
            
        except Exception as e:
            # Fallback to simple keyword matching
            fallback_score = self._keyword_overlap_score(query, response)
            return {
                'score': fallback_score,
                'reason': f"Fallback evaluation using keyword overlap: {fallback_score:.3f}",
                'method': 'fallback'
            }
    
    def _evaluate_faithfulness(self, response: str, context: List[str]) -> Dict:
        """
        Evaluate how faithful the response is to the provided context
        Checks for hallucinations and unsupported claims
        """
        if not context:
            return {
                'score': 0.5,
                'reason': "No context provided for faithfulness evaluation"
            }
        
        try:
            # Extract claims from response
            claims = self._extract_claims(response)
            
            if not claims:
                return {
                    'score': 0.8,
                    'reason': "Response contains no specific claims to verify"
                }
            
            # Verify each claim against context
            supported_claims = 0
            total_claims = len(claims)
            
            context_text = " ".join(context)
            
            for claim in claims:
                if self._is_claim_supported(claim, context_text):
                    supported_claims += 1
            
            faithfulness_score = supported_claims / total_claims if total_claims > 0 else 0.8
            
            return {
                'score': round(faithfulness_score, 3),
                'reason': f"Faithfulness evaluation: {supported_claims}/{total_claims} claims are supported by the context. {'High faithfulness' if faithfulness_score > 0.8 else 'Moderate faithfulness' if faithfulness_score > 0.6 else 'Low faithfulness'} detected.",
                'supported_claims': supported_claims,
                'total_claims': total_claims
            }
            
        except Exception as e:
            # Fallback to context overlap
            fallback_score = self._context_overlap_score(response, context)
            return {
                'score': fallback_score,
                'reason': f"Fallback faithfulness evaluation using context overlap: {fallback_score:.3f}",
                'method': 'fallback'
            }
    
    def _evaluate_contextual_relevancy(self, query: str, context: List[str]) -> Dict:
        """
        Evaluate how relevant the retrieved context is to the query
        """
        if not context:
            return {
                'score': 0.0,
                'reason': "No context provided for relevancy evaluation"
            }
        
        try:
            relevant_chunks = 0
            total_chunks = len(context)
            
            for chunk in context:
                relevancy = self._semantic_similarity(query, chunk)
                if relevancy > 0.3:  # Threshold for relevancy
                    relevant_chunks += 1
            
            contextual_relevancy_score = relevant_chunks / total_chunks if total_chunks > 0 else 0.0
            
            return {
                'score': round(contextual_relevancy_score, 3),
                'reason': f"Contextual relevancy: {relevant_chunks}/{total_chunks} chunks are relevant to the query. {'High relevancy' if contextual_relevancy_score > 0.7 else 'Moderate relevancy' if contextual_relevancy_score > 0.4 else 'Low relevancy'} of retrieved context.",
                'relevant_chunks': relevant_chunks,
                'total_chunks': total_chunks
            }
            
        except Exception as e:
            # Fallback to keyword matching
            fallback_score = sum(self._keyword_overlap_score(query, chunk) for chunk in context) / len(context)
            return {
                'score': fallback_score,
                'reason': f"Fallback contextual relevancy using keyword matching: {fallback_score:.3f}",
                'method': 'fallback'
            }
    
    def _evaluate_contextual_recall(self, query: str, response: str, context: List[str]) -> Dict:
        """
        Evaluate how well the response utilizes the available context
        Measures completeness of information retrieval
        """
        if not context:
            return {
                'score': 0.5,
                'reason': "No context provided for recall evaluation"
            }
        
        try:
            # Generate expected answer from context
            expected_answer = self._generate_expected_answer(query, context)
            
            # Compare response with expected answer
            recall_score = self._semantic_similarity(response, expected_answer)
            
            # Also check information coverage
            coverage_score = self._information_coverage(response, context)
            
            # Combine scores
            final_score = (recall_score * 0.6) + (coverage_score * 0.4)
            
            return {
                'score': round(final_score, 3),
                'reason': f"Contextual recall: Response covers {final_score:.1%} of available relevant information. {'Comprehensive' if final_score > 0.8 else 'Adequate' if final_score > 0.6 else 'Limited'} use of context.",
                'recall_score': recall_score,
                'coverage_score': coverage_score
            }
            
        except Exception as e:
            # Fallback to simple coverage
            fallback_score = self._simple_coverage_score(response, context)
            return {
                'score': fallback_score,
                'reason': f"Fallback contextual recall using coverage analysis: {fallback_score:.3f}",
                'method': 'fallback'
            }
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if not self.embedding_model:
            return self._keyword_overlap_score(text1, text2)
        
        try:
            embeddings = self.embedding_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return max(0.0, min(1.0, similarity))
        except Exception:
            return self._keyword_overlap_score(text1, text2)
    
    def _llm_answer_relevancy(self, query: str, response: str) -> float:
        """Use LLM to evaluate answer relevancy"""
        prompt = f"""
        Evaluate how relevant this response is to the given query on a scale of 0.0 to 1.0.
        
        Query: {query}
        Response: {response}
        
        Consider:
        - Does the response directly address the query?
        - Is the information provided relevant and useful?
        - How completely does it answer what was asked?
        
        Respond with only a decimal number between 0.0 and 1.0 (e.g., 0.85)
        """
        
        try:
            result = self.client.generate_content(prompt)
            score_text = result.text.strip()
            
            # Extract numeric score
            score_match = re.search(r'(\d+\.?\d*)', score_text)
            if score_match:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))
            
        except Exception:
            pass
        
        return self._keyword_overlap_score(query, response)
    
    def _extract_claims(self, response: str) -> List[str]:
        """Extract factual claims from response"""
        prompt = f"""
        Extract the main factual claims from this response. List each claim as a separate sentence.
        
        Response: {response}
        
        Return only the claims, one per line, without numbering or bullets.
        """
        
        try:
            result = self.client.generate_content(prompt)
            claims_text = result.text.strip()
            
            # Split into individual claims
            claims = [claim.strip() for claim in claims_text.split('\n') if claim.strip()]
            return claims[:5]  # Limit to 5 claims for efficiency
            
        except Exception:
            # Fallback: split response into sentences
            sentences = re.split(r'[.!?]+', response)
            return [s.strip() for s in sentences if len(s.strip()) > 10][:3]
    
    def _is_claim_supported(self, claim: str, context: str) -> bool:
        """Check if a claim is supported by the context"""
        prompt = f"""
        Is this claim supported by the given context? Answer only "YES" or "NO".
        
        Claim: {claim}
        Context: {context}
        
        Answer:
        """
        
        try:
            result = self.client.generate_content(prompt)
            answer = result.text.strip().upper()
            return "YES" in answer
            
        except Exception:
            # Fallback: check for keyword overlap
            return self._keyword_overlap_score(claim, context) > 0.3
    
    def _generate_expected_answer(self, query: str, context: List[str]) -> str:
        """Generate expected answer from context"""
        context_text = "\n".join(context[:3])  # Use top 3 chunks
        
        prompt = f"""
        Based on the provided context, generate a comprehensive answer to this query:
        
        Query: {query}
        Context: {context_text}
        
        Provide a complete, factual answer based only on the context provided.
        """
        
        try:
            result = self.client.generate_content(prompt)
            return result.text.strip()
            
        except Exception:
            # Fallback: return context summary
            return " ".join(context[:2])
    
    def _information_coverage(self, response: str, context: List[str]) -> float:
        """Measure how much of the context information is covered in response"""
        if not context:
            return 0.5
        
        context_text = " ".join(context)
        
        # Extract key information from context
        context_words = set(context_text.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        context_words -= common_words
        response_words -= common_words
        
        if not context_words:
            return 0.5
        
        overlap = len(context_words.intersection(response_words))
        coverage = overlap / len(context_words)
        
        return min(1.0, coverage)
    
    def _keyword_overlap_score(self, text1: str, text2: str) -> float:
        """Fallback scoring using keyword overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        return overlap / total if total > 0 else 0.0
    
    def _context_overlap_score(self, response: str, context: List[str]) -> float:
        """Fallback faithfulness using context overlap"""
        if not context:
            return 0.5
        
        context_text = " ".join(context)
        return self._keyword_overlap_score(response, context_text)
    
    def _simple_coverage_score(self, response: str, context: List[str]) -> float:
        """Simple coverage score for fallback"""
        if not context:
            return 0.5
        
        total_score = 0
        for chunk in context:
            total_score += self._keyword_overlap_score(response, chunk)
        
        return total_score / len(context) if context else 0.5

def get_custom_evaluator(api_key: str) -> CustomRAGEvaluator:
    """Get a custom RAG evaluator instance"""
    return CustomRAGEvaluator(api_key)