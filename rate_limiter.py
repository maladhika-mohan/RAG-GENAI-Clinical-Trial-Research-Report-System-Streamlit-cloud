"""
Rate Limiter for Gemini API
Handles rate limiting for Gemini 2.0 Flash-Lite: 200 RPM, 1M TPM, 301M RPD
"""

import time
import threading
from collections import deque
from typing import Optional
from datetime import datetime, timedelta
import tiktoken

class GeminiRateLimiter:
    """
    Rate limiter for Gemini API with support for RPM, TPM, and RPD limits
    """
    
    def __init__(
        self,
        requests_per_minute: int = 200,
        tokens_per_minute: int = 1_000_000,
        requests_per_day: int = 301_000_000,
        buffer_factor: float = 0.9  # Use 90% of limits for safety
    ):
        """
        Initialize rate limiter
        
        Args:
            requests_per_minute: Maximum requests per minute
            tokens_per_minute: Maximum tokens per minute
            requests_per_day: Maximum requests per day
            buffer_factor: Safety factor to stay below limits
        """
        self.rpm_limit = int(requests_per_minute * buffer_factor)
        self.tpm_limit = int(tokens_per_minute * buffer_factor)
        self.rpd_limit = int(requests_per_day * buffer_factor)
        
        # Request tracking
        self.request_times = deque()
        self.daily_requests = deque()
        
        # Token tracking
        self.token_usage = deque()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Token encoder for estimation
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoder = None
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Rough estimation: ~4 characters per token
            return len(text) // 4
    
    def _cleanup_old_records(self):
        """Remove old records outside the time windows"""
        current_time = time.time()
        current_date = datetime.now().date()
        
        # Clean up minute-based records (RPM and TPM)
        minute_ago = current_time - 60
        
        while self.request_times and self.request_times[0] < minute_ago:
            self.request_times.popleft()
        
        while self.token_usage and self.token_usage[0][0] < minute_ago:
            self.token_usage.popleft()
        
        # Clean up daily records (RPD)
        while self.daily_requests and self.daily_requests[0].date() < current_date:
            self.daily_requests.popleft()
    
    def _get_current_usage(self):
        """Get current usage statistics"""
        self._cleanup_old_records()
        
        # Current requests per minute
        current_rpm = len(self.request_times)
        
        # Current tokens per minute
        current_tpm = sum(tokens for _, tokens in self.token_usage)
        
        # Current requests per day
        current_rpd = len(self.daily_requests)
        
        return current_rpm, current_tpm, current_rpd
    
    def can_make_request(self, estimated_tokens: int = 0) -> tuple[bool, str]:
        """
        Check if a request can be made without exceeding limits
        
        Args:
            estimated_tokens: Estimated tokens for the request
            
        Returns:
            Tuple of (can_make_request, reason_if_not)
        """
        with self._lock:
            current_rpm, current_tpm, current_rpd = self._get_current_usage()
            
            # Check RPM limit
            if current_rpm >= self.rpm_limit:
                return False, f"RPM limit exceeded ({current_rpm}/{self.rpm_limit})"
            
            # Check TPM limit
            if current_tpm + estimated_tokens > self.tpm_limit:
                return False, f"TPM limit would be exceeded ({current_tpm + estimated_tokens}/{self.tpm_limit})"
            
            # Check RPD limit
            if current_rpd >= self.rpd_limit:
                return False, f"RPD limit exceeded ({current_rpd}/{self.rpd_limit})"
            
            return True, "OK"
    
    def wait_if_needed(self, estimated_tokens: int = 0, max_wait: float = 300) -> bool:
        """
        Wait if necessary to respect rate limits
        
        Args:
            estimated_tokens: Estimated tokens for the request
            max_wait: Maximum time to wait in seconds
            
        Returns:
            True if can proceed, False if max_wait exceeded
        """
        start_time = time.time()
        
        while True:
            can_proceed, reason = self.can_make_request(estimated_tokens)
            
            if can_proceed:
                return True
            
            # Check if we've waited too long
            if time.time() - start_time > max_wait:
                return False
            
            # Calculate wait time based on the limiting factor
            if "RPM" in reason:
                # Wait until oldest request is more than a minute old
                if self.request_times:
                    wait_time = 61 - (time.time() - self.request_times[0])
                    wait_time = max(1, min(wait_time, 60))
                else:
                    wait_time = 1
            elif "TPM" in reason:
                # Wait until enough tokens are freed up
                wait_time = 30  # Wait 30 seconds and retry
            elif "RPD" in reason:
                # Daily limit exceeded - wait until next day
                now = datetime.now()
                tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                wait_time = (tomorrow - now).total_seconds()
                if wait_time > max_wait:
                    return False
            else:
                wait_time = 1
            
            print(f"Rate limit hit: {reason}. Waiting {wait_time:.1f} seconds...")
            time.sleep(min(wait_time, max_wait - (time.time() - start_time)))
    
    def wait_for_server_availability(self, max_wait: float = 120) -> bool:
        """
        Wait for server availability (handles 503 errors)
        
        Args:
            max_wait: Maximum time to wait in seconds
            
        Returns:
            True if should retry, False if max_wait exceeded
        """
        # Exponential backoff for server overload
        wait_times = [5, 10, 20, 30, 60]  # Progressive wait times
        
        for wait_time in wait_times:
            if wait_time > max_wait:
                return False
            
            print(f"Server overloaded. Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
            
            # Simple availability check - just return True to allow retry
            return True
        
        return False
    
    def record_request(self, tokens_used: int = 0):
        """
        Record a successful request
        
        Args:
            tokens_used: Number of tokens used in the request
        """
        with self._lock:
            current_time = time.time()
            current_datetime = datetime.now()
            
            # Record request time for RPM tracking
            self.request_times.append(current_time)
            
            # Record token usage for TPM tracking
            if tokens_used > 0:
                self.token_usage.append((current_time, tokens_used))
            
            # Record request for RPD tracking
            self.daily_requests.append(current_datetime)
            
            # Clean up old records
            self._cleanup_old_records()
    
    def get_usage_stats(self) -> dict:
        """Get current usage statistics"""
        with self._lock:
            current_rpm, current_tpm, current_rpd = self._get_current_usage()
            
            return {
                'requests_per_minute': {
                    'current': current_rpm,
                    'limit': self.rpm_limit,
                    'percentage': (current_rpm / self.rpm_limit) * 100
                },
                'tokens_per_minute': {
                    'current': current_tpm,
                    'limit': self.tpm_limit,
                    'percentage': (current_tpm / self.tpm_limit) * 100
                },
                'requests_per_day': {
                    'current': current_rpd,
                    'limit': self.rpd_limit,
                    'percentage': (current_rpd / self.rpd_limit) * 100
                }
            }

class RateLimitedEvaluator:
    """
    Wrapper for RAG evaluator with rate limiting
    """
    
    def __init__(self, evaluator, rate_limiter: Optional[GeminiRateLimiter] = None):
        """
        Initialize rate-limited evaluator
        
        Args:
            evaluator: RAG evaluator instance
            rate_limiter: Rate limiter instance (creates default if None)
        """
        self.evaluator = evaluator
        self.rate_limiter = rate_limiter or GeminiRateLimiter()
    
    def evaluate_single_response_with_rate_limit(
        self,
        input_query: str,
        actual_output: str,
        retrieval_context: list,
        expected_output: str = None,
        metrics_to_use: list = None,
        max_wait: float = 300
    ) -> dict:
        """
        Evaluate single response with rate limiting
        
        Args:
            input_query: User's question
            actual_output: RAG system's response
            retrieval_context: Retrieved document chunks
            expected_output: Expected ideal response (optional)
            metrics_to_use: List of metric names to use
            max_wait: Maximum time to wait for rate limits
            
        Returns:
            Dictionary with evaluation results
        """
        # Estimate tokens for the request
        text_to_evaluate = f"{input_query}\n{actual_output}\n" + "\n".join(retrieval_context)
        if expected_output:
            text_to_evaluate += f"\n{expected_output}"
        
        estimated_tokens = self.rate_limiter.estimate_tokens(text_to_evaluate)
        
        # Wait if needed to respect rate limits
        if not self.rate_limiter.wait_if_needed(estimated_tokens, max_wait):
            return {
                'error': 'Rate limit exceeded and max wait time reached',
                'estimated_tokens': estimated_tokens,
                'usage_stats': self.rate_limiter.get_usage_stats()
            }
        
        try:
            # Perform the evaluation
            results = self.evaluator._evaluate_single_response_direct(
                input_query=input_query,
                actual_output=actual_output,
                retrieval_context=retrieval_context,
                expected_output=expected_output,
                metrics_to_use=metrics_to_use
            )
            
            # Record the successful request
            self.rate_limiter.record_request(estimated_tokens)
            
            # Add usage stats to results
            results['usage_stats'] = self.rate_limiter.get_usage_stats()
            results['tokens_used'] = estimated_tokens
            
            return results
            
        except Exception as e:
            return {
                'error': f'Evaluation failed: {str(e)}',
                'estimated_tokens': estimated_tokens,
                'usage_stats': self.rate_limiter.get_usage_stats()
            }
    
    def evaluate_batch_with_rate_limit(
        self,
        test_cases: list,
        metrics_to_use: list = None,
        max_wait_per_request: float = 60,
        progress_callback=None
    ) -> list:
        """
        Evaluate batch with rate limiting and progress tracking
        
        Args:
            test_cases: List of test case dictionaries
            metrics_to_use: List of metric names to use
            max_wait_per_request: Maximum wait time per request
            progress_callback: Function to call with progress updates
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, case in enumerate(test_cases):
            if progress_callback:
                progress_callback(i, len(test_cases))
            
            result = self.evaluate_single_response_with_rate_limit(
                input_query=case['input'],
                actual_output=case['actual_output'],
                retrieval_context=case['retrieval_context'],
                expected_output=case.get('expected_output'),
                metrics_to_use=metrics_to_use,
                max_wait=max_wait_per_request
            )
            
            # Add case metadata
            result['case_id'] = i
            result['input'] = case['input']
            result['actual_output'] = case['actual_output']
            
            results.append(result)
            
            # Small delay between requests to be extra safe
            time.sleep(0.1)
        
        return results

# Global rate limiter instance
_global_rate_limiter = None

def get_global_rate_limiter() -> GeminiRateLimiter:
    """Get or create global rate limiter instance"""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = GeminiRateLimiter()
    return _global_rate_limiter

if __name__ == "__main__":
    # Test the rate limiter
    limiter = GeminiRateLimiter(requests_per_minute=5, tokens_per_minute=1000)  # Lower limits for testing
    
    print("Testing rate limiter...")
    
    for i in range(10):
        can_proceed, reason = limiter.can_make_request(100)
        print(f"Request {i+1}: {can_proceed} - {reason}")
        
        if can_proceed:
            limiter.record_request(100)
            print(f"Usage: {limiter.get_usage_stats()}")
        else:
            print("Waiting for rate limit...")
            if limiter.wait_if_needed(100, max_wait=10):
                limiter.record_request(100)
                print(f"After wait - Usage: {limiter.get_usage_stats()}")
            else:
                print("Max wait exceeded")
        
        time.sleep(1)