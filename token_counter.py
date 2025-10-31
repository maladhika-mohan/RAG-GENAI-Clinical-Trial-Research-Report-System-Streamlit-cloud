"""
Token Counter Utility for RAG Application
Provides token counting and optimization utilities for efficient prompt construction
"""

import tiktoken
from typing import List, Dict, Optional, Tuple
import re
import streamlit as st


class TokenCounter:
    """Utility class for counting and managing tokens in text"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize token counter with specific model encoding
        
        Args:
            model_name: Model name for token encoding (default: gpt-3.5-turbo)
        """
        self.model_name = model_name
        self.encoding = self._get_encoding()
        
        # Token limits for different models
        self.model_limits = {
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gemini-pro": 30720,
            "gemini-2.0-flash-lite": 1048576,  # 1M tokens
            "claude-3": 200000
        }
    
    def _get_encoding(self):
        """Get the appropriate encoding for the model"""
        try:
            # Try to get model-specific encoding
            if "gpt-4" in self.model_name.lower():
                return tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in self.model_name.lower():
                return tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Default to cl100k_base for most modern models
                return tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback to cl100k_base
            return tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            int: Number of tokens
        """
        if not text:
            return 0
        
        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """
        Count tokens for multiple texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List[int]: Token counts for each text
        """
        return [self.count_tokens(text) for text in texts]
    
    def get_model_limit(self, model_name: Optional[str] = None) -> int:
        """
        Get token limit for a specific model
        
        Args:
            model_name: Model name (uses instance model if None)
            
        Returns:
            int: Token limit for the model
        """
        model = model_name or self.model_name
        
        # Check for exact match first
        if model in self.model_limits:
            return self.model_limits[model]
        
        # Check for partial matches
        for key, limit in self.model_limits.items():
            if key.lower() in model.lower():
                return limit
        
        # Default fallback
        return 4096
    
    def estimate_tokens_from_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        Estimate tokens for a list of chat messages
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            int: Estimated total tokens
        """
        total_tokens = 0
        
        for message in messages:
            # Count tokens in content
            content_tokens = self.count_tokens(message.get('content', ''))
            
            # Add overhead for message structure (role, formatting, etc.)
            # Rough estimate: 4 tokens per message for structure
            total_tokens += content_tokens + 4
        
        # Add overhead for conversation structure
        total_tokens += 2
        
        return total_tokens
    
    def truncate_text_to_tokens(self, text: str, max_tokens: int, 
                               preserve_end: bool = False) -> str:
        """
        Truncate text to fit within token limit
        
        Args:
            text: Input text to truncate
            max_tokens: Maximum number of tokens allowed
            preserve_end: If True, preserve end of text instead of beginning
            
        Returns:
            str: Truncated text
        """
        if not text:
            return text
        
        current_tokens = self.count_tokens(text)
        
        if current_tokens <= max_tokens:
            return text
        
        try:
            tokens = self.encoding.encode(text)
            
            if preserve_end:
                # Keep the end of the text
                truncated_tokens = tokens[-max_tokens:]
            else:
                # Keep the beginning of the text
                truncated_tokens = tokens[:max_tokens]
            
            return self.encoding.decode(truncated_tokens)
        
        except Exception:
            # Fallback: character-based truncation
            # Rough estimate: 1 token ≈ 4 characters
            max_chars = max_tokens * 4
            
            if preserve_end:
                return text[-max_chars:] if len(text) > max_chars else text
            else:
                return text[:max_chars] if len(text) > max_chars else text
    
    def split_text_by_tokens(self, text: str, max_tokens_per_chunk: int, 
                            overlap_tokens: int = 0) -> List[str]:
        """
        Split text into chunks based on token count
        
        Args:
            text: Input text to split
            max_tokens_per_chunk: Maximum tokens per chunk
            overlap_tokens: Number of tokens to overlap between chunks
            
        Returns:
            List[str]: List of text chunks
        """
        if not text:
            return []
        
        try:
            tokens = self.encoding.encode(text)
            chunks = []
            
            start = 0
            while start < len(tokens):
                end = min(start + max_tokens_per_chunk, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = self.encoding.decode(chunk_tokens)
                chunks.append(chunk_text)
                
                # Move start position with overlap
                start = end - overlap_tokens
                if start >= end:  # Prevent infinite loop
                    break
            
            return chunks
        
        except Exception:
            # Fallback: character-based splitting
            max_chars = max_tokens_per_chunk * 4
            overlap_chars = overlap_tokens * 4
            
            chunks = []
            start = 0
            
            while start < len(text):
                end = min(start + max_chars, len(text))
                chunk = text[start:end]
                chunks.append(chunk)
                
                start = end - overlap_chars
                if start >= end:
                    break
            
            return chunks
    
    def get_token_efficiency_stats(self, original_text: str, 
                                  optimized_text: str) -> Dict[str, any]:
        """
        Calculate token efficiency statistics
        
        Args:
            original_text: Original text before optimization
            optimized_text: Optimized text after processing
            
        Returns:
            Dict: Statistics about token efficiency
        """
        original_tokens = self.count_tokens(original_text)
        optimized_tokens = self.count_tokens(optimized_text)
        
        if original_tokens == 0:
            reduction_percentage = 0
        else:
            reduction_percentage = ((original_tokens - optimized_tokens) / original_tokens) * 100
        
        return {
            'original_tokens': original_tokens,
            'optimized_tokens': optimized_tokens,
            'tokens_saved': original_tokens - optimized_tokens,
            'reduction_percentage': round(reduction_percentage, 2),
            'efficiency_ratio': round(optimized_tokens / original_tokens, 3) if original_tokens > 0 else 0
        }


@st.cache_resource
def get_token_counter(model_name: str = "gemini-2.0-flash-lite") -> TokenCounter:
    """
    Get cached token counter instance
    
    Args:
        model_name: Model name for token counting
        
    Returns:
        TokenCounter: Cached token counter instance
    """
    return TokenCounter(model_name)


def format_token_count(token_count: int) -> str:
    """
    Format token count for display
    
    Args:
        token_count: Number of tokens
        
    Returns:
        str: Formatted token count string
    """
    if token_count < 1000:
        return f"{token_count} tokens"
    elif token_count < 1000000:
        return f"{token_count/1000:.1f}K tokens"
    else:
        return f"{token_count/1000000:.1f}M tokens"


def estimate_cost(token_count: int, model_name: str = "gemini-2.0-flash-lite") -> float:
    """
    Estimate cost based on token count and model
    
    Args:
        token_count: Number of tokens
        model_name: Model name for pricing
        
    Returns:
        float: Estimated cost in USD
    """
    # Rough pricing estimates (as of 2024)
    pricing = {
        "gpt-3.5-turbo": 0.002 / 1000,  # $0.002 per 1K tokens
        "gpt-4": 0.03 / 1000,           # $0.03 per 1K tokens
        "gemini-pro": 0.00025 / 1000,   # $0.00025 per 1K tokens
        "gemini-2.0-flash-lite": 0.000075 / 1000,  # $0.000075 per 1K tokens
        "claude-3": 0.015 / 1000        # $0.015 per 1K tokens
    }
    
    # Find matching pricing
    cost_per_token = 0.001 / 1000  # Default fallback
    
    for model, price in pricing.items():
        if model.lower() in model_name.lower():
            cost_per_token = price
            break
    
    return token_count * cost_per_token
