"""
Token-Efficient Prompt Construction for RAG Applications
Optimizes prompts to reduce token usage while maintaining quality
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import streamlit as st
from token_counter import TokenCounter, get_token_counter


@dataclass
class PromptTemplate:
    """Template for efficient prompt construction"""
    name: str
    template: str
    required_variables: List[str]
    optional_variables: List[str]
    max_tokens: int
    description: str


class PromptOptimizer:
    """Token-efficient prompt construction and optimization"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash-lite", 
                 max_tokens: int = 4000):
        """
        Initialize prompt optimizer
        
        Args:
            model_name: Target model name for token counting
            max_tokens: Maximum tokens allowed in prompts
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.token_counter = get_token_counter(model_name)
        self.templates = self._initialize_templates()
        
        # Compression strategies
        self.compression_strategies = {
            'remove_redundancy': self._remove_redundant_text,
            'summarize_context': self._summarize_long_context,
            'prioritize_relevant': self._prioritize_relevant_chunks,
            'compress_whitespace': self._compress_whitespace
        }
    
    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize optimized prompt templates"""
        templates = {}
        
        # RAG Query Template - Optimized
        templates['rag_query'] = PromptTemplate(
            name="rag_query",
            template="""Context: {context}

Q: {query}
A:""",
            required_variables=['context', 'query'],
            optional_variables=[],
            max_tokens=3500,
            description="Minimal RAG query template"
        )
        
        # RAG Query with Instructions - Optimized
        templates['rag_query_detailed'] = PromptTemplate(
            name="rag_query_detailed",
            template="""Based on the context below, answer the question. Be concise and accurate.

Context:
{context}

Question: {query}

Answer:""",
            required_variables=['context', 'query'],
            optional_variables=[],
            max_tokens=3800,
            description="Detailed RAG query with instructions"
        )
        
        # Multi-document RAG Template
        templates['multi_doc_rag'] = PromptTemplate(
            name="multi_doc_rag",
            template="""Sources:
{context}

Query: {query}
Response:""",
            required_variables=['context', 'query'],
            optional_variables=[],
            max_tokens=3600,
            description="Multi-document RAG template"
        )
        
        # Evaluation Template - Optimized
        templates['evaluation'] = PromptTemplate(
            name="evaluation",
            template="""Evaluate this response:

Query: {query}
Response: {response}
Context: {context}

Rate {metric} (0-1): """,
            required_variables=['query', 'response', 'context', 'metric'],
            optional_variables=[],
            max_tokens=2000,
            description="Response evaluation template"
        )
        
        return templates
    
    def build_optimized_prompt(self, template_name: str, variables: Dict[str, str],
                              target_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Build an optimized prompt from template
        
        Args:
            template_name: Name of the template to use
            variables: Variables to fill in the template
            target_tokens: Target token count (uses template max if None)
            
        Returns:
            Dict with optimized prompt and metadata
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name]
        target_tokens = target_tokens or template.max_tokens
        
        # Check required variables
        missing_vars = [var for var in template.required_variables 
                       if var not in variables]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Initial prompt construction
        prompt = template.template.format(**variables)
        initial_tokens = self.token_counter.count_tokens(prompt)
        
        # Optimize if needed
        if initial_tokens > target_tokens:
            prompt = self._optimize_prompt(prompt, template, variables, target_tokens)
        
        final_tokens = self.token_counter.count_tokens(prompt)
        
        return {
            'prompt': prompt,
            'template_name': template_name,
            'initial_tokens': initial_tokens,
            'final_tokens': final_tokens,
            'tokens_saved': initial_tokens - final_tokens,
            'target_tokens': target_tokens,
            'optimization_applied': initial_tokens > target_tokens,
            'efficiency_ratio': final_tokens / initial_tokens if initial_tokens > 0 else 1.0
        }
    
    def _optimize_prompt(self, prompt: str, template: PromptTemplate, 
                        variables: Dict[str, str], target_tokens: int) -> str:
        """
        Optimize prompt to fit target token count
        
        Args:
            prompt: Original prompt
            template: Template used
            variables: Template variables
            target_tokens: Target token count
            
        Returns:
            str: Optimized prompt
        """
        # Strategy 1: Compress context if it exists
        if 'context' in variables:
            optimized_context = self._optimize_context(
                variables['context'], 
                target_tokens - 200  # Reserve tokens for query and template
            )
            
            # Rebuild prompt with optimized context
            optimized_vars = variables.copy()
            optimized_vars['context'] = optimized_context
            prompt = template.template.format(**optimized_vars)
        
        # Strategy 2: Apply general compression
        current_tokens = self.token_counter.count_tokens(prompt)
        if current_tokens > target_tokens:
            prompt = self._apply_compression_strategies(prompt, target_tokens)
        
        # Strategy 3: Final truncation if still too long
        current_tokens = self.token_counter.count_tokens(prompt)
        if current_tokens > target_tokens:
            prompt = self.token_counter.truncate_text_to_tokens(
                prompt, target_tokens, preserve_end=False
            )
        
        return prompt
    
    def _optimize_context(self, context: str, max_context_tokens: int) -> str:
        """
        Optimize context to fit within token limit
        
        Args:
            context: Original context text
            max_context_tokens: Maximum tokens for context
            
        Returns:
            str: Optimized context
        """
        current_tokens = self.token_counter.count_tokens(context)
        
        if current_tokens <= max_context_tokens:
            return context
        
        # Strategy 1: Split into chunks and prioritize
        if '\n\n' in context:
            chunks = context.split('\n\n')
            return self._prioritize_relevant_chunks(chunks, max_context_tokens)
        
        # Strategy 2: Sentence-level optimization
        sentences = re.split(r'[.!?]+', context)
        if len(sentences) > 1:
            return self._prioritize_relevant_sentences(sentences, max_context_tokens)
        
        # Strategy 3: Simple truncation
        return self.token_counter.truncate_text_to_tokens(
            context, max_context_tokens, preserve_end=False
        )
    
    def _prioritize_relevant_chunks(self, chunks: List[str], 
                                   max_tokens: int) -> str:
        """
        Prioritize and select most relevant chunks
        
        Args:
            chunks: List of text chunks
            max_tokens: Maximum tokens allowed
            
        Returns:
            str: Optimized context from selected chunks
        """
        # Score chunks by length and content quality
        scored_chunks = []
        
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            tokens = self.token_counter.count_tokens(chunk)
            
            # Simple relevance scoring
            score = self._calculate_chunk_relevance(chunk)
            
            scored_chunks.append({
                'text': chunk.strip(),
                'tokens': tokens,
                'score': score,
                'efficiency': score / tokens if tokens > 0 else 0
            })
        
        # Sort by efficiency (relevance per token)
        scored_chunks.sort(key=lambda x: x['efficiency'], reverse=True)
        
        # Select chunks that fit within token limit
        selected_chunks = []
        total_tokens = 0
        
        for chunk_data in scored_chunks:
            if total_tokens + chunk_data['tokens'] <= max_tokens:
                selected_chunks.append(chunk_data['text'])
                total_tokens += chunk_data['tokens']
            else:
                # Try to fit a truncated version
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 50:  # Minimum useful chunk size
                    truncated = self.token_counter.truncate_text_to_tokens(
                        chunk_data['text'], remaining_tokens
                    )
                    selected_chunks.append(truncated)
                break
        
        return '\n\n'.join(selected_chunks)
    
    def _prioritize_relevant_sentences(self, sentences: List[str], 
                                      max_tokens: int) -> str:
        """
        Prioritize and select most relevant sentences
        
        Args:
            sentences: List of sentences
            max_tokens: Maximum tokens allowed
            
        Returns:
            str: Optimized context from selected sentences
        """
        # Score sentences
        scored_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            tokens = self.token_counter.count_tokens(sentence)
            score = self._calculate_sentence_relevance(sentence)
            
            scored_sentences.append({
                'text': sentence,
                'tokens': tokens,
                'score': score,
                'efficiency': score / tokens if tokens > 0 else 0
            })
        
        # Sort by efficiency
        scored_sentences.sort(key=lambda x: x['efficiency'], reverse=True)
        
        # Select sentences that fit
        selected_sentences = []
        total_tokens = 0
        
        for sent_data in scored_sentences:
            if total_tokens + sent_data['tokens'] <= max_tokens:
                selected_sentences.append(sent_data['text'])
                total_tokens += sent_data['tokens']
            else:
                break
        
        return '. '.join(selected_sentences) + '.'
    
    def _calculate_chunk_relevance(self, chunk: str) -> float:
        """
        Calculate relevance score for a text chunk
        
        Args:
            chunk: Text chunk
            
        Returns:
            float: Relevance score
        """
        score = 0.0
        
        # Length penalty for very short or very long chunks
        length = len(chunk)
        if 50 <= length <= 500:
            score += 1.0
        elif length < 50:
            score += 0.3
        else:
            score += 0.7
        
        # Content quality indicators
        if any(word in chunk.lower() for word in 
               ['study', 'research', 'analysis', 'result', 'conclusion']):
            score += 0.5
        
        # Sentence structure
        sentence_count = len(re.split(r'[.!?]+', chunk))
        if 1 <= sentence_count <= 5:
            score += 0.3
        
        return score
    
    def _calculate_sentence_relevance(self, sentence: str) -> float:
        """
        Calculate relevance score for a sentence
        
        Args:
            sentence: Sentence text
            
        Returns:
            float: Relevance score
        """
        score = 0.0
        
        # Length scoring
        length = len(sentence)
        if 20 <= length <= 200:
            score += 1.0
        elif length < 20:
            score += 0.2
        else:
            score += 0.6
        
        # Content indicators
        if any(word in sentence.lower() for word in 
               ['important', 'significant', 'key', 'main', 'primary']):
            score += 0.4
        
        return score
    
    def _apply_compression_strategies(self, text: str, target_tokens: int) -> str:
        """
        Apply various compression strategies
        
        Args:
            text: Input text
            target_tokens: Target token count
            
        Returns:
            str: Compressed text
        """
        compressed_text = text
        
        # Apply strategies in order
        for strategy_name, strategy_func in self.compression_strategies.items():
            current_tokens = self.token_counter.count_tokens(compressed_text)
            if current_tokens <= target_tokens:
                break
            
            compressed_text = strategy_func(compressed_text)
        
        return compressed_text
    
    def _remove_redundant_text(self, text: str) -> str:
        """Remove redundant phrases and repetitions"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common redundant phrases
        redundant_phrases = [
            r'\b(as mentioned|as stated|as discussed) (above|before|earlier)\b',
            r'\b(it should be noted that|it is important to note that)\b',
            r'\b(in other words|that is to say)\b',
        ]
        
        for pattern in redundant_phrases:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _summarize_long_context(self, text: str) -> str:
        """Summarize long context sections"""
        # Simple summarization by keeping key sentences
        sentences = re.split(r'[.!?]+', text)
        
        if len(sentences) <= 3:
            return text
        
        # Keep first and last sentences, plus any with key terms
        key_sentences = []
        
        # Always keep first sentence
        if sentences[0].strip():
            key_sentences.append(sentences[0].strip())
        
        # Keep sentences with important terms
        for sentence in sentences[1:-1]:
            sentence = sentence.strip()
            if any(term in sentence.lower() for term in 
                   ['result', 'conclusion', 'finding', 'significant', 'important']):
                key_sentences.append(sentence)
        
        # Always keep last sentence if different from first
        if len(sentences) > 1 and sentences[-1].strip() != sentences[0].strip():
            key_sentences.append(sentences[-1].strip())
        
        return '. '.join(key_sentences) + '.'
    
    def _compress_whitespace(self, text: str) -> str:
        """Compress excessive whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\n+', '\n\n', text)
        
        # Remove trailing whitespace from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        
        return text.strip()
    
    def get_optimization_stats(self, original_text: str, 
                              optimized_text: str) -> Dict[str, Any]:
        """
        Get optimization statistics
        
        Args:
            original_text: Original text
            optimized_text: Optimized text
            
        Returns:
            Dict with optimization statistics
        """
        return self.token_counter.get_token_efficiency_stats(
            original_text, optimized_text
        )


@st.cache_resource
def get_prompt_optimizer(model_name: str = "gemini-2.0-flash-lite", 
                        max_tokens: int = 4000) -> PromptOptimizer:
    """
    Get cached prompt optimizer instance
    
    Args:
        model_name: Target model name
        max_tokens: Maximum tokens allowed
        
    Returns:
        PromptOptimizer: Cached optimizer instance
    """
    return PromptOptimizer(model_name, max_tokens)
