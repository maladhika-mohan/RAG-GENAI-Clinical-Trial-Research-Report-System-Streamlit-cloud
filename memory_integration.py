"""
LangChain Memory Integration for RAG Applications
Provides conversation memory, document memory, and summary capabilities
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import streamlit as st

# LangChain imports
from langchain.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationBufferMemory
)
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage as CoreBaseMessage

# For LLM integration
import google.generativeai as genai
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Mapping, Optional


class GeminiLLMWrapper(LLM):
    """LangChain-compatible wrapper for Gemini API"""

    api_key: str
    model_name: str = "gemini-2.0-flash-lite"
    client: Any = None

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-lite", **kwargs):
        """
        Initialize Gemini LLM wrapper

        Args:
            api_key: Gemini API key
            model_name: Model name
        """
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model_name = model_name
        self._setup_client()

    def _setup_client(self):
        """Setup Gemini client"""
        try:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
        except Exception as e:
            print(f"Error setting up Gemini client: {e}")
            self.client = None

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "gemini"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate response for LangChain compatibility

        Args:
            prompt: Input prompt
            stop: Stop sequences (not used for Gemini)
            run_manager: Callback manager

        Returns:
            str: Generated response
        """
        if not self.client:
            return "Error: Gemini client not configured"

        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name}


class DocumentMemory:
    """Memory system for document-specific conversations"""
    
    def __init__(self, max_entries_per_doc: int = 20):
        """
        Initialize document memory
        
        Args:
            max_entries_per_doc: Maximum memory entries per document
        """
        self.max_entries_per_doc = max_entries_per_doc
        self.document_memories: Dict[str, List[Dict[str, Any]]] = {}
    
    def add_interaction(self, document_id: str, query: str, response: str, 
                       metadata: Optional[Dict[str, Any]] = None):
        """
        Add interaction to document memory
        
        Args:
            document_id: Document identifier
            query: User query
            response: AI response
            metadata: Optional metadata
        """
        if document_id not in self.document_memories:
            self.document_memories[document_id] = []
        
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'metadata': metadata or {}
        }
        
        self.document_memories[document_id].append(interaction)
        
        # Maintain size limit
        if len(self.document_memories[document_id]) > self.max_entries_per_doc:
            self.document_memories[document_id] = \
                self.document_memories[document_id][-self.max_entries_per_doc:]
    
    def get_document_history(self, document_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get conversation history for a document
        
        Args:
            document_id: Document identifier
            limit: Maximum number of interactions to return
            
        Returns:
            List of recent interactions
        """
        if document_id not in self.document_memories:
            return []
        
        return self.document_memories[document_id][-limit:]
    
    def search_document_interactions(self, document_id: str, 
                                   query_keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Search for relevant interactions in document memory
        
        Args:
            document_id: Document identifier
            query_keywords: Keywords to search for
            
        Returns:
            List of relevant interactions
        """
        if document_id not in self.document_memories:
            return []
        
        relevant_interactions = []
        
        for interaction in self.document_memories[document_id]:
            query_text = interaction['query'].lower()
            response_text = interaction['response'].lower()
            
            # Check if any keyword appears in query or response
            if any(keyword.lower() in query_text or keyword.lower() in response_text 
                   for keyword in query_keywords):
                relevant_interactions.append(interaction)
        
        return relevant_interactions


class RAGMemoryManager:
    """Comprehensive memory management for RAG applications"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-lite"):
        """
        Initialize RAG memory manager

        Args:
            api_key: Gemini API key for LLM operations
            model_name: Model name
        """
        self.api_key = api_key
        self.model_name = model_name
        self.llm = None
        self.conversation_buffer = None
        self.conversation_summary = None
        self.memory_enabled = False

        try:
            self.llm = GeminiLLMWrapper(api_key=api_key, model_name=model_name)

            # Initialize different memory types
            self.conversation_buffer = ConversationBufferWindowMemory(
                k=10,  # Keep last 10 interactions
                return_messages=True,
                memory_key="chat_history"
            )

            self.conversation_summary = ConversationSummaryMemory(
                llm=self.llm,
                return_messages=True,
                memory_key="chat_summary"
            )

            self.memory_enabled = True

        except Exception as e:
            print(f"Warning: Could not initialize LangChain memory: {e}")
            print("Falling back to basic memory functionality")
            # Initialize basic buffer memory without summary
            self.conversation_buffer = ConversationBufferWindowMemory(
                k=10,
                return_messages=True,
                memory_key="chat_history"
            )
            self.conversation_summary = None

        self.document_memory = DocumentMemory()

        # Session management
        self.session_memories: Dict[str, Dict[str, Any]] = {}
        self.current_session_id: Optional[str] = None
    
    def create_session(self, session_id: str) -> None:
        """
        Create a new memory session

        Args:
            session_id: Unique session identifier
        """
        session_data = {
            'buffer_memory': ConversationBufferWindowMemory(
                k=10, return_messages=True, memory_key="chat_history"
            ),
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat()
        }

        # Only add summary memory if LLM is available
        if self.llm and self.memory_enabled:
            try:
                session_data['summary_memory'] = ConversationSummaryMemory(
                    llm=self.llm, return_messages=True, memory_key="chat_summary"
                )
            except Exception as e:
                print(f"Warning: Could not create summary memory for session: {e}")
                session_data['summary_memory'] = None
        else:
            session_data['summary_memory'] = None

        self.session_memories[session_id] = session_data
        self.current_session_id = session_id
    
    def set_current_session(self, session_id: str) -> None:
        """
        Set the current active session
        
        Args:
            session_id: Session identifier
        """
        if session_id not in self.session_memories:
            self.create_session(session_id)
        
        self.current_session_id = session_id
        self.session_memories[session_id]['last_accessed'] = datetime.now().isoformat()
    
    def add_conversation(self, human_input: str, ai_response: str, 
                        document_ids: Optional[List[str]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add conversation to memory systems
        
        Args:
            human_input: User's input
            ai_response: AI's response
            document_ids: List of document IDs involved
            metadata: Optional metadata
        """
        # Add to current session buffer memory
        if self.current_session_id and self.current_session_id in self.session_memories:
            session_memory = self.session_memories[self.current_session_id]
            session_memory['buffer_memory'].save_context(
                {"input": human_input},
                {"output": ai_response}
            )
            # Only save to summary memory if it exists
            if session_memory.get('summary_memory'):
                try:
                    session_memory['summary_memory'].save_context(
                        {"input": human_input},
                        {"output": ai_response}
                    )
                except Exception as e:
                    print(f"Warning: Could not save to summary memory: {e}")

        # Add to global memory
        if self.conversation_buffer:
            self.conversation_buffer.save_context(
                {"input": human_input},
                {"output": ai_response}
            )

        if self.conversation_summary:
            try:
                self.conversation_summary.save_context(
                    {"input": human_input},
                    {"output": ai_response}
                )
            except Exception as e:
                print(f"Warning: Could not save to global summary memory: {e}")
        
        # Add to document-specific memory
        if document_ids:
            for doc_id in document_ids:
                self.document_memory.add_interaction(
                    doc_id, human_input, ai_response, metadata
                )
    
    def get_relevant_history(self, current_query: str,
                           document_ids: Optional[List[str]] = None,
                           max_interactions: int = 5) -> Dict[str, Any]:
        """
        Get relevant conversation history for current query

        Args:
            current_query: Current user query
            document_ids: Document IDs to search in
            max_interactions: Maximum interactions to return

        Returns:
            Dict with relevant history from different memory types
        """
        history = {
            'buffer_history': [],
            'summary': '',
            'document_history': [],
            'relevant_interactions': [],
            'is_history_query': self._is_asking_about_history(current_query),
            'formatted_history': ''
        }
        
        # Get buffer history from current session
        if self.current_session_id and self.current_session_id in self.session_memories:
            session_memory = self.session_memories[self.current_session_id]

            # Get buffer history
            if session_memory.get('buffer_memory'):
                try:
                    buffer_vars = session_memory['buffer_memory'].load_memory_variables({})
                    if 'chat_history' in buffer_vars:
                        history['buffer_history'] = buffer_vars['chat_history'][-max_interactions:]
                except Exception as e:
                    print(f"Warning: Could not load buffer memory: {e}")

            # Get summary if available
            if session_memory.get('summary_memory'):
                try:
                    summary_vars = session_memory['summary_memory'].load_memory_variables({})
                    if 'chat_summary' in summary_vars:
                        history['summary'] = summary_vars['chat_summary']
                except Exception as e:
                    print(f"Warning: Could not load summary memory: {e}")
        
        # Get document-specific history
        if document_ids:
            query_keywords = self._extract_keywords(current_query)
            
            for doc_id in document_ids:
                doc_history = self.document_memory.get_document_history(doc_id, max_interactions)
                history['document_history'].extend(doc_history)
                
                # Search for relevant interactions
                relevant = self.document_memory.search_document_interactions(
                    doc_id, query_keywords
                )
                history['relevant_interactions'].extend(relevant)
        
        # Format history for easy consumption
        if history['is_history_query']:
            history['formatted_history'] = self._format_history_for_query(history, current_query)

        return history

    def _is_asking_about_history(self, query: str) -> bool:
        """
        Check if the query is asking about conversation history

        Args:
            query: User query

        Returns:
            bool: True if asking about history
        """
        history_keywords = [
            'previous question', 'last question', 'what did i ask',
            'my previous', 'my last', 'earlier question', 'before',
            'what was my', 'conversation history', 'chat history',
            'what did we discuss', 'our conversation', 'talked about',
            'mentioned before', 'said earlier', 'discussed previously'
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in history_keywords)

    def _format_history_for_query(self, history: Dict[str, Any], query: str) -> str:
        """
        Format conversation history for answering history-related queries

        Args:
            history: History dictionary
            query: Current query

        Returns:
            str: Formatted history string
        """
        formatted_parts = []

        # Add recent conversation history
        if history['buffer_history']:
            formatted_parts.append("Recent conversation history:")

            for i, message in enumerate(history['buffer_history'][-10:], 1):
                if hasattr(message, 'content'):
                    role = "You" if isinstance(message, HumanMessage) else "Assistant"
                    content = message.content[:200] + "..." if len(message.content) > 200 else message.content
                    formatted_parts.append(f"{i}. {role}: {content}")

        # Add summary if available
        if history['summary']:
            formatted_parts.append(f"\nConversation summary: {history['summary']}")

        # Add document-specific history if relevant
        if history['document_history']:
            formatted_parts.append("\nDocument-related questions:")
            for interaction in history['document_history'][-5:]:
                formatted_parts.append(f"Q: {interaction['query']}")
                formatted_parts.append(f"A: {interaction['response'][:150]}...")

        return "\n".join(formatted_parts) if formatted_parts else "No previous conversation history found."
    
    def get_conversation_context(self, max_tokens: int = 1000) -> str:
        """
        Get formatted conversation context for prompt inclusion
        
        Args:
            max_tokens: Maximum tokens for context
            
        Returns:
            str: Formatted conversation context
        """
        context_parts = []
        
        # Get recent buffer history
        if self.current_session_id and self.current_session_id in self.session_memories:
            session_memory = self.session_memories[self.current_session_id]

            if session_memory.get('buffer_memory'):
                try:
                    buffer_vars = session_memory['buffer_memory'].load_memory_variables({})

                    if 'chat_history' in buffer_vars and buffer_vars['chat_history']:
                        context_parts.append("Recent conversation:")

                        for message in buffer_vars['chat_history'][-3:]:  # Last 3 interactions
                            if hasattr(message, 'content'):
                                role = "Human" if isinstance(message, HumanMessage) else "Assistant"
                                context_parts.append(f"{role}: {message.content}")
                except Exception as e:
                    print(f"Warning: Could not load buffer history for context: {e}")

        # Add summary if available
        if self.current_session_id and self.current_session_id in self.session_memories:
            session_memory = self.session_memories[self.current_session_id]

            if session_memory.get('summary_memory'):
                try:
                    summary_vars = session_memory['summary_memory'].load_memory_variables({})

                    if 'chat_summary' in summary_vars and summary_vars['chat_summary']:
                        context_parts.append(f"Conversation summary: {summary_vars['chat_summary']}")
                except Exception as e:
                    print(f"Warning: Could not load summary for context: {e}")
        
        context = "\n".join(context_parts)
        
        # Truncate if too long (rough token estimation)
        if len(context) > max_tokens * 4:  # Rough: 1 token ≈ 4 chars
            context = context[:max_tokens * 4] + "..."
        
        return context
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from query for memory search
        
        Args:
            query: User query
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction
        import re
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'what', 'when', 'where', 'why', 'how', 'who', 'which'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Limit to 10 keywords
    
    def clear_session_memory(self, session_id: Optional[str] = None) -> None:
        """
        Clear memory for a specific session
        
        Args:
            session_id: Session to clear (current session if None)
        """
        target_session = session_id or self.current_session_id
        
        if target_session and target_session in self.session_memories:
            del self.session_memories[target_session]
        
        if target_session == self.current_session_id:
            self.current_session_id = None
    
    def export_session_memory(self, session_id: str) -> Dict[str, Any]:
        """
        Export session memory for persistence
        
        Args:
            session_id: Session to export
            
        Returns:
            Dict with exportable session data
        """
        if session_id not in self.session_memories:
            return {}
        
        session = self.session_memories[session_id]
        
        # Extract messages from buffer memory
        buffer_vars = session['buffer_memory'].load_memory_variables({})
        messages = []
        
        if 'chat_history' in buffer_vars:
            for msg in buffer_vars['chat_history']:
                if hasattr(msg, 'content'):
                    messages.append({
                        'type': 'human' if isinstance(msg, HumanMessage) else 'ai',
                        'content': msg.content
                    })
        
        # Get summary
        summary_vars = session['summary_memory'].load_memory_variables({})
        summary = summary_vars.get('chat_summary', '')
        
        return {
            'session_id': session_id,
            'created_at': session['created_at'],
            'last_accessed': session['last_accessed'],
            'messages': messages,
            'summary': summary
        }
    
    def import_session_memory(self, session_data: Dict[str, Any]) -> str:
        """
        Import session memory from exported data
        
        Args:
            session_data: Exported session data
            
        Returns:
            str: Imported session ID
        """
        session_id = session_data['session_id']
        
        # Create new session
        self.create_session(session_id)
        
        # Restore messages
        for msg_data in session_data.get('messages', []):
            if msg_data['type'] == 'human':
                # Find corresponding AI message
                ai_msg = None
                msg_index = session_data['messages'].index(msg_data)
                if msg_index + 1 < len(session_data['messages']):
                    next_msg = session_data['messages'][msg_index + 1]
                    if next_msg['type'] == 'ai':
                        ai_msg = next_msg['content']
                
                if ai_msg:
                    self.add_conversation(msg_data['content'], ai_msg)
        
        return session_id
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics
        
        Returns:
            Dict with memory statistics
        """
        stats = {
            'total_sessions': len(self.session_memories),
            'current_session': self.current_session_id,
            'document_memories': len(self.document_memory.document_memories),
            'total_document_interactions': sum(
                len(interactions) 
                for interactions in self.document_memory.document_memories.values()
            )
        }
        
        # Session-specific stats
        if self.current_session_id and self.current_session_id in self.session_memories:
            session = self.session_memories[self.current_session_id]

            if session.get('buffer_memory'):
                try:
                    buffer_vars = session['buffer_memory'].load_memory_variables({})
                    stats['current_session_messages'] = len(buffer_vars.get('chat_history', []))
                except Exception as e:
                    print(f"Warning: Could not get session message count: {e}")
                    stats['current_session_messages'] = 0
            else:
                stats['current_session_messages'] = 0

            stats['session_created'] = session['created_at']
            stats['session_last_accessed'] = session['last_accessed']
        
        return stats


@st.cache_resource
def get_memory_manager(api_key: str, model_name: str = "gemini-2.0-flash-lite") -> RAGMemoryManager:
    """
    Get cached memory manager instance
    
    Args:
        api_key: Gemini API key
        model_name: Model name
        
    Returns:
        RAGMemoryManager: Cached memory manager instance
    """
    return RAGMemoryManager(api_key, model_name)
