"""
Embedding Generation Module for Clinical Trial RAG Application
Uses Hugging Face sentence-transformers/all-MiniLM-L6-v2 for text embeddings
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Union
import streamlit as st


class EmbeddingGenerator:
    """Handles text embedding generation using sentence-transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2
    
    @st.cache_resource
    def load_model(_self):
        """
        Load the sentence transformer model (cached for efficiency)
        
        Returns:
            SentenceTransformer: Loaded model
        """
        try:
            model = SentenceTransformer(_self.model_name)
            return model
        except Exception as e:
            raise Exception(f"Error loading embedding model: {str(e)}")
    
    def get_model(self):
        """Get the loaded model instance"""
        if self.model is None:
            self.model = self.load_model()
        return self.model
    
    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            np.ndarray: Embeddings array
        """
        try:
            model = self.get_model()
            
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
            
            # Generate embeddings
            embeddings = model.encode(texts, convert_to_numpy=True)
            
            return embeddings
            
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def generate_chunk_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for document chunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            
        Returns:
            List[Dict]: Chunks with added 'embedding' field
        """
        try:
            # Extract text from chunks
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk['embedding'] = embeddings[i].tolist()  # Convert to list for JSON serialization
            
            return chunks
            
        except Exception as e:
            raise Exception(f"Error generating chunk embeddings: {str(e)}")
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query
        
        Args:
            query: Search query text
            
        Returns:
            np.ndarray: Query embedding
        """
        try:
            embedding = self.generate_embeddings(query)
            return embedding[0]  # Return single embedding
            
        except Exception as e:
            raise Exception(f"Error generating query embedding: {str(e)}")
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            float: Cosine similarity score
        """
        try:
            # Ensure embeddings are numpy arrays
            if isinstance(embedding1, list):
                embedding1 = np.array(embedding1)
            if isinstance(embedding2, list):
                embedding2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            raise Exception(f"Error calculating similarity: {str(e)}")
    
    def find_similar_chunks(self, query_embedding: np.ndarray, chunks: List[Dict], 
                           top_k: int = 5) -> List[Dict]:
        """
        Find most similar chunks to a query embedding
        
        Args:
            query_embedding: Query embedding vector
            chunks: List of chunks with embeddings
            top_k: Number of top similar chunks to return
            
        Returns:
            List[Dict]: Top similar chunks with similarity scores
        """
        try:
            similarities = []
            
            for chunk in chunks:
                if 'embedding' in chunk:
                    similarity = self.calculate_similarity(query_embedding, chunk['embedding'])
                    chunk_with_score = chunk.copy()
                    chunk_with_score['similarity_score'] = similarity
                    similarities.append(chunk_with_score)
            
            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Return top k results
            return similarities[:top_k]
            
        except Exception as e:
            raise Exception(f"Error finding similar chunks: {str(e)}")


# Global embedding generator instance
@st.cache_resource
def get_embedding_generator():
    """Get cached embedding generator instance"""
    return EmbeddingGenerator()
