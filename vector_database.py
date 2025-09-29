"""
Vector Database Module for Clinical Trial RAG Application
Uses local Qdrant library with in-memory storage
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, HnswConfigDiff
from qdrant_client.models import Filter, FieldCondition, MatchValue
import uuid
from typing import List, Dict, Optional
import streamlit as st
import os


class VectorDatabase:
    """Handles vector storage and retrieval using Qdrant"""
    
    def __init__(self, collection_name: str = "clinical_trials"):
        """
        Initialize the vector database
        
        Args:
            collection_name: Name of the Qdrant collection
        """
        self.collection_name = collection_name
        self.client = None
        self.embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2
    
    def get_client(self):
        """Get or create Qdrant client using local in-memory storage"""
        if self.client is None:
            # Use local in-memory Qdrant client (no server required)
            self.client = QdrantClient(":memory:")
            st.info("🗄️ Using local Qdrant in-memory storage")
        return self.client
    
    def create_collection(self):
        """Create collection if it doesn't exist"""
        try:
            client = self.get_client()
            
            # Check if collection exists
            collections = client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                # Create collection with explicit HNSW indexing configuration
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE,
                        hnsw_config=HnswConfigDiff(
                            m=16,  # Number of bi-directional links for each node
                            ef_construct=200,  # Size of the dynamic candidate list
                            full_scan_threshold=10000,  # Threshold for full scan vs HNSW
                            max_indexing_threads=0,  # Use all available threads
                            on_disk=False  # Keep index in memory for faster access
                        )
                    )
                )
                st.success(f"Created collection: {self.collection_name}")
            
        except Exception as e:
            raise Exception(f"Error creating collection: {str(e)}")
    
    def store_document_chunks(self, chunks: List[Dict], document_id: str,
                            filename: str, additional_metadata: Dict = None) -> List[str]:
        """
        Store document chunks in the unified vector database

        Args:
            chunks: List of chunks with embeddings
            document_id: Unique document identifier
            filename: Original filename
            additional_metadata: Additional metadata to store with chunks

        Returns:
            List[str]: List of stored point IDs
        """
        try:
            client = self.get_client()
            self.create_collection()

            points = []
            point_ids = []

            for chunk in chunks:
                if 'embedding' not in chunk:
                    continue

                point_id = str(uuid.uuid4())
                point_ids.append(point_id)

                # Enhanced metadata for cross-document search
                payload = {
                    'document_id': document_id,
                    'filename': filename,
                    'chunk_id': chunk['id'],
                    'text': chunk['text'],
                    'length': chunk['length'],
                    'document_type': 'clinical_trial',  # Can be extended for different document types
                    'upload_timestamp': additional_metadata.get('upload_timestamp') if additional_metadata else None,
                    'file_size': additional_metadata.get('file_size') if additional_metadata else None
                }

                # Add any additional metadata
                if additional_metadata:
                    for key, value in additional_metadata.items():
                        if key not in payload:  # Don't override existing keys
                            payload[key] = value

                # Create point with enhanced metadata
                point = PointStruct(
                    id=point_id,
                    vector=chunk['embedding'],
                    payload=payload
                )
                points.append(point)

            # Store points in batch
            if points:
                client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )

            return point_ids

        except Exception as e:
            raise Exception(f"Error storing document chunks: {str(e)}")
    
    def search_similar_chunks(self, query_embedding: List[float],
                            top_k: int = 5, score_threshold: float = 0.0,
                            document_filter: List[str] = None) -> List[Dict]:
        """
        Search for similar chunks across all documents using vector similarity

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            document_filter: Optional list of document IDs to filter results

        Returns:
            List[Dict]: Similar chunks with scores and enhanced metadata
        """
        try:
            client = self.get_client()

            # Prepare search filter if document filtering is requested
            search_filter = None
            if document_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=doc_id)
                        ) for doc_id in document_filter
                    ]
                )

            # Perform cross-document vector search
            search_results = client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=search_filter
            )

            # Format results with enhanced metadata
            results = []
            for result in search_results:
                chunk_data = {
                    'id': result.id,
                    'score': result.score,
                    'text': result.payload['text'],
                    'document_id': result.payload['document_id'],
                    'filename': result.payload['filename'],
                    'chunk_id': result.payload['chunk_id'],
                    'length': result.payload['length'],
                    'document_type': result.payload.get('document_type', 'unknown'),
                    'upload_timestamp': result.payload.get('upload_timestamp'),
                    'file_size': result.payload.get('file_size')
                }
                results.append(chunk_data)

            return results

        except Exception as e:
            raise Exception(f"Error searching similar chunks: {str(e)}")

    def get_all_documents(self) -> List[Dict]:
        """
        Get list of all documents in the collection

        Returns:
            List[Dict]: List of unique documents with metadata
        """
        try:
            client = self.get_client()

            # Get all points to extract unique documents
            scroll_result = client.scroll(
                collection_name=self.collection_name,
                limit=10000  # Adjust based on expected collection size
            )

            documents = {}
            for point in scroll_result[0]:
                doc_id = point.payload['document_id']
                if doc_id not in documents:
                    documents[doc_id] = {
                        'document_id': doc_id,
                        'filename': point.payload['filename'],
                        'document_type': point.payload.get('document_type', 'unknown'),
                        'upload_timestamp': point.payload.get('upload_timestamp'),
                        'file_size': point.payload.get('file_size'),
                        'chunk_count': 0
                    }
                documents[doc_id]['chunk_count'] += 1

            return list(documents.values())

        except Exception as e:
            return []
    
    def get_document_chunks(self, document_id: str) -> List[Dict]:
        """
        Get all chunks for a specific document
        
        Args:
            document_id: Document identifier
            
        Returns:
            List[Dict]: All chunks for the document
        """
        try:
            client = self.get_client()
            
            # Search with filter for specific document
            search_results = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=1000  # Adjust based on expected document size
            )
            
            # Format results
            chunks = []
            for point in search_results[0]:  # scroll returns (points, next_page_offset)
                chunk_data = {
                    'id': point.id,
                    'text': point.payload['text'],
                    'chunk_id': point.payload['chunk_id'],
                    'length': point.payload['length']
                }
                chunks.append(chunk_data)
            
            return chunks
            
        except Exception as e:
            raise Exception(f"Error getting document chunks: {str(e)}")
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a specific document
        
        Args:
            document_id: Document identifier
            
        Returns:
            bool: True if successful
        """
        try:
            client = self.get_client()
            
            # Delete points with matching document_id
            client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
            
            return True
            
        except Exception as e:
            raise Exception(f"Error deleting document: {str(e)}")
    
    def get_collection_info(self) -> Dict:
        """
        Get information about the collection
        
        Returns:
            Dict: Collection information
        """
        try:
            client = self.get_client()
            
            # Get collection info
            info = client.get_collection(self.collection_name)
            
            return {
                'name': info.config.params.vectors.size,
                'vectors_count': info.vectors_count,
                'points_count': info.points_count,
                'status': info.status
            }
            
        except Exception as e:
            return {'error': str(e)}


# Global vector database instance
@st.cache_resource
def get_vector_database():
    """Get cached vector database instance"""
    return VectorDatabase()
