"""
Document Processing Module for Clinical Trial RAG Application
Handles PDF text extraction and paragraph-based chunking
"""

import PyPDF2
import re
from typing import List, Dict
import io


class DocumentProcessor:
    """Handles PDF document processing and text chunking"""
    
    def __init__(self):
        self.min_chunk_length = 100  # Minimum characters per chunk
        self.max_chunk_length = 1000  # Maximum characters per chunk
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract text from uploaded PDF file
        
        Args:
            pdf_file: Streamlit uploaded file object or file path
            
        Returns:
            str: Extracted text from PDF
        """
        try:
            if hasattr(pdf_file, 'read'):
                # Handle Streamlit uploaded file
                pdf_reader = PyPDF2.PdfReader(pdf_file)
            else:
                # Handle file path
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text with medical terminology preservation (BLEU optimization)

        Args:
            text: Raw extracted text

        Returns:
            str: Cleaned text with preserved medical terminology
        """
        # Medical terminology patterns to preserve
        medical_patterns = {
            # Preserve statistical notations
            r'(p\s*[<>=]\s*0\.\d+)': lambda m: m.group(1).replace(' ', ''),
            r'(95%\s*CI[:\s]*[\d\.\-,\s]+)': lambda m: m.group(1),
            r'(HbA1c)': 'HbA1c',  # Standardize HbA1c
            r'(\d+\.?\d*%\s*vs\.?\s*\d+\.?\d*%)': lambda m: m.group(1).replace('vs.', 'vs'),
        }

        # Apply medical terminology preservation
        for pattern, replacement in medical_patterns.items():
            if callable(replacement):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            else:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)

        # Fix common OCR issues while preserving medical terms
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')

        # Preserve important medical abbreviations
        medical_abbreviations = ['CI', 'AE', 'SAE', 'ITT', 'PP', 'FDA', 'EMA', 'ICH', 'GCP']
        for abbrev in medical_abbreviations:
            # Ensure consistent capitalization
            text = re.sub(f'\\b{abbrev.lower()}\\b', abbrev, text, flags=re.IGNORECASE)

        return text.strip()
    
    def chunk_by_paragraphs_with_overlap(self, text: str, overlap_sentences: int = 2) -> List[Dict[str, str]]:
        """
        Split text into paragraph-based chunks with overlap for better retrieval (BLEU optimization)

        Args:
            text: Cleaned text to chunk
            overlap_sentences: Number of sentences to overlap between chunks

        Returns:
            List[Dict]: List of chunks with metadata and overlap
        """
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = ""
        chunk_id = 0
        previous_sentences = []  # Store sentences for overlap

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph would exceed max length, save current chunk
            if len(current_chunk) + len(paragraph) > self.max_chunk_length and current_chunk:
                if len(current_chunk) >= self.min_chunk_length:
                    # Extract last few sentences for overlap
                    sentences = re.split(r'[.!?]+', current_chunk)
                    sentences = [s.strip() for s in sentences if s.strip()]

                    chunks.append({
                        'id': chunk_id,
                        'text': current_chunk.strip(),
                        'length': len(current_chunk),
                        'overlap_start': len(previous_sentences) > 0,
                        'overlap_sentences': previous_sentences.copy() if previous_sentences else []
                    })

                    # Store last sentences for next chunk overlap
                    previous_sentences = sentences[-overlap_sentences:] if len(sentences) >= overlap_sentences else sentences
                    chunk_id += 1

                # Start new chunk with overlap from previous chunk
                if previous_sentences:
                    overlap_text = '. '.join(previous_sentences) + '. '
                    current_chunk = overlap_text + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add the last chunk if it meets minimum length
        if current_chunk and len(current_chunk) >= self.min_chunk_length:
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'length': len(current_chunk),
                'overlap_start': len(previous_sentences) > 0,
                'overlap_sentences': previous_sentences.copy() if previous_sentences else []
            })

        return chunks

    def chunk_by_paragraphs(self, text: str) -> List[Dict[str, str]]:
        """
        Split text into paragraph-based chunks (enhanced with overlap support)

        Args:
            text: Cleaned text to chunk

        Returns:
            List[Dict]: List of chunks with metadata
        """
        # Use the enhanced chunking method with overlap
        return self.chunk_by_paragraphs_with_overlap(text, overlap_sentences=2)
    
    def process_document(self, pdf_file, filename: str = None) -> Dict:
        """
        Complete document processing pipeline
        
        Args:
            pdf_file: PDF file to process
            filename: Optional filename for metadata
            
        Returns:
            Dict: Processed document with chunks and metadata
        """
        try:
            # Extract text
            raw_text = self.extract_text_from_pdf(pdf_file)
            
            # Clean text
            cleaned_text = self.clean_text(raw_text)
            
            # Create chunks
            chunks = self.chunk_by_paragraphs(cleaned_text)
            
            # Prepare document metadata
            document = {
                'filename': filename or 'uploaded_document.pdf',
                'total_text_length': len(cleaned_text),
                'num_chunks': len(chunks),
                'chunks': chunks,
                'raw_text': raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text  # Preview
            }
            
            return document
            
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")


def validate_pdf_file(uploaded_file) -> bool:
    """
    Validate that uploaded file is a valid PDF
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        bool: True if valid PDF, False otherwise
    """
    if uploaded_file is None:
        return False
    
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False
    
    try:
        # Try to read the PDF
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        # Reset file pointer
        uploaded_file.seek(0)
        return len(pdf_reader.pages) > 0
    except:
        return False
