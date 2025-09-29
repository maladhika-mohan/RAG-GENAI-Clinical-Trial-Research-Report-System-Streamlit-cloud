"""
Simple RAG Evaluation with Custom Metrics
Clean interface focused on Gemini-based evaluation metrics only
"""

import streamlit as st
import os
import uuid
import time
from typing import List, Dict, Optional
import google.generativeai as genai
from document_processor import DocumentProcessor
from embedding_generator import get_embedding_generator
from vector_database import get_vector_database
# Import custom evaluator (our own implementation)
from custom_evaluator import get_custom_evaluator

# Load environment variables if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Detect if running in Streamlit Cloud
IS_STREAMLIT_CLOUD = os.getenv('STREAMLIT_SHARING_MODE') == 'true' or 'streamlit.app' in os.getenv('HOSTNAME', '')

# Configure page
st.set_page_config(
    page_title="Clinical Trial RAG Evaluation",
    page_icon="🏥",
    layout="wide"
)

def load_css():
    """Load custom CSS styling with 3D protruding effects matching the reference image"""
    st.markdown("""
    <style>
    /* Clean header layout */
    .main-header-container {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* Ensure proper content spacing */
    .main .block-container {
        padding-top: 1rem !important;
    }

    /* Dark text for main content */
    .stMarkdown, .stText, p, div, span {
        color: #1f2937 !important;
    }

    /* Sidebar text - dark except navigation */
    .css-1d391kg p, .css-1d391kg div {
        color: #1f2937 !important;
    }

    /* Navigation section styling - keep light */
    .css-1d391kg h2 {
        color: #9ca3af !important;
        font-weight: 500 !important;
    }

    /* Metric card text - dark */
    .metric-card h3 {
        color: #1f2937 !important;
        font-weight: 600 !important;
    }

    .metric-card p {
        color: #374151 !important;
    }

    /* Main content headers - dark */
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
    }

    /* Sidebar content except navigation buttons - dark */
    .css-1d391kg .stMarkdown:not(:first-child) {
        color: #1f2937 !important;
    }

    /* 3D Protruding Effects for Cards - Smaller Size */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 50%, #e2e8f0 100%);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.8rem 0;
        box-shadow: 
            0 4px 16px rgba(0, 0, 0, 0.08),
            0 2px 8px rgba(0, 0, 0, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.6),
            inset 0 -1px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.3);
        position: relative;
        transition: all 0.3s ease;
        transform: translateZ(0);
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(145deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        border-radius: 12px;
        pointer-events: none;
    }

    .metric-card:hover {
        transform: translateY(-2px) translateZ(0);
        box-shadow: 
            0 8px 24px rgba(0, 0, 0, 0.12),
            0 4px 12px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.7),
            inset 0 -1px 0 rgba(0, 0, 0, 0.08);
    }

    /* Enhanced Response Display with 3D Effect */
    .response-container {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border-left: 4px solid #6366f1;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 
            0 12px 40px rgba(99, 102, 241, 0.15),
            0 4px 16px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
        position: relative;
        transform: translateZ(0);
    }

    .response-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 4px;
        right: 0;
        bottom: 0;
        background: linear-gradient(145deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%);
        border-radius: 0 12px 12px 0;
        pointer-events: none;
    }

    /* Chunk Display with Raised Effect */
    .chunk-display {
        background: linear-gradient(145deg, #ffffff 0%, #fafbfc 100%);
        border-radius: 10px;
        padding: 1.4rem;
        margin: 1rem 0;
        box-shadow: 
            0 6px 20px rgba(0, 0, 0, 0.08),
            0 2px 8px rgba(0, 0, 0, 0.04),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
        border: 1px solid rgba(226, 232, 240, 0.6);
        position: relative;
        transition: all 0.3s ease;
    }

    .chunk-display:hover {
        transform: translateY(-1px);
        box-shadow: 
            0 8px 25px rgba(0, 0, 0, 0.12),
            0 3px 12px rgba(0, 0, 0, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
    }

    /* Evaluation Results Cards - Smaller Size */
    .evaluation-card {
        background: linear-gradient(145deg, #ffffff 0%, #f1f5f9 100%);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 
            0 6px 20px rgba(0, 0, 0, 0.08),
            0 3px 10px rgba(0, 0, 0, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.6),
            inset 0 -1px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.4);
        position: relative;
        transform: translateZ(0);
    }

    .evaluation-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(145deg, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0.04) 100%);
        border-radius: 16px;
        pointer-events: none;
    }

    /* Header Styling with Enhanced 3D Effect */
    .clinical-logo {
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        margin-right: 1.5rem;
        box-shadow: 
            0 8px 25px rgba(139, 92, 246, 0.3),
            0 4px 12px rgba(0, 0, 0, 0.1),
            inset 0 2px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        z-index: 1;
        border: 3px solid rgba(255, 255, 255, 0.3);
    }
    
    .clinical-logo::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.05) 100%);
        border-radius: 50%;
        pointer-events: none;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        text-shadow: 0 2px 4px rgba(139, 92, 246, 0.2);
        position: relative;
        z-index: 1;
    }

    .subtitle {
        font-size: 0.95rem;
        color: #6b7280;
        margin: 0.4rem 0;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }

    .tagline {
        font-size: 1.1rem;
        color: #9ca3af;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
        position: relative;
        z-index: 1;
        font-style: italic;
    }

    .header-content {
        display: flex;
        align-items: center;
        padding: 2rem 2.5rem;
        background: linear-gradient(135deg, #e8e5ff 0%, #f3f1ff 50%, #e8e5ff 100%);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 
            0 4px 20px rgba(139, 92, 246, 0.15),
            0 2px 10px rgba(0, 0, 0, 0.05);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(139, 92, 246, 0.1);
    }
    
    .header-content::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.4) 0%, rgba(255,255,255,0.1) 100%);
        pointer-events: none;
    }

    .logo-container {
        margin-right: 1.5rem;
        padding-left: 1rem;
    }

    .header-text {
        flex: 1;
        padding-right: 1rem;
    }

    /* Metric Score Cards - Smaller Size */
    .metric-score-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 50%, #e2e8f0 100%);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 
            0 4px 16px rgba(0, 0, 0, 0.08),
            0 2px 8px rgba(0, 0, 0, 0.06),
            inset 0 1px 0 rgba(255, 255, 255, 0.7),
            inset 0 -1px 0 rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.5);
        position: relative;
        transition: all 0.3s ease;
        transform: translateZ(0);
    }

    .metric-score-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(145deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%);
        border-radius: 14px;
        pointer-events: none;
    }

    .metric-score-card:hover {
        transform: translateY(-2px) translateZ(0);
        box-shadow: 
            0 6px 20px rgba(0, 0, 0, 0.12),
            0 3px 10px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.8),
            inset 0 -1px 0 rgba(0, 0, 0, 0.1);
    }

    /* Navigation buttons - keep light and subtle */
    .css-1d391kg .stButton > button {
        background: linear-gradient(145deg, #ffffff 0%, #f1f5f9 100%) !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        border-radius: 10px !important;
        color: #6b7280 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        box-shadow: 
            0 4px 12px rgba(0, 0, 0, 0.1),
            0 2px 6px rgba(0, 0, 0, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.5) !important;
        transition: all 0.3s ease !important;
    }

    .css-1d391kg .stButton > button:hover {
        transform: translateY(-2px) !important;
        color: #4b5563 !important;
        box-shadow: 
            0 8px 20px rgba(0, 0, 0, 0.15),
            0 4px 10px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.6) !important;
    }

    /* Other buttons in main content - dark text */
    .stButton > button {
        background: linear-gradient(145deg, #ffffff 0%, #f1f5f9 100%) !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        border-radius: 10px !important;
        color: #1f2937 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 
            0 4px 12px rgba(0, 0, 0, 0.1),
            0 2px 6px rgba(0, 0, 0, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.5) !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        color: #111827 !important;
        box-shadow: 
            0 8px 20px rgba(0, 0, 0, 0.15),
            0 4px 10px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.6) !important;
    }

    /* Primary button styling for active navigation */
    .stButton > button[kind="primary"] {
        background: linear-gradient(145deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: 1px solid rgba(99, 102, 241, 0.8) !important;
        box-shadow: 
            0 6px 16px rgba(99, 102, 241, 0.3),
            0 3px 8px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(145deg, #5b21b6 0%, #6d28d9 100%) !important;
        color: #ffffff !important;
        box-shadow: 
            0 8px 24px rgba(99, 102, 241, 0.4),
            0 4px 12px rgba(0, 0, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    }

    /* Secondary button styling */
    .stButton > button[kind="secondary"] {
        background: linear-gradient(145deg, #f8fafc 0%, #e2e8f0 100%) !important;
        color: #475569 !important;
        font-weight: 500 !important;
        border: 1px solid rgba(148, 163, 184, 0.6) !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: linear-gradient(145deg, #e2e8f0 0%, #cbd5e1 100%) !important;
        color: #334155 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def display_navigation():
    """Display navigation sidebar"""
    st.sidebar.markdown("## 🧭 Navigation")
    
    # Navigation options
    pages = {
        "💬 Chat & Evaluation": "Chat & Evaluation",
        "📊 Evaluation Logs": "Evaluation Logs", 
        "⚙️ Settings": "Settings"
    }
    
    # Create navigation buttons
    for page_icon, page_name in pages.items():
        if st.sidebar.button(
            page_icon,
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary" if st.session_state.current_page == page_name else "secondary"
        ):
            st.session_state.current_page = page_name
            st.rerun()
    
    st.sidebar.markdown("---")

def display_header():
    """Display the professional header with logo and branding"""
    # Theme toggle in header
    col1, col2 = st.columns([8, 1])

    with col1:
        st.markdown(f"""
        <div class="main-header-container">
            <div class="header-content">
                <div class="logo-container">
                    <div class="clinical-logo">🏥</div>
                </div>
                <div class="header-text">
                    <h1 class="main-title">Clinical Trial RAG Evaluation</h1>
                    <p class="subtitle">Custom Metrics • Gemini 2.0 Flash-Lite • Real-time Evaluation • RAG Assessment</p>
                    <p class="tagline">Advanced RAG Evaluation System with Comprehensive Metrics</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Placeholder for consistency
        st.empty()

def initialize_session_state():
    """Initialize session state variables"""
    if 'processed_documents' not in st.session_state:
        st.session_state.processed_documents = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'gemini_client' not in st.session_state:
        st.session_state.gemini_client = None
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = None
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 5
    if 'enable_evaluation' not in st.session_state:
        st.session_state.enable_evaluation = False
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = []
    if 'evaluation_logs' not in st.session_state:
        st.session_state.evaluation_logs = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Chat & Evaluation"
    if 'evaluator_type' not in st.session_state:
        st.session_state.evaluator_type = "custom"

def setup_gemini_api():
    """Setup Gemini API configuration"""
    st.sidebar.header("🔑 Gemini API Configuration")
    
    # Try to get API key from environment first
    env_api_key = os.getenv('GEMINI_API_KEY')
    
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="Enter your Google Gemini API key",
        placeholder="AIza...",
        value=env_api_key if env_api_key and env_api_key != 'your_gemini_api_key_here' else ""
    )
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            st.session_state.gemini_client = genai.GenerativeModel('gemini-2.0-flash-lite')
            
            # Setup evaluator if evaluation is enabled
            if st.session_state.enable_evaluation:
                if st.session_state.evaluator is None:
                    # Use our custom evaluator
                    st.session_state.evaluator = get_custom_evaluator(api_key)
                    st.session_state.evaluator_type = "custom"
            
            st.sidebar.success("✅ Gemini API configured")
            return True, api_key
        except Exception as e:
            st.sidebar.error(f"❌ Error configuring Gemini: {str(e)}")
            return False, None
    else:
        st.sidebar.info("Please enter your Gemini API key")
        st.sidebar.info("💡 Get your key from: [Google AI Studio](https://makersuite.google.com/app/apikey)")
        return False, None

def setup_evaluation_settings(api_key_available: bool):
    """Setup evaluation configuration"""
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Evaluation Settings")
    
    # Enable/disable evaluation
    enable_eval = st.sidebar.checkbox(
        "Enable Real-time Evaluation",
        value=st.session_state.enable_evaluation,
        help="Evaluate each response in real-time using custom RAG metrics",
        disabled=not api_key_available
    )
    
    st.session_state.enable_evaluation = enable_eval
    
    if not api_key_available and enable_eval:
        st.sidebar.warning("⚠️ API key required for evaluation")
    
    if enable_eval and api_key_available:
        # Show evaluator type
        evaluator_type = getattr(st.session_state, 'evaluator_type', 'custom')
        if evaluator_type == 'custom':
            st.sidebar.success("🚀 Using Custom RAG Evaluator")
            st.sidebar.caption("Full metrics: Answer Relevancy, Faithfulness, Contextual Relevancy & Recall")
        else:
            st.sidebar.info("🔧 Using basic evaluator")
        
        # Evaluation metrics selection
        st.sidebar.subheader("Metrics")
        
        available_metrics = {
            'answer_relevancy': 'Answer Relevancy',
            'faithfulness': 'Faithfulness', 
            'contextual_relevancy': 'Contextual Relevancy',
            'contextual_recall': 'Contextual Recall'
        }
        
        # Add metric descriptions and compatibility info
        metric_info = {
            'answer_relevancy': '✅ Most reliable - measures response relevance to question',
            'faithfulness': '✅ Reliable - checks if response aligns with context',
            'contextual_relevancy': '✅ Fixed - measures context relevance to question',
            'contextual_recall': '✅ Auto-generated expected output - measures context completeness'
        }
        
        selected_metrics = []
        for metric_key, metric_name in available_metrics.items():
            default_selected = metric_key in ['answer_relevancy', 'faithfulness']  # Changed default
            if st.sidebar.checkbox(
                metric_name, 
                value=default_selected,
                help=metric_info[metric_key]
            ):
                selected_metrics.append(metric_key)
        
        st.session_state.selected_metrics = selected_metrics
        
        # Note: Expected output for contextual_recall is now auto-generated
        
        # Evaluation threshold
        threshold = st.sidebar.slider(
            "Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Minimum score for passing evaluation"
        )
        
        st.session_state.eval_threshold = threshold
    
    return enable_eval

def upload_documents():
    """Handle document upload and processing"""
    st.sidebar.header("📄 Document Upload")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDF documents to chat about"
    )
    
    if uploaded_files:
        if st.sidebar.button("🔄 Process Documents"):
            process_documents(uploaded_files)

def process_documents(uploaded_files):
    """Process uploaded documents"""
    try:
        doc_processor = DocumentProcessor()
        embedding_gen = get_embedding_generator()
        vector_db = get_vector_database()
        
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Extract text
            document = doc_processor.process_document(uploaded_file, uploaded_file.name)
            
            # Generate embeddings
            chunks_with_embeddings = embedding_gen.generate_chunk_embeddings(document['chunks'])
            
            # Store in vector database
            document_id = str(uuid.uuid4())
            vector_db.store_document_chunks(
                chunks_with_embeddings,
                document_id,
                uploaded_file.name,
                {'processed_at': time.strftime("%Y-%m-%d %H:%M:%S")}
            )
            
            # Store in session state
            st.session_state.processed_documents[document_id] = {
                'filename': uploaded_file.name,
                'chunks': chunks_with_embeddings,
                'metadata': document
            }
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("✅ Processing complete!")
        st.sidebar.success(f"Processed {len(uploaded_files)} documents")
        
    except Exception as e:
        st.sidebar.error(f"Error processing documents: {str(e)}")

def generate_response(query: str, relevant_chunks: List[Dict]) -> str:
    """Generate response using Gemini API with relevant chunks"""
    if not st.session_state.gemini_client:
        return "Please configure Gemini API key first."
    
    # Prepare context from chunks
    context = "\n\n".join([
        f"Source: {chunk['filename']}\nContent: {chunk['text']}"
        for chunk in relevant_chunks
    ])
    
    # Create prompt
    prompt = f"""
Based on the following document context, answer the user's question.

Document Context:
{context}

User Question: {query}

Please provide a helpful answer based on the documents. If you reference specific information, mention which document it came from.
"""
    
    try:
        response = st.session_state.gemini_client.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def test_api_connection():
    """Test if the Gemini API is responding"""
    if not st.session_state.evaluator:
        st.error("No evaluator configured")
        return
    
    with st.spinner("Testing API connection..."):
        try:
            # Simple test evaluation
            test_results = st.session_state.evaluator.evaluate_response(
                query="test",
                response="test response",
                context=["test context"],
                metrics=['answer_relevancy']
            )
            
            if test_results and 'error' not in test_results:
                st.success("✅ API is responding normally")
            elif test_results and 'error' in test_results:
                error_msg = test_results['error']
                if '503' in error_msg:
                    st.warning("⚠️ API is overloaded - try again later")
                else:
                    st.error(f"API error: {error_msg}")
            else:
                st.error("❌ Unexpected response from API")
                
        except Exception as e:
            error_msg = str(e)
            if '503' in error_msg:
                st.warning("⚠️ API is currently overloaded")
            else:
                st.error(f"API test failed: {error_msg}")

def search_documents(query: str, top_k: int = None) -> List[Dict]:
    """Search for relevant document chunks"""
    if not st.session_state.processed_documents:
        return []
    
    if top_k is None:
        top_k = getattr(st.session_state, 'top_k', 5)
    
    try:
        embedding_gen = get_embedding_generator()
        vector_db = get_vector_database()
        
        # Generate query embedding
        query_embedding = embedding_gen.generate_query_embedding(query)
        
        # Search for similar chunks
        similar_chunks = vector_db.search_similar_chunks(
            query_embedding.tolist(),
            top_k=top_k,
            score_threshold=0.0
        )
        
        return similar_chunks
    except Exception as e:
        st.error(f"Error searching documents: {str(e)}")
        return []



def evaluate_response(query: str, response: str, relevant_chunks: List[Dict]) -> Optional[Dict]:
    """Evaluate a single response using available evaluator"""
    if not st.session_state.evaluator or not st.session_state.enable_evaluation:
        return None
    
    try:
        # Prepare retrieval context
        context_chunks = [chunk['text'] for chunk in relevant_chunks]
        
        # Get selected metrics
        selected_metrics = getattr(st.session_state, 'selected_metrics', ['answer_relevancy', 'faithfulness'])
        
        if not selected_metrics:
            return None
        
        # Use our custom evaluator
        results = st.session_state.evaluator.evaluate_response(
            query=query,
            response=response,
            context=context_chunks,
            metrics=selected_metrics
        )
        
        return results
        
    except Exception as e:
        error_msg = str(e)
        
        # Handle specific API errors
        if '503' in error_msg or 'overloaded' in error_msg.lower():
            st.warning("⚠️ **Gemini API Overloaded**")
            st.info("The Gemini API is currently experiencing high load. Please try again in a few minutes.")
            
            # Show retry suggestion
            with st.expander("💡 Suggestions"):
                st.write("- Wait 2-3 minutes and try again")
                st.write("- Try asking shorter questions")
                st.write("- Use fewer evaluation metrics")
                st.write("- Try during off-peak hours")
                
        elif '400' in error_msg and ('invalid' in error_msg.lower() or 'api_key' in error_msg.lower()):
            st.error("❌ **Invalid API Key**")
            st.info("Please check your Gemini API key in the sidebar.")
            
        elif '429' in error_msg or 'quota' in error_msg.lower():
            st.warning("⚠️ **Rate Limit Exceeded**")
            st.info("You've exceeded the API rate limits. Please wait before trying again.")
            
        else:
            st.error(f"Evaluation error: {error_msg}")
            
        return None

def display_evaluation_results(eval_results: Dict, threshold: float = 0.7):
    """Display evaluation results in the chat interface"""
    if not eval_results:
        return
    
    # Store evaluation in logs for the logs page
    if eval_results and 'error' not in eval_results:
        log_entry = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'results': eval_results,
            'threshold': threshold,
            'top_k': st.session_state.get('top_k', 5),
            'status': eval_results.get('status', 'success')
        }
        st.session_state.evaluation_logs.append(log_entry)
    
    # Check if results contain an error
    if 'error' in eval_results:
        with st.expander("🔍 Response Evaluation", expanded=False):
            error_msg = eval_results['error']
            error_type = eval_results.get('error_type', 'general_error')
            
            if error_type == 'quota_exceeded':
                st.error("❌ **API Quota Exceeded**")
                st.warning("You've reached the free tier limit of 200 requests per day.")
                st.info("💡 **Solutions:**")
                st.write("- Wait until tomorrow for quota reset")
                st.write("- Upgrade to a paid Gemini API plan")
                st.write("- Disable evaluation temporarily")
                
            elif error_type == 'api_overloaded':
                st.warning("⚠️ **API Temporarily Overloaded**")
                st.info("The Gemini API is experiencing high traffic. Try again in a few minutes.")
                
            elif '400' in error_msg and 'invalid' in error_msg.lower():
                st.error("❌ Invalid API key. Please check your Gemini API key.")
                
            else:
                st.error(f"Evaluation failed: {error_msg}")
        return
    
    # Filter out non-metric keys
    metric_results = {k: v for k, v in eval_results.items() 
                     if isinstance(v, dict) and 'score' in v}
    
    if not metric_results:
        return
    
    # Enhanced evaluation results with 3D cards
    st.markdown("""
    <div class="evaluation-card">
        <h2 style="margin-top: 0; color: #1e293b; font-weight: 600;">🔍 RAG Evaluation Results</h2>
    """, unsafe_allow_html=True)
    
    # Show evaluation summary
    total_metrics = len(metric_results)
    successful_count = sum(1 for result in metric_results.values() 
                         if result.get('score', 0) > 0 or 'Error' not in result.get('reason', ''))
    failed_count = total_metrics - successful_count
    
    if failed_count > 0:
        st.warning(f"⚠️ {successful_count}/{total_metrics} metrics completed successfully. {failed_count} failed.")
    else:
        st.success(f"✅ All {total_metrics} metrics completed successfully!")
    
    # Create metrics display with enhanced 3D cards
    cols = st.columns(len(metric_results))
    
    for i, (metric_name, result) in enumerate(metric_results.items()):
            with cols[i]:
                # Enhanced metric card with 3D effect
                score = result.get('score', 0)
                status_color = "#10b981" if score >= threshold else "#f59e0b" if score >= threshold - 0.2 else "#ef4444"
                status_text = "✅ Pass" if score >= threshold else "⚠️ Fair" if score >= threshold - 0.2 else "❌ Fail"
                
                st.markdown(f"""
                <div class="metric-score-card">
                    <h3 style="margin: 0 0 0.5rem 0; color: #1e293b; font-size: 1.1rem; font-weight: 600;">
                        {metric_name.replace('_', ' ').title()}
                    </h3>
                    <div style="font-size: 2rem; font-weight: 700; color: {status_color}; margin: 0.5rem 0;">
                        {score:.3f}
                    </div>
                    <div style="color: {status_color}; font-weight: 500; font-size: 0.9rem;">
                        {status_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Show detailed analysis
    st.subheader(" Detailed Analysis")
        
    # Separate successful and failed metrics
    successful_metrics = []
    failed_metrics = []
        
    for metric_name, result in metric_results.items():
        if result.get('reason'):
            if result.get('score', 0) > 0 or 'Error' not in result.get('reason', ''):
                successful_metrics.append((metric_name, result))
            else:
                failed_metrics.append((metric_name, result))
        
    # Show successful metrics first
    for metric_name, result in successful_metrics:
        st.write(f"**{metric_name.replace('_', ' ').title()}:**")
        st.write(result['reason'])
        
        # Add note for auto-generated expected output
        if metric_name == 'contextual_recall':
            st.caption("💡 Expected output was automatically generated from retrieval context")
        
        st.divider()
    
    # Close the evaluation card
    st.markdown("</div>", unsafe_allow_html=True)

def display_evaluation_logs_page():
    """Display the evaluation logs page"""
    st.title("📊 Evaluation Logs & Performance Analysis")
    
    # Clean up any inconsistent log entries
    if hasattr(st.session_state, 'evaluation_logs'):
        valid_logs = []
        for log in st.session_state.evaluation_logs:
            if 'results' in log or 'metrics' in log:
                valid_logs.append(log)
        st.session_state.evaluation_logs = valid_logs
    
    if not st.session_state.evaluation_logs:
        st.info("🔍 No evaluation logs yet. Run some evaluations in the Chat & Evaluation page to see performance analysis here.")
        
        # Show instructions
        st.markdown("""
        ### How to Generate Evaluation Logs:
        1. **Configure API**: Set up your Gemini API key in the sidebar
        2. **Upload Documents**: Add PDF documents to chat about
        3. **Enable Evaluation**: Turn on real-time evaluation in settings
        4. **Ask Questions**: Chat with your documents to generate evaluations
        5. **View Analysis**: Return here to see comprehensive performance metrics
        """)
        return
    
    # Summary Statistics
    st.markdown("### 📈 Summary Statistics")
    
    total_evaluations = len(st.session_state.evaluation_logs)
    
    # Calculate average scores across all evaluations
    all_scores = []
    metric_averages = {}
    top_k_performance = {}
    
    for log in st.session_state.evaluation_logs:
        # Handle both log formats
        if 'results' in log:
            # Format 1: from display_evaluation_results
            results = log['results']
        elif 'metrics' in log:
            # Format 2: from chat interface
            results = log['metrics']
        else:
            continue  # Skip invalid log entries
        
        top_k = log.get('top_k', 5)
        
        if top_k not in top_k_performance:
            top_k_performance[top_k] = []
        
        for metric_name, result in results.items():
            if isinstance(result, dict) and 'score' in result:
                score = result['score']
                all_scores.append(score)
                top_k_performance[top_k].append(score)
                
                if metric_name not in metric_averages:
                    metric_averages[metric_name] = []
                metric_averages[metric_name].append(score)
    
    # Display summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Evaluations</h3>
            <div style="font-size: 2rem; font-weight: 700; color: #6366f1;">
                {total_evaluations}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>Average Score</h3>
            <div style="font-size: 2rem; font-weight: 700; color: #10b981;">
                {avg_score:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        best_top_k = max(top_k_performance.keys(), 
                        key=lambda k: sum(top_k_performance[k])/len(top_k_performance[k])) if top_k_performance else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Best Top-K</h3>
            <div style="font-size: 2rem; font-weight: 700; color: #f59e0b;">
                {best_top_k}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        latest_score = all_scores[-1] if all_scores else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>Latest Score</h3>
            <div style="font-size: 2rem; font-weight: 700; color: #8b5cf6;">
                {latest_score:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Logs Table
    st.markdown("### 📋 Detailed Evaluation Logs")
    
    # Show last 10 evaluations
    recent_logs = st.session_state.evaluation_logs[-10:]
    
    # Create table data
    table_data = []
    for i, log in enumerate(reversed(recent_logs)):
        results = log['results']
        timestamp = log['timestamp']
        top_k = log.get('top_k', 5)
        threshold = log.get('threshold', 0.7)
        
        # Calculate average score for this evaluation
        scores = [result['score'] for result in results.values() 
                 if isinstance(result, dict) and 'score' in result]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Count passing metrics
        passing = sum(1 for result in results.values() 
                     if isinstance(result, dict) and result.get('score', 0) >= threshold)
        total_metrics = len([result for result in results.values() 
                           if isinstance(result, dict) and 'score' in result])
        
        table_data.append({
            'Evaluation': f"#{len(st.session_state.evaluation_logs) - i}",
            'Timestamp': timestamp,
            'Top-K': top_k,
            'Avg Score': f"{avg_score:.3f}",
            'Pass Rate': f"{passing}/{total_metrics}",
            'Metrics': ', '.join([name.replace('_', ' ').title() 
                                for name in results.keys() 
                                if isinstance(results[name], dict)])
        })
    
    # Display table
    if table_data:
        import pandas as pd
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    
    # Top-K Performance Analysis
    if len(top_k_performance) > 1:
        st.markdown("### 🎯 Top-K Performance Comparison")
        
        # Create comparison chart data
        comparison_data = []
        for top_k, scores in top_k_performance.items():
            avg_score = sum(scores) / len(scores)
            comparison_data.append({
                'Top-K': top_k,
                'Average Score': avg_score,
                'Evaluations': len(scores)
            })
        
        # Sort by average score
        comparison_data.sort(key=lambda x: x['Average Score'], reverse=True)
        
        # Display comparison
        for data in comparison_data:
            col1, col2, col3 = st.columns([2, 3, 2])
            with col1:
                st.metric("Top-K", data['Top-K'])
            with col2:
                st.metric("Average Score", f"{data['Average Score']:.3f}")
            with col3:
                st.metric("Evaluations", data['Evaluations'])
        
        # Recommendation
        best_k = comparison_data[0]['Top-K']
        best_score = comparison_data[0]['Average Score']
        st.success(f"🎯 **Recommendation**: Use Top-K = {best_k} for best performance (avg score: {best_score:.3f})")
    
    # Export and Management
    st.markdown("### 🔧 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Logs (CSV)", use_container_width=True):
            # Create CSV data
            csv_data = []
            for i, log in enumerate(st.session_state.evaluation_logs):
                results = log['results']
                base_row = {
                    'evaluation_id': i + 1,
                    'timestamp': log['timestamp'],
                    'top_k': log.get('top_k', 5),
                    'threshold': log.get('threshold', 0.7)
                }
                
                for metric_name, result in results.items():
                    if isinstance(result, dict) and 'score' in result:
                        row = base_row.copy()
                        row['metric'] = metric_name
                        row['score'] = result['score']
                        row['reason'] = result.get('reason', '')
                        csv_data.append(row)
            
            if csv_data:
                import pandas as pd
                df = pd.DataFrame(csv_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"evaluation_logs_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        if st.button("🗑️ Clear All Logs", use_container_width=True, type="secondary"):
            if st.session_state.get('confirm_clear_logs', False):
                st.session_state.evaluation_logs = []
                st.session_state.confirm_clear_logs = False
                st.success("✅ Evaluation logs cleared!")
                st.rerun()
            else:
                st.session_state.confirm_clear_logs = True
                st.warning("⚠️ Click again to confirm clearing all logs")

def display_settings_page():
    """Display the settings page"""
    st.title("⚙️ Settings & Configuration")
    
    # System Configuration
    st.markdown("### 🔧 System Configuration")
    
    # Current settings display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 RAG Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        current_top_k = st.session_state.get('top_k', 5)
        st.write(f"**Top-K Retrieval**: {current_top_k}")
        
        evaluation_enabled = st.session_state.get('enable_evaluation', False)
        st.write(f"**Evaluation Enabled**: {'✅ Yes' if evaluation_enabled else '❌ No'}")
        
        if evaluation_enabled:
            metrics = st.session_state.get('selected_metrics', [])
            st.write(f"**Active Metrics**: {', '.join(metrics) if metrics else 'None'}")
            
            threshold = st.session_state.get('eval_threshold', 0.7)
            st.write(f"**Score Threshold**: {threshold}")
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Evaluation Status</h3>
        </div>
        """, unsafe_allow_html=True)
        
        total_logs = len(st.session_state.get('evaluation_logs', []))
        st.write(f"**Total Evaluations**: {total_logs}")
        
        api_configured = st.session_state.gemini_client is not None
        st.write(f"**API Status**: {'✅ Configured' if api_configured else '❌ Not Configured'}")
        
        evaluator_ready = st.session_state.evaluator is not None
        st.write(f"**Evaluator Status**: {'✅ Ready' if evaluator_ready else '❌ Not Ready'}")
    
    # Data Management
    st.markdown("### 🗂️ Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>💬 Chat History</h3>
        </div>
        """, unsafe_allow_html=True)
        
        chat_count = len(st.session_state.get('chat_history', []))
        st.write(f"**Messages**: {chat_count}")
        
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.success("✅ Chat history cleared!")
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Evaluation Logs</h3>
        </div>
        """, unsafe_allow_html=True)
        
        log_count = len(st.session_state.get('evaluation_logs', []))
        st.write(f"**Log Entries**: {log_count}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Evaluation Logs", use_container_width=True):
                st.session_state.evaluation_logs = []
                st.success("✅ Evaluation logs cleared!")
        
        with col2:
            if st.button("Fix Corrupted Logs", use_container_width=True):
                if hasattr(st.session_state, 'evaluation_logs'):
                    original_count = len(st.session_state.evaluation_logs)
                    valid_logs = []
                    for log in st.session_state.evaluation_logs:
                        if 'results' in log or 'metrics' in log:
                            valid_logs.append(log)
                    st.session_state.evaluation_logs = valid_logs
                    removed_count = original_count - len(valid_logs)
                    if removed_count > 0:
                        st.success(f"✅ Removed {removed_count} corrupted log entries!")
                    else:
                        st.info("ℹ️ No corrupted logs found!")
                else:
                    st.info("ℹ️ No logs to fix!")
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📚 Documents</h3>
        </div>
        """, unsafe_allow_html=True)
        
        doc_count = len(st.session_state.get('processed_documents', {}))
        st.write(f"**Documents**: {doc_count}")
        
        if st.button("Clear Documents", use_container_width=True):
            st.session_state.processed_documents = {}
            st.success("✅ Documents cleared!")
    
    # System Information
    st.markdown("### ℹ️ System Information")
    
    # Calculate document statistics
    total_chunks = 0
    total_documents = len(st.session_state.get('processed_documents', {}))
    
    for doc_data in st.session_state.get('processed_documents', {}).values():
        if 'chunks' in doc_data:
            total_chunks += len(doc_data['chunks'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", total_documents)
    
    with col2:
        st.metric("Total Chunks", total_chunks)
    
    with col3:
        avg_chunks = total_chunks / total_documents if total_documents > 0 else 0
        st.metric("Avg Chunks/Doc", f"{avg_chunks:.1f}")
    
    # Reset All Data
    st.markdown("### 🔄 Reset System")
    
    st.warning("⚠️ **Danger Zone**: This will clear all data and reset the system to initial state.")
    
    if st.button("🔄 Reset All Data", type="secondary"):
        if st.session_state.get('confirm_reset', False):
            # Reset all session state
            keys_to_keep = ['current_page']  # Keep navigation state
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
            
            # Reinitialize
            initialize_session_state()
            st.success("✅ System reset complete!")
            st.rerun()
        else:
            st.session_state.confirm_reset = True
            st.error("⚠️ Click again to confirm complete system reset")

def display_chat_interface():
    """Display the main chat interface"""
    
    # Check if API is configured
    if not st.session_state.gemini_client:
        st.warning("⚠️ Please configure your Gemini API key in the sidebar to start chatting.")
        return
    
    # Display document status with enhanced styling
    if st.session_state.processed_documents:
        doc_count = len(st.session_state.processed_documents)
        st.markdown(f"""
        <div class="metric-card">
            <h3>📚 Knowledge Base Status</h3>
            <p>✅ <strong>{doc_count} document(s)</strong> loaded and ready for intelligent evaluation</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <h3>📚 Knowledge Base</h3>
            <p>💡 Upload documents in the sidebar to chat about them, or ask general questions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Evaluation status with enhanced styling
    if st.session_state.enable_evaluation:
        if st.session_state.evaluator:
            metrics = getattr(st.session_state, 'selected_metrics', [])
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔍 Evaluation Status</h3>
                <p>✅ <strong>Enabled</strong> with {len(metrics)} metrics: {', '.join(metrics)}</p>
                <p style="font-size: 0.9rem; color: #64748b;">💡 If evaluation fails due to API overload, responses will still be generated without evaluation.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Test API button
            if st.button("🔄 Test API", help="Test if Gemini API is responding"):
                test_api_connection()
        else:
            st.markdown("""
            <div class="metric-card">
                <h3>⚠️ Evaluation Status</h3>
                <p>Evaluation enabled but evaluator not configured</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat history display
    for i, chat in enumerate(st.session_state.chat_history):
        # User message
        with st.chat_message("user"):
            st.write(chat['query'])
        
        # Assistant message
        with st.chat_message("assistant"):
            st.write(chat['response'])
            
            # Show evaluation results if available
            if 'evaluation' in chat and chat['evaluation']:
                threshold = getattr(st.session_state, 'eval_threshold', 0.7)
                display_evaluation_results(chat['evaluation'], threshold)
            
            # Show sources if available
            if 'sources' in chat and chat['sources']:
                with st.expander(f"📚 Sources ({len(chat['sources'])} chunks)"):
                    for j, source in enumerate(chat['sources']):
                        st.write(f"**{j+1}. {source['filename']}** (Score: {source['score']:.3f})")
                        st.write(source['text'][:200] + "..." if len(source['text']) > 200 else source['text'])
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                # Search for relevant chunks if documents are available
                relevant_chunks = []
                if st.session_state.processed_documents:
                    relevant_chunks = search_documents(prompt)
                
                # Generate response with enhanced styling
                response = generate_response(prompt, relevant_chunks)
                st.markdown(f"""
                <div class="response-container">
                    {response}
                </div>
                """, unsafe_allow_html=True)
                
                # Evaluate response if enabled
                evaluation_results = None
                if st.session_state.enable_evaluation:
                    with st.spinner("Evaluating response..."):
                        evaluation_results = evaluate_response(prompt, response, relevant_chunks)
                        if evaluation_results:
                            threshold = getattr(st.session_state, 'eval_threshold', 0.7)
                            display_evaluation_results(evaluation_results, threshold)
                
                # Show sources with enhanced styling
                if relevant_chunks:
                    with st.expander(f"📚 Sources ({len(relevant_chunks)} chunks)"):
                        for i, chunk in enumerate(relevant_chunks):
                            st.markdown(f"""
                            <div class="chunk-display">
                                <h4>📄 {chunk['filename']} (Score: {chunk['score']:.3f})</h4>
                                <p>{chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']}</p>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Add to chat history
        chat_entry = {
            'query': prompt,
            'response': response,
            'sources': relevant_chunks,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if evaluation_results:
            chat_entry['evaluation'] = evaluation_results
        
        st.session_state.chat_history.append(chat_entry)
        
        # Store evaluation results
        if evaluation_results:
            st.session_state.evaluation_results.append({
                'query': prompt,
                'response': response,
                'evaluation': evaluation_results,
                'timestamp': chat_entry['timestamp']
            })
            
            # Note: Evaluation logging is handled in display_evaluation_results function
        
        # Rerun to update the display
        st.rerun()



def display_sidebar_info():
    """Display additional information in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Settings")
    
    # Top-K setting
    top_k = st.sidebar.slider(
        "Number of relevant chunks",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of document chunks to retrieve"
    )
    
    st.session_state.top_k = top_k
    

    
    # Evaluation stats
    if st.session_state.evaluation_results:
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Evaluation Stats")
        
        total_evals = len(st.session_state.evaluation_results)
        st.sidebar.metric("Total Evaluations", total_evals)
        
        # Calculate average scores
        avg_scores = {}
        for result in st.session_state.evaluation_results:
            if 'evaluation' in result:
                for metric, data in result['evaluation'].items():
                    if isinstance(data, dict) and 'score' in data:
                        if metric not in avg_scores:
                            avg_scores[metric] = []
                        avg_scores[metric].append(data['score'])
        
        for metric, scores in avg_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                st.sidebar.metric(
                    f"Avg {metric.replace('_', ' ').title()}",
                    f"{avg_score:.3f}"
                )
    
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.info("""
    Simple RAG evaluation with:
    - Gemini 2.0 Flash-Lite
    - Custom RAG metrics
    - Document Q&A
    - Real-time evaluation
    """)
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.evaluation_results = []
        st.rerun()

def main():
    """Main application function"""
    initialize_session_state()
    
    # Load custom CSS
    load_css()
    
    # Display professional header
    display_header()
    
    # Display navigation in sidebar
    display_navigation()
    
    # Setup API and evaluation in sidebar (only for Chat & Evaluation page)
    if st.session_state.current_page == "Chat & Evaluation":
        api_configured, api_key = setup_gemini_api()
        setup_evaluation_settings(api_configured)
        upload_documents()
        display_sidebar_info()
    
    # Display selected page
    if st.session_state.current_page == "Chat & Evaluation":
        display_chat_interface()
    elif st.session_state.current_page == "Evaluation Logs":
        display_evaluation_logs_page()
    elif st.session_state.current_page == "Settings":
        display_settings_page()

if __name__ == "__main__":
    main()
