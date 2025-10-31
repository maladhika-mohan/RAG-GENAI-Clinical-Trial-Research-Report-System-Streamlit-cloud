"""
Simple RAG Evaluation with Custom Metrics
Clean interface focused on Gemini-based evaluation metrics only
"""

import streamlit as st
import os
import uuid
import time
from typing import List, Dict, Optional, Tuple
import google.generativeai as genai
from document_processor import DocumentProcessor
from embedding_generator import get_embedding_generator
from vector_database import get_vector_database
# Import custom evaluator (our own implementation)
from custom_evaluator import get_custom_evaluator

# Import new features
from prompt_optimizer import get_prompt_optimizer, PromptOptimizer
from memory_integration import get_memory_manager, RAGMemoryManager
from token_counter import get_token_counter, format_token_count, estimate_cost
from mcp_client import get_mcp_client, process_query_with_agentic_rag, run_async_in_streamlit, check_mcp_server_health
from feedback_system import get_feedback_system

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

    # New feature states
    if 'enable_agentic_rag' not in st.session_state:
        st.session_state.enable_agentic_rag = False
    if 'enable_prompt_optimization' not in st.session_state:
        st.session_state.enable_prompt_optimization = False
    if 'enable_memory' not in st.session_state:
        st.session_state.enable_memory = False
    if 'mcp_server_running' not in st.session_state:
        st.session_state.mcp_server_running = False
    if 'mcp_client' not in st.session_state:
        st.session_state.mcp_client = None
    if 'prompt_optimizer' not in st.session_state:
        st.session_state.prompt_optimizer = None
    if 'memory_manager' not in st.session_state:
        st.session_state.memory_manager = None
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = str(uuid.uuid4())
    if 'current_context_id' not in st.session_state:
        st.session_state.current_context_id = None
    if 'token_usage_stats' not in st.session_state:
        st.session_state.token_usage_stats = []
    if 'feedback_system' not in st.session_state:
        st.session_state.feedback_system = None
    if 'feedback_enabled' not in st.session_state:
        st.session_state.feedback_enabled = False
    if 'pending_feedback' not in st.session_state:
        st.session_state.pending_feedback = {}

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
            st.session_state.api_key = api_key  # Store for feedback system

            # Setup evaluator if evaluation is enabled
            if st.session_state.enable_evaluation:
                if st.session_state.evaluator is None:
                    # Use our custom evaluator
                    st.session_state.evaluator = get_custom_evaluator(api_key)
                    st.session_state.evaluator_type = "custom"

            # Setup new features if enabled
            setup_new_features(api_key)

            st.sidebar.success("✅ Gemini API configured")
            return True, api_key
        except Exception as e:
            st.sidebar.error(f"❌ Error configuring Gemini: {str(e)}")
            return False, None
    else:
        st.sidebar.info("Please enter your Gemini API key")
        st.sidebar.info("💡 Get your key from: [Google AI Studio](https://makersuite.google.com/app/apikey)")
        return False, None


def setup_new_features(api_key: str):
    """Setup new features (Agentic RAG, Prompt Optimization, Memory)"""
    try:
        # Setup Agentic RAG with MCP Server
        if st.session_state.enable_agentic_rag and st.session_state.mcp_client is None:
            st.session_state.mcp_client = get_mcp_client()
            # Check server health
            health_status = check_mcp_server_health()
            st.session_state.mcp_server_running = health_status['status'] == 'running'

            # Configure Gemini API if server is running and API key is available
            if st.session_state.mcp_server_running and api_key:
                try:
                    # Connect to server and configure API
                    with st.spinner("Connecting to MCP server..."):
                        connection_success = run_async_in_streamlit(st.session_state.mcp_client.connect())

                    if connection_success:
                        with st.spinner("Configuring Gemini API..."):
                            config_success = run_async_in_streamlit(
                                st.session_state.mcp_client.configure_gemini_api(api_key)
                            )
                        if config_success:
                            st.success("✅ Gemini API configured on MCP server")
                        else:
                            st.warning("⚠️ Failed to configure Gemini API on MCP server")
                    else:
                        st.warning("⚠️ Could not connect to MCP server. Make sure it's running on port 8050.")
                except Exception as e:
                    st.warning(f"MCP server configuration error: {str(e)}")

        # Setup Prompt Optimizer
        if st.session_state.enable_prompt_optimization and st.session_state.prompt_optimizer is None:
            st.session_state.prompt_optimizer = get_prompt_optimizer("gemini-2.0-flash-lite", 4000)

        # Setup Memory Manager
        if st.session_state.enable_memory and st.session_state.memory_manager is None:
            try:
                st.session_state.memory_manager = get_memory_manager(api_key, "gemini-2.0-flash-lite")
                st.session_state.memory_manager.set_current_session(st.session_state.current_session_id)
            except Exception as e:
                st.sidebar.warning(f"⚠️ Memory integration partially available: {str(e)}")
                # Still create the manager for basic functionality
                try:
                    st.session_state.memory_manager = get_memory_manager(api_key, "gemini-2.0-flash-lite")
                    st.session_state.memory_manager.set_current_session(st.session_state.current_session_id)
                except Exception as e2:
                    st.sidebar.error(f"❌ Could not initialize memory: {str(e2)}")
                    st.session_state.memory_manager = None

    except Exception as e:
        st.sidebar.warning(f"⚠️ Error setting up new features: {str(e)}")


def setup_advanced_features():
    """Setup advanced features configuration in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.header("🚀 Advanced Features")

    # Agentic RAG with MCP Server
    enable_agentic_rag = st.sidebar.checkbox(
        "Enable Agentic RAG with MCP Server",
        value=st.session_state.enable_agentic_rag,
        help="AI agent with intelligent tool calling for enhanced RAG operations"
    )
    st.session_state.enable_agentic_rag = enable_agentic_rag

    if enable_agentic_rag:
        st.sidebar.caption("🤖 Entity extraction, query refinement, and relevance checking")
        st.sidebar.caption("⏱️ Note: Adds 5-10 seconds per query (multiple AI calls)")

    # Prompt Optimization
    enable_prompt_opt = st.sidebar.checkbox(
        "Enable Token-Efficient Prompts",
        value=st.session_state.enable_prompt_optimization,
        help="Optimize prompts to reduce token usage"
    )
    st.session_state.enable_prompt_optimization = enable_prompt_opt

    if enable_prompt_opt:
        st.sidebar.caption("⚡ Dynamic context compression and optimization")

    # Memory Integration
    enable_memory = st.sidebar.checkbox(
        "Enable LangChain Memory",
        value=st.session_state.enable_memory,
        help="Conversation memory and document-specific history"
    )
    st.session_state.enable_memory = enable_memory

    if enable_memory:
        st.sidebar.caption("🧠 Conversation buffer and summary memory")

    # Feature status
    if any([enable_agentic_rag, enable_prompt_opt, enable_memory]):
        st.sidebar.markdown("**Active Features:**")
        if enable_agentic_rag:
            st.sidebar.write("• Agentic RAG with MCP Server")
        if enable_prompt_opt:
            st.sidebar.write("• Token Optimization")
        if enable_memory:
            st.sidebar.write("• Conversation Memory")

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

def generate_response(query: str, relevant_chunks: List[Dict]) -> Tuple[str, Dict]:
    """Generate response using Gemini API with relevant chunks and new features"""
    if not st.session_state.gemini_client:
        return "Please configure Gemini API key first.", {}

    # Initialize response metadata
    response_metadata = {
        'original_tokens': 0,
        'optimized_tokens': 0,
        'tokens_saved': 0,
        'memory_used': False,
        'agentic_tools_used': [],
        'optimization_applied': False
    }

    # Prepare context from chunks
    context = "\n\n".join([
        f"Source: {chunk['filename']}\nContent: {chunk['text']}"
        for chunk in relevant_chunks
    ])

    # Add memory context if enabled
    memory_context = ""
    memory_history = None
    if st.session_state.enable_memory and st.session_state.memory_manager:
        try:
            document_ids = list(set(chunk.get('document_id', 'unknown') for chunk in relevant_chunks))
            memory_history = st.session_state.memory_manager.get_relevant_history(
                query, document_ids, max_interactions=5
            )

            # If this is a history query, prioritize memory over documents
            if memory_history.get('is_history_query', False):
                memory_context = memory_history.get('formatted_history', '')
                response_metadata['memory_query_type'] = 'history_query'
                response_metadata['memory_used'] = True

                # For history queries, we might want to use less document context
                if memory_context:
                    context = context[:500] + "..." if len(context) > 500 else context
            else:
                if memory_history['buffer_history'] or memory_history['summary']:
                    memory_context = st.session_state.memory_manager.get_conversation_context(500)
                    response_metadata['memory_query_type'] = 'context_query'
                    response_metadata['memory_used'] = True
        except Exception as e:
            st.warning(f"Memory integration error: {str(e)}")

    # Create prompt with or without optimization
    if st.session_state.enable_prompt_optimization and st.session_state.prompt_optimizer:
        try:
            # Use optimized prompt construction
            variables = {
                'context': context,
                'query': query
            }

            # Add memory context if available and handle history queries
            if memory_context:
                if memory_history and memory_history.get('is_history_query', False):
                    # For history queries, prioritize memory context
                    variables['context'] = f"{memory_context}\n\nAdditional Document Context:\n{context[:500]}..."
                else:
                    variables['context'] = f"{memory_context}\n\nDocument Context:\n{context}"

            prompt_result = st.session_state.prompt_optimizer.build_optimized_prompt(
                'rag_query_detailed', variables, target_tokens=3500
            )

            prompt = prompt_result['prompt']
            response_metadata.update({
                'original_tokens': prompt_result['initial_tokens'],
                'optimized_tokens': prompt_result['final_tokens'],
                'tokens_saved': prompt_result['tokens_saved'],
                'optimization_applied': prompt_result['optimization_applied']
            })

        except Exception as e:
            st.warning(f"Prompt optimization error: {str(e)}")
            # Fallback to standard prompt
            is_history_query = memory_history.get('is_history_query', False) if memory_history else False
            prompt = create_standard_prompt(query, context, memory_context, is_history_query)
    else:
        # Standard prompt creation
        is_history_query = memory_history.get('is_history_query', False) if memory_history else False
        prompt = create_standard_prompt(query, context, memory_context, is_history_query)

        # Count tokens for metadata
        if st.session_state.enable_prompt_optimization:
            token_counter = get_token_counter()
            response_metadata['original_tokens'] = token_counter.count_tokens(prompt)
            response_metadata['optimized_tokens'] = response_metadata['original_tokens']

    # Agentic RAG processing with MCP tools
    if st.session_state.enable_agentic_rag and st.session_state.mcp_client:
        try:
            # Connect to MCP server if not connected
            if not st.session_state.mcp_client.connected:
                connection_success = run_async_in_streamlit(st.session_state.mcp_client.connect())
                if connection_success:
                    st.session_state.mcp_server_running = True

            # Process query with agentic tools if connected
            if st.session_state.mcp_client.connected:
                # Get document chunks for processing (reduced to 3 for speed)
                doc_chunks = [chunk.get('text', chunk.get('content', '')) for chunk in relevant_chunks[:3]]

                # Process with agentic RAG - this takes 5-10 seconds due to multiple AI calls
                # Note: Each tool (entity extraction, query enhancement, relevance check) calls Gemini API
                refined_query, tools_used, agentic_metadata = run_async_in_streamlit(
                    process_query_with_agentic_rag(st.session_state.mcp_client, query, doc_chunks)
                )

                # Update query if refined
                if refined_query != query and tools_used:
                    query = refined_query
                    response_metadata['query_refined'] = True
                    response_metadata['original_query'] = agentic_metadata.get('original_query', query)

                response_metadata['agentic_tools_used'] = tools_used
                response_metadata['agentic_metadata'] = agentic_metadata

            else:
                response_metadata['agentic_tools_used'] = []

        except Exception as e:
            st.warning(f"Agentic RAG processing error: {str(e)}")
            response_metadata['agentic_tools_used'] = []
    else:
        response_metadata['agentic_tools_used'] = []

    try:
        # Generate response
        response = st.session_state.gemini_client.generate_content(prompt)
        response_text = response.text

        # Placeholder for Agentic RAG response processing
        if st.session_state.enable_agentic_rag:
            # Will be implemented with tool usage tracking
            pass

        # Add to memory if enabled
        if st.session_state.enable_memory and st.session_state.memory_manager:
            try:
                document_ids = list(set(chunk.get('document_id', 'unknown') for chunk in relevant_chunks))
                st.session_state.memory_manager.add_conversation(
                    query, response_text, document_ids,
                    {'chunks_used': len(relevant_chunks), 'optimization_applied': response_metadata['optimization_applied']}
                )
            except Exception as e:
                st.warning(f"Memory storage error: {str(e)}")

        return response_text, response_metadata

    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"

        # Placeholder for error logging in Agentic RAG
        if st.session_state.enable_agentic_rag:
            # Will be implemented with proper error handling
            pass

        return error_msg, response_metadata


def create_standard_prompt(query: str, context: str, memory_context: str = "", is_history_query: bool = False) -> str:
    """Create standard prompt without optimization"""
    prompt_parts = []

    if is_history_query and memory_context:
        # For history queries, prioritize conversation history
        prompt_parts.extend([
            "Conversation History:",
            memory_context,
            "",
            f"User Question: {query}",
            "",
            "Please answer the question based on our conversation history above. If the question is about previous questions, discussions, or what was mentioned before, refer to the conversation history."
        ])

        # Add document context as supplementary if available
        if context:
            prompt_parts.extend([
                "",
                "Additional Document Context (if relevant):",
                context[:500] + "..." if len(context) > 500 else context
            ])
    else:
        # Standard document-based query
        if memory_context:
            prompt_parts.append(f"Previous conversation context:\n{memory_context}\n")

        prompt_parts.extend([
            "Based on the following document context, answer the user's question.",
            "",
            "Document Context:",
            context,
            "",
            f"User Question: {query}",
            "",
            "Please provide a helpful answer based on the documents. If you reference specific information, mention which document it came from."
        ])

    return "\n".join(prompt_parts)

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

    # Token usage statistics
    token_stats = st.session_state.get('token_usage_stats', [])
    if token_stats:
        st.markdown("### ⚡ Token Optimization Statistics")

        total_original = sum(stat['original_tokens'] for stat in token_stats)
        total_optimized = sum(stat['optimized_tokens'] for stat in token_stats)
        total_saved = sum(stat['tokens_saved'] for stat in token_stats)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Original Tokens</h3>
                <div style="font-size: 2rem; font-weight: 700; color: #6366f1;">
                    {format_token_count(total_original)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Optimized Tokens</h3>
                <div style="font-size: 2rem; font-weight: 700; color: #10b981;">
                    {format_token_count(total_optimized)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Tokens Saved</h3>
                <div style="font-size: 2rem; font-weight: 700; color: #f59e0b;">
                    {format_token_count(total_saved)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            efficiency = (total_saved / total_original * 100) if total_original > 0 else 0
            cost_saved = estimate_cost(total_saved, "gemini-2.0-flash-lite")
            st.markdown(f"""
            <div class="metric-card">
                <h3>Efficiency</h3>
                <div style="font-size: 2rem; font-weight: 700; color: #8b5cf6;">
                    {efficiency:.1f}%
                </div>
                <div style="font-size: 0.9rem; color: #64748b;">
                    ~${cost_saved:.4f} saved
                </div>
            </div>
            """, unsafe_allow_html=True)
    
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

    # Feedback System Configuration
    st.markdown("### 📝 Feedback System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Feedback Collection</h3>
        </div>
        """, unsafe_allow_html=True)
        
        feedback_enabled = st.checkbox(
            "Enable Feedback Collection",
            value=st.session_state.get('feedback_enabled', False),
            help="Allow users to provide feedback on responses"
        )
        st.session_state.feedback_enabled = feedback_enabled
        
        if feedback_enabled:
            if st.session_state.gemini_client and not st.session_state.feedback_system:
                # Initialize feedback system
                api_key = st.session_state.get('api_key')
                if api_key:
                    smtp_config = st.session_state.get('smtp_config')
                    st.session_state.feedback_system = get_feedback_system(api_key, smtp_config)
                    st.success("✅ Feedback system initialized")
            
            if st.session_state.feedback_system:
                stats = st.session_state.feedback_system.get_feedback_statistics()
                st.write(f"**Total Feedback**: {stats.get('total_feedback', 0)}")
                
                if stats.get('sentiment_distribution'):
                    st.write("**Sentiment Distribution**:")
                    for sentiment, count in stats['sentiment_distribution'].items():
                        st.write(f"  • {sentiment.title()}: {count}")
                
                if stats.get('average_rating'):
                    st.write(f"**Average Rating**: {stats['average_rating']:.2f}/5")
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📧 Email Notifications</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if feedback_enabled:
            st.write("Configure email notifications for feedback:")
            
            recipient_email = st.text_input(
                "Recipient Email",
                value=st.session_state.get('feedback_recipient_email', ''),
                placeholder="manager@company.com",
                help="Email address to receive feedback notifications"
            )
            st.session_state.feedback_recipient_email = recipient_email
            
            recipient_name = st.text_input(
                "Recipient Name",
                value=st.session_state.get('feedback_recipient_name', 'Team'),
                placeholder="Product Manager",
                help="Name or role of the recipient"
            )
            st.session_state.feedback_recipient_name = recipient_name
            
            with st.expander("📧 SMTP Configuration (Optional)"):
                st.markdown("""
                **For Gmail users:**
                1. Enable 2-Factor Authentication on your Google Account
                2. Generate an App Password at: https://myaccount.google.com/apppasswords
                3. Use the 16-character App Password below (not your regular password)
                
                See `GMAIL_APP_PASSWORD_SETUP.md` for detailed instructions.
                """)
                
                smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
                smtp_port = st.number_input("SMTP Port", value=587, min_value=1, max_value=65535)
                smtp_username = st.text_input("Username/Email", placeholder="your.email@gmail.com")
                smtp_password = st.text_input(
                    "Password (App Password for Gmail)", 
                    type="password",
                    placeholder="16-character App Password"
                )
                use_tls = st.checkbox("Use TLS", value=True)
                from_email = st.text_input("From Email", value=smtp_username)
                
                if st.button("Save SMTP Configuration"):
                    if smtp_server and smtp_username and smtp_password:
                        st.session_state.smtp_config = {
                            'smtp_server': smtp_server,
                            'smtp_port': smtp_port,
                            'username': smtp_username,
                            'password': smtp_password,
                            'use_tls': use_tls,
                            'from_email': from_email or smtp_username
                        }
                        
                        # Reinitialize feedback system with SMTP config
                        if st.session_state.feedback_system:
                            api_key = st.session_state.get('api_key')
                            st.session_state.feedback_system = get_feedback_system(
                                api_key, 
                                st.session_state.smtp_config
                            )
                        
                        st.success("✅ SMTP configuration saved! Test it by submitting feedback.")
                    else:
                        st.error("Please fill in all SMTP fields")
        else:
            st.info("Enable feedback collection to configure email notifications")
    
    if feedback_enabled and st.session_state.feedback_system:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Feedback Data", use_container_width=True):
                result = st.session_state.feedback_system.export_feedback()
                if result['success']:
                    st.success(f"✅ Exported {result['count']} feedback records to {result['filepath']}")
                else:
                    st.error(f"❌ Export failed: {result['error']}")
        
        with col2:
            if st.button("🗑️ Clear Feedback History", use_container_width=True):
                st.session_state.feedback_system.feedback_history = []
                st.success("✅ Feedback history cleared!")
    
    st.markdown("---")
    
    # Advanced Features Configuration
    st.markdown("### 🚀 Advanced Features Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Agentic RAG with MCP Server</h3>
        </div>
        """, unsafe_allow_html=True)

        agentic_enabled = st.session_state.get('enable_agentic_rag', False)
        st.write(f"**Status**: {'✅ Enabled' if agentic_enabled else '❌ Disabled'}")

        if agentic_enabled:
            server_running = st.session_state.get('mcp_server_running', False)
            st.write(f"**MCP Server**: {'🟢 Running' if server_running else '🔴 Stopped'}")
            st.write(f"**Gemini API**: {'✅ Configured' if st.session_state.get('api_key') else '❌ Not Set'}")
            st.write(f"**Available Tools**: Entity Extraction, Query Refinement, Relevance Check")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Check Server Status", use_container_width=True):
                health_status = check_mcp_server_health()
                if health_status['status'] == 'running':
                    st.success(f"✅ Server running with {health_status['tools_available']} tools")
                    st.session_state.mcp_server_running = True
                elif health_status['status'] == 'stopped':
                    st.error("❌ Server not running")
                    st.session_state.mcp_server_running = False
                else:
                    st.warning("⚠️ MCP libraries not installed")

        with col2:
            if st.button("Configure API", use_container_width=True, disabled=not (agentic_enabled and st.session_state.get('api_key'))):
                if st.session_state.get('mcp_client') and st.session_state.get('api_key'):
                    try:
                        # Check if server is running first
                        if not st.session_state.mcp_client.is_server_running():
                            st.error("❌ MCP server is not running. Please start it first with: python mcp_server.py")
                        else:
                            # Connect and configure API
                            with st.spinner("Connecting to MCP server..."):
                                connection_success = run_async_in_streamlit(st.session_state.mcp_client.connect())

                            if connection_success:
                                with st.spinner("Configuring Gemini API..."):
                                    config_success = run_async_in_streamlit(
                                        st.session_state.mcp_client.configure_gemini_api(st.session_state.api_key)
                                    )
                                if config_success:
                                    st.success("✅ Gemini API configured on MCP server")
                                else:
                                    st.error("❌ Failed to configure Gemini API")
                            else:
                                st.error("❌ Could not connect to MCP server")
                    except Exception as e:
                        st.error(f"Configuration error: {str(e)}")
                else:
                    st.warning("⚠️ API key required")

        st.info("🚀 To start the MCP server, run in terminal:")
        st.code("python mcp_server.py", language="bash")

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Token Optimization</h3>
        </div>
        """, unsafe_allow_html=True)

        opt_enabled = st.session_state.get('enable_prompt_optimization', False)
        st.write(f"**Status**: {'✅ Enabled' if opt_enabled else '❌ Disabled'}")

        if opt_enabled:
            token_stats = st.session_state.get('token_usage_stats', [])
            total_saved = sum(stat['tokens_saved'] for stat in token_stats)
            st.write(f"**Total Tokens Saved**: {format_token_count(total_saved)}")

            if token_stats:
                avg_efficiency = sum(stat['tokens_saved'] / stat['original_tokens']
                                   for stat in token_stats if stat['original_tokens'] > 0) / len(token_stats) * 100
                st.write(f"**Avg Efficiency**: {avg_efficiency:.1f}%")

        if st.button("Clear Token Stats", use_container_width=True):
            st.session_state.token_usage_stats = []
            st.success("✅ Token stats cleared!")

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 Memory Integration</h3>
        </div>
        """, unsafe_allow_html=True)

        memory_enabled = st.session_state.get('enable_memory', False)
        st.write(f"**Status**: {'✅ Enabled' if memory_enabled else '❌ Disabled'}")

        if memory_enabled and st.session_state.get('memory_manager'):
            try:
                memory_stats = st.session_state.memory_manager.get_memory_stats()
                st.write(f"**Sessions**: {memory_stats.get('total_sessions', 0)}")
                st.write(f"**Messages**: {memory_stats.get('current_session_messages', 0)}")
                st.write(f"**Doc Memories**: {memory_stats.get('document_memories', 0)}")
            except Exception as e:
                st.write("**Status**: Error loading stats")

        if st.button("Clear Memory", use_container_width=True):
            if st.session_state.get('memory_manager'):
                st.session_state.memory_manager.clear_session_memory()
                st.success("✅ Memory cleared!")
    
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
    for idx, chat in enumerate(st.session_state.chat_history):
        # User message
        with st.chat_message("user"):
            st.write(chat['query'])
        
        # Assistant message
        with st.chat_message("assistant"):
            st.write(chat['response'])
            
            # Show advanced features used (MCP tools, token optimization, memory)
            if 'metadata' in chat:
                response_metadata = chat['metadata']
                if any([
                    response_metadata.get('optimization_applied'),
                    response_metadata.get('memory_used'),
                    response_metadata.get('agentic_tools_used')
                ]):
                    with st.expander("🚀 Advanced Features Used"):
                        # Token optimization
                        if response_metadata.get('optimization_applied'):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Original Tokens", format_token_count(response_metadata.get('original_tokens', 0)))
                            with col2:
                                st.metric("Optimized Tokens", format_token_count(response_metadata.get('optimized_tokens', 0)))
                            with col3:
                                st.metric("Tokens Saved", format_token_count(response_metadata.get('tokens_saved', 0)))
                            
                            if response_metadata.get('tokens_saved', 0) > 0:
                                efficiency = (response_metadata['tokens_saved'] / response_metadata['original_tokens']) * 100
                                st.success(f"⚡ {efficiency:.1f}% token reduction achieved!")
                        
                        # Memory usage
                        if response_metadata.get('memory_used'):
                            st.info("🧠 Used conversation memory for enhanced context")
                        
                        # Agentic RAG tools
                        if response_metadata.get('agentic_tools_used'):
                            tools_used = response_metadata['agentic_tools_used']
                            if tools_used:
                                st.info(f"🤖 Agentic Tools Used: {', '.join(tools_used)}")
                                
                                # Show agentic metadata
                                agentic_meta = response_metadata.get('agentic_metadata', {})
                                if agentic_meta.get('entities'):
                                    with st.expander("🔍 Extracted Entities"):
                                        for entity_type, entities in agentic_meta['entities'].items():
                                            if entities:
                                                st.write(f"**{entity_type}**: {', '.join(entities)}")
                                
                                if agentic_meta.get('avg_relevance'):
                                    relevance = agentic_meta['avg_relevance']
                                    st.metric("Average Document Relevance", f"{relevance:.2f}")
                                
                                if response_metadata.get('query_refined'):
                                    with st.expander("✨ Query Refinement"):
                                        st.write(f"**Original**: {response_metadata.get('original_query', 'N/A')}")
                                        st.write(f"**Refined**: {chat['query']}")
            
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
            
            # Show feedback UI for each message in history
            if st.session_state.feedback_enabled and st.session_state.feedback_system:
                # Check if feedback already submitted for this message
                feedback_key = f"feedback_submitted_{idx}"
                if not st.session_state.get(feedback_key, False):
                    with st.expander("📝 Provide Feedback on this Response", expanded=False):
                        st.write("Help us improve! Share your thoughts on this response.")
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            feedback_text = st.text_area(
                                "Your Feedback",
                                placeholder="Was this response helpful? Any issues or suggestions?",
                                key=f"history_feedback_text_{idx}",
                                height=100
                            )
                        
                        with col2:
                            rating = st.select_slider(
                                "Rating",
                                options=[1, 2, 3, 4, 5],
                                value=3,
                                key=f"history_rating_{idx}"
                            )
                            st.caption("1=Poor, 5=Excellent")
                        
                        user_email = st.text_input(
                            "Your Email",
                            placeholder="your.email@example.com",
                            key=f"history_user_email_{idx}"
                        )
                        
                        if st.button("Submit Feedback", key=f"history_submit_feedback_{idx}"):
                            if feedback_text.strip():
                                with st.spinner("Analyzing feedback..."):
                                    # Collect feedback
                                    feedback_record = st.session_state.feedback_system.collect_feedback(
                                        query=chat['query'],
                                        response=chat['response'],
                                        feedback_text=feedback_text,
                                        rating=rating,
                                        user_email=user_email if user_email else None,
                                        metadata=chat.get('metadata', {})
                                    )
                                    
                                    # Display sentiment analysis
                                    if feedback_record['sentiment_analysis']['success']:
                                        sentiment_data = feedback_record['sentiment_analysis']['sentiment_data']
                                        
                                        st.success("✅ Thank you for your feedback!")
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Sentiment", sentiment_data.get('sentiment', 'N/A').title())
                                        with col2:
                                            st.metric("Satisfaction", f"{sentiment_data.get('satisfaction_score', 'N/A')}/5")
                                        with col3:
                                            st.metric("Urgency", sentiment_data.get('urgency', 'N/A').title())
                                        
                                        # Send email
                                        smtp_config = st.session_state.get('smtp_config')
                                        recipient_email = st.session_state.get('feedback_recipient_email')
                                        
                                        if smtp_config and recipient_email:
                                            try:
                                                with st.spinner("Sending email..."):
                                                    email_result = st.session_state.feedback_system.send_feedback_email(
                                                        feedback_record,
                                                        recipient_email,
                                                        st.session_state.get('feedback_recipient_name', 'Team')
                                                    )
                                                
                                                if email_result.get('success'):
                                                    st.success(f"📧 Email sent to {recipient_email}")
                                                else:
                                                    st.error(f"❌ Email failed: {email_result.get('error')}")
                                            except Exception as e:
                                                st.error(f"❌ Email error: {str(e)}")
                                        elif recipient_email and not smtp_config:
                                            st.warning("⚠️ Configure SMTP in sidebar to send emails")
                                        
                                        # Mark as submitted
                                        st.session_state[feedback_key] = True
                                        st.rerun()
                                    else:
                                        st.warning("⚠️ Sentiment analysis failed")
                            else:
                                st.warning("Please enter feedback text")
                else:
                    st.info("✅ Feedback already submitted for this response")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            # Search for relevant chunks if documents are available
            relevant_chunks = []
            if st.session_state.processed_documents:
                with st.spinner("Searching documents..."):
                    relevant_chunks = search_documents(prompt)

            # Generate response with new features
            with st.spinner("Generating response..."):
                response, response_metadata = generate_response(prompt, relevant_chunks)

            # Display response with enhanced styling
            st.markdown(f"""
            <div class="response-container">
                {response}
            </div>
            """, unsafe_allow_html=True)

            # Display new feature information
            if any([st.session_state.enable_prompt_optimization, st.session_state.enable_memory, st.session_state.enable_agentic_rag]):
                with st.expander("🚀 Advanced Features Used"):
                    if st.session_state.enable_prompt_optimization and response_metadata.get('optimization_applied'):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Original Tokens", format_token_count(response_metadata['original_tokens']))
                            with col2:
                                st.metric("Optimized Tokens", format_token_count(response_metadata['optimized_tokens']))
                            with col3:
                                st.metric("Tokens Saved", format_token_count(response_metadata['tokens_saved']))

                            if response_metadata['tokens_saved'] > 0:
                                efficiency = (response_metadata['tokens_saved'] / response_metadata['original_tokens']) * 100
                                st.success(f"⚡ {efficiency:.1f}% token reduction achieved!")

                    if st.session_state.enable_memory and response_metadata.get('memory_used'):
                        st.info("🧠 Used conversation memory for enhanced context")

                    if st.session_state.enable_agentic_rag and response_metadata.get('agentic_tools_used'):
                            tools_used = response_metadata['agentic_tools_used']
                            if tools_used:
                                st.info(f"🤖 Agentic Tools Used: {', '.join(tools_used)}")

                                # Show additional agentic metadata
                                agentic_meta = response_metadata.get('agentic_metadata', {})
                                if agentic_meta.get('entities'):
                                    with st.expander("🔍 Extracted Entities"):
                                        for entity_type, entities in agentic_meta['entities'].items():
                                            if entities:
                                                st.write(f"**{entity_type}**: {', '.join(entities)}")

                                if agentic_meta.get('avg_relevance'):
                                    relevance = agentic_meta['avg_relevance']
                                    st.metric("Average Document Relevance", f"{relevance:.2f}")

                                if response_metadata.get('query_refined'):
                                    with st.expander("✨ Query Refinement"):
                                        st.write(f"**Original**: {response_metadata.get('original_query', 'N/A')}")
                                        st.write(f"**Refined**: {prompt}")
                            else:
                                st.info("🤖 Agentic RAG enabled (tools will be used when MCP server is running)")
                
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
                    for chunk in relevant_chunks:
                            st.markdown(f"""
                            <div class="chunk-display">
                                <h4>📄 {chunk['filename']} (Score: {chunk['score']:.3f})</h4>
                                <p>{chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
            # Feedback Collection Section
            if st.session_state.feedback_enabled and st.session_state.feedback_system:
                    with st.expander("📝 Provide Feedback on this Response", expanded=False):
                        st.write("Help us improve! Share your thoughts on this response.")
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            feedback_text = st.text_area(
                                "Your Feedback",
                                placeholder="Was this response helpful? Any issues or suggestions?",
                                key=f"feedback_text_{len(st.session_state.chat_history)}",
                                height=100
                            )
                        
                        with col2:
                            rating = st.select_slider(
                                "Rating",
                                options=[1, 2, 3, 4, 5],
                                value=3,
                                key=f"rating_{len(st.session_state.chat_history)}"
                            )
                            st.caption("1=Poor, 5=Excellent")
                        
                        user_email = st.text_input(
                            "Your Email",
                            placeholder="your.email@example.com",
                            key=f"user_email_{len(st.session_state.chat_history)}"
                        )
                        
                        if st.button("Submit Feedback", key=f"submit_feedback_{len(st.session_state.chat_history)}"):
                            if feedback_text.strip():
                                with st.spinner("Analyzing feedback..."):
                                    # Collect feedback with sentiment analysis
                                    feedback_record = st.session_state.feedback_system.collect_feedback(
                                        query=prompt,
                                        response=response,
                                        feedback_text=feedback_text,
                                        rating=rating,
                                        user_email=user_email if user_email else None,
                                        metadata={
                                            'sources_count': len(relevant_chunks),
                                            'evaluation': evaluation_results if evaluation_results else None,
                                            'response_metadata': response_metadata
                                        }
                                    )
                                    
                                    # Display sentiment analysis
                                    if feedback_record['sentiment_analysis']['success']:
                                        sentiment_data = feedback_record['sentiment_analysis']['sentiment_data']
                                        
                                        st.success("✅ Thank you for your feedback!")
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Sentiment", sentiment_data.get('sentiment', 'N/A').title())
                                        with col2:
                                            st.metric("Satisfaction", f"{sentiment_data.get('satisfaction_score', 'N/A')}/5")
                                        with col3:
                                            st.metric("Urgency", sentiment_data.get('urgency', 'N/A').title())
                                        
                                        # Send email notification if configured
                                        smtp_config = st.session_state.get('smtp_config')
                                        recipient_email = st.session_state.get('feedback_recipient_email')
                                        
                                        if smtp_config and recipient_email:
                                            try:
                                                with st.spinner("Sending email notification..."):
                                                    email_result = st.session_state.feedback_system.send_feedback_email(
                                                        feedback_record,
                                                        recipient_email,
                                                        st.session_state.get('feedback_recipient_name', 'Team')
                                                    )
                                                
                                                if email_result.get('success'):
                                                    st.success(f"📧 Email sent to {recipient_email}")
                                                else:
                                                    st.error(f"❌ Email failed: {email_result.get('error', 'Unknown error')}")
                                            except Exception as e:
                                                st.error(f"❌ Email error: {str(e)}")
                                        elif recipient_email and not smtp_config:
                                            st.warning("⚠️ Configure SMTP settings in sidebar to send emails")
                                    else:
                                        st.warning("⚠️ Feedback collected but sentiment analysis failed")
                            else:
                                st.warning("Please enter your feedback before submitting")
        
        # Add to chat history
        chat_entry = {
            'query': prompt,
            'response': response,
            'sources': relevant_chunks,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'metadata': response_metadata
        }

        if evaluation_results:
            chat_entry['evaluation'] = evaluation_results

        st.session_state.chat_history.append(chat_entry)

        # Store token usage stats
        if response_metadata.get('optimized_tokens', 0) > 0:
            token_stat = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'original_tokens': response_metadata['original_tokens'],
                'optimized_tokens': response_metadata['optimized_tokens'],
                'tokens_saved': response_metadata['tokens_saved'],
                'optimization_applied': response_metadata['optimization_applied']
            }
            st.session_state.token_usage_stats.append(token_stat)
        
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
    
    # Feedback System Configuration in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("📝 Feedback System")
    
    feedback_enabled = st.sidebar.checkbox(
        "Enable Feedback Collection",
        value=st.session_state.get('feedback_enabled', False),
        help="Allow users to provide feedback on responses"
    )
    st.session_state.feedback_enabled = feedback_enabled
    
    if feedback_enabled:
        # Initialize feedback system if not already done
        if st.session_state.gemini_client and not st.session_state.feedback_system:
            api_key = st.session_state.get('api_key')
            if api_key:
                smtp_config = st.session_state.get('smtp_config')
                st.session_state.feedback_system = get_feedback_system(api_key, smtp_config)
                st.sidebar.success("✅ Feedback system ready")
        
        # Email Configuration
        with st.sidebar.expander("📧 Email Notifications", expanded=False):
            recipient_email = st.text_input(
                "Recipient Email",
                value=st.session_state.get('feedback_recipient_email', ''),
                placeholder="manager@company.com",
                help="Email to receive feedback notifications",
                key="sidebar_recipient_email"
            )
            st.session_state.feedback_recipient_email = recipient_email
            
            recipient_name = st.text_input(
                "Recipient Name",
                value=st.session_state.get('feedback_recipient_name', 'Team'),
                placeholder="Product Manager",
                key="sidebar_recipient_name"
            )
            st.session_state.feedback_recipient_name = recipient_name
            
            st.markdown("**SMTP Configuration:**")
            
            smtp_server = st.text_input(
                "SMTP Server",
                value=st.session_state.get('smtp_server', 'smtp.gmail.com'),
                key="sidebar_smtp_server"
            )
            
            smtp_port = st.number_input(
                "SMTP Port",
                value=st.session_state.get('smtp_port', 587),
                min_value=1,
                max_value=65535,
                key="sidebar_smtp_port"
            )
            
            smtp_username = st.text_input(
                "Email Username",
                value=st.session_state.get('smtp_username', ''),
                placeholder="your.email@gmail.com",
                key="sidebar_smtp_username"
            )
            
            smtp_password = st.text_input(
                "Email Password",
                type="password",
                value=st.session_state.get('smtp_password', ''),
                placeholder="App Password",
                help="For Gmail, use App Password not regular password",
                key="sidebar_smtp_password"
            )
            
            use_tls = st.checkbox(
                "Use TLS",
                value=st.session_state.get('smtp_use_tls', True),
                key="sidebar_smtp_tls"
            )
            
            if st.button("💾 Save Email Config", key="save_smtp_sidebar"):
                if smtp_server and smtp_username and smtp_password and recipient_email:
                    # Store SMTP config
                    st.session_state.smtp_config = {
                        'smtp_server': smtp_server,
                        'smtp_port': smtp_port,
                        'username': smtp_username,
                        'password': smtp_password,
                        'use_tls': use_tls,
                        'from_email': smtp_username
                    }
                    
                    # Store individual values for persistence
                    st.session_state.smtp_server = smtp_server
                    st.session_state.smtp_port = smtp_port
                    st.session_state.smtp_username = smtp_username
                    st.session_state.smtp_password = smtp_password
                    st.session_state.smtp_use_tls = use_tls
                    
                    # Reinitialize feedback system with SMTP config
                    if st.session_state.get('api_key'):
                        st.session_state.feedback_system = get_feedback_system(
                            st.session_state.api_key,
                            st.session_state.smtp_config
                        )
                    
                    st.sidebar.success("✅ Email config saved!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Please fill all fields")
            
            st.caption("💡 Gmail users: Use App Password from Google Account settings")
        
        # Feedback Statistics
        if st.session_state.feedback_system:
            stats = st.session_state.feedback_system.get_feedback_statistics()
            if stats.get('total_feedback', 0) > 0:
                st.sidebar.markdown("**📊 Statistics:**")
                st.sidebar.write(f"Total: {stats['total_feedback']}")
                if stats.get('average_rating'):
                    st.sidebar.write(f"Avg Rating: {stats['average_rating']:.1f}/5")
    

    
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
        api_configured, _ = setup_gemini_api()
        setup_advanced_features()
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
