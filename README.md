# 🏥 Clinical Trial RAG Evaluation System

A comprehensive RAG (Retrieval-Augmented Generation) evaluation system with real-time metrics, document chat, and performance analysis using Google Gemini API and DeepEval.

## Features

- **🤖 Google Gemini 2.0 Flash-Lite**: Latest Gemini model for fast, accurate responses
- **📊 Real-time Evaluation**: Custom RAG metrics with comprehensive analysis
- **📄 Document Processing**: Upload and chat with PDF documents
- **🧭 Multi-page Navigation**: Separate sections for chat, logs, and settings
- **📈 Performance Analytics**: Track and compare evaluation metrics over time
- **🎯 Top-K Optimization**: Find optimal retrieval parameters
- **💬 Chat Memory**: Maintains conversation history and context
- **📚 Source References**: Shows document chunks used for answers

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```

3. **Get your Gemini API key**:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Enter it in the sidebar of the app

4. **Start using the system**:
   - Navigate between Chat, Evaluation Logs, and Settings
   - Upload PDF documents for analysis
   - Enable real-time evaluation to track performance
   - Ask questions and view comprehensive metrics

## Navigation System

### 💬 Chat & Evaluation
- Interactive chat interface with document Q&A
- Real-time evaluation metrics (Answer Relevancy, Faithfulness, etc.)
- Source attribution and chunk analysis
- Configurable Top-K retrieval settings

### 📊 Evaluation Logs
- Comprehensive performance analytics
- Historical evaluation data with trends
- Top-K performance comparison
- CSV export functionality
- Performance recommendations

### ⚙️ Settings
- System configuration overview
- Data management (clear history, logs, documents)
- API status monitoring
- System information and statistics

## How It Works

1. **Document Processing**: PDFs are processed into text chunks with embeddings
2. **Smart Retrieval**: Finds relevant chunks using semantic similarity
3. **Response Generation**: Gemini generates context-aware responses
4. **Real-time Evaluation**: DeepEval metrics assess response quality
5. **Performance Tracking**: Logs and analyzes evaluation results over time

## Evaluation Metrics

Our custom-built evaluation system provides comprehensive RAG assessment:

- **Answer Relevancy**: Combines semantic similarity and LLM judgment to measure response relevance
- **Faithfulness**: Extracts and verifies factual claims against provided context
- **Contextual Relevancy**: Evaluates how well retrieved chunks match the query
- **Contextual Recall**: Measures completeness by comparing with auto-generated expected answers

### Evaluation Methods

Each metric uses multiple approaches for robust assessment:

1. **Semantic Analysis**: Sentence transformers for embedding-based similarity
2. **LLM Judgment**: Gemini API for nuanced evaluation
3. **Fallback Methods**: Keyword overlap and statistical measures for reliability

## Configuration

- **API Key**: Enter your Gemini API key in the sidebar
- **Document Upload**: Upload PDFs for analysis
- **Evaluation Settings**: Enable/disable metrics and set thresholds
- **Top-K Settings**: Adjust number of retrieved chunks (1-20)

## Technical Stack

- **LLM**: Google Gemini 2.0 Flash-Lite
- **Evaluation**: Custom RAG evaluation system (built from scratch)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Vector Database**: Qdrant (in-memory)
- **Framework**: Streamlit with custom CSS
- **Document Processing**: PyPDF2
- **ML Libraries**: scikit-learn for similarity calculations

## Project Structure

```
├── app.py                    # Main Streamlit application
├── document_processor.py     # PDF processing and text extraction
├── embedding_generator.py    # Embedding generation utilities
├── custom_evaluator.py      # Custom RAG evaluation metrics
├── evaluation_fallback.py   # Fallback evaluation system
├── rate_limiter.py          # API rate limiting
├── vector_database.py       # Qdrant vector database operations
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Troubleshooting

### Common Issues

1. **API Overload (503 errors)**: Gemini API is experiencing high load
   - Wait 2-3 minutes and try again
   - Use fewer evaluation metrics
   - Try during off-peak hours

2. **Invalid API Key (400 errors)**: Check your Gemini API key
   - Verify the key is correct in the sidebar
   - Ensure the key has proper permissions

3. **Rate Limiting (429 errors)**: Too many requests
   - Built-in rate limiting should prevent this
   - Wait before making more requests

### Performance Tips

- Start with fewer evaluation metrics to test the system
- Use Top-K values between 3-7 for optimal performance
- Upload smaller documents initially to test functionality
- Monitor the Evaluation Logs page for performance insights

## Advanced Features

- **Auto-generated Expected Output**: For contextual recall evaluation
- **Performance Comparison**: Compare different Top-K values
- **Export Functionality**: Download evaluation logs as CSV
- **3D UI Effects**: Modern, professional interface design
- **Dual Color System**: Light navigation, dark content for optimal UX

## Deployment

### Streamlit Cloud Deployment

The app includes automatic fallback for Streamlit Cloud compatibility:

1. **Fork this repository** to your GitHub account
2. **Connect to Streamlit Cloud** at [share.streamlit.io](https://share.streamlit.io)
3. **Deploy** using the main `app.py` file
4. **Add your Gemini API key** in the app interface

### Environment Variables (Optional)

Set these in Streamlit Cloud secrets or your environment:

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your_api_key_here"
```

### Troubleshooting Deployment

**Evaluation System Benefits:**
- Built from scratch for maximum Streamlit Cloud compatibility
- No complex async dependencies that cause deployment issues
- Robust fallback mechanisms for reliable operation
- Full suite of RAG metrics without external framework dependencies

**API Quota Issues:**
- Free Gemini API: 200 requests/day limit
- Upgrade to paid plan for higher limits
- Monitor usage in the Evaluation Logs page