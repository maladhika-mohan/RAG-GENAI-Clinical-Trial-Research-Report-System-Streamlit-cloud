# 🏥 Clinical Trial RAG Evaluation System

A comprehensive RAG (Retrieval-Augmented Generation) evaluation system with real-time metrics, document chat, AI-powered feedback system, and agentic RAG capabilities using Google Gemini API.

## 📋 Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [Deployment (Free Cloud)](#deployment-free-cloud)
- [MCP Server (Agentic RAG)](#mcp-server-agentic-rag)
- [Feedback System](#feedback-system)
- [Gmail Setup for Email](#gmail-setup-for-email)
- [Technical Stack](#technical-stack)
- [Troubleshooting](#troubleshooting)

---

## Features

### Core Features
- **🤖 Google Gemini 2.0 Flash-Lite**: Latest Gemini model for fast, accurate responses
- **📊 Real-time Evaluation**: Custom RAG metrics with comprehensive analysis
- **📄 Document Processing**: Upload and chat with PDF documents
- **🧭 Multi-page Navigation**: Separate sections for chat, logs, and settings
- **📈 Performance Analytics**: Track and compare evaluation metrics over time
- **🎯 Top-K Optimization**: Find optimal retrieval parameters
- **💬 Chat Memory**: Maintains conversation history and context
- **📚 Source References**: Shows document chunks used for answers

### 🚀 Advanced Features
- **🤖 Agentic RAG with MCP Server**: AI-powered query enhancement with 6 clinical trial tools
  - Validate medical queries
  - Extract clinical entities (drugs, conditions, trial phases)
  - Enhance queries for better search
  - Check document relevance
  - Validate answer accuracy
- **⚡ Token-Efficient Prompts**: Intelligent prompt optimization to reduce token usage by up to 50%
- **🧠 LangChain Memory Integration**: Persistent conversation memory with document-specific history
- **📊 Token Usage Analytics**: Track optimization savings and efficiency metrics
- **💬 Feedback System**: AI-powered sentiment analysis with email notifications
- **🎛️ Feature Toggle Controls**: Enable/disable advanced features independently

---

## Quick Start

### Option 1: Basic RAG (No MCP Server)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
streamlit run app.py

# 3. Configure in UI
# - Enter your Gemini API key in the sidebar
# - Upload PDF documents
# - Start chatting!
```

### Option 2: With Agentic RAG (MCP Server)

```bash
# Terminal 1 - Start MCP Server
python mcp_server.py

# Terminal 2 - Start Streamlit App
streamlit run app.py

# Then in the Streamlit UI:
# 1. Enter your Gemini API key in the sidebar
# 2. Enable "Agentic RAG with MCP Server" in sidebar
# 3. Go to Settings page and click "Configure API"
# 4. Upload PDF documents
# 5. Start chatting with AI-enhanced query processing!
```

### Get Your Gemini API Key
- Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
- Create a new API key
- Enter it in the sidebar of the app

---

## Deployment (Free Cloud)

### Recommended: Streamlit Cloud ⭐

**Why?** Free forever, zero config, perfect for Streamlit apps, automatic updates from GitHub.

**Limitation:** MCP Server won't work (needs multiple processes), but all core RAG features work perfectly.

#### Quick Deployment Steps:

**1. Create `.streamlit/config.toml`:**
```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#1e293b"
textColor = "#e2e8f0"

[server]
headless = true
port = 8501
enableCORS = false

[browser]
gatherUsageStats = false
```

**2. Create `.gitignore`:**
```
__pycache__/
*.pyc
.streamlit/secrets.toml
*.pdf
*.csv
evaluation_logs/
uploaded_documents/
vector_db/
.vscode/
.DS_Store
```

**3. Push to GitHub:**
```bash
git init
git add .
git commit -m "Deploy Clinical Trials RAG"
git remote add origin https://github.com/YOUR_USERNAME/clinical-trials-rag.git
git push -u origin main
```

**4. Deploy:**
1. Go to **https://share.streamlit.io**
2. Click **"New app"**
3. Connect GitHub
4. Select your repo
5. Main file: `app.py`
6. Click **"Deploy"**

Done! Your app will be live at: `https://YOUR_USERNAME-clinical-trials-rag.streamlit.app`

### Alternative Options

**Hugging Face Spaces:**
- Free unlimited public spaces
- Better resources (2 vCPU, 16GB storage)
- Create space at huggingface.co/spaces
- Choose Streamlit SDK
- Push your code

**Railway (If you need MCP Server):**
- $5 credit/month (~500 hours)
- Can run multiple processes (MCP Server works!)
- Deploy from GitHub at railway.app
- Create `Procfile`: `web: streamlit run app.py --server.port $PORT`

### What Works on Free Tier

✅ Core RAG functionality
✅ Document upload & processing
✅ Chat interface with memory
✅ Evaluation metrics
✅ Feedback system with sentiment analysis
✅ Email notifications
✅ All UI features
✅ Token optimization
✅ LangChain memory

❌ MCP Server (only on Railway or paid tiers)

---

## MCP Server (Agentic RAG)

### What is MCP?

**Model Context Protocol (MCP)** gives your RAG system "superpowers" through specialized AI tools. Think of it as an AI assistant that helps process queries intelligently.

### The 6 MCP Tools

| Tool | Purpose | What It Does |
|------|---------|--------------|
| `configure_gemini_api` | Setup | Configures Gemini API key on server |
| `validate_medical_query` | Validation | Checks if query is medical/clinical |
| `enhance_clinical_query` | Enhancement | Improves query for better search |
| `extract_clinical_entities` | Extraction | Finds drugs, conditions, trial phases |
| `check_clinical_relevance` | Filtering | Evaluates document chunk relevance |
| `validate_clinical_answer` | Verification | Validates answer accuracy |

### How It Works

**Without Agentic RAG (Basic):**
```
User Query → Vector Search → Retrieve Chunks → Generate Response
```

**With Agentic RAG (Enhanced):**
```
User Query: "What are aspirin side effects?"
    ↓
Extract Entities: {drug: "aspirin", adverse_events: "side effects"}
    ↓
Enhance Query: "What are the documented adverse events, safety profile,
                and contraindications of aspirin in Phase II/III trials?"
    ↓
Vector Search (with enhanced query)
    ↓
Check Relevance: Filter chunks (keep 0.92, 0.88; remove 0.45)
    ↓
Generate Response (with better context)
    ↓
Validate Answer: 95% confidence, strong evidence
    ↓
Result: Precise, clinically-relevant response
```

### Architecture

```
Streamlit App (app.py)
    ↓ HTTP/SSE
MCP Client (mcp_client.py)
    ↓ Tool calls
MCP Server (mcp_server.py) - Port 8050
    ↓ AI processing
Gemini API
```

### Quick Commands

```bash
# Start MCP Server (Terminal 1)
python mcp_server.py

# Start Streamlit (Terminal 2)
streamlit run app.py

# In UI: Enable "Agentic RAG with MCP Server"
# In Settings: Click "Configure API"
```

### Troubleshooting MCP

**Server not starting:**
```bash
# Check if port 8050 is in use
netstat -ano | findstr :8050

# Kill process if needed
taskkill /PID <process_id> /F
```

**Connection failed:**
- Ensure MCP server is running first
- Check server URL: `http://localhost:8050/sse`
- Verify no firewall blocking port 8050

**Tools not working:**
- Enter API key in sidebar
- Enable Agentic RAG
- Go to Settings → Click "Configure API"
- Check server logs for confirmation

---

## Feedback System

### Features

- **AI-Powered Sentiment Analysis**: Uses Gemini to analyze feedback
- **Email Notifications**: Beautiful HTML emails to stakeholders
- **Analytics**: Track sentiment distribution and ratings
- **Export**: Download feedback data as JSON

### Setup

**1. Enable Feedback System:**
- Go to Settings page
- Check "Enable Feedback Collection"
- System initializes automatically

**2. Configure Email (Optional):**
- Set recipient email and name
- Expand "SMTP Configuration"
- Fill in SMTP details (see Gmail setup below)
- Click "Save SMTP Configuration"

### Using Feedback

**For Users:**
1. Ask a question in chat
2. Review the response
3. Expand "Provide Feedback on this Response"
4. Fill in feedback text and rating (1-5 stars)
5. Click "Submit Feedback"
6. See sentiment analysis results immediately

**What You Get:**
- **Sentiment**: positive/negative/neutral/mixed
- **Confidence**: 0.0-1.0
- **Emotions**: satisfied, frustrated, confused, etc.
- **Satisfaction Score**: 1-5
- **Key Points**: Main feedback points
- **Issues Mentioned**: Problems identified
- **Suggestions**: User recommendations
- **Urgency**: low/medium/high
- **Category**: accuracy/relevance/completeness/usability

### Email Format

Emails include:
- Sentiment badge (color-coded)
- Urgency level and priority
- Complete user interaction (query, response, feedback)
- Sentiment analysis details
- Actionable insights

---

## Gmail Setup for Email

### The Problem

If you see this error:
```
❌ Email failed: (535, b'5.7.8 Username and Password not accepted')
```

Gmail requires an **App Password** instead of your regular password.

### The Solution (3 Steps)

**1. Get Gmail App Password:**
- Go to: https://myaccount.google.com/apppasswords
- Enable 2-Factor Authentication first if needed
- Generate an App Password
- Copy the 16-character password (e.g., `abcd efgh ijkl mnop`)

**2. Configure SMTP in App:**
```
📧 SMTP Configuration
├── SMTP Server: smtp.gmail.com
├── SMTP Port: 587
├── Username/Email: your.email@gmail.com
├── Password: abcdefghijklmnop  ← App Password (remove spaces)
├── ✅ Use TLS
└── From Email: your.email@gmail.com
```

**3. Save and Test:**
- Click "Save SMTP Configuration"
- Submit feedback to test email delivery

### Other Email Providers

**Outlook/Hotmail:**
- Server: `smtp-mail.outlook.com`
- Port: 587
- Use TLS: Yes

**Yahoo Mail:**
- Server: `smtp.mail.yahoo.com`
- Port: 587
- Use TLS: Yes

---

## Technical Stack

- **LLM**: Google Gemini 2.0 Flash-Lite
- **Evaluation**: Custom RAG evaluation system (built from scratch)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Framework**: Streamlit with custom CSS
- **Document Processing**: PyPDF2
- **MCP Server**: FastMCP for agentic RAG tools
- **Memory**: LangChain for conversation history
- **ML Libraries**: scikit-learn for similarity calculations
- **Feedback**: AI-powered sentiment analysis with email notifications

## Project Structure

```
├── app.py                    # Main Streamlit application
├── document_processor.py     # PDF processing and text extraction
├── embedding_generator.py    # Embedding generation utilities
├── vector_database.py        # FAISS vector database operations
├── custom_evaluator.py       # Custom RAG evaluation metrics
├── evaluation_fallback.py    # Fallback evaluation system
├── rate_limiter.py           # API rate limiting
│
├── Advanced Features
├── mcp_server.py             # FastMCP server with 6 clinical trial tools
├── mcp_client.py             # MCP client for server communication
├── prompt_optimizer.py       # Token-efficient prompt construction
├── memory_integration.py     # LangChain memory integration
├── token_counter.py          # Token counting and optimization utilities
├── feedback_system.py        # Feedback collection with sentiment analysis
│
├── Testing
├── test_integration.py       # Integration tests for all features
├── test_memory_history.py    # Memory feature tests
├── test_feedback_system.py   # Feedback system tests
│
└── requirements.txt          # Python dependencies
```

## Evaluation Metrics

Our custom-built evaluation system provides comprehensive RAG assessment:

- **Answer Relevancy**: Combines semantic similarity and LLM judgment
- **Faithfulness**: Extracts and verifies factual claims against context
- **Contextual Relevancy**: Evaluates how well retrieved chunks match query
- **Contextual Recall**: Measures completeness by comparing with expected answers

### Evaluation Methods

1. **Semantic Analysis**: Sentence transformers for embedding-based similarity
2. **LLM Judgment**: Gemini API for nuanced evaluation
3. **Fallback Methods**: Keyword overlap and statistical measures for reliability

---

## Troubleshooting

### Common Issues

**API Overload (503 errors):**
- Wait 2-3 minutes and try again
- Use fewer evaluation metrics
- Try during off-peak hours

**Invalid API Key (400 errors):**
- Verify the key is correct in the sidebar
- Ensure the key has proper permissions

**Rate Limiting (429 errors):**
- Built-in rate limiting should prevent this
- Wait before making more requests

**Out of Memory (Cloud deployment):**
- Reduce MAX_CHUNKS to 500
- Limit file size to 10MB
- Clear cache more frequently

**Email Not Sending:**
- For Gmail: Use App Password, not regular password
- Check SMTP configuration is saved
- Verify recipient email is set
- Check firewall allows SMTP connections

**MCP Server Issues:**
- Start server before Streamlit app
- Check port 8050 is not in use
- Verify API key is configured on server
- Review server logs for errors

### Performance Tips

- Start with fewer evaluation metrics to test the system
- Use Top-K values between 3-7 for optimal performance
- Upload smaller documents initially to test functionality
- Monitor the Evaluation Logs page for performance insights
- Enable only needed advanced features

---

## Configuration

- **API Key**: Enter your Gemini API key in the sidebar
- **Document Upload**: Upload PDFs for analysis
- **Evaluation Settings**: Enable/disable metrics and set thresholds
- **Top-K Settings**: Adjust number of retrieved chunks (1-20)
- **Advanced Features**: Toggle MCP, memory, token optimization
- **Feedback System**: Configure email notifications and recipients

---

## Navigation System

### 💬 Chat & Evaluation
- Interactive chat interface with document Q&A
- Real-time evaluation metrics
- Source attribution and chunk analysis
- Configurable Top-K retrieval settings
- Feedback collection

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
- Feedback system configuration
- System information and statistics

---

## Security & Privacy

- **API Keys**: Stored in session state only (not persisted)
- **Feedback Data**: Stored in memory (session-based)
- **Email**: Uses TLS encryption, App Passwords recommended
- **Documents**: Processed locally, not sent to external services
- **User Data**: Optional email collection, privacy-focused design

---

## License

MIT License - Feel free to use and modify for your projects!

---

## Support

- **Issues**: Open an issue on GitHub
- **Questions**: Check troubleshooting section
- **Contributions**: Pull requests welcome!

---

**Built with ❤️ for clinical trial research and RAG evaluation**
