"""
Clinical Trial RAG MCP Server with Agentic Capabilities
Implements FastMCP server with Gemini-powered tools for enhanced RAG operations
"""

import os
import json
import datetime
import asyncio
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Check if FastMCP is available
try:
    from mcp.server.fastmcp import FastMCP
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    print("WARNING: FastMCP not available. Install with: pip install mcp[cli]")

# Setup Gemini client (will be configured dynamically)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_client = None

def configure_gemini(api_key: str):
    """Configure Gemini client with provided API key"""
    global gemini_client
    try:
        genai.configure(api_key=api_key)
        # Use Gemini 2.5 Flash for full function calling support, fallback to 2.0 Flash-Lite
        try:
            gemini_client = genai.GenerativeModel('gemini-2.5-flash')
        except:
            gemini_client = genai.GenerativeModel('gemini-2.0-flash-lite')
        return True
    except Exception as e:
        print(f"Failed to configure Gemini: {e}")
        return False

# Try to configure from environment if available
if GEMINI_API_KEY:
    configure_gemini(GEMINI_API_KEY)
    print("SUCCESS: Gemini configured from environment variables")
else:
    print("WARNING: GEMINI_API_KEY not found in environment - will be configured via Streamlit UI")

# Create MCP server (only if FastMCP is available)
if FASTMCP_AVAILABLE:
    mcp = FastMCP(
        name="Clinical Trial RAG Server",
        host="0.0.0.0",
        port=8050,
    )

    # Configure Gemini API tool
    @mcp.tool()
    def configure_gemini_api(api_key: str) -> str:
        """Configure Gemini API with the provided key."""
        if configure_gemini(api_key):
            return json.dumps({"status": "success", "message": "Gemini API configured successfully"})
        else:
            return json.dumps({"status": "error", "message": "Failed to configure Gemini API"})

    # Clinical Trial MCP Tools
    @mcp.tool()
    def validate_medical_query(query: str) -> str:
        """Check if query is medical/clinical trial related using Gemini"""
        if not gemini_client:
            return json.dumps({"error": "Gemini API not configured"})

        try:
            prompt = f"""
            You are a medical domain validator.

            Check if this query is related to clinical trials or medical research:
            "{query}"

            Return ONLY a JSON object (no markdown, no explanation):
            {{
                "is_medical_query": true/false,
                "medical_terms_found": ["term1", "term2"],
                "query_clarity": "clear/ambiguous/unclear",
                "confidence": 0.0-1.0,
                "recommendations": "suggestions if any"
            }}
            """

            response = gemini_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=500
                )
            )

            result_text = response.text.strip()

            # Remove markdown code blocks if present
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_text = result_text.strip()

            # Validate JSON
            json.loads(result_text)  # Check if valid JSON
            return result_text

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def enhance_clinical_query(original_query: str) -> str:
        """Enhance query for better clinical trial search using Gemini"""
        if not gemini_client:
            return json.dumps({"error": "Gemini API not configured"})

        try:
            prompt = f"""
            You are a clinical trial research query optimizer.

            Improve this query to be more specific for searching clinical trial documents:
            "{original_query}"

            Guidelines:
            - Add medical context
            - Be more specific about what aspect (safety, efficacy, methodology)
            - Keep concise
            - Don't add random years or numbers

            Return ONLY a JSON object:
            {{
                "enhanced_query": "improved query here",
                "improvements": ["improvement1", "improvement2"]
            }}
            """

            response = gemini_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=300
                )
            )

            result_text = response.text.strip()

            # Remove markdown if present
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_text = result_text.strip()

            return result_text

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def extract_clinical_entities(query: str) -> str:
        """Extract clinical entities from query using Gemini"""
        if not gemini_client:
            return json.dumps({"error": "Gemini API not configured"})

        try:
            prompt = f"""
            Extract clinical entities from this medical query.

            Query: "{query}"

            Return ONLY JSON (include only if found):
            {{
                "drug_names": [],
                "medical_conditions": [],
                "adverse_events": [],
                "patient_populations": [],
                "trial_aspects": [],
                "trial_phases": [],
                "endpoints": [],
                "biomarkers": []
            }}
            """

            response = gemini_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=500
                )
            )

            result_text = response.text.strip()

            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_text = result_text.strip()

            return result_text

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def check_clinical_relevance(question: str, document_chunk: str, trial_phase: str = "any") -> str:
        """Check clinical relevance of document using Gemini"""
        if not gemini_client:
            return json.dumps({"error": "Gemini API not configured"})

        try:
            # Truncate long documents
            content_preview = document_chunk[:1000] if len(document_chunk) > 1000 else document_chunk

            prompt = f"""
            You are a clinical trial research evaluator.

            Question: "{question}"
            Trial Phase Expected: {trial_phase}
            Document Preview: "{content_preview}"

            Evaluate relevance considering:
            1. Does it answer the question?
            2. Is it from the correct trial phase?
            3. Is medical context appropriate?
            4. Does it have actionable information?

            Return ONLY JSON:
            {{
                "relevant": true/false,
                "relevance_score": 0.0-1.0,
                "clinical_relevance": 0.0-1.0,
                "reason": "explanation",
                "trial_phase_match": true/false,
                "key_info": "key finding if relevant"
            }}
            """

            response = gemini_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=500
                )
            )

            result_text = response.text.strip()

            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_text = result_text.strip()

            return result_text

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    def validate_clinical_answer(question: str, answer: str, source_documents: str) -> str:
        """Validate clinical answer accuracy using Gemini"""
        if not gemini_client:
            return json.dumps({"error": "Gemini API not configured"})

        try:
            prompt = f"""
            You are a clinical research validator.

            Question: "{question}"
            Answer: "{answer}"
            Supporting Sources: {source_documents}

            Check:
            1. Is answer supported by sources?
            2. Any unsupported claims?
            3. Appropriate medical language?
            4. Any potential safety issues?

            Return ONLY JSON:
            {{
                "is_valid": true/false,
                "confidence": 0.0-1.0,
                "issues": [],
                "recommendations": "if any",
                "evidence_level": "strong/moderate/weak"
            }}
            """

            response = gemini_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=500
                )
            )

            result_text = response.text.strip()

            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_text = result_text.strip()

            return result_text

        except Exception as e:
            return json.dumps({"error": str(e)})



    # Run the server
    def run_server():
        """Run the MCP server"""
        if not FASTMCP_AVAILABLE:
            print("ERROR: Cannot run server: FastMCP not available")
            return
            
        print("Starting Clinical Trial RAG MCP Server...")
        print(f"Server running on http://localhost:8050")
        print("Available tools:")
        print("   - configure_gemini_api - Configure Gemini API key")
        print("   - validate_medical_query - Check if query is medical/clinical trial related")
        print("   - enhance_clinical_query - Enhance query for better clinical trial search")
        print("   - extract_clinical_entities - Extract clinical entities from query")
        print("   - check_clinical_relevance - Check clinical relevance of document")
        print("   - validate_clinical_answer - Validate clinical answer accuracy")
        print("")
        if gemini_client:
            print("SUCCESS: Gemini API configured and ready")
        else:
            print("WARNING: Gemini API not configured - configure via Streamlit UI")
        
        mcp.run(transport="sse")

else:
    def run_server():
        print("ERROR: FastMCP not available. Please install with: pip install mcp[cli]")

if __name__ == "__main__":
    run_server()
