"""
MCP Client for Clinical Trial RAG Application
Handles communication with the FastMCP server and integrates with Gemini function calling
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
import streamlit as st

# Check if MCP client is available
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    import httpx
    import nest_asyncio
    MCP_AVAILABLE = True
    
    # Allow nested event loops for Streamlit compatibility
    nest_asyncio.apply()
    
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️ MCP client not available. Install with: pip install mcp[cli] httpx nest_asyncio")


class MCPAgenticRAGClient:
    """Client for interacting with the Clinical Trial RAG MCP Server"""
    
    def __init__(self, server_url: str = "http://localhost:8050/sse"):
        """
        Initialize MCP client
        
        Args:
            server_url: URL of the MCP server SSE endpoint
        """
        self.server_url = server_url
        self.session = None
        self.session_context = None
        self.sse_context = None
        self.read = None
        self.write = None
        self.available_tools = []
        self.connected = False
        
    async def connect(self) -> bool:
        """
        Connect to the MCP server

        Returns:
            bool: True if connection successful, False otherwise
        """
        if not MCP_AVAILABLE:
            return False

        try:
            # Disconnect first if already connected
            if self.connected:
                await self.disconnect()

            # Connect to MCP server via SSE
            self.sse_context = sse_client(self.server_url)
            self.read, self.write = await self.sse_context.__aenter__()

            # Create session
            self.session_context = ClientSession(self.read, self.write)
            self.session = await self.session_context.__aenter__()

            # Initialize session
            await self.session.initialize()

            # Get available tools
            tools_result = await self.session.list_tools()
            self.available_tools = [tool.name for tool in tools_result.tools]

            self.connected = True
            print(f"✅ Connected to MCP server with {len(self.available_tools)} tools")
            return True

        except Exception as e:
            print(f"Failed to connect to MCP server: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Disconnect from the MCP server"""
        try:
            if self.session_context:
                await self.session_context.__aexit__(None, None, None)
                self.session_context = None
                self.session = None

            if self.sse_context:
                await self.sse_context.__aexit__(None, None, None)
                self.sse_context = None
                self.read = None
                self.write = None

            self.connected = False
            print("🔌 Disconnected from MCP server")
        except Exception as e:
            print(f"Error during disconnect: {e}")
            self.connected = False
    
    async def disconnect(self):
        """Disconnect from the MCP server"""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
            self.connected = False
        except Exception as e:
            print(f"Error disconnecting: {e}")

    async def get_server_health(self) -> Dict[str, Any]:
        """
        Get server health status via MCP tool

        Returns:
            Dictionary with server health information
        """
        if not self.connected:
            return {"status": "disconnected"}

        try:
            result = await self.session.call_tool("server_health_check", arguments={})
            health_data = json.loads(result.content[0].text)
            return health_data
        except Exception as e:
            print(f"Error getting server health: {e}")
            return {"status": "error", "message": str(e)}

    async def configure_gemini_api(self, api_key: str) -> bool:
        """
        Configure Gemini API on the MCP server

        Args:
            api_key: Gemini API key

        Returns:
            bool: True if configuration successful, False otherwise
        """
        if not self.connected:
            return False

        try:
            result = await self.session.call_tool(
                "configure_gemini_api",
                arguments={"api_key": api_key}
            )

            response = json.loads(result.content[0].text)
            return response.get("status") == "success"

        except Exception as e:
            print(f"Error configuring Gemini API: {e}")
            return False
    
    async def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract clinical entities from a query using the MCP server
        
        Args:
            query: The query to analyze
            
        Returns:
            Dictionary of entity types and their values
        """
        if not self.connected:
            return {}
            
        try:
            result = await self.session.call_tool(
                "extract_clinical_entities",
                arguments={"query": query}
            )
            
            # Parse JSON response
            entities = json.loads(result.content[0].text)
            return entities if isinstance(entities, dict) else {}
            
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return {}
    
    async def refine_query(self, original_query: str) -> str:
        """
        Enhance a query for better clinical trial search
        
        Args:
            original_query: The original query to refine
            
        Returns:
            Enhanced query string
        """
        if not self.connected:
            return original_query
            
        try:
            result = await self.session.call_tool(
                "enhance_clinical_query",
                arguments={"original_query": original_query}
            )
            
            # Parse JSON response
            response_data = json.loads(result.content[0].text)
            enhanced = response_data.get('enhanced_query', original_query)
            return enhanced if enhanced else original_query
            
        except Exception as e:
            print(f"Error refining query: {e}")
            return original_query
    
    async def check_relevance(self, question: str, text_chunk: str) -> float:
        """
        Check clinical relevance of a document chunk to a question
        
        Args:
            question: The user's question
            text_chunk: Document chunk to evaluate
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not self.connected:
            return 0.5
            
        try:
            result = await self.session.call_tool(
                "check_clinical_relevance",
                arguments={"question": question, "document_chunk": text_chunk, "trial_phase": "any"}
            )
            
            # Parse JSON response
            response_data = json.loads(result.content[0].text)
            score = response_data.get('relevance_score', 0.5)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"Error checking relevance: {e}")
            return 0.5
    
    def is_server_running(self) -> bool:
        """
        Check if the MCP server is running

        Returns:
            bool: True if server is accessible, False otherwise
        """
        if not MCP_AVAILABLE:
            return False

        try:
            import socket
            # Try to connect to the server port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', 8050))
            sock.close()
            return result == 0
        except:
            return False

    async def answer_clinical_query(self, user_query: str, rag_search_function, gemini_client) -> Dict[str, Any]:
        """
        7-step clinical trial RAG workflow using the 5 MCP tools

        Args:
            user_query: User's clinical trial query
            rag_search_function: Function to search RAG documents
            gemini_client: Gemini client for final answer generation

        Returns:
            Dictionary containing the complete workflow result
        """
        if not self.connected:
            return {"error": "MCP client not connected"}

        try:
            print(f"\n{'='*70}")
            print(f"User Query: {user_query}")
            print('='*70)

            # STEP 1: Validate medical query
            print("\n[STEP 1] Validating medical query...")
            validation_result = await self.session.call_tool(
                "validate_medical_query",
                arguments={"query": user_query}
            )
            validation_data = json.loads(validation_result.content[0].text)
            print(f"Valid: {validation_data.get('is_medical_query', False)}")

            if not validation_data.get('is_medical_query'):
                print("❌ Not a medical query")
                return {
                    "error": "Not a medical query",
                    "validation": validation_data,
                    "final_answer": "This query does not appear to be related to clinical trials or medical research."
                }

            # STEP 2: Extract entities
            print("\n[STEP 2] Extracting clinical entities...")
            entities_result = await self.session.call_tool(
                "extract_clinical_entities",
                arguments={"query": user_query}
            )
            entities_data = json.loads(entities_result.content[0].text)
            print(f"Entities: {json.dumps(entities_data, indent=2)}")

            # STEP 3: Enhance query
            print("\n[STEP 3] Enhancing query...")
            enhancement_result = await self.session.call_tool(
                "enhance_clinical_query",
                arguments={"original_query": user_query}
            )
            enhancement_data = json.loads(enhancement_result.content[0].text)
            enhanced_query = enhancement_data.get("enhanced_query", user_query)
            print(f"Enhanced: {enhanced_query}")

            # STEP 4: Search RAG
            print("\n[STEP 4] Searching RAG...")
            rag_results = rag_search_function(enhanced_query)
            print(f"Found {len(rag_results)} results")

            # STEP 5: Check relevance
            print("\n[STEP 5] Checking clinical relevance...")
            relevant_results = []
            trial_phase = entities_data.get("trial_phases", ["any"])[0] if entities_data.get("trial_phases") else "any"

            for i, result in enumerate(rag_results[:5]):  # Limit to top 5
                relevance_check = await self.session.call_tool(
                    "check_clinical_relevance",
                    arguments={
                        "question": user_query,
                        "document_chunk": result.get('content', result.get('text', '')),
                        "trial_phase": trial_phase
                    }
                )
                relevance_data = json.loads(relevance_check.content[0].text)

                if relevance_data.get('relevant') and relevance_data.get('relevance_score', 0) > 0.5:
                    relevant_results.append(result)
                    print(f"✅ Result {i+1}: Score {relevance_data.get('relevance_score')}")
                else:
                    print(f"❌ Result {i+1}: Not relevant")

            if not relevant_results:
                print("No relevant results")
                return {
                    "validation": validation_data,
                    "entities": entities_data,
                    "enhanced_query": enhanced_query,
                    "final_answer": "No relevant clinical trial documents found for this query."
                }

            # STEP 6: Generate answer with Gemini
            print("\n[STEP 6] Generating answer with Gemini...")
            context = "\n\n".join([f"Source {i+1}: {r.get('content', r.get('text', ''))}" for i, r in enumerate(relevant_results)])

            prompt = f"""You are a clinical trial research assistant. Answer this query using the provided clinical trial documents.

Query: {user_query}

Clinical Context:
{json.dumps(entities_data, indent=2)}

Supporting Documents:
{context}

Provide a clear, evidence-based answer that:
1. Directly answers the question
2. Cites which trial documents support your answer
3. Mentions any limitations or caveats
4. Is appropriate for clinical research professionals"""

            import google.generativeai as genai
            response = gemini_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=1000
                )
            )

            answer = response.text
            print(f"Answer: {answer[:200]}...")

            # STEP 7: Validate answer
            print("\n[STEP 7] Validating answer...")
            validation_check = await self.session.call_tool(
                "validate_clinical_answer",
                arguments={
                    "question": user_query,
                    "answer": answer,
                    "source_documents": context[:500]
                }
            )
            validation_data_final = json.loads(validation_check.content[0].text)
            print(f"Valid: {validation_data_final.get('is_valid')}")
            print(f"Confidence: {validation_data_final.get('confidence')}")

            return {
                "validation": validation_data,
                "entities": entities_data,
                "enhanced_query": enhanced_query,
                "relevant_documents": len(relevant_results),
                "final_answer": answer,
                "answer_validation": validation_data_final
            }

        except Exception as e:
            return {
                "error": f"Clinical query processing failed: {str(e)}",
                "final_answer": f"Error processing query: {str(e)}"
            }


# Streamlit integration functions
@st.cache_resource
def get_mcp_client() -> MCPAgenticRAGClient:
    """Get cached MCP client instance"""
    return MCPAgenticRAGClient()


async def process_query_with_agentic_rag(
    client: MCPAgenticRAGClient,
    query: str,
    document_chunks: List[str]
) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Process a query using agentic RAG with MCP tools
    
    Args:
        client: MCP client instance
        query: User's query
        document_chunks: List of document chunks to process
        
    Returns:
        Tuple of (refined_query, tools_used, metadata)
    """
    tools_used = []
    metadata = {}
    
    if not client.connected:
        return query, tools_used, metadata
    
    try:
        # Step 1: Extract entities from the query
        entities = await client.extract_entities(query)
        if entities:
            tools_used.append("extract_clinical_entities")
            metadata['entities'] = entities
        
        # Step 2: Refine the query for better search
        refined_query = await client.refine_query(query)
        if refined_query != query:
            tools_used.append("enhance_clinical_query")
            metadata['query_refined'] = True
            metadata['original_query'] = query
        
        # Step 3: Check relevance of document chunks (sample first few)
        if document_chunks:
            relevance_scores = []
            sample_chunks = document_chunks[:3]  # Sample first 3 chunks
            
            for chunk in sample_chunks:
                score = await client.check_relevance(query, chunk)
                relevance_scores.append(score)
            
            if relevance_scores:
                tools_used.append("check_clinical_relevance")
                metadata['avg_relevance'] = sum(relevance_scores) / len(relevance_scores)
                metadata['relevance_scores'] = relevance_scores
        
        return refined_query, tools_used, metadata
        
    except Exception as e:
        print(f"Error in agentic RAG processing: {e}")
        return query, tools_used, metadata


def run_async_in_streamlit(coro):
    """
    Run async function in Streamlit environment
    
    Args:
        coro: Coroutine to run
        
    Returns:
        Result of the coroutine
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


# Health check function
def check_mcp_server_health() -> Dict[str, Any]:
    """
    Check MCP server health and return status information
    
    Returns:
        Dictionary with server status information
    """
    if not MCP_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "MCP client libraries not installed",
            "tools_available": 0
        }
    
    client = get_mcp_client()
    
    if client.is_server_running():
        # Try to get tool count by connecting briefly
        try:
            async def get_tool_count():
                temp_client = MCPAgenticRAGClient()
                if await temp_client.connect():
                    tool_count = len(temp_client.available_tools)
                    tools = temp_client.available_tools
                    await temp_client.disconnect()
                    return tool_count, tools
                return 0, []

            tool_count, tools = run_async_in_streamlit(get_tool_count())
            return {
                "status": "running",
                "message": "MCP server is accessible",
                "tools_available": tool_count,
                "tools": tools
            }
        except:
            return {
                "status": "running",
                "message": "MCP server is accessible (tools unknown)",
                "tools_available": 7,  # Known tool count
                "tools": []
            }
    else:
        return {
            "status": "stopped",
            "message": "MCP server is not running",
            "tools_available": 0
        }
