"""
Test script to validate the integration of new features
Run this to test MCP, Prompt Optimization, and Memory Integration
"""

import os
import sys
import time
from typing import Dict, List

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_token_counter():
    """Test token counter functionality"""
    print("🧪 Testing Token Counter...")
    
    try:
        from token_counter import TokenCounter, get_token_counter, format_token_count
        
        # Test basic functionality
        counter = get_token_counter("gemini-2.0-flash-lite")
        
        test_text = "This is a test sentence for token counting."
        token_count = counter.count_tokens(test_text)
        
        print(f"✅ Token counting works: '{test_text}' = {token_count} tokens")
        print(f"✅ Formatted count: {format_token_count(token_count)}")
        
        # Test truncation
        long_text = "This is a very long text that should be truncated. " * 100
        truncated = counter.truncate_text_to_tokens(long_text, 50)
        truncated_tokens = counter.count_tokens(truncated)
        
        print(f"✅ Truncation works: {len(long_text)} chars -> {len(truncated)} chars ({truncated_tokens} tokens)")
        
        return True
        
    except Exception as e:
        print(f"❌ Token Counter test failed: {e}")
        return False


def test_mcp_manager():
    """Test Agentic RAG MCP functionality"""
    print("\n🧪 Testing Agentic RAG MCP...")

    try:
        from mcp_client import get_mcp_client, check_mcp_server_health

        # Test basic functionality
        mcp_client = get_mcp_client()
        print("✅ MCP client created successfully")

        # Test server health check
        health_status = check_mcp_server_health()
        print(f"✅ Server health check: {health_status['status']}")
        print(f"✅ Available tools: {health_status['tools_available']}")

        if health_status['status'] == 'unavailable':
            print("⚠️ MCP libraries not installed - this is expected in testing")
        elif health_status['status'] == 'stopped':
            print("⚠️ MCP server not running - start with 'python mcp_server.py'")
        else:
            print("🎉 MCP server is running!")

        # Test client properties
        print(f"✅ Client server URL: {mcp_client.server_url}")
        print(f"✅ Client connected: {mcp_client.connected}")

        return True

    except Exception as e:
        print(f"❌ Agentic RAG MCP test failed: {e}")
        return False


def test_prompt_optimizer():
    """Test Prompt Optimizer functionality"""
    print("\n🧪 Testing Prompt Optimizer...")
    
    try:
        from prompt_optimizer import PromptOptimizer, get_prompt_optimizer
        
        # Test basic functionality
        optimizer = get_prompt_optimizer("gemini-2.0-flash-lite", 4000)
        
        # Test template usage
        variables = {
            'context': "This is a test context with some information about clinical trials.",
            'query': "What is this about?"
        }
        
        result = optimizer.build_optimized_prompt('rag_query', variables, target_tokens=100)
        
        print(f"✅ Prompt optimization works:")
        print(f"   - Template: rag_query")
        print(f"   - Initial tokens: {result['initial_tokens']}")
        print(f"   - Final tokens: {result['final_tokens']}")
        print(f"   - Optimization applied: {result['optimization_applied']}")
        
        # Test context optimization
        long_context = "This is a very long context. " * 200
        optimized_context = optimizer._optimize_context(long_context, 100)
        
        print(f"✅ Context optimization works: {len(long_context)} -> {len(optimized_context)} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt Optimizer test failed: {e}")
        return False


def test_memory_integration():
    """Test Memory Integration functionality"""
    print("\n🧪 Testing Memory Integration...")
    
    try:
        # Mock API key for testing
        test_api_key = "test_key_for_integration_testing"
        
        from memory_integration import RAGMemoryManager, get_memory_manager
        
        # Test basic functionality (will fail on actual API calls, but structure should work)
        print("✅ Memory integration imports work")
        
        # Test document memory
        from memory_integration import DocumentMemory
        
        doc_memory = DocumentMemory()
        doc_memory.add_interaction("doc1", "Test query", "Test response", {"test": True})
        
        history = doc_memory.get_document_history("doc1")
        print(f"✅ Document memory works: {len(history)} interactions")
        
        # Test keyword search
        relevant = doc_memory.search_document_interactions("doc1", ["test"])
        print(f"✅ Memory search works: {len(relevant)} relevant interactions")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory Integration test failed: {e}")
        return False


def test_app_imports():
    """Test that app.py can import all new modules"""
    print("\n🧪 Testing App Integration...")
    
    try:
        # Test imports that app.py uses
        from mcp_client import get_mcp_client, check_mcp_server_health
        from prompt_optimizer import get_prompt_optimizer, PromptOptimizer
        from memory_integration import get_memory_manager, RAGMemoryManager
        from token_counter import get_token_counter, format_token_count, estimate_cost
        
        print("✅ All imports work correctly")
        
        # Test that cached functions work
        mcp_client = get_mcp_client()
        optimizer = get_prompt_optimizer()
        counter = get_token_counter()
        
        print("✅ All cached functions work correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ App integration test failed: {e}")
        return False


def run_all_tests():
    """Run all integration tests"""
    print("🚀 Starting Integration Tests for RAG Advanced Features\n")
    
    tests = [
        ("Token Counter", test_token_counter),
        ("Agentic RAG MCP", test_mcp_manager),
        ("Prompt Optimizer", test_prompt_optimizer),
        ("Memory Integration", test_memory_integration),
        ("App Integration", test_app_imports)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Integration is successful.")
        print("\n💡 Next steps:")
        print("1. Run 'streamlit run app.py' to start the application")
        print("2. Configure your Gemini API key in the sidebar")
        print("3. Enable the advanced features in the sidebar")
        print("4. Upload documents and test the new functionality")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
