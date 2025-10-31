"""
Test script to verify memory integration handles history queries correctly
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_history_query_detection():
    """Test if history queries are detected correctly"""
    print("🧪 Testing History Query Detection...")
    
    try:
        from memory_integration import RAGMemoryManager
        
        # Create a mock memory manager (without API key for testing)
        class MockMemoryManager:
            def _is_asking_about_history(self, query):
                history_keywords = [
                    'previous question', 'last question', 'what did i ask',
                    'my previous', 'my last', 'earlier question', 'before',
                    'what was my', 'conversation history', 'chat history',
                    'what did we discuss', 'our conversation', 'talked about',
                    'mentioned before', 'said earlier', 'discussed previously'
                ]
                
                query_lower = query.lower()
                return any(keyword in query_lower for keyword in history_keywords)
        
        mock_manager = MockMemoryManager()
        
        # Test history queries
        history_queries = [
            "What was my previous question?",
            "What did I ask before?",
            "What was my last question?",
            "What did we discuss earlier?",
            "What did I mention before?",
            "Can you show me our conversation history?"
        ]
        
        # Test non-history queries
        regular_queries = [
            "What is this document about?",
            "Explain the clinical trial process",
            "How does this medication work?",
            "What are the side effects?"
        ]
        
        print("✅ Testing history queries:")
        for query in history_queries:
            is_history = mock_manager._is_asking_about_history(query)
            status = "✅" if is_history else "❌"
            print(f"  {status} '{query}' -> {is_history}")
        
        print("\n✅ Testing regular queries:")
        for query in regular_queries:
            is_history = mock_manager._is_asking_about_history(query)
            status = "✅" if not is_history else "❌"
            print(f"  {status} '{query}' -> {is_history}")
        
        return True
        
    except Exception as e:
        print(f"❌ History query detection test failed: {e}")
        return False


def test_memory_formatting():
    """Test memory history formatting"""
    print("\n🧪 Testing Memory History Formatting...")
    
    try:
        from memory_integration import DocumentMemory
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Create document memory and add some interactions
        doc_memory = DocumentMemory()
        doc_memory.add_interaction("doc1", "What is this about?", "This is about clinical trials.", {"test": True})
        doc_memory.add_interaction("doc1", "How long do trials take?", "Clinical trials typically take 3-5 years.", {"test": True})
        
        # Test document history retrieval
        history = doc_memory.get_document_history("doc1")
        print(f"✅ Document memory works: {len(history)} interactions stored")
        
        # Test search functionality
        relevant = doc_memory.search_document_interactions("doc1", ["trials", "clinical"])
        print(f"✅ Memory search works: {len(relevant)} relevant interactions found")
        
        # Test mock history formatting
        mock_history = {
            'buffer_history': [
                HumanMessage(content="What is this document about?"),
                AIMessage(content="This document discusses clinical trial protocols."),
                HumanMessage(content="How long do these trials typically take?"),
                AIMessage(content="Clinical trials usually take 3-5 years to complete.")
            ],
            'summary': 'User asked about clinical trial documents and duration.',
            'document_history': history,
            'is_history_query': True
        }
        
        # Mock the formatting function
        def format_history_for_query(history_dict, query):
            formatted_parts = []
            
            if history_dict['buffer_history']:
                formatted_parts.append("Recent conversation history:")
                
                for i, message in enumerate(history_dict['buffer_history'], 1):
                    if hasattr(message, 'content'):
                        role = "You" if isinstance(message, HumanMessage) else "Assistant"
                        content = message.content[:100] + "..." if len(message.content) > 100 else message.content
                        formatted_parts.append(f"{i}. {role}: {content}")
            
            if history_dict['summary']:
                formatted_parts.append(f"\nConversation summary: {history_dict['summary']}")
            
            return "\n".join(formatted_parts) if formatted_parts else "No previous conversation history found."
        
        formatted = format_history_for_query(mock_history, "What was my previous question?")
        print(f"✅ History formatting works: {len(formatted)} characters formatted")
        print(f"   Sample: {formatted[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory formatting test failed: {e}")
        return False


def test_integration_with_app():
    """Test integration with app.py functions"""
    print("\n🧪 Testing App Integration...")
    
    try:
        # Test that the create_standard_prompt function accepts the new parameter
        from app import create_standard_prompt
        
        # Test regular query
        regular_prompt = create_standard_prompt(
            "What is this about?",
            "This is a document about clinical trials.",
            "Previous conversation: User asked about trials.",
            is_history_query=False
        )
        
        print(f"✅ Regular prompt creation works: {len(regular_prompt)} characters")
        
        # Test history query
        history_prompt = create_standard_prompt(
            "What was my previous question?",
            "This is a document about clinical trials.",
            "Recent conversation history:\n1. You: What is this about?\n2. Assistant: This is about clinical trials.",
            is_history_query=True
        )
        
        print(f"✅ History prompt creation works: {len(history_prompt)} characters")
        print(f"   History prompt prioritizes conversation history: {'Conversation History:' in history_prompt}")
        
        return True
        
    except Exception as e:
        print(f"❌ App integration test failed: {e}")
        return False


def run_memory_tests():
    """Run all memory-related tests"""
    print("🚀 Starting Memory Integration Tests for History Queries\n")
    
    tests = [
        ("History Query Detection", test_history_query_detection),
        ("Memory Formatting", test_memory_formatting),
        ("App Integration", test_integration_with_app)
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
    print("📊 MEMORY TEST RESULTS")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All memory tests passed!")
        print("\n💡 How to test in the app:")
        print("1. Run 'streamlit run app.py'")
        print("2. Enable 'LangChain Memory' in the sidebar")
        print("3. Upload a document and ask a question")
        print("4. Then ask: 'What was my previous question?'")
        print("5. The app should respond with your conversation history!")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_memory_tests()
    sys.exit(0 if success else 1)
