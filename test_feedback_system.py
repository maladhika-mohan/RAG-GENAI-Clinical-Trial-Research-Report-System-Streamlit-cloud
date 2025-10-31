"""
Test script for Feedback System
Run this to test sentiment analysis without Streamlit
"""

import os
from feedback_system import FeedbackSystem

def test_feedback_system():
    """Test the feedback system"""
    
    print("=" * 70)
    print("Testing Feedback System with Sentiment Analysis")
    print("=" * 70)
    
    # Get API key from environment or input
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        api_key = input("\nEnter your Gemini API key: ").strip()
    
    if not api_key:
        print("❌ API key required!")
        return
    
    # Initialize feedback system (without SMTP for testing)
    print("\n📝 Initializing feedback system...")
    feedback_system = FeedbackSystem(api_key)
    print("✅ Feedback system initialized")
    
    # Test cases
    test_cases = [
        {
            "query": "What are the side effects of aspirin?",
            "response": "Aspirin may cause gastrointestinal bleeding, nausea, and stomach upset.",
            "feedback": "Very helpful and accurate! The response covered all major side effects.",
            "rating": 5
        },
        {
            "query": "How does metformin work?",
            "response": "Metformin works by reducing glucose production in the liver.",
            "feedback": "The answer is incomplete. It doesn't mention insulin sensitivity or other mechanisms.",
            "rating": 2
        },
        {
            "query": "What is diabetes?",
            "response": "Diabetes is a metabolic disorder characterized by high blood sugar levels.",
            "feedback": "Good basic explanation but could use more detail about types and causes.",
            "rating": 3
        }
    ]
    
    print("\n" + "=" * 70)
    print("Running Test Cases")
    print("=" * 70)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─' * 70}")
        print(f"Test Case {i}")
        print(f"{'─' * 70}")
        
        print(f"\n📝 Query: {test_case['query']}")
        print(f"🤖 Response: {test_case['response']}")
        print(f"💬 Feedback: {test_case['feedback']}")
        print(f"⭐ Rating: {test_case['rating']}/5")
        
        print("\n🔄 Analyzing sentiment...")
        
        # Collect feedback
        feedback_record = feedback_system.collect_feedback(
            query=test_case['query'],
            response=test_case['response'],
            feedback_text=test_case['feedback'],
            rating=test_case['rating']
        )
        
        # Display results
        if feedback_record['sentiment_analysis']['success']:
            sentiment_data = feedback_record['sentiment_analysis']['sentiment_data']
            
            print("\n✅ Sentiment Analysis Results:")
            print(f"   • Sentiment: {sentiment_data.get('sentiment', 'N/A').upper()}")
            print(f"   • Confidence: {sentiment_data.get('confidence', 0):.0%}")
            print(f"   • Satisfaction Score: {sentiment_data.get('satisfaction_score', 'N/A')}/5")
            print(f"   • Urgency: {sentiment_data.get('urgency', 'N/A').upper()}")
            print(f"   • Category: {sentiment_data.get('category', 'N/A').title()}")
            
            emotions = sentiment_data.get('emotions', [])
            if emotions:
                print(f"   • Emotions: {', '.join(emotions)}")
            
            key_points = sentiment_data.get('key_points', [])
            if key_points:
                print(f"\n   Key Points:")
                for point in key_points:
                    print(f"      - {point}")
            
            issues = sentiment_data.get('issues_mentioned', [])
            if issues:
                print(f"\n   Issues Mentioned:")
                for issue in issues:
                    print(f"      ⚠️ {issue}")
            
            suggestions = sentiment_data.get('suggestions', [])
            if suggestions:
                print(f"\n   Suggestions:")
                for suggestion in suggestions:
                    print(f"      💡 {suggestion}")
        else:
            print(f"\n❌ Sentiment analysis failed: {feedback_record['sentiment_analysis'].get('error')}")
    
    # Display statistics
    print("\n" + "=" * 70)
    print("Feedback Statistics")
    print("=" * 70)
    
    stats = feedback_system.get_feedback_statistics()
    print(f"\n📊 Total Feedback: {stats['total_feedback']}")
    
    if stats.get('sentiment_distribution'):
        print("\n📈 Sentiment Distribution:")
        for sentiment, count in stats['sentiment_distribution'].items():
            print(f"   • {sentiment.title()}: {count}")
    
    if stats.get('average_rating'):
        print(f"\n⭐ Average Rating: {stats['average_rating']:.2f}/5")
    
    # Export feedback
    print("\n" + "=" * 70)
    print("Exporting Feedback")
    print("=" * 70)
    
    export_result = feedback_system.export_feedback("test_feedback_export.json")
    if export_result['success']:
        print(f"\n✅ Exported {export_result['count']} feedback records to {export_result['filepath']}")
    else:
        print(f"\n❌ Export failed: {export_result['error']}")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
    print("\n💡 To test email notifications, configure SMTP in the Streamlit app")
    print("💡 Run: streamlit run app.py")


if __name__ == "__main__":
    test_feedback_system()
