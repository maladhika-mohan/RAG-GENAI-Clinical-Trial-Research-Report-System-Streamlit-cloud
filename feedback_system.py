"""
Feedback System with Sentiment Analysis and Email Notifications
Uses Google Gemini for sentiment analysis and SMTP for email delivery
"""

import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional
import google.generativeai as genai


class FeedbackSystem:
    """Handles user feedback collection, sentiment analysis, and email notifications"""
    
    def __init__(self, gemini_api_key: str, smtp_config: Optional[Dict] = None):
        """
        Initialize feedback system
        
        Args:
            gemini_api_key: Google Gemini API key for sentiment analysis
            smtp_config: SMTP configuration for email (optional)
        """
        # Configure Gemini for sentiment analysis
        genai.configure(api_key=gemini_api_key)
        self.gemini_client = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # SMTP configuration
        self.smtp_config = smtp_config or {}
        
        # Feedback storage
        self.feedback_history = []
    
    def analyze_sentiment(self, feedback_text: str) -> Dict:
        """
        Analyze sentiment of feedback using Gemini
        
        Args:
            feedback_text: User's feedback text
            
        Returns:
            Dict with sentiment analysis results
        """
        try:
            prompt = f"""
            Analyze the sentiment of this user feedback about a RAG system response.
            
            Feedback: "{feedback_text}"
            
            Provide a detailed sentiment analysis in JSON format:
            {{
                "sentiment": "positive/negative/neutral/mixed",
                "confidence": 0.0-1.0,
                "emotions": ["satisfied", "frustrated", "confused", etc.],
                "key_points": ["point1", "point2"],
                "satisfaction_score": 1-5,
                "issues_mentioned": ["issue1", "issue2"],
                "suggestions": ["suggestion1", "suggestion2"],
                "urgency": "low/medium/high",
                "category": "accuracy/relevance/completeness/usability/other"
            }}
            
            Return ONLY valid JSON, no markdown or explanation.
            """
            
            response = self.gemini_client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=800
                )
            )
            
            result_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_text = result_text.strip()
            
            # Parse JSON
            sentiment_data = json.loads(result_text)
            
            return {
                "success": True,
                "sentiment_data": sentiment_data,
                "raw_feedback": feedback_text,
                "analyzed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "raw_feedback": feedback_text
            }
    
    def collect_feedback(self, 
                        query: str,
                        response: str,
                        feedback_text: str,
                        rating: Optional[int] = None,
                        user_email: Optional[str] = None,
                        metadata: Optional[Dict] = None) -> Dict:
        """
        Collect and process user feedback
        
        Args:
            query: Original user query
            response: RAG system response
            feedback_text: User's feedback text
            rating: Optional rating (1-5)
            user_email: Optional user email
            metadata: Additional metadata (sources, evaluation scores, etc.)
            
        Returns:
            Dict with feedback processing results
        """
        # Analyze sentiment
        sentiment_analysis = self.analyze_sentiment(feedback_text)
        
        # Create feedback record
        feedback_record = {
            "feedback_id": f"FB_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response[:500] + "..." if len(response) > 500 else response,
            "feedback_text": feedback_text,
            "rating": rating,
            "user_email": user_email,
            "sentiment_analysis": sentiment_analysis,
            "metadata": metadata or {}
        }
        
        # Store feedback
        self.feedback_history.append(feedback_record)
        
        return feedback_record
    
    def send_feedback_email(self, 
                           feedback_record: Dict,
                           recipient_email: str,
                           recipient_name: str = "Team") -> Dict:
        """
        Send feedback notification email to specific persona
        
        Args:
            feedback_record: Feedback record from collect_feedback()
            recipient_email: Email address to send notification
            recipient_name: Name of recipient (e.g., "Product Manager", "Data Scientist")
            
        Returns:
            Dict with email sending status
        """
        if not self.smtp_config:
            return {
                "success": False,
                "error": "SMTP configuration not provided"
            }
        
        try:
            # Extract sentiment data
            sentiment = feedback_record.get('sentiment_analysis', {}).get('sentiment_data', {})
            sentiment_label = sentiment.get('sentiment', 'unknown')
            satisfaction_score = sentiment.get('satisfaction_score', 'N/A')
            urgency = sentiment.get('urgency', 'low')
            
            # Determine email priority based on sentiment and urgency
            priority_map = {
                'high': '1 (Highest)',
                'medium': '3 (Normal)',
                'low': '5 (Lowest)'
            }
            
            # Create email
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{urgency.upper()} PRIORITY] RAG Feedback: {sentiment_label.title()} - {feedback_record['feedback_id']}"
            msg['From'] = self.smtp_config.get('from_email', 'noreply@rag-system.com')
            msg['To'] = recipient_email
            msg['X-Priority'] = priority_map.get(urgency, '3 (Normal)')
            
            # Create HTML email body
            html_body = self._create_email_html(feedback_record, recipient_name)
            
            # Attach HTML body
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)
            
            # Send email with detailed error handling
            try:
                with smtplib.SMTP(self.smtp_config['smtp_server'], self.smtp_config['smtp_port'], timeout=30) as server:
                    server.set_debuglevel(0)  # Set to 1 for debugging
                    
                    if self.smtp_config.get('use_tls', True):
                        server.starttls()
                    
                    if 'username' in self.smtp_config and 'password' in self.smtp_config:
                        server.login(self.smtp_config['username'], self.smtp_config['password'])
                    
                    server.send_message(msg)
            except smtplib.SMTPAuthenticationError as e:
                return {
                    "success": False,
                    "error": f"Authentication failed. Check your email and password. For Gmail, use App Password not regular password. Error: {str(e)}"
                }
            except smtplib.SMTPException as e:
                return {
                    "success": False,
                    "error": f"SMTP error: {str(e)}"
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Connection error: {str(e)}"
                }
            
            return {
                "success": True,
                "message": f"Feedback email sent to {recipient_email}",
                "feedback_id": feedback_record['feedback_id']
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_email_html(self, feedback_record: Dict, recipient_name: str) -> str:
        """Create HTML email body"""
        sentiment = feedback_record.get('sentiment_analysis', {}).get('sentiment_data', {})
        sentiment_label = sentiment.get('sentiment', 'unknown')
        confidence = sentiment.get('confidence', 0)
        satisfaction_score = sentiment.get('satisfaction_score', 'N/A')
        emotions = sentiment.get('emotions', [])
        key_points = sentiment.get('key_points', [])
        issues = sentiment.get('issues_mentioned', [])
        suggestions = sentiment.get('suggestions', [])
        urgency = sentiment.get('urgency', 'low')
        category = sentiment.get('category', 'other')
        
        # Color coding based on sentiment
        sentiment_colors = {
            'positive': '#10b981',
            'negative': '#ef4444',
            'neutral': '#6b7280',
            'mixed': '#f59e0b'
        }
        sentiment_color = sentiment_colors.get(sentiment_label, '#6b7280')
        
        # Urgency badge color
        urgency_colors = {
            'high': '#ef4444',
            'medium': '#f59e0b',
            'low': '#10b981'
        }
        urgency_color = urgency_colors.get(urgency, '#6b7280')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
                .header h1 {{ margin: 0; font-size: 24px; }}
                .header p {{ margin: 5px 0 0 0; opacity: 0.9; }}
                .badge {{ display: inline-block; padding: 5px 15px; border-radius: 20px; 
                         font-size: 12px; font-weight: bold; color: white; margin: 5px; }}
                .section {{ background: #f9fafb; padding: 20px; border-radius: 8px; 
                           margin-bottom: 20px; border-left: 4px solid {sentiment_color}; }}
                .section h2 {{ margin-top: 0; color: #1f2937; font-size: 18px; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .metric-label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: {sentiment_color}; }}
                .list-item {{ background: white; padding: 10px; margin: 5px 0; 
                             border-radius: 5px; border-left: 3px solid #e5e7eb; }}
                .query-box {{ background: #eff6ff; padding: 15px; border-radius: 8px; 
                             border-left: 4px solid #3b82f6; margin: 10px 0; }}
                .response-box {{ background: #f0fdf4; padding: 15px; border-radius: 8px; 
                                border-left: 4px solid #10b981; margin: 10px 0; }}
                .feedback-box {{ background: #fef3c7; padding: 15px; border-radius: 8px; 
                                border-left: 4px solid #f59e0b; margin: 10px 0; }}
                .footer {{ text-align: center; color: #6b7280; font-size: 12px; 
                          margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; }}
                .alert {{ background: #fee2e2; border: 1px solid #fecaca; color: #991b1b; 
                         padding: 15px; border-radius: 8px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔔 New RAG System Feedback</h1>
                    <p>Hello {recipient_name}, you have received new user feedback</p>
                </div>
                
                <div class="section">
                    <h2>📊 Feedback Summary</h2>
                    <span class="badge" style="background-color: {sentiment_color};">
                        {sentiment_label.upper()}
                    </span>
                    <span class="badge" style="background-color: {urgency_color};">
                        {urgency.upper()} URGENCY
                    </span>
                    <span class="badge" style="background-color: #6366f1;">
                        {category.upper()}
                    </span>
                    
                    <div style="margin-top: 20px;">
                        <div class="metric">
                            <div class="metric-label">Satisfaction Score</div>
                            <div class="metric-value">{satisfaction_score}/5</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Confidence</div>
                            <div class="metric-value">{confidence:.0%}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Rating</div>
                            <div class="metric-value">{feedback_record.get('rating', 'N/A')}/5</div>
                        </div>
                    </div>
                </div>
                
                {"<div class='alert'><strong>⚠️ High Priority:</strong> This feedback requires immediate attention due to " + urgency + " urgency level.</div>" if urgency == 'high' else ""}
                
                <div class="section">
                    <h2>💬 User Interaction</h2>
                    
                    <div class="query-box">
                        <strong>🔍 User Query:</strong><br>
                        {feedback_record['query']}
                    </div>
                    
                    <div class="response-box">
                        <strong>🤖 System Response:</strong><br>
                        {feedback_record['response']}
                    </div>
                    
                    <div class="feedback-box">
                        <strong>📝 User Feedback:</strong><br>
                        {feedback_record['feedback_text']}
                    </div>
                </div>
                
                <div class="section">
                    <h2>🧠 Sentiment Analysis</h2>
                    
                    <p><strong>Detected Emotions:</strong></p>
                    <div>
                        {' '.join([f'<span class="badge" style="background-color: #8b5cf6;">{emotion}</span>' for emotion in emotions])}
                    </div>
                    
                    <p style="margin-top: 20px;"><strong>Key Points:</strong></p>
                    {''.join([f'<div class="list-item">• {point}</div>' for point in key_points])}
                    
                    {f'''
                    <p style="margin-top: 20px;"><strong>Issues Mentioned:</strong></p>
                    {''.join([f'<div class="list-item" style="border-left-color: #ef4444;">⚠️ {issue}</div>' for issue in issues])}
                    ''' if issues else ''}
                    
                    {f'''
                    <p style="margin-top: 20px;"><strong>Suggestions:</strong></p>
                    {''.join([f'<div class="list-item" style="border-left-color: #10b981;">💡 {suggestion}</div>' for suggestion in suggestions])}
                    ''' if suggestions else ''}
                </div>
                
                <div class="section">
                    <h2>📋 Metadata</h2>
                    <p><strong>Feedback ID:</strong> {feedback_record['feedback_id']}</p>
                    <p><strong>Timestamp:</strong> {feedback_record['timestamp']}</p>
                    {f"<p><strong>User Email:</strong> {feedback_record.get('user_email', 'Not provided')}</p>" if feedback_record.get('user_email') else ''}
                </div>
                
                <div class="footer">
                    <p>This is an automated notification from the RAG Feedback System</p>
                    <p>Feedback ID: {feedback_record['feedback_id']} | Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def get_feedback_statistics(self) -> Dict:
        """Get statistics about collected feedback"""
        if not self.feedback_history:
            return {"total_feedback": 0}
        
        total = len(self.feedback_history)
        sentiments = {}
        avg_rating = 0
        rating_count = 0
        
        for feedback in self.feedback_history:
            sentiment_data = feedback.get('sentiment_analysis', {}).get('sentiment_data', {})
            sentiment = sentiment_data.get('sentiment', 'unknown')
            sentiments[sentiment] = sentiments.get(sentiment, 0) + 1
            
            if feedback.get('rating'):
                avg_rating += feedback['rating']
                rating_count += 1
        
        return {
            "total_feedback": total,
            "sentiment_distribution": sentiments,
            "average_rating": avg_rating / rating_count if rating_count > 0 else None,
            "latest_feedback": self.feedback_history[-1]['timestamp'] if self.feedback_history else None
        }
    
    def export_feedback(self, filepath: str = "feedback_export.json"):
        """Export all feedback to JSON file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_history, f, indent=2, ensure_ascii=False)
            return {"success": True, "filepath": filepath, "count": len(self.feedback_history)}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Helper function for Streamlit integration
def get_feedback_system(api_key: str, smtp_config: Optional[Dict] = None):
    """Get or create feedback system instance"""
    return FeedbackSystem(api_key, smtp_config)
