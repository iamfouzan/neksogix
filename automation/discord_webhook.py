"""
Discord webhook integration for sentiment analysis system.

This module handles automated Discord notifications with
formatted sentiment analysis results and summaries.
"""

import requests
import json
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from config import config

logger = logging.getLogger(__name__)

class DiscordWebhook:
    """
    Handles Discord webhook notifications for sentiment analysis results.
    """
    
    def __init__(self, webhook_url: str = None):
        """
        Initialize Discord webhook.
        
        Args:
            webhook_url (str): Discord webhook URL
        """
        self.webhook_url = webhook_url or config.DISCORD_WEBHOOK_URL
        self.session = requests.Session()
        
    def send_sentiment_report(self, movie_name: str, predictions: List[Dict[str, Any]],
                            summary_stats: Dict[str, Any]) -> bool:
        """
        Send comprehensive sentiment analysis report to Discord.
        
        Args:
            movie_name (str): Name of the analyzed movie
            predictions (List[Dict[str, Any]]): Prediction results
            summary_stats (Dict[str, Any]): Summary statistics
            
        Returns:
            bool: True if message sent successfully
        """
        try:
            if not self.webhook_url:
                logger.warning("Discord webhook URL not configured")
                return False
                
            # Create embed message
            embed = self._create_sentiment_embed(movie_name, predictions, summary_stats)
            
            # Send message
            payload = {
                "embeds": [embed]
            }
            
            response = self.session.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 204:
                logger.info("Discord notification sent successfully")
                return True
            else:
                logger.error(f"Failed to send Discord notification: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
            return False
            
    def _create_sentiment_embed(self, movie_name: str, predictions: List[Dict[str, Any]],
                               summary_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Discord embed for sentiment analysis results.
        
        Args:
            movie_name (str): Movie name
            predictions (List[Dict[str, Any]]): Prediction results
            summary_stats (Dict[str, Any]): Summary statistics
            
        Returns:
            Dict[str, Any]: Discord embed object
        """
        # Get top positive and negative reviews
        positive_reviews = [p for p in predictions if p.get('final_sentiment') in ['positive', 'very positive']]
        negative_reviews = [p for p in predictions if p.get('final_sentiment') in ['negative', 'very negative']]
        
        # Sort by confidence
        positive_reviews.sort(key=lambda x: x.get('final_sentiment', 0), reverse=True)
        negative_reviews.sort(key=lambda x: x.get('final_sentiment', 0))
        
        # Create embed
        embed = {
            "title": f"🎬 Sentiment Analysis Results: {movie_name}",
            "description": f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "color": 0x00ff00,  # Green color
            "fields": []
        }
        
        # Summary statistics
        total_predictions = summary_stats.get('total_predictions', 0)
        sentiment_dist = summary_stats.get('sentiment_distribution', {})
        
        summary_text = f"**Total Reviews Analyzed:** {total_predictions}\n"
        for sentiment, count in sentiment_dist.items():
            percentage = (count / total_predictions * 100) if total_predictions > 0 else 0
            summary_text += f"• {sentiment.title()}: {count} ({percentage:.1f}%)\n"
            
        embed["fields"].append({
            "name": "📊 Summary Statistics",
            "value": summary_text,
            "inline": False
        })
        
        # Top positive reviews
        if positive_reviews:
            positive_text = ""
            for i, review in enumerate(positive_reviews[:3], 1):
                text = review.get('text', '')[:100] + "..." if len(review.get('text', '')) > 100 else review.get('text', '')
                confidence = review.get('final_sentiment', 'neutral')
                positive_text += f"{i}. {text}\n   Sentiment: {confidence}\n\n"
                
            embed["fields"].append({
                "name": "👍 Top Positive Reviews",
                "value": positive_text,
                "inline": False
            })
            
        # Top negative reviews
        if negative_reviews:
            negative_text = ""
            for i, review in enumerate(negative_reviews[:3], 1):
                text = review.get('text', '')[:100] + "..." if len(review.get('text', '')) > 100 else review.get('text', '')
                confidence = review.get('final_sentiment', 'neutral')
                negative_text += f"{i}. {text}\n   Sentiment: {confidence}\n\n"
                
            embed["fields"].append({
                "name": "👎 Top Negative Reviews",
                "value": negative_text,
                "inline": False
            })
            
        # Model performance
        bert_confidence = summary_stats.get('bert_confidence_mean', 0)
        svm_confidence = summary_stats.get('svm_confidence_mean', 0)
        agreement_rate = summary_stats.get('model_agreement_rate', 0)
        
        performance_text = f"**BERT Avg Confidence:** {bert_confidence:.3f}\n"
        performance_text += f"**SVM Avg Confidence:** {svm_confidence:.3f}\n"
        performance_text += f"**Model Agreement Rate:** {agreement_rate:.1%}"
        
        embed["fields"].append({
            "name": "🤖 Model Performance",
            "value": performance_text,
            "inline": False
        })
        
        # Footer
        embed["footer"] = {
            "text": "Sentiment Analysis System"
        }
        
        return embed
        
    def send_error_notification(self, error_message: str, movie_name: str = None) -> bool:
        """
        Send error notification to Discord.
        
        Args:
            error_message (str): Error message
            movie_name (str): Optional movie name
            
        Returns:
            bool: True if message sent successfully
        """
        try:
            if not self.webhook_url:
                return False
                
            embed = {
                "title": "❌ Sentiment Analysis Error",
                "description": f"An error occurred during sentiment analysis",
                "color": 0xff0000,  # Red color
                "fields": [
                    {
                        "name": "Error Details",
                        "value": error_message[:1000],  # Limit length
                        "inline": False
                    }
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            if movie_name:
                embed["fields"].append({
                    "name": "Movie",
                    "value": movie_name,
                    "inline": True
                })
                
            payload = {
                "embeds": [embed]
            }
            
            response = self.session.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            return response.status_code == 204
            
        except Exception as e:
            logger.error(f"Error sending error notification: {e}")
            return False
            
    def send_training_completion(self, training_stats: Dict[str, Any]) -> bool:
        """
        Send training completion notification.
        
        Args:
            training_stats (Dict[str, Any]): Training statistics
            
        Returns:
            bool: True if message sent successfully
        """
        try:
            if not self.webhook_url:
                return False
                
            embed = {
                "title": "🎯 Model Training Completed",
                "description": "Custom SVM model training has finished",
                "color": 0x00ff00,  # Green color
                "fields": [
                    {
                        "name": "Training Statistics",
                        "value": f"**Samples:** {training_stats.get('n_samples', 0)}\n"
                                f"**Features:** {training_stats.get('n_features', 0)}\n"
                                f"**CV F1 Score:** {training_stats.get('cv_mean', 0):.4f}\n"
                                f"**Best Parameters:** {str(training_stats.get('best_params', {}))}",
                        "inline": False
                    }
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            payload = {
                "embeds": [embed]
            }
            
            response = self.session.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            return response.status_code == 204
            
        except Exception as e:
            logger.error(f"Error sending training notification: {e}")
            return False
            
    def test_webhook(self) -> bool:
        """
        Test Discord webhook connectivity.
        
        Returns:
            bool: True if webhook is working
        """
        try:
            if not self.webhook_url:
                logger.warning("Discord webhook URL not configured")
                return False
                
            embed = {
                "title": "🧪 Webhook Test",
                "description": "Discord webhook is working correctly",
                "color": 0x0099ff,  # Blue color
                "timestamp": datetime.now().isoformat()
            }
            
            payload = {
                "embeds": [embed]
            }
            
            response = self.session.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 204:
                logger.info("Discord webhook test successful")
                return True
            else:
                logger.error(f"Discord webhook test failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Discord webhook test error: {e}")
            return False 