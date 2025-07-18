"""
Streamlit web interface for sentiment analysis system.

This module provides a user-friendly web interface for
interactive sentiment analysis of movie reviews.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from scraper.imdb_spider import IMDbSpider
from nlp.preprocessor import TextPreprocessor
from ml.predictor import SentimentPredictor
from automation.discord_webhook import DiscordWebhook
from utils.helpers import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def _create_bert_summary(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create summary statistics for BERT-only predictions.
    
    Args:
        predictions (List[Dict[str, Any]]): List of prediction results
        
    Returns:
        Dict[str, Any]: Summary statistics
    """
    if not predictions:
        return {}
    
    # Count sentiments
    sentiment_counts = {}
    total_confidence = 0
    
    for pred in predictions:
        sentiment = pred.get('bert_sentiment', 'neutral')
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        total_confidence += pred.get('bert_confidence', 0)
    
    # Calculate percentages
    total_predictions = len(predictions)
    sentiment_percentages = {
        sentiment: (count / total_predictions) * 100 
        for sentiment, count in sentiment_counts.items()
    }
    
    # Find most common sentiment
    most_common = max(sentiment_counts.items(), key=lambda x: x[1])[0] if sentiment_counts else 'neutral'
    
    # Average confidence
    avg_confidence = total_confidence / total_predictions if total_predictions > 0 else 0
    
    return {
        'total_predictions': total_predictions,
        'sentiment_distribution': sentiment_counts,
        'sentiment_percentages': sentiment_percentages,
        'most_common_sentiment': most_common,
        'average_bert_confidence': avg_confidence,
        'average_svm_confidence': 0.0,  # Not available
        'model_agreement_rate': 1.0,  # Always 100% since only one model
        'bert_confidence_mean': avg_confidence,
        'svm_confidence_mean': 0.0
    }

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Movie Sentiment Analysis",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sentiment-positive { color: #28a745; }
    .sentiment-negative { color: #dc3545; }
    .sentiment-neutral { color: #6c757d; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Input method selection
        input_method = st.radio(
            "Input Method:",
            ["Movie Name", "IMDb ID"],
            help="Choose how to specify the movie"
        )
        
        if input_method == "Movie Name":
            # Movie input
            movie_name = st.text_input(
                "Enter Movie Name:",
                placeholder="e.g., The Shawshank Redemption",
                help="Enter the exact movie name as it appears on IMDb"
            )
            movie_id = None
        else:
            # IMDb ID input
            movie_id = st.text_input(
                "Enter IMDb ID:",
                placeholder="e.g., tt0111161",
                help="Enter the IMDb ID (found in the movie's URL)"
            )
            movie_name = None
        
        # Number of reviews
        num_reviews = st.slider(
            "Number of Reviews to Analyze:",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="More reviews provide better accuracy but take longer to process"
        )
        
        # Send Discord notification
        send_discord = st.checkbox(
            "Send Discord Notification",
            value=True,
            help="Send results to Discord webhook"
        )
        
        # Process button
        process_button = st.button(
            "🚀 Analyze Sentiment",
            type="primary",
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        
        # Load model info
        predictor = SentimentPredictor()
        model_loaded = predictor.load_models()
        
        if model_loaded:
            st.success("✅ Models loaded successfully")
        else:
            st.warning("⚠️ Models not found. Using BERT-only analysis.")
            
        st.markdown("---")
        st.markdown("### 📈 Recent Activity")
        
        # Show recent activity (placeholder)
        st.info("No recent activity")
        
    # Main content area
    if process_button and (movie_name or movie_id):
        try:
            # Initialize components
            predictor = SentimentPredictor()
            discord_webhook = DiscordWebhook()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Get movie ID if needed
            if movie_name and not movie_id:
                status_text.text("🔍 Converting movie name to IMDb ID...")
                progress_bar.progress(10)
                
                # Use the same mapping as simple_predict.py
                movie_mapping = {
                    "The Shawshank Redemption": "tt0111161",
                    "The Godfather": "tt0068646",
                    "Pulp Fiction": "tt0110912",
                    "Fight Club": "tt0133093",
                    "Forrest Gump": "tt0109830",
                    "The Matrix": "tt0133093",
                    "Goodfellas": "tt0099685",
                    "The Silence of the Lambs": "tt0102926",
                    "Interstellar": "tt0816692",
                    "The Dark Knight": "tt0468569",
                    "Inception": "tt1375666",
                    "Joker": "tt7286456",
                    "Parasite": "tt6751668",
                    "Avengers: Endgame": "tt4154796",
                    "Black Swan": "tt0947798",
                    "Memento": "tt0209144",
                    "No Country for Old Men": "tt0477348",
                    "Reservoir Dogs": "tt0105236",
                    "Se7en": "tt0114369",
                    "V for Vendetta": "tt0434409",
                    "The Sixth Sense": "tt0167404",
                    "Good Will Hunting": "tt0119217",
                    "Eternal Sunshine of the Spotless Mind": "tt0338013",
                    "Titanic": "tt0120338",
                    "Avatar": "tt0499549",
                    "The Lion King": "tt0110357",
                    "Frozen": "tt2294629",
                    "Toy Story": "tt0114709",
                    "Iron Man": "tt0371746",
                    "The Avengers": "tt0848228",
                    "Black Panther": "tt1825683",
                    "Spider-Man": "tt0145487",
                    "The Batman": "tt1877830",
                    "Wonder Woman": "tt0451279",
                    "Deadpool": "tt1431045",
                    "Logan": "tt3315342",
                    "X-Men": "tt0120903",
                    "Star Wars: Episode IV - A New Hope": "tt0076759",
                    "Star Wars: Episode V - The Empire Strikes Back": "tt0080684",
                    "Star Wars: Episode VI - Return of the Jedi": "tt0086190",
                    "The Lord of the Rings: The Fellowship of the Ring": "tt0120737",
                    "The Lord of the Rings: The Two Towers": "tt0167261",
                    "The Lord of the Rings: The Return of the King": "tt0167260",
                    "The Hobbit: An Unexpected Journey": "tt0903624",
                    "The Hobbit: The Desolation of Smaug": "tt1170358",
                    "The Hobbit: The Battle of the Five Armies": "tt2310332",
                }
                
                movie_id = movie_mapping.get(movie_name)
                if not movie_id:
                    st.error(f"❌ Could not find IMDb ID for '{movie_name}'. Please try using IMDb ID directly.")
                    return
                
                display_name = movie_name
            else:
                display_name = movie_id
            
            # Step 2: Scrape movie reviews
            status_text.text("📝 Scraping reviews...")
            progress_bar.progress(30)
            
            spider = IMDbSpider(movie_id=movie_id)
            spider.scrape_movie_reviews(movie_id, num_reviews)
            
            # Read scraped data from file (similar to simple_predict.py)
            import pandas as pd
            try:
                df = pd.read_csv('data/raw_comments.csv')
                # Filter for the current movie and get the latest reviews
                movie_data = df[df['movie_id'] == movie_id].tail(num_reviews).to_dict('records')
                
                if not movie_data:
                    st.error(f"❌ No reviews found for '{display_name}'. Please check the IMDb ID.")
                    return
                
                # Filter out invalid data (JavaScript code, etc.)
                valid_movie_data = []
                for item in movie_data:
                    text = str(item.get('text', ''))
                    # Skip JavaScript code and other invalid content
                    if (len(text) > 10 and 
                        not text.startswith('window.') and 
                        not text.startswith('ue.count') and
                        not text.isdigit() and
                        'CSMLibrarySize' not in text):
                        valid_movie_data.append(item)
                
                if not valid_movie_data:
                    st.error(f"❌ No valid reviews found for '{display_name}' after filtering.")
                    return
                
                movie_data = valid_movie_data
                logger.info(f"Successfully loaded {len(movie_data)} valid reviews from file")
                
            except FileNotFoundError:
                st.error("❌ No reviews found - scraping may have failed")
                return
            except Exception as e:
                st.error(f"❌ Error reading scraped data: {e}")
                return
                
            # Step 3: Preprocess and predict
            status_text.text("🧠 Processing with BERT...")
            progress_bar.progress(50)
            
            # Extract texts and ratings
            texts = [item['text'] for item in movie_data]
            ratings = [item.get('rating', 0) for item in movie_data]
            
            # Check if models are loaded
            model_loaded = predictor.load_models()
            
            if model_loaded:
                # Use both BERT and SVM
                status_text.text("🤖 Running ML predictions...")
                progress_bar.progress(70)
                predictions = predictor.predict_sentiment(texts, ratings)
            else:
                # Use BERT only
                status_text.text("🧠 Running BERT analysis...")
                progress_bar.progress(70)
                
                # Use BERT analyzer directly
                from nlp.sentiment_analyzer import BertSentimentAnalyzer
                bert_analyzer = BertSentimentAnalyzer()
                bert_results = bert_analyzer.predict(texts)
                
                # Convert to prediction format
                predictions = []
                for i, (text, bert_result) in enumerate(zip(texts, bert_results)):
                    prediction = {
                        'text': text,
                        'cleaned_text': text.lower(),
                        'bert_sentiment': bert_result['label'],
                        'bert_confidence': bert_result['confidence'],
                        'svm_sentiment': 'neutral',  # Default since no SVM
                        'svm_confidence': 0.0,
                        'final_sentiment': bert_result['label'],
                        'rating': ratings[i] if i < len(ratings) else 0.0
                    }
                    predictions.append(prediction)
            
            if not predictions:
                st.error("❌ Failed to make predictions.")
                return
                
            # Step 4: Generate summary
            status_text.text("📊 Generating summary...")
            progress_bar.progress(90)
            
            summary_stats = predictor.get_prediction_summary(predictions) if model_loaded else _create_bert_summary(predictions)
            
            # Step 5: Send Discord notification
            if send_discord:
                status_text.text("📨 Sending Discord notification...")
                try:
                    discord_webhook.send_sentiment_report(display_name, predictions, summary_stats)
                except Exception as e:
                    st.warning(f"⚠️ Discord notification failed: {e}")
                
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Display results
            display_results(display_name, predictions, summary_stats)
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            logger.error(f"Streamlit app error: {e}")
            
            # Only try Discord notification if webhook was initialized
            if send_discord and 'discord_webhook' in locals():
                try:
                    discord_webhook.send_error_notification(str(e), display_name if 'display_name' in locals() else "Unknown")
                except Exception as discord_error:
                    st.warning(f"⚠️ Discord error notification also failed: {discord_error}")
                
    elif process_button and not movie_name and not movie_id:
        st.warning("⚠️ Please enter a movie name or IMDb ID.")
        
    else:
        # Show welcome message
        st.markdown("""
        ## Welcome to Movie Sentiment Analysis! 🎬
        
        This application analyzes sentiment in movie reviews using advanced AI models:
        
        - **BERT Model**: State-of-the-art transformer model for sentiment analysis
        - **Custom SVM Model**: Trained on movie review data for domain-specific accuracy
        - **Ensemble Approach**: Combines both models for optimal results
        
        ### How to use:
        1. Enter a movie name in the sidebar
        2. Adjust the number of reviews to analyze
        3. Click "Analyze Sentiment" to start
        4. View detailed results and visualizations
        
        ### Features:
        - Real-time sentiment analysis
        - Interactive visualizations
        - Discord notifications
        - Download results as CSV
        """)
        
        # Show sample results (placeholder)
        st.markdown("### 📊 Sample Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Positive Reviews", "65%", "↑ 5%")
        with col2:
            st.metric("Negative Reviews", "20%", "↓ 3%")
        with col3:
            st.metric("Neutral Reviews", "15%", "→ 0%")

def display_results(movie_name: str, predictions: List[Dict[str, Any]], 
                  summary_stats: Dict[str, Any]):
    """Display analysis results with visualizations."""
    
    st.markdown(f"## 📊 Results for: {movie_name}")
    st.markdown("---")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total = summary_stats.get('total_predictions', 0)
        st.metric("Total Reviews", total)
        
    with col2:
        positive_pct = summary_stats.get('sentiment_percentages', {}).get('positive', 0) + \
                      summary_stats.get('sentiment_percentages', {}).get('very positive', 0)
        st.metric("Positive", f"{positive_pct:.1f}%")
        
    with col3:
        negative_pct = summary_stats.get('sentiment_percentages', {}).get('negative', 0) + \
                      summary_stats.get('sentiment_percentages', {}).get('very negative', 0)
        st.metric("Negative", f"{negative_pct:.1f}%")
        
    with col4:
        neutral_pct = summary_stats.get('sentiment_percentages', {}).get('neutral', 0)
        st.metric("Neutral", f"{neutral_pct:.1f}%")
    
    # Sentiment distribution chart
    st.markdown("### 📈 Sentiment Distribution")
    
    sentiment_dist = summary_stats.get('sentiment_distribution', {})
    if sentiment_dist:
        fig = px.pie(
            values=list(sentiment_dist.values()),
            names=list(sentiment_dist.keys()),
            title="Sentiment Distribution",
            color_discrete_map={
                'very positive': '#28a745',
                'positive': '#20c997',
                'neutral': '#6c757d',
                'negative': '#fd7e14',
                'very negative': '#dc3545'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model performance
    st.markdown("### 🤖 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bert_conf = summary_stats.get('average_bert_confidence', 0)
        st.metric("BERT Confidence", f"{bert_conf:.3f}")
        
    with col2:
        svm_conf = summary_stats.get('average_svm_confidence', 0)
        st.metric("SVM Confidence", f"{svm_conf:.3f}")
    
    # Top reviews
    st.markdown("### 🏆 Top Reviews")
    
    # Positive reviews
    positive_reviews = [p for p in predictions if p.get('final_sentiment') in ['positive', 'very positive']]
    negative_reviews = [p for p in predictions if p.get('final_sentiment') in ['negative', 'very negative']]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👍 Top Positive Reviews")
        for i, review in enumerate(positive_reviews[:3], 1):
            with st.expander(f"Review {i} - {review.get('final_sentiment', 'neutral').title()}"):
                st.write(review.get('text', '')[:200] + "..." if len(review.get('text', '')) > 200 else review.get('text', ''))
                st.caption(f"Confidence: {review.get('final_sentiment', 'neutral')}")
    
    with col2:
        st.markdown("#### 👎 Top Negative Reviews")
        for i, review in enumerate(negative_reviews[:3], 1):
            with st.expander(f"Review {i} - {review.get('final_sentiment', 'neutral').title()}"):
                st.write(review.get('text', '')[:200] + "..." if len(review.get('text', '')) > 200 else review.get('text', ''))
                st.caption(f"Confidence: {review.get('final_sentiment', 'neutral')}")
    
    # Download results
    st.markdown("### 💾 Download Results")
    
    # Create DataFrame for download
    df = pd.DataFrame(predictions)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"sentiment_analysis_{movie_name.replace(' ', '_')}.csv",
        mime="text/csv"
    )
    
    # Raw data table
    with st.expander("📋 View Raw Data"):
        st.dataframe(df)

if __name__ == "__main__":
    main() 