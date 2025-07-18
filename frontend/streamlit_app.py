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
from scraper.imdb_spider import ImdbSpider
from nlp.preprocessor import TextPreprocessor
from ml.predictor import SentimentPredictor
from automation.discord_webhook import DiscordWebhook
from utils.helpers import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

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
        
        # Movie input
        movie_name = st.text_input(
            "Enter Movie Name:",
            placeholder="e.g., The Shawshank Redemption",
            help="Enter the exact movie name as it appears on IMDb"
        )
        
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
            st.error("❌ Models not found. Please train models first.")
            
        st.markdown("---")
        st.markdown("### 📈 Recent Activity")
        
        # Show recent activity (placeholder)
        st.info("No recent activity")
        
    # Main content area
    if process_button and movie_name:
        try:
            # Initialize components
            spider = ImdbSpider()
            predictor = SentimentPredictor()
            discord_webhook = DiscordWebhook()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Search and scrape movie reviews
            status_text.text("🔍 Searching for movie...")
            progress_bar.progress(10)
            
            movie_data = spider.scrape_movie_reviews(movie_name, num_reviews)
            
            if not movie_data:
                st.error(f"❌ No reviews found for '{movie_name}'. Please check the movie name.")
                return
                
            status_text.text("📝 Scraping reviews...")
            progress_bar.progress(30)
            
            # Step 2: Preprocess and predict
            status_text.text("🧠 Processing with BERT...")
            progress_bar.progress(50)
            
            # Extract texts and ratings
            texts = [item['text'] for item in movie_data]
            ratings = [item.get('rating', 0) for item in movie_data]
            
            # Make predictions
            predictions = predictor.predict_sentiment(texts, ratings)
            
            if not predictions:
                st.error("❌ Failed to make predictions. Please check if models are loaded.")
                return
                
            status_text.text("🤖 Running ML predictions...")
            progress_bar.progress(70)
            
            # Step 3: Generate summary
            status_text.text("📊 Generating summary...")
            progress_bar.progress(90)
            
            summary_stats = predictor.get_prediction_summary(predictions)
            
            # Step 4: Send Discord notification
            if send_discord:
                status_text.text("📨 Sending Discord notification...")
                discord_webhook.send_sentiment_report(movie_name, predictions, summary_stats)
                
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Display results
            display_results(movie_name, predictions, summary_stats)
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            logger.error(f"Streamlit app error: {e}")
            
            if send_discord:
                discord_webhook.send_error_notification(str(e), movie_name)
                
    elif process_button and not movie_name:
        st.warning("⚠️ Please enter a movie name.")
        
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