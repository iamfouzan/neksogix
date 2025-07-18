"""
Main application for sentiment analysis system.

This module provides the main entry point for training and prediction modes,
orchestrating the entire sentiment analysis pipeline.
"""

import argparse
import logging
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from config import config
from utils.helpers import setup_logging
from ml.data_preparation import DataPreparation
from ml.trainer import ModelTrainer
from ml.predictor import SentimentPredictor
from ml.model_evaluation import ModelEvaluator
from scraper.imdb_spider import IMDbSpider
from automation.discord_webhook import DiscordWebhook
from database.connection import get_db_session
from database.models import Movie, Comment, Prediction

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class SentimentAnalysisPipeline:
    """
    Main pipeline for sentiment analysis system.
    """
    
    def __init__(self):
        """Initialize the sentiment analysis pipeline."""
        self.data_prep = DataPreparation()
        self.trainer = ModelTrainer()
        self.predictor = SentimentPredictor()
        self.evaluator = ModelEvaluator()
        self.discord_webhook = DiscordWebhook()
        
    def _create_spider(self, movie_name: str = None, movie_id: str = None):
        """Create IMDbSpider instance with proper parameters."""
        if movie_name:
            return IMDbSpider(movie_name=movie_name)
        elif movie_id:
            return IMDbSpider(movie_id=movie_id)
        else:
            raise ValueError("Either movie_name or movie_id must be provided")
            
    def train_mode(self) -> Dict[str, Any]:
        """
        Execute training mode to collect data and train models.
        
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        try:
            logger.info("Starting training mode...")
            
            # Step 1: Collect training data
            logger.info("Collecting training data...")
            training_df = self.data_prep.collect_training_data(min_comments_per_movie=1)
            
            if training_df.empty:
                logger.error("No training data collected - all movies failed to scrape")
                return {"error": "No training data collected - all movies failed to scrape. Please check your internet connection and try again."}
            
            if len(training_df) < 10:
                logger.warning(f"Only collected {len(training_df)} samples, which is below the minimum threshold of 10")
                return {"error": f"Insufficient training data: only {len(training_df)} samples collected. Need at least 10 samples to train the model."}
                
            # Step 2: Validate data
            if not self.data_prep.validate_data(training_df):
                logger.error("Training data validation failed")
                return {"error": "Training data validation failed - data quality issues detected"}
                
            # Step 3: Prepare features
            logger.info("Preparing features...")
            features, labels = self.data_prep.prepare_features(training_df)
            
            # Step 4: Split dataset
            X_train, X_test, y_train, y_test = self.data_prep.split_dataset(features, labels)
            
            # Step 5: Train model
            logger.info("Training SVM model...")
            training_results = self.trainer.train_model(X_train, y_train)
            
            # Step 6: Evaluate model
            logger.info("Evaluating model...")
            evaluation_results = self.trainer.evaluate_model(X_test, y_test)
            
            # Step 7: Save model
            self.trainer.save_model()
            
            # Step 8: Create evaluation plots
            self.evaluator.create_evaluation_plots(training_df.to_dict('records'))
            
            # Step 9: Save evaluation report
            combined_results = {
                "training": training_results,
                "evaluation": evaluation_results,
                "data_stats": {
                    "total_samples": len(training_df),
                    "sentiment_distribution": training_df['bert_sentiment'].value_counts().to_dict()
                }
            }
            
            self.evaluator.save_evaluation_report(combined_results)
            
            # Step 10: Send Discord notification
            self.discord_webhook.send_training_completion(training_results)
            
            logger.info("Training mode completed successfully")
            return combined_results
            
        except Exception as e:
            logger.error(f"Training mode failed: {e}")
            return {"error": str(e)}
            
    def _get_movie_id(self, movie_name: str) -> Optional[str]:
        """
        Get IMDb ID for a movie name using predefined mapping.
        
        Args:
            movie_name (str): Name of the movie to search for
            
        Returns:
            Optional[str]: IMDb ID if found, None otherwise
        """
        # Predefined mapping of popular movies
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
        
        return movie_mapping.get(movie_name)

    def predict_mode(self, movie_name: str, num_reviews: int = 50) -> Dict[str, Any]:
        """
        Execute prediction mode for a single movie.
        
        Args:
            movie_name (str): Name of the movie to analyze
            num_reviews (int): Number of reviews to analyze
            
        Returns:
            Dict[str, Any]: Prediction results and summary
        """
        try:
            logger.info(f"Starting prediction mode for: {movie_name}")
            
            # Step 1: Load models
            if not self.predictor.load_models():
                logger.error("Failed to load models")
                return {"error": "Models not loaded"}
            
            # Step 2: Get movie ID
            movie_id = self._get_movie_id(movie_name)
            if not movie_id:
                logger.error(f"Could not find IMDb ID for movie: {movie_name}")
                return {"error": f"Could not find IMDb ID for movie: {movie_name}. Please check the movie name and try again."}
            
            logger.info(f"Found IMDb ID: {movie_id} for movie: {movie_name}")
                
            # Step 3: Scrape movie reviews using the movie ID
            logger.info("Scraping movie reviews...")
            spider = IMDbSpider(movie_id=movie_id)
            movie_data = spider.scrape_movie_reviews(movie_id, num_reviews)
            
            if not movie_data:
                logger.error(f"No reviews found for {movie_name}")
                return {"error": f"No reviews found for {movie_name}"}
                
            # Step 4: Make predictions
            logger.info("Making predictions...")
            texts = [item['text'] for item in movie_data]
            ratings = [item.get('rating', 0) for item in movie_data]
            
            predictions = self.predictor.predict_sentiment(texts, ratings)
            
            if not predictions:
                logger.error("Failed to make predictions")
                return {"error": "Failed to make predictions"}
                
            # Step 5: Generate summary
            summary_stats = self.predictor.get_prediction_summary(predictions)
            
            # Step 6: Save to database
            self._save_to_database(movie_name, movie_data, predictions)
            
            # Step 7: Send Discord notification
            self.discord_webhook.send_sentiment_report(movie_name, predictions, summary_stats)
            
            # Step 8: Save results to CSV
            self._save_results_to_csv(movie_name, predictions)
            
            logger.info("Prediction mode completed successfully")
            
            return {
                "movie_name": movie_name,
                "movie_id": movie_id,
                "predictions": predictions,
                "summary": summary_stats,
                "total_reviews": len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Prediction mode failed: {e}")
            self.discord_webhook.send_error_notification(str(e), movie_name)
            return {"error": str(e)}
            
    def _save_to_database(self, movie_name: str, movie_data: List[Dict[str, Any]], 
                         predictions: List[Dict[str, Any]]) -> None:
        """Save results to database."""
        try:
            with get_db_session() as session:
                # Create or get movie
                movie = session.query(Movie).filter(Movie.name == movie_name).first()
                if not movie:
                    movie = Movie(name=movie_name, processed_at=datetime.now())
                    session.add(movie)
                    session.commit()
                    session.refresh(movie)
                
                # Save comments and predictions
                for i, (data, pred) in enumerate(zip(movie_data, predictions)):
                    # Create comment
                    comment = Comment(
                        movie_id=movie.id,
                        text=data['text'],
                        cleaned_text=pred.get('cleaned_text', ''),
                        username=data.get('username', ''),
                        rating=data.get('rating', 0),
                        date=data.get('date', datetime.now())
                    )
                    session.add(comment)
                    session.commit()
                    session.refresh(comment)
                    
                    # Create prediction
                    prediction = Prediction(
                        comment_id=comment.id,
                        bert_sentiment=pred.get('bert_sentiment', 'neutral'),
                        bert_confidence=pred.get('bert_confidence', 0),
                        svm_sentiment=pred.get('svm_sentiment', 'neutral'),
                        svm_confidence=pred.get('svm_confidence', 0),
                        final_sentiment=pred.get('final_sentiment', 'neutral'),
                        created_at=datetime.now()
                    )
                    session.add(prediction)
                
                session.commit()
                logger.info(f"Saved {len(predictions)} predictions to database")
                
        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
            
    def _save_results_to_csv(self, movie_name: str, predictions: List[Dict[str, Any]]) -> None:
        """Save results to CSV file."""
        try:
            import pandas as pd
            
            df = pd.DataFrame(predictions)
            filename = f"data/output_{movie_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            df.to_csv(filename, index=False)
            logger.info(f"Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Sentiment Analysis System")
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        required=True,
        help="Mode to run: train (collect data and train models) or predict (analyze movie)"
    )
    parser.add_argument(
        "--movie",
        type=str,
        help="Movie name for prediction mode"
    )
    parser.add_argument(
        "--reviews",
        type=int,
        default=50,
        help="Number of reviews to analyze (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SentimentAnalysisPipeline()
    
    if args.mode == "train":
        logger.info("Running in training mode...")
        results = pipeline.train_mode()
        
        if "error" in results:
            logger.error(f"Training failed: {results['error']}")
            sys.exit(1)
        else:
            logger.info("Training completed successfully")
            print("Training completed successfully!")
            
    elif args.mode == "predict":
        if not args.movie:
            logger.error("Movie name is required for prediction mode")
            sys.exit(1)
            
        logger.info(f"Running in prediction mode for: {args.movie}")
        results = pipeline.predict_mode(args.movie, args.reviews)
        
        if "error" in results:
            logger.error(f"Prediction failed: {results['error']}")
            sys.exit(1)
        else:
            logger.info("Prediction completed successfully")
            print(f"Analysis completed for {args.movie}!")
            print(f"Total reviews analyzed: {results['total_reviews']}")

if __name__ == "__main__":
    main() 