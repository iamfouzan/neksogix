"""
Prediction utilities for sentiment analysis system.

This module handles loading trained models and making predictions
on new text data with confidence scores.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
import joblib
import os

from config import config
from nlp.preprocessor import TextPreprocessor
from nlp.sentiment_analyzer import BertSentimentAnalyzer

logger = logging.getLogger(__name__)

class SentimentPredictor:
    """
    Handles sentiment predictions using trained models.
    """
    
    def __init__(self):
        """Initialize the sentiment predictor."""
        self.preprocessor = TextPreprocessor()
        self.bert_analyzer = BertSentimentAnalyzer()
        self.svm_model = None
        self.tfidf_vectorizer = None
        self.scaler = None
        self.label_mapping = {
            0: 'very negative',
            1: 'negative',
            2: 'neutral', 
            3: 'positive',
            4: 'very positive'
        }
        
    def load_models(self) -> bool:
        """
        Load trained models and components.
        
        Returns:
            bool: True if loading successful
        """
        try:
            # Load SVM model
            if os.path.exists(config.MODEL_PATH):
                self.svm_model = joblib.load(config.MODEL_PATH)
                logger.info("SVM model loaded successfully")
            else:
                logger.warning("SVM model file not found")
                return False
                
            # Load TF-IDF vectorizer
            if os.path.exists(config.TFIDF_VECTORIZER_FILE):
                self.tfidf_vectorizer = joblib.load(config.TFIDF_VECTORIZER_FILE)
                logger.info("TF-IDF vectorizer loaded successfully")
            else:
                logger.warning("TF-IDF vectorizer file not found")
                return False
                
            # Load scaler
            if os.path.exists(config.SCALER_PATH):
                self.scaler = joblib.load(config.SCALER_PATH)
                logger.info("Scaler loaded successfully")
            else:
                logger.warning("Scaler file not found")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
            
    def predict_sentiment(self, texts: List[str], ratings: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a list of texts using both BERT and SVM.
        
        Args:
            texts (List[str]): List of input texts
            ratings (Optional[List[float]]): Optional ratings for each text
            
        Returns:
            List[Dict[str, Any]]: List of prediction results
        """
        if not self.svm_model or not self.tfidf_vectorizer or not self.scaler:
            logger.error("Models not loaded. Call load_models() first.")
            return []
            
        try:
            results = []
            
            # Preprocess texts
            cleaned_texts = [self.preprocessor.preprocess_text(text) for text in texts]
            
            # Get BERT predictions
            bert_results = self.bert_analyzer.predict(texts)
            
            # Prepare features for SVM
            features = self._prepare_features(cleaned_texts, ratings)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get SVM predictions
            svm_predictions = self.svm_model.predict(features_scaled)
            svm_probabilities = self.svm_model.predict_proba(features_scaled)
            
            # Combine results
            for i, (text, cleaned_text) in enumerate(zip(texts, cleaned_texts)):
                bert_result = bert_results[i] if i < len(bert_results) else None
                svm_pred = svm_predictions[i] if i < len(svm_predictions) else None
                svm_proba = svm_probabilities[i] if i < len(svm_probabilities) else None
                
                # Get confidence scores
                bert_confidence = bert_result['confidence'] if bert_result else 0.0
                svm_confidence = max(svm_proba) if svm_proba is not None else 0.0
                
                # Determine final sentiment (ensemble approach)
                final_sentiment = self._ensemble_sentiment(
                    bert_result['label'] if bert_result else 'neutral',
                    self.label_mapping.get(svm_pred, 'neutral'),
                    bert_confidence,
                    svm_confidence
                )
                
                result = {
                    'text': text,
                    'cleaned_text': cleaned_text,
                    'bert_sentiment': bert_result['label'] if bert_result else 'neutral',
                    'bert_confidence': bert_confidence,
                    'svm_sentiment': self.label_mapping.get(svm_pred, 'neutral'),
                    'svm_confidence': svm_confidence,
                    'final_sentiment': final_sentiment,
                    'rating': ratings[i] if ratings and i < len(ratings) else 0.0
                }
                results.append(result)
                
            logger.info(f"Predicted sentiment for {len(texts)} texts")
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []
            
    def _prepare_features(self, cleaned_texts: List[str], ratings: Optional[List[float]] = None) -> np.ndarray:
        """
        Prepare features for SVM prediction.
        
        Args:
            cleaned_texts (List[str]): Preprocessed texts
            ratings (Optional[List[float]]): Optional ratings
            
        Returns:
            np.ndarray: Feature matrix
        """
        # TF-IDF features
        text_features = self.tfidf_vectorizer.transform(cleaned_texts).toarray()
        
        # Additional features
        text_length = np.array([len(text) for text in cleaned_texts]).reshape(-1, 1)
        
        if ratings:
            rating_features = np.array(ratings).reshape(-1, 1)
        else:
            rating_features = np.zeros((len(cleaned_texts), 1))
            
        # Default BERT confidence (will be updated in predict_sentiment)
        bert_confidence = np.ones((len(cleaned_texts), 1)) * 0.5
        
        # Combine features
        features = np.hstack([
            text_features,
            text_length,
            rating_features,
            bert_confidence
        ])
        
        return features
        
    def _ensemble_sentiment(self, bert_sentiment: str, svm_sentiment: str,
                           bert_confidence: float, svm_confidence: float) -> str:
        """
        Combine BERT and SVM predictions using confidence-weighted ensemble.
        
        Args:
            bert_sentiment (str): BERT prediction
            svm_sentiment (str): SVM prediction
            bert_confidence (float): BERT confidence score
            svm_confidence (float): SVM confidence score
            
        Returns:
            str: Final ensemble prediction
        """
        # If one model has much higher confidence, use its prediction
        confidence_threshold = 0.2
        
        if abs(bert_confidence - svm_confidence) > confidence_threshold:
            return bert_sentiment if bert_confidence > svm_confidence else svm_sentiment
            
        # If confidences are similar, use weighted voting
        sentiment_scores = {
            'very negative': 0,
            'negative': 1,
            'neutral': 2,
            'positive': 3,
            'very positive': 4
        }
        
        bert_score = sentiment_scores.get(bert_sentiment, 2)
        svm_score = sentiment_scores.get(svm_sentiment, 2)
        
        # Weighted average
        weighted_score = (bert_score * bert_confidence + svm_score * svm_confidence) / (bert_confidence + svm_confidence)
        
        # Map back to sentiment
        if weighted_score < 1.5:
            return 'very negative'
        elif weighted_score < 2.5:
            return 'negative'
        elif weighted_score < 3.5:
            return 'neutral'
        elif weighted_score < 4.5:
            return 'positive'
        else:
            return 'very positive'
            
    def predict_batch(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a batch of data with metadata.
        
        Args:
            data (List[Dict[str, Any]]): List of data dictionaries with 'text' key
            
        Returns:
            List[Dict[str, Any]]: Prediction results with metadata
        """
        texts = [item['text'] for item in data]
        ratings = [item.get('rating', 0.0) for item in data]
        
        predictions = self.predict_sentiment(texts, ratings)
        
        # Add metadata back to results
        for i, prediction in enumerate(predictions):
            if i < len(data):
                prediction.update({
                    'username': data[i].get('username', ''),
                    'date': data[i].get('date', ''),
                    'movie_id': data[i].get('movie_id', None)
                })
                
        return predictions
        
    def get_prediction_summary(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics for predictions.
        
        Args:
            predictions (List[Dict[str, Any]]): List of predictions
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        if not predictions:
            return {}
            
        # Count sentiments
        sentiment_counts = {}
        bert_sentiment_counts = {}
        svm_sentiment_counts = {}
        
        for pred in predictions:
            # Final sentiment
            final_sentiment = pred.get('final_sentiment', 'neutral')
            sentiment_counts[final_sentiment] = sentiment_counts.get(final_sentiment, 0) + 1
            
            # BERT sentiment
            bert_sentiment = pred.get('bert_sentiment', 'neutral')
            bert_sentiment_counts[bert_sentiment] = bert_sentiment_counts.get(bert_sentiment, 0) + 1
            
            # SVM sentiment
            svm_sentiment = pred.get('svm_sentiment', 'neutral')
            svm_sentiment_counts[svm_sentiment] = svm_sentiment_counts.get(svm_sentiment, 0) + 1
            
        # Calculate percentages
        total = len(predictions)
        sentiment_percentages = {k: (v/total)*100 for k, v in sentiment_counts.items()}
        
        # Average confidence scores
        avg_bert_confidence = np.mean([p.get('bert_confidence', 0) for p in predictions])
        avg_svm_confidence = np.mean([p.get('svm_confidence', 0) for p in predictions])
        
        return {
            'total_predictions': total,
            'sentiment_distribution': sentiment_counts,
            'sentiment_percentages': sentiment_percentages,
            'bert_sentiment_distribution': bert_sentiment_counts,
            'svm_sentiment_distribution': svm_sentiment_counts,
            'average_bert_confidence': avg_bert_confidence,
            'average_svm_confidence': avg_svm_confidence
        } 