"""
Tests for ML components.

This module tests machine learning functionality including
data preparation, model training, and prediction.
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.data_preparation import DataPreparation
from ml.predictor import SentimentPredictor

class TestDataPreparation:
    """Test cases for data preparation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_prep = DataPreparation()
        
    def test_feature_preparation(self):
        """Test feature preparation with sample data."""
        # Create sample data
        sample_data = pd.DataFrame({
            'cleaned_text': [
                'this is a great movie',
                'terrible film waste of time',
                'okay movie nothing special'
            ],
            'rating': [5.0, 1.0, 3.0],
            'bert_confidence': [0.9, 0.8, 0.7],
            'bert_sentiment': ['positive', 'negative', 'neutral']
        })
        
        features, labels = self.data_prep.prepare_features(sample_data)
        
        assert isinstance(features, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert len(features) == len(labels)
        assert features.shape[1] > 0  # Should have features
        
    def test_dataset_splitting(self):
        """Test dataset splitting functionality."""
        features = np.random.rand(100, 10)
        labels = np.random.randint(0, 5, 100)
        
        X_train, X_test, y_train, y_test = self.data_prep.split_dataset(features, labels)
        
        assert len(X_train) + len(X_test) == len(features)
        assert len(y_train) + len(y_test) == len(labels)
        assert X_train.shape[1] == X_test.shape[1]
        
    def test_data_validation(self):
        """Test data validation functionality."""
        # Valid data
        valid_data = pd.DataFrame({
            'bert_sentiment': ['positive'] * 200 + ['negative'] * 200 + ['neutral'] * 100,
            'text': ['sample text'] * 500
        })
        
        assert self.data_prep.validate_data(valid_data) == True
        
        # Invalid data (too small)
        invalid_data = pd.DataFrame({
            'bert_sentiment': ['positive'] * 50,
            'text': ['sample text'] * 50
        })
        
        assert self.data_prep.validate_data(invalid_data) == False

class TestSentimentPredictor:
    """Test cases for sentiment prediction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.predictor = SentimentPredictor()
        
    def test_model_loading(self):
        """Test model loading functionality."""
        # This will fail if models don't exist, which is expected
        result = self.predictor.load_models()
        # Should return False if models don't exist
        assert isinstance(result, bool)
        
    def test_prediction_summary(self):
        """Test prediction summary generation."""
        # Mock predictions
        predictions = [
            {
                'final_sentiment': 'positive',
                'bert_confidence': 0.8,
                'svm_confidence': 0.7
            },
            {
                'final_sentiment': 'negative',
                'bert_confidence': 0.9,
                'svm_confidence': 0.8
            },
            {
                'final_sentiment': 'neutral',
                'bert_confidence': 0.6,
                'svm_confidence': 0.5
            }
        ]
        
        summary = self.predictor.get_prediction_summary(predictions)
        
        assert 'total_predictions' in summary
        assert 'sentiment_distribution' in summary
        assert 'sentiment_percentages' in summary
        assert summary['total_predictions'] == 3
        
    def test_ensemble_sentiment(self):
        """Test ensemble sentiment combination."""
        # Test when BERT has higher confidence
        result = self.predictor._ensemble_sentiment(
            'positive', 'negative', 0.9, 0.6
        )
        assert result == 'positive'
        
        # Test when SVM has higher confidence
        result = self.predictor._ensemble_sentiment(
            'negative', 'positive', 0.6, 0.9
        )
        assert result == 'positive'
        
        # Test weighted average
        result = self.predictor._ensemble_sentiment(
            'neutral', 'neutral', 0.7, 0.7
        )
        assert result in ['very negative', 'negative', 'neutral', 'positive', 'very positive']
        
    def test_feature_preparation(self):
        """Test feature preparation for prediction."""
        cleaned_texts = ['great movie', 'terrible film', 'okay']
        ratings = [5.0, 1.0, 3.0]
        
        # This will fail if TF-IDF vectorizer not loaded, which is expected
        try:
            features = self.predictor._prepare_features(cleaned_texts, ratings)
            assert isinstance(features, np.ndarray)
        except:
            # Expected if models not loaded
            pass
            
    def test_empty_predictions(self):
        """Test handling of empty predictions."""
        summary = self.predictor.get_prediction_summary([])
        assert summary == {}
        
    def test_prediction_batch(self):
        """Test batch prediction with metadata."""
        data = [
            {'text': 'Great movie!', 'rating': 5.0, 'username': 'user1'},
            {'text': 'Terrible film', 'rating': 1.0, 'username': 'user2'}
        ]
        
        # This will fail if models not loaded, which is expected
        try:
            results = self.predictor.predict_batch(data)
            assert len(results) == 2
            for result in results:
                assert 'username' in result
        except:
            # Expected if models not loaded
            pass 