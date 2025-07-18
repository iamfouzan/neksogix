"""
Tests for NLP components.

This module tests text preprocessing and sentiment analysis functionality.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.preprocessor import TextPreprocessor
from nlp.sentiment_analyzer import BertSentimentAnalyzer

class TestTextPreprocessor:
    """Test cases for text preprocessing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
        
    def test_basic_preprocessing(self):
        """Test basic text preprocessing."""
        text = "<b>This is a GREAT movie!</b>"
        result = self.preprocessor.preprocess_text(text)
        
        assert "great" in result.lower()
        assert "<b>" not in result
        assert "!" not in result
        
    def test_empty_text(self):
        """Test preprocessing of empty text."""
        result = self.preprocessor.preprocess_text("")
        assert result == ""
        
    def test_special_characters(self):
        """Test handling of special characters."""
        text = "Movie with @#$%^&*() characters"
        result = self.preprocessor.preprocess_text(text)
        
        # Should contain only alphanumeric and spaces
        assert all(c.isalnum() or c.isspace() for c in result)
        
    def test_extra_whitespace(self):
        """Test removal of extra whitespace."""
        text = "   multiple    spaces   "
        result = self.preprocessor.preprocess_text(text)
        
        # Should have single spaces
        assert "  " not in result
        
    def test_minimum_length(self):
        """Test minimum text length validation."""
        short_text = "Hi"
        result = self.preprocessor.preprocess_text(short_text)
        
        # Should be filtered out if too short
        assert len(result) >= 10 or result == ""

class TestBertSentimentAnalyzer:
    """Test cases for BERT sentiment analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = BertSentimentAnalyzer()
        
    def test_model_loading(self):
        """Test that BERT model loads correctly."""
        assert self.analyzer.model is not None
        assert self.analyzer.tokenizer is not None
        
    def test_single_prediction(self):
        """Test single text prediction."""
        text = "This is a great movie!"
        results = self.analyzer.predict([text])
        
        assert len(results) == 1
        assert 'label' in results[0]
        assert 'confidence' in results[0]
        assert 'text' in results[0]
        
    def test_batch_prediction(self):
        """Test batch prediction."""
        texts = [
            "This movie is amazing!",
            "I hated this film.",
            "It was okay, nothing special."
        ]
        
        results = self.analyzer.predict(texts)
        
        assert len(results) == 3
        for result in results:
            assert 'label' in result
            assert 'confidence' in result
            assert 0 <= result['confidence'] <= 1
            
    def test_label_mapping(self):
        """Test sentiment label mapping."""
        # Test all possible labels
        labels = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
        
        for i in range(5):
            label = self.analyzer._label_from_index(i)
            assert label in labels
            
    def test_empty_batch(self):
        """Test prediction with empty batch."""
        results = self.analyzer.predict([])
        assert results == []
        
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with None values
        results = self.analyzer.predict([None, "valid text"])
        
        # Should handle gracefully
        assert isinstance(results, list)
        
    def test_confidence_scores(self):
        """Test that confidence scores are reasonable."""
        text = "This is a fantastic movie!"
        results = self.analyzer.predict([text])
        
        confidence = results[0]['confidence']
        assert 0 <= confidence <= 1
        assert confidence > 0.1  # Should have some confidence 