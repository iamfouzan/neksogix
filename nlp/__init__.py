"""
NLP module for sentiment analysis system.

This module contains text preprocessing and BERT-based sentiment analysis utilities.
"""

from .preprocessor import preprocess_text, clean_text, remove_stopwords
from .sentiment_analyzer import BertSentimentAnalyzer

__all__ = ['preprocess_text', 'clean_text', 'remove_stopwords', 'BertSentimentAnalyzer'] 