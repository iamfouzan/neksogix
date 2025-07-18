"""
Machine Learning module for sentiment analysis system.

This module contains data preparation, model training, prediction,
and evaluation components for the custom SVM sentiment classifier.
"""

from .data_preparation import DataPreparation
from .trainer import ModelTrainer
from .predictor import SentimentPredictor
from .model_evaluation import ModelEvaluator

__all__ = [
    'DataPreparation',
    'ModelTrainer', 
    'SentimentPredictor',
    'ModelEvaluator'
] 