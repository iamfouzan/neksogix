"""
Database module for sentiment analysis system.

This module contains database models, connection management,
and database operations for the sentiment analysis system.
"""

from .models import Movie, Comment, Prediction
from .connection import DatabaseConnection

__all__ = ['Movie', 'Comment', 'Prediction', 'DatabaseConnection'] 