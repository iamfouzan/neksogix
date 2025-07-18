"""
Text preprocessing utilities for sentiment analysis system.

This module provides functions for HTML tag removal, special character cleaning,
stop word removal, normalization, and tokenization.
"""

import re
from typing import List
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))


class TextPreprocessor:
    """
    Text preprocessing class for sentiment analysis.
    """
    
    def __init__(self):
        """Initialize the text preprocessor."""
        pass
        
    def preprocess_text(self, text: str) -> str:
        """
        Full preprocessing pipeline: HTML removal, cleaning, normalization, stopword removal.
        Args:
            text (str): Raw text
        Returns:
            str: Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ''
        text = self.remove_html_tags(text)
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        return text
        
    def remove_html_tags(self, text: str) -> str:
        """
        Remove HTML tags from text.
        Args:
            text (str): Input text
        Returns:
            str: Text without HTML tags
        """
        return BeautifulSoup(text, "html.parser").get_text()

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and normalizing whitespace.
        Args:
            text (str): Input text
        Returns:
            str: Cleaned text
        """
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def remove_stopwords(self, text: str) -> str:
        """
        Remove stop words from text.
        Args:
            text (str): Input text
        Returns:
            str: Text without stop words
        """
        tokens = text.split()
        filtered = [word for word in tokens if word not in STOP_WORDS]
        return ' '.join(filtered)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        Args:
            text (str): Input text
        Returns:
            List[str]: List of tokens
        """
        return text.split()


# Legacy functions for backward compatibility
def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.
    Args:
        text (str): Input text
    Returns:
        str: Text without HTML tags
    """
    return BeautifulSoup(text, "html.parser").get_text()


def clean_text(text: str) -> str:
    """
    Clean text by removing special characters and normalizing whitespace.
    Args:
        text (str): Input text
    Returns:
        str: Cleaned text
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text: str) -> str:
    """
    Remove stop words from text.
    Args:
        text (str): Input text
    Returns:
        str: Text without stop words
    """
    tokens = text.split()
    filtered = [word for word in tokens if word not in STOP_WORDS]
    return ' '.join(filtered)


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline: HTML removal, cleaning, normalization, stopword removal.
    Args:
        text (str): Raw text
    Returns:
        str: Preprocessed text
    """
    if not text or not isinstance(text, str):
        return ''
    text = remove_html_tags(text)
    text = clean_text(text)
    text = remove_stopwords(text)
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words.
    Args:
        text (str): Input text
    Returns:
        List[str]: List of tokens
    """
    return text.split() 