"""
Helper utilities for sentiment analysis system.

This module contains logging configuration, data validation functions,
error handling decorators, and common utility functions.
"""

import os
import re
import logging
import logging.handlers
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
import time
from datetime import datetime
import json

from config import config


def setup_logging(name: str = None, log_file: str = None) -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Args:
        name (str): Logger name
        log_file (str): Log file path
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        >>> logger = setup_logging('sentiment_analyzer')
        >>> logger.info('Application started')
    """
    if name is None:
        name = __name__
    
    if log_file is None:
        log_file = config.LOG_FILE
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # Create file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def validate_text(text: str, min_length: int = None) -> bool:
    """
    Validate text for sentiment analysis.
    
    Args:
        text (str): Text to validate
        min_length (int): Minimum text length
        
    Returns:
        bool: True if text is valid, False otherwise
        
    Example:
        >>> validate_text("This is a great movie!", min_length=10)
        True
        >>> validate_text("", min_length=10)
        False
    """
    if not text or not isinstance(text, str):
        return False
    
    # Remove whitespace and check length
    cleaned_text = text.strip()
    if not cleaned_text:
        return False
    
    if min_length and len(cleaned_text) < min_length:
        return False
    
    return True


def safe_filename(filename: str) -> str:
    """
    Convert filename to safe format for file system.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Safe filename
        
    Example:
        >>> safe_filename("Movie Title (2023).csv")
        'Movie_Title_2023.csv'
    """
    # Remove or replace unsafe characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    safe_name = re.sub(r'\s+', '_', safe_name)
    safe_name = re.sub(r'_+', '_', safe_name)
    safe_name = safe_name.strip('_')
    
    return safe_name


def retry_on_error(max_retries: int = 3, delay: float = 1.0, 
                   exceptions: tuple = (Exception,)):
    """
    Decorator to retry function on error.
    
    Args:
        max_retries (int): Maximum number of retries
        delay (float): Delay between retries in seconds
        exceptions (tuple): Exceptions to catch
        
    Returns:
        Callable: Decorated function
        
    Example:
        >>> @retry_on_error(max_retries=3, delay=2.0)
        ... def fetch_data():
        ...     # Some operation that might fail
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        raise last_exception
            
            return None
        return wrapper
    return decorator


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func (Callable): Function to measure
        
    Returns:
        Callable: Decorated function with timing
        
    Example:
        >>> @timing_decorator
        ... def slow_function():
        ...     time.sleep(1)
        ...     return "Done"
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger(func.__module__)
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        
        return result
    return wrapper


def save_json(data: Dict[str, Any], filepath: str) -> bool:
    """
    Save data to JSON file.
    
    Args:
        data (Dict[str, Any]): Data to save
        filepath (str): File path
        
    Returns:
        bool: True if successful, False otherwise
        
    Example:
        >>> data = {"accuracy": 0.85, "precision": 0.82}
        >>> save_json(data, "results.json")
        True
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Failed to save JSON file {filepath}: {e}")
        return False


def load_json(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load data from JSON file.
    
    Args:
        filepath (str): File path
        
    Returns:
        Optional[Dict[str, Any]]: Loaded data or None if failed
        
    Example:
        >>> data = load_json("results.json")
        >>> print(data)
        {'accuracy': 0.85, 'precision': 0.82}
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON file {filepath}: {e}")
        return None


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable string.
    
    Args:
        seconds (float): Duration in seconds
        
    Returns:
        str: Formatted duration string
        
    Example:
        >>> format_duration(3661)
        '1h 1m 1s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = int(seconds % 60)
        return f"{hours}h {remaining_minutes}m {remaining_seconds}s"


def validate_movie_name(movie_name: str) -> bool:
    """
    Validate movie name for IMDb search.
    
    Args:
        movie_name (str): Movie name to validate
        
    Returns:
        bool: True if valid, False otherwise
        
    Example:
        >>> validate_movie_name("The Shawshank Redemption")
        True
        >>> validate_movie_name("")
        False
    """
    if not movie_name or not isinstance(movie_name, str):
        return False
    
    # Remove extra whitespace
    cleaned_name = movie_name.strip()
    if not cleaned_name:
        return False
    
    # Check minimum length
    if len(cleaned_name) < 2:
        return False
    
    # Check for valid characters (allow letters, numbers, spaces, and common punctuation)
    if not re.match(r'^[a-zA-Z0-9\s\-\.\,\&\'\"\(\)]+$', cleaned_name):
        return False
    
    return True


def sanitize_text(text: str) -> str:
    """
    Sanitize text for safe storage and processing.
    
    Args:
        text (str): Text to sanitize
        
    Returns:
        str: Sanitized text
        
    Example:
        >>> sanitize_text("<script>alert('xss')</script>Hello World!")
        'Hello World!'
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    return text.strip()


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Args:
        lst (List[Any]): List to split
        chunk_size (int): Size of each chunk
        
    Returns:
        List[List[Any]]: List of chunks
        
    Example:
        >>> chunk_list([1, 2, 3, 4, 5, 6], 2)
        [[1, 2], [3, 4], [5, 6]]
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_file_size_mb(filepath: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        filepath (str): File path
        
    Returns:
        float: File size in MB
        
    Example:
        >>> get_file_size_mb("data.csv")
        2.5
    """
    try:
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0


def create_progress_callback(total: int) -> Callable[[int], None]:
    """
    Create a progress callback function.
    
    Args:
        total (int): Total number of items
        
    Returns:
        Callable[[int], None]: Progress callback function
        
    Example:
        >>> callback = create_progress_callback(100)
        >>> callback(50)  # 50% complete
    """
    def callback(current: int) -> None:
        percentage = (current / total) * 100
        logger = logging.getLogger(__name__)
        logger.info(f"Progress: {current}/{total} ({percentage:.1f}%)")
    
    return callback 