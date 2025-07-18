"""
Configuration module for sentiment analysis system.

This module handles all configuration settings including database connections,
model parameters, scraping settings, and environment variables.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for sentiment analysis system."""
    
    # Database Configuration
    DATABASE_URL: str = os.getenv(
        'DATABASE_URL', 
        'postgresql://user:password@localhost/sentiment_db'
    )
    DATABASE_POOL_SIZE: int = int(os.getenv('DATABASE_POOL_SIZE', '10'))
    DATABASE_MAX_OVERFLOW: int = int(os.getenv('DATABASE_MAX_OVERFLOW', '20'))
    
    # Discord Webhook Configuration
    DISCORD_WEBHOOK_URL: Optional[str] = os.getenv('DISCORD_WEBHOOK_URL')
    
    # Hugging Face Model Configuration
    HUGGINGFACE_MODEL_NAME: str = os.getenv(
        'HUGGINGFACE_MODEL_NAME',
        'nlptown/bert-base-multilingual-uncased-sentiment'
    )
    BERT_BATCH_SIZE: int = int(os.getenv('BERT_BATCH_SIZE', '16'))
    BERT_MAX_LENGTH: int = int(os.getenv('BERT_MAX_LENGTH', '512'))
    
    # Scraping Configuration
    SCRAPING_DELAY: float = float(os.getenv('SCRAPING_DELAY', '2.0'))
    SCRAPING_TIMEOUT: int = int(os.getenv('SCRAPING_TIMEOUT', '30'))
    SCRAPING_RETRY_TIMES: int = int(os.getenv('SCRAPING_RETRY_TIMES', '3'))
    SCRAPING_CONCURRENT_REQUESTS: int = int(os.getenv('SCRAPING_CONCURRENT_REQUESTS', '1'))
    
    # User Agents for Rotation
    USER_AGENTS: list = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15'
    ]
    
    # File Paths
    MODELS_DIR: str = os.getenv('MODELS_DIR', 'models')
    DATA_DIR: str = os.getenv('DATA_DIR', 'data')
    LOGS_DIR: str = os.getenv('LOGS_DIR', 'logs')
    
    # Model File Names
    SENTIMENT_MODEL_FILE: str = os.path.join(MODELS_DIR, 'sentiment_model.pkl')
    TFIDF_VECTORIZER_FILE: str = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
    
    # Data File Names
    RAW_COMMENTS_FILE: str = os.path.join(DATA_DIR, 'raw_comments.csv')
    TRAINING_DATA_FILE: str = os.path.join(DATA_DIR, 'training_data.csv')
    OUTPUT_FILE: str = os.path.join(DATA_DIR, 'output.csv')
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: str = os.path.join(LOGS_DIR, 'app.log')
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # ML Model Configuration
    TFIDF_MAX_FEATURES: int = int(os.getenv('TFIDF_MAX_FEATURES', '5000'))
    SVM_KERNEL: str = os.getenv('SVM_KERNEL', 'rbf')
    SVM_C: float = float(os.getenv('SVM_C', '1.0'))
    SVM_GAMMA: str = os.getenv('SVM_GAMMA', 'scale')
    
    # Training Configuration
    TRAIN_TEST_SPLIT: float = float(os.getenv('TRAIN_TEST_SPLIT', '0.8'))
    CROSS_VALIDATION_FOLDS: int = int(os.getenv('CROSS_VALIDATION_FOLDS', '5'))
    MIN_TRAINING_SAMPLES: int = int(os.getenv('MIN_TRAINING_SAMPLES', '1000'))
    
    # Sentiment Analysis Configuration
    MIN_CONFIDENCE_THRESHOLD: float = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.8'))
    MIN_TEXT_LENGTH: int = int(os.getenv('MIN_TEXT_LENGTH', '10'))
    
    # Streamlit Configuration
    STREAMLIT_PORT: int = int(os.getenv('STREAMLIT_PORT', '8501'))
    STREAMLIT_HOST: str = os.getenv('STREAMLIT_HOST', 'localhost')
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """Get database configuration dictionary."""
        return {
            'url': cls.DATABASE_URL,
            'pool_size': cls.DATABASE_POOL_SIZE,
            'max_overflow': cls.DATABASE_MAX_OVERFLOW,
            'echo': False
        }
    
    @classmethod
    def get_scraping_config(cls) -> Dict[str, Any]:
        """Get scraping configuration dictionary."""
        return {
            'delay': cls.SCRAPING_DELAY,
            'timeout': cls.SCRAPING_TIMEOUT,
            'retry_times': cls.SCRAPING_RETRY_TIMES,
            'concurrent_requests': cls.SCRAPING_CONCURRENT_REQUESTS,
            'user_agents': cls.USER_AGENTS
        }
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration dictionary."""
        return {
            'huggingface_model': cls.HUGGINGFACE_MODEL_NAME,
            'batch_size': cls.BERT_BATCH_SIZE,
            'max_length': cls.BERT_MAX_LENGTH,
            'tfidf_max_features': cls.TFIDF_MAX_FEATURES,
            'svm_kernel': cls.SVM_KERNEL,
            'svm_c': cls.SVM_C,
            'svm_gamma': cls.SVM_GAMMA
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings."""
        required_dirs = [cls.MODELS_DIR, cls.DATA_DIR, cls.LOGS_DIR]
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        
        if not cls.DISCORD_WEBHOOK_URL:
            print("Warning: DISCORD_WEBHOOK_URL not set. Discord notifications disabled.")
        
        return True


# Create global config instance
config = Config() 