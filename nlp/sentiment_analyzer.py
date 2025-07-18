"""
BERT sentiment analysis utilities for sentiment analysis system.

This module provides a class for running BERT-based sentiment analysis
with batch processing, confidence score extraction, and error handling.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Tuple, Dict, Any
import logging

from config import config

logger = logging.getLogger(__name__)

class BertSentimentAnalyzer:
    """
    BERT-based sentiment analyzer using Hugging Face Transformers.
    """
    def __init__(self, model_name: str = None, device: str = None, batch_size: int = None):
        """
        Initialize the BERT sentiment analyzer.
        Args:
            model_name (str): Hugging Face model name
            device (str): 'cuda' or 'cpu'
            batch_size (int): Batch size for inference
        """
        self.model_name = model_name or config.HUGGINGFACE_MODEL_NAME
        self.batch_size = batch_size or config.BERT_BATCH_SIZE
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded BERT model: {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise

    def predict(self, texts: List[str], timeout: int = 60) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a list of texts.
        Args:
            texts (List[str]): List of input texts
            timeout (int): Timeout in seconds for inference
        Returns:
            List[Dict[str, Any]]: List of dicts with 'label' and 'confidence'
        """
        results = []
        try:
            with torch.no_grad():
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i+self.batch_size]
                    encodings = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=config.BERT_MAX_LENGTH,
                        return_tensors='pt'
                    )
                    encodings = {k: v.to(self.device) for k, v in encodings.items()}
                    outputs = self.model(**encodings)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                    confs, preds = torch.max(probs, dim=1)
                    for j, text in enumerate(batch):
                        label = self._label_from_index(preds[j].item())
                        confidence = confs[j].item()
                        results.append({
                            'text': text,
                            'label': label,
                            'confidence': confidence
                        })
        except Exception as e:
            logger.error(f"BERT prediction error: {e}")
        return results

    def _label_from_index(self, idx: int) -> str:
        """
        Map model output index to sentiment label.
        Args:
            idx (int): Output index
        Returns:
            str: Sentiment label
        """
        # nlptown/bert-base-multilingual-uncased-sentiment: 0=1 star, 4=5 stars
        mapping = {
            0: 'very negative',
            1: 'negative',
            2: 'neutral',
            3: 'positive',
            4: 'very positive'
        }
        return mapping.get(idx, 'unknown') 