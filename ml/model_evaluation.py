"""
Model evaluation utilities for sentiment analysis system.

This module provides comprehensive evaluation metrics and visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from config import config

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and comparison utilities."""
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_results = {}
        self.sentiment_labels = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
        
    def evaluate_model_performance(self, y_true: List[str], y_pred: List[str],
                                 model_name: str = "model") -> Dict[str, Any]:
        """Evaluate model performance with comprehensive metrics."""
        try:
            logger.info(f"Evaluating {model_name} performance...")
            
            accuracy = accuracy_score(y_true, y_pred)
            report = classification_report(y_true, y_pred, target_names=self.sentiment_labels, output_dict=True)
            cm = confusion_matrix(y_true, y_pred, labels=self.sentiment_labels)
            
            results = {
                'model_name': model_name,
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'total_samples': len(y_true),
                'evaluation_date': datetime.now().isoformat()
            }
            
            self.evaluation_results[model_name] = results
            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {}
            
    def compare_models(self, bert_predictions: List[Dict[str, Any]], 
                      svm_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare performance between BERT and SVM models."""
        try:
            logger.info("Comparing BERT and SVM model performance...")
            
            bert_sentiments = [p.get('bert_sentiment', 'neutral') for p in bert_predictions]
            svm_sentiments = [p.get('svm_sentiment', 'neutral') for p in svm_predictions]
            final_sentiments = [p.get('final_sentiment', 'neutral') for p in bert_predictions]
            
            svm_vs_bert = self.evaluate_model_performance(bert_sentiments, svm_sentiments, "SVM_vs_BERT")
            ensemble_vs_bert = self.evaluate_model_performance(bert_sentiments, final_sentiments, "Ensemble_vs_BERT")
            
            bert_confidences = [p.get('bert_confidence', 0) for p in bert_predictions]
            svm_confidences = [p.get('svm_confidence', 0) for p in svm_predictions]
            
            agreement_count = sum(1 for b, s in zip(bert_sentiments, svm_sentiments) if b == s)
            agreement_rate = agreement_count / len(bert_sentiments)
            
            comparison_results = {
                'svm_vs_bert': svm_vs_bert,
                'ensemble_vs_bert': ensemble_vs_bert,
                'bert_avg_confidence': np.mean(bert_confidences),
                'svm_avg_confidence': np.mean(svm_confidences),
                'agreement_rate': agreement_rate,
                'total_predictions': len(bert_predictions)
            }
            
            logger.info(f"Model agreement rate: {agreement_rate:.4f}")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {}
            
    def create_evaluation_plots(self, predictions: List[Dict[str, Any]]) -> None:
        """Create comprehensive evaluation plots."""
        try:
            if not predictions:
                return
                
            bert_sentiments = [p.get('bert_sentiment', 'neutral') for p in predictions]
            svm_sentiments = [p.get('svm_sentiment', 'neutral') for p in predictions]
            bert_confidences = [p.get('bert_confidence', 0) for p in predictions]
            svm_confidences = [p.get('svm_confidence', 0) for p in predictions]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Sentiment Analysis Model Evaluation', fontsize=16)
            
            # Sentiment distribution
            bert_counts = Counter(bert_sentiments)
            svm_counts = Counter(svm_sentiments)
            
            labels = list(bert_counts.keys())
            x = np.arange(len(labels))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, [bert_counts.get(l, 0) for l in labels], width, label='BERT')
            axes[0, 0].bar(x + width/2, [svm_counts.get(l, 0) for l in labels], width, label='SVM')
            axes[0, 0].set_title('Sentiment Distribution')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].legend()
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(labels, rotation=45)
            
            # Confidence distribution
            axes[0, 1].hist(bert_confidences, alpha=0.7, label='BERT', bins=20)
            axes[0, 1].hist(svm_confidences, alpha=0.7, label='SVM', bins=20)
            axes[0, 1].set_title('Confidence Distribution')
            axes[0, 1].set_xlabel('Confidence Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            
            # Confidence correlation
            axes[1, 0].scatter(bert_confidences, svm_confidences, alpha=0.6)
            axes[1, 0].set_xlabel('BERT Confidence')
            axes[1, 0].set_ylabel('SVM Confidence')
            axes[1, 0].set_title('BERT vs SVM Confidence')
            
            # Confusion matrix
            cm = confusion_matrix(bert_sentiments, svm_sentiments, labels=self.sentiment_labels)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                       xticklabels=self.sentiment_labels, yticklabels=self.sentiment_labels)
            axes[1, 1].set_title('BERT vs SVM Confusion Matrix')
            axes[1, 1].set_xlabel('SVM Prediction')
            axes[1, 1].set_ylabel('BERT Prediction')
            
            plt.tight_layout()
            plt.savefig(config.EVALUATION_PLOTS_PATH, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Evaluation plots saved to {config.EVALUATION_PLOTS_PATH}")
            
        except Exception as e:
            logger.error(f"Failed to create evaluation plots: {e}")
            
    def save_evaluation_report(self, results: Dict[str, Any]) -> None:
        """Save evaluation report to JSON."""
        try:
            with open(config.EVALUATION_REPORT_PATH, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Evaluation report saved to {config.EVALUATION_REPORT_PATH}")
        except Exception as e:
            logger.error(f"Failed to save evaluation report: {e}")
            
    def generate_summary_statistics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for the evaluation."""
        if not predictions:
            return {}
            
        total_predictions = len(predictions)
        bert_dist = Counter(p.get('bert_sentiment', 'neutral') for p in predictions)
        svm_dist = Counter(p.get('svm_sentiment', 'neutral') for p in predictions)
        
        bert_confidences = [p.get('bert_confidence', 0) for p in predictions]
        svm_confidences = [p.get('svm_confidence', 0) for p in predictions]
        
        agreement_count = sum(1 for p in predictions 
                           if p.get('bert_sentiment') == p.get('svm_sentiment'))
        agreement_rate = agreement_count / total_predictions
        
        return {
            'total_predictions': total_predictions,
            'bert_sentiment_distribution': dict(bert_dist),
            'svm_sentiment_distribution': dict(svm_dist),
            'bert_confidence_mean': np.mean(bert_confidences),
            'svm_confidence_mean': np.mean(svm_confidences),
            'model_agreement_rate': agreement_rate,
            'evaluation_timestamp': datetime.now().isoformat()
        } 