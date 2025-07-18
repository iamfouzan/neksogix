"""
Model training utilities for sentiment analysis system.

This module handles training of the custom SVM model with
hyperparameter tuning, cross-validation, and model serialization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import logging
import joblib
import json
from datetime import datetime, timedelta
import time
from cuml.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.auto import tqdm

from config import config
from ml.data_preparation import DataPreparation

logger = logging.getLogger(__name__)

class TqdmJoblibProgressBar:
    """Progress bar for joblib parallel tasks using tqdm."""
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="GridSearchCV Progress", unit="fit")
        self.start_time = time.time()
    def __call__(self, *args, **kwargs):
        self.pbar.update()
        elapsed = time.time() - self.start_time
        avg = elapsed / self.pbar.n if self.pbar.n else 0
        remaining = avg * (self.pbar.total - self.pbar.n)
        self.pbar.set_postfix_str(f"ETA: {int(remaining // 60)}m {int(remaining % 60)}s")
    def close(self):
        self.pbar.close()

def run_grid_search_with_progress(grid_search, X, y, total):
    """Run GridSearchCV with a live tqdm progress bar and ETA."""
    progress = TqdmJoblibProgressBar(total)
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    class BatchCompletionCallBackWithTqdm(old_batch_callback):
        def __call__(self, *args, **kwargs):
            progress()
            return super().__call__(*args, **kwargs)
    joblib.parallel.BatchCompletionCallBack = BatchCompletionCallBackWithTqdm
    try:
        grid_search.fit(X, y)
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        progress.close()

class ModelTrainer:
    """
    Handles training of the custom SVM sentiment classifier.
    """
    
    def __init__(self):
        """Initialize the model trainer."""
        self.data_prep = DataPreparation()
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        self.training_history = {}
        
    def train_model(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Train the SVM model with hyperparameter tuning.
        
        Args:
            features (np.ndarray): Training features
            labels (np.ndarray): Training labels
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING MODEL TRAINING PROCESS")
            logger.info("=" * 60)
            logger.info(f"Input features shape: {features.shape}")
            logger.info(f"Input labels shape: {labels.shape}")
            logger.info(f"Unique labels: {np.unique(labels, return_counts=True)}")
            
            # Scale features
            logger.info("Scaling features using StandardScaler...")
            features_scaled = self.scaler.fit_transform(features)
            logger.info(f"Features scaled successfully. Shape: {features_scaled.shape}")
            logger.info(f"Scaled features mean: {np.mean(features_scaled, axis=0)[:5]}...")
            logger.info(f"Scaled features std: {np.std(features_scaled, axis=0)[:5]}...")
            
            # Define parameter grid for GridSearchCV
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }
            total_combinations = len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])
            total_folds = total_combinations * 5  # 5-fold CV
            
            logger.info("Parameter grid for hyperparameter tuning:")
            logger.info(f"C values: {param_grid['C']}")
            logger.info(f"Gamma values: {param_grid['gamma']}")
            logger.info(f"Kernel values: {param_grid['kernel']}")
            logger.info(f"Total parameter combinations: {total_combinations}")
            logger.info(f"Total folds to process: {total_folds}")
            
            # Initialize SVM with probability estimation
            logger.info("Initializing cuML SVM classifier (GPU, no probability outputs)...")
            svm = SVC(random_state=42)
            
            # Perform grid search with cross-validation
            logger.info("Starting GridSearchCV with 5-fold cross-validation...")
            logger.info("This may take several minutes depending on dataset size...")
            
            # Start timing
            start_time = time.time()
            
            grid_search = GridSearchCV(
                svm, param_grid,
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=0  # tqdm will handle progress
            )
            logger.info("Performing hyperparameter tuning with live progress bar...")
            run_grid_search_with_progress(grid_search, features_scaled, labels, total_folds)
            
            # Calculate total training time
            total_time = time.time() - start_time
            avg_time_per_fold = total_time / total_folds
            
            logger.info("Grid search completed!")
            logger.info(f"Total training time: {timedelta(seconds=int(total_time))}")
            logger.info(f"Average time per fold: {avg_time_per_fold:.2f} seconds")
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            logger.info(f"Best parameters found: {self.best_params}")
            logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            # Log all CV results
            logger.info("All cross-validation results:")
            for i, (params, score) in enumerate(zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'])):
                logger.info(f"  {i+1:2d}. {params} -> Score: {score:.4f}")
            
            # Cross-validation scores
            logger.info("Performing additional 5-fold cross-validation on best model...")
            logger.info("This will take approximately 5 folds...")
            
            cv_start_time = time.time()
            cv_scores = cross_val_score(
                self.model, features_scaled, labels,
                cv=5, scoring='f1_weighted'
            )
            cv_total_time = time.time() - cv_start_time
            
            logger.info(f"Additional CV completed in {timedelta(seconds=int(cv_total_time))}")
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"CV mean score: {cv_scores.mean():.4f}")
            logger.info(f"CV std score: {cv_scores.std():.4f}")
            logger.info(f"CV 95% confidence interval: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Training metrics
            logger.info("Calculating training set predictions and metrics...")
            train_predictions = self.model.predict(features_scaled)
            train_report = classification_report(
                labels, train_predictions,
                output_dict=True
            )
            
            logger.info("Training set classification report:")
            logger.info(f"  Accuracy: {train_report['accuracy']:.4f}")
            logger.info(f"  Macro avg F1: {train_report['macro avg']['f1-score']:.4f}")
            logger.info(f"  Weighted avg F1: {train_report['weighted avg']['f1-score']:.4f}")
            
            # Log per-class metrics
            logger.info("Per-class training metrics:")
            for class_name, metrics in train_report.items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    logger.info(f"  {class_name}: Precision={metrics['precision']:.4f}, "
                              f"Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
            
            # Store training history
            self.training_history = {
                'best_params': self.best_params,
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_report': train_report,
                'training_date': datetime.now().isoformat(),
                'n_samples': len(features),
                'n_features': features.shape[1],
                'total_training_time': total_time,
                'avg_time_per_fold': avg_time_per_fold,
                'additional_cv_time': cv_total_time
            }
            
            logger.info("=" * 60)
            logger.info("MODEL TRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            logger.info(f"Training samples: {len(features)}")
            logger.info(f"Features: {features.shape[1]}")
            logger.info(f"Total training time: {timedelta(seconds=int(total_time))}")
            logger.info(f"Average time per fold: {avg_time_per_fold:.2f} seconds")
            logger.info("=" * 60)
            
            return self.training_history
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        try:
            logger.info("=" * 50)
            logger.info("STARTING MODEL EVALUATION")
            logger.info("=" * 50)
            logger.info(f"Test features shape: {X_test.shape}")
            logger.info(f"Test labels shape: {y_test.shape}")
            logger.info(f"Test labels distribution: {np.unique(y_test, return_counts=True)}")
            
            # Scale test features
            logger.info("Scaling test features...")
            X_test_scaled = self.scaler.transform(X_test)
            logger.info("Test features scaled successfully")
            
            # Make predictions
            logger.info("Making predictions on test set...")
            y_pred = self.model.predict(X_test_scaled)
            # cuML SVC does not support predict_proba
            # y_pred_proba = self.model.predict_proba(X_test_scaled)
            logger.info(f"Predictions shape: {y_pred.shape}")
            # logger.info(f"Prediction probabilities shape: {y_pred_proba.shape}")
            logger.info(f"Predicted labels distribution: {np.unique(y_pred, return_counts=True)}")
            
            # Calculate metrics
            logger.info("Calculating evaluation metrics...")
            test_report = classification_report(
                y_test, y_pred,
                output_dict=True
            )
            
            # Confusion matrix
            logger.info("Computing confusion matrix...")
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"Confusion matrix shape: {cm.shape}")
            
            # Store evaluation results
            evaluation_results = {
                'test_report': test_report,
                'confusion_matrix': cm.tolist(),
                'predictions': y_pred.tolist(),
                # 'probabilities': y_pred_proba.tolist(),  # Not available in cuML
                'test_accuracy': test_report['accuracy'],
                'test_f1_weighted': test_report['weighted avg']['f1-score']
            }
            
            logger.info("Test set classification report:")
            logger.info(f"  Overall Accuracy: {test_report['accuracy']:.4f}")
            logger.info(f"  Macro avg F1: {test_report['macro avg']['f1-score']:.4f}")
            logger.info(f"  Weighted avg F1: {test_report['weighted avg']['f1-score']:.4f}")
            
            # Log per-class metrics
            logger.info("Per-class test metrics:")
            for class_name, metrics in test_report.items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    logger.info(f"  {class_name}: Precision={metrics['precision']:.4f}, "
                              f"Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
            
            # Generate all visualizations
            logger.info("Generating comprehensive visualizations...")
            self._plot_confusion_matrix(cm)
            self._plot_accuracy_metrics(test_report)
            self._plot_f1_scores(test_report)
            self._plot_precision_recall_curves(test_report)
            self._plot_class_distribution(y_test, y_pred)
            self._plot_training_metrics_dashboard(test_report)
            self._plot_training_time_analysis()
            
            logger.info("=" * 50)
            logger.info("MODEL EVALUATION COMPLETED")
            logger.info("=" * 50)
            logger.info(f"Test Accuracy: {test_report['accuracy']:.4f}")
            logger.info(f"Test F1 Score: {test_report['weighted avg']['f1-score']:.4f}")
            logger.info("=" * 50)
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            
    def save_model(self) -> None:
        """Save the trained model and related components."""
        try:
            logger.info("=" * 40)
            logger.info("SAVING MODEL AND COMPONENTS")
            logger.info("=" * 40)
            
            # Create models directory if it doesn't exist
            model_dir = os.path.dirname(config.MODEL_PATH)
            logger.info(f"Creating model directory: {model_dir}")
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            logger.info(f"Saving model to: {config.MODEL_PATH}")
            joblib.dump(self.model, config.MODEL_PATH)
            logger.info("Model saved successfully")
            
            # Save scaler
            logger.info(f"Saving scaler to: {config.SCALER_PATH}")
            joblib.dump(self.scaler, config.SCALER_PATH)
            logger.info("Scaler saved successfully")
            
            # Save training history
            logger.info(f"Saving training history to: {config.TRAINING_HISTORY_PATH}")
            with open(config.TRAINING_HISTORY_PATH, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            logger.info("Training history saved successfully")
            
            # Log file sizes
            model_size = os.path.getsize(config.MODEL_PATH) / (1024 * 1024)  # MB
            scaler_size = os.path.getsize(config.SCALER_PATH) / 1024  # KB
            history_size = os.path.getsize(config.TRAINING_HISTORY_PATH) / 1024  # KB
            
            logger.info(f"Model file size: {model_size:.2f} MB")
            logger.info(f"Scaler file size: {scaler_size:.2f} KB")
            logger.info(f"History file size: {history_size:.2f} KB")
            
            logger.info("=" * 40)
            logger.info("ALL COMPONENTS SAVED SUCCESSFULLY")
            logger.info("=" * 40)
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            
    def load_model(self) -> bool:
        """
        Load the trained model and related components.
        
        Returns:
            bool: True if loading successful
        """
        try:
            logger.info("=" * 40)
            logger.info("LOADING MODEL AND COMPONENTS")
            logger.info("=" * 40)
            
            if not os.path.exists(config.MODEL_PATH):
                logger.warning(f"Model file not found: {config.MODEL_PATH}")
                return False
                
            logger.info(f"Loading model from: {config.MODEL_PATH}")
            self.model = joblib.load(config.MODEL_PATH)
            logger.info("Model loaded successfully")
            
            logger.info(f"Loading scaler from: {config.SCALER_PATH}")
            self.scaler = joblib.load(config.SCALER_PATH)
            logger.info("Scaler loaded successfully")
            
            # Log model information
            logger.info(f"Model type: {type(self.model).__name__}")
            logger.info(f"Model parameters: {self.model.get_params()}")
            
            logger.info("=" * 40)
            logger.info("MODEL LOADED SUCCESSFULLY")
            logger.info("=" * 40)
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
            
    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """
        Plot and save confusion matrix.
        
        Args:
            cm (np.ndarray): Confusion matrix
        """
        try:
            logger.info("Creating confusion matrix visualization...")
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, annot=True, fmt='d',
                cmap='Blues',
                xticklabels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'],
                yticklabels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            logger.info(f"Saving confusion matrix to: {config.CONFUSION_MATRIX_PATH}")
            plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log confusion matrix details
            logger.info("Confusion matrix details:")
            logger.info(f"  Shape: {cm.shape}")
            logger.info(f"  Total predictions: {np.sum(cm)}")
            logger.info(f"  Correct predictions: {np.trace(cm)}")
            logger.info(f"  Accuracy from confusion matrix: {np.trace(cm) / np.sum(cm):.4f}")
            
            logger.info("Confusion matrix saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _plot_accuracy_metrics(self, report: Dict[str, Any]) -> None:
        """
        Plot and save accuracy metrics.
        
        Args:
            report (Dict[str, Any]): Classification report
        """
        try:
            logger.info("Creating accuracy metrics visualization...")
            plt.figure(figsize=(8, 6))
            accuracy_data = {
                'Overall': report['accuracy'],
                'Macro Avg': report['macro avg']['accuracy'],
                'Weighted Avg': report['weighted avg']['accuracy']
            }
            plt.bar(accuracy_data.keys(), accuracy_data.values())
            plt.ylim(0, 1)
            plt.title('Accuracy Metrics')
            plt.ylabel('Accuracy')
            plt.xlabel('Metric')
            plt.tight_layout()
            
            logger.info(f"Saving accuracy metrics to: {config.ACCURACY_METRICS_PATH}")
            plt.savefig(config.ACCURACY_METRICS_PATH, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Accuracy metrics visualization saved successfully")
        except Exception as e:
            logger.error(f"Failed to plot accuracy metrics: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _plot_f1_scores(self, report: Dict[str, Any]) -> None:
        """
        Plot and save F1 score metrics.
        
        Args:
            report (Dict[str, Any]): Classification report
        """
        try:
            logger.info("Creating F1 score metrics visualization...")
            plt.figure(figsize=(8, 6))
            f1_data = {
                'Overall': report['weighted avg']['f1-score'],
                'Macro Avg': report['macro avg']['f1-score'],
                'Weighted Avg': report['weighted avg']['f1-score']
            }
            plt.bar(f1_data.keys(), f1_data.values())
            plt.ylim(0, 1)
            plt.title('F1 Score Metrics')
            plt.ylabel('F1 Score')
            plt.xlabel('Metric')
            plt.tight_layout()
            
            logger.info(f"Saving F1 score metrics to: {config.F1_SCORES_PATH}")
            plt.savefig(config.F1_SCORES_PATH, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("F1 score metrics visualization saved successfully")
        except Exception as e:
            logger.error(f"Failed to plot F1 scores: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _plot_precision_recall_curves(self, report: Dict[str, Any]) -> None:
        """
        Plot and save precision-recall metrics.
        
        Args:
            report (Dict[str, Any]): Classification report
        """
        try:
            logger.info("Creating precision-recall metrics visualization...")
            plt.figure(figsize=(10, 6))
            
            # Extract precision and recall for each class
            classes = []
            precisions = []
            recalls = []
            
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    classes.append(class_name)
                    precisions.append(metrics['precision'])
                    recalls.append(metrics['recall'])
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Precision bar plot
            ax1.bar(classes, precisions)
            ax1.set_title('Precision by Class')
            ax1.set_ylabel('Precision')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
            
            # Recall bar plot
            ax2.bar(classes, recalls)
            ax2.set_title('Recall by Class')
            ax2.set_ylabel('Recall')
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            logger.info(f"Saving precision-recall metrics to: {config.PRECISION_RECALL_CURVES_PATH}")
            plt.savefig(config.PRECISION_RECALL_CURVES_PATH, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Precision-recall metrics visualization saved successfully")
        except Exception as e:
            logger.error(f"Failed to plot precision-recall metrics: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _plot_class_distribution(self, y_test: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot and save class distribution.
        
        Args:
            y_test (np.ndarray): Test labels
            y_pred (np.ndarray): Predicted labels
        """
        try:
            logger.info("Creating class distribution visualization...")
            plt.figure(figsize=(12, 6))
            
            # Create subplots for true vs predicted
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # True labels distribution
            unique_true, counts_true = np.unique(y_test, return_counts=True)
            ax1.bar(unique_true, counts_true)
            ax1.set_title('True Labels Distribution')
            ax1.set_ylabel('Count')
            ax1.set_xlabel('Class')
            
            # Predicted labels distribution
            unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
            ax2.bar(unique_pred, counts_pred)
            ax2.set_title('Predicted Labels Distribution')
            ax2.set_ylabel('Count')
            ax2.set_xlabel('Class')
            
            plt.tight_layout()
            
            logger.info(f"Saving class distribution to: {config.CLASS_DISTRIBUTION_PLOT_PATH}")
            plt.savefig(config.CLASS_DISTRIBUTION_PLOT_PATH, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Class distribution visualization saved successfully")
        except Exception as e:
            logger.error(f"Failed to plot class distribution: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _plot_training_metrics_dashboard(self, report: Dict[str, Any]) -> None:
        """
        Plot and save a comprehensive metrics dashboard.
        
        Args:
            report (Dict[str, Any]): Classification report
        """
        try:
            logger.info("Creating comprehensive metrics dashboard...")
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 1. Overall Accuracy
            axes[0, 0].bar(['Accuracy'], [report['accuracy']])
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].set_title('Overall Accuracy')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].grid(True)

            # 2. F1 Scores
            f1_scores = []
            class_names = []
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and 'f1-score' in metrics:
                    f1_scores.append(metrics['f1-score'])
                    class_names.append(class_name)
            
            axes[0, 1].bar(class_names, f1_scores)
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].set_title('F1 Scores by Class')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True)

            # 3. Precision by Class
            precisions = []
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    precisions.append(metrics['precision'])
            
            axes[0, 2].bar(class_names, precisions)
            axes[0, 2].set_ylim(0, 1)
            axes[0, 2].set_title('Precision by Class')
            axes[0, 2].set_ylabel('Precision')
            axes[0, 2].tick_params(axis='x', rotation=45)
            axes[0, 2].grid(True)

            # 4. Recall by Class
            recalls = []
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and 'recall' in metrics:
                    recalls.append(metrics['recall'])
            
            axes[1, 0].bar(class_names, recalls)
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].set_title('Recall by Class')
            axes[1, 0].set_ylabel('Recall')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True)

            # 5. Macro vs Weighted Averages
            macro_avg = report['macro avg']
            weighted_avg = report['weighted avg']
            
            metrics_names = ['Precision', 'Recall', 'F1-Score']
            macro_values = [macro_avg['precision'], macro_avg['recall'], macro_avg['f1-score']]
            weighted_values = [weighted_avg['precision'], weighted_avg['recall'], weighted_avg['f1-score']]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, macro_values, width, label='Macro Avg')
            axes[1, 1].bar(x + width/2, weighted_values, width, label='Weighted Avg')
            axes[1, 1].set_title('Macro vs Weighted Averages')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(metrics_names)
            axes[1, 1].legend()
            axes[1, 1].grid(True)

            # 6. Support (number of samples per class)
            supports = []
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and 'support' in metrics:
                    supports.append(metrics['support'])
            
            axes[1, 2].bar(class_names, supports)
            axes[1, 2].set_title('Support (Samples per Class)')
            axes[1, 2].set_ylabel('Number of Samples')
            axes[1, 2].tick_params(axis='x', rotation=45)
            axes[1, 2].grid(True)

            plt.tight_layout()
            
            logger.info(f"Saving comprehensive metrics dashboard to: {config.METRICS_DASHBOARD_PATH}")
            plt.savefig(config.METRICS_DASHBOARD_PATH, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Comprehensive metrics dashboard saved successfully")
        except Exception as e:
            logger.error(f"Failed to plot comprehensive metrics dashboard: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _plot_training_time_analysis(self) -> None:
        """
        Plot and save training time analysis.
        """
        try:
            logger.info("Creating training time analysis visualization...")
            
            if not hasattr(self, 'training_history') or not self.training_history:
                logger.warning("No training history available for time analysis")
                return
            
            plt.figure(figsize=(12, 8))
            
            # Create subplots for different time metrics
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # 1. Total training time
            if 'total_training_time' in self.training_history:
                total_time = self.training_history['total_training_time']
                axes[0, 0].bar(['Total Training'], [total_time])
                axes[0, 0].set_title('Total Training Time')
                axes[0, 0].set_ylabel('Time (seconds)')
                axes[0, 0].grid(True)
            
            # 2. Average time per fold
            if 'avg_time_per_fold' in self.training_history:
                avg_time = self.training_history['avg_time_per_fold']
                axes[0, 1].bar(['Avg per Fold'], [avg_time])
                axes[0, 1].set_title('Average Time per Fold')
                axes[0, 1].set_ylabel('Time (seconds)')
                axes[0, 1].grid(True)
            
            # 3. Cross-validation scores
            if 'cv_scores' in self.training_history:
                cv_scores = self.training_history['cv_scores']
                axes[1, 0].plot(range(1, len(cv_scores) + 1), cv_scores, 'bo-')
                axes[1, 0].set_title('Cross-Validation Scores')
                axes[1, 0].set_xlabel('Fold')
                axes[1, 0].set_ylabel('F1 Score')
                axes[1, 0].grid(True)
                axes[1, 0].set_ylim(0, 1)
            
            # 4. Training samples vs features
            if 'n_samples' in self.training_history and 'n_features' in self.training_history:
                n_samples = self.training_history['n_samples']
                n_features = self.training_history['n_features']
                
                axes[1, 1].scatter(n_features, n_samples, s=100, alpha=0.7)
                axes[1, 1].set_title('Dataset Size')
                axes[1, 1].set_xlabel('Number of Features')
                axes[1, 1].set_ylabel('Number of Samples')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            logger.info(f"Saving training time analysis to: {config.TRAINING_TIME_ANALYSIS_PATH}")
            plt.savefig(config.TRAINING_TIME_ANALYSIS_PATH, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Training time analysis visualization saved successfully")
        except Exception as e:
            logger.error(f"Failed to plot training time analysis: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        if self.model is None:
            logger.warning("No model loaded - cannot provide model info")
            return {"error": "No model loaded"}
            
        logger.info("Retrieving model information...")
        model_info = {
            "model_type": type(self.model).__name__,
            "best_params": self.best_params,
            "training_history": self.training_history,
            "feature_names": getattr(self.model, 'feature_names_in_', None),
            "n_features": getattr(self.model, 'n_features_in_', None)
        }
        
        logger.info(f"Model type: {model_info['model_type']}")
        logger.info(f"Number of features: {model_info['n_features']}")
        logger.info(f"Best parameters: {model_info['best_params']}")
        
        return model_info 