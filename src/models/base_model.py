"""
Base model class for all machine learning models.
"""

from abc import ABC, abstractmethod
import joblib
import numpy as np
from pathlib import Path


class BaseModel(ABC):
    """
    Abstract base class for all models in the system.
    Provides common interface for training, prediction, and evaluation.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def build_model(self):
        """Build the specific model architecture."""
        pass
    
    def train(self, X, y, **kwargs):
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            **kwargs: Additional training parameters
        """
        if self.model is None:
            self.build_model()
        
        print(f"Training {self.model_name}...")
        self.model.fit(X, y)
        self.is_trained = True
        print(f"{self.model_name} training completed.")
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError(f"{self.model_name} is not trained yet.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities (for classification models).
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError(f"{self.model_name} is not trained yet.")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError(f"{self.model_name} does not support probability prediction.")
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError(f"{self.model_name} is not trained yet.")
        
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                    f1_score, mean_squared_error, mean_absolute_error, r2_score)
        from sklearn.utils.multiclass import type_of_target

        y_pred = self.predict(X)

        # Normalize y to numpy array / pandas Series for type checking
        try:
            import numpy as _np
            if hasattr(y, 'values'):
                y_vals = y.values
            else:
                y_vals = _np.asarray(y)
        except Exception:
            y_vals = y

        # Use sklearn's type_of_target to decide classification vs regression
        is_classification = False
        try:
            ttype = type_of_target(y_vals)
            if ttype in ('binary', 'multiclass', 'multilabel-indicator'):
                is_classification = True
        except Exception:
            # Fallback heuristic: if dtype is object or categorical, treat as classification
            try:
                if hasattr(y, 'dtype') and (str(y.dtype) in ['object', 'category'] or getattr(y, 'cat', None) is not None):
                    is_classification = True
            except Exception:
                is_classification = False

        if is_classification:
            # Classification metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        else:
            # Regression metrics
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)

            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2
            }
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None or not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"{self.model_name} saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"{self.model_name} loaded from {filepath}")
    
    def get_feature_importance(self):
        """
        Get feature importance from the model.
        
        Returns:
            Feature importance array or None
        """
        if self.model is None or not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return None
    
    def __repr__(self):
        return f"{self.model_name}(trained={self.is_trained})"

