"""
Data preprocessing utilities.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
import joblib


class DataPreprocessor:
    """
    Unified data preprocessing class.
    """
    
    def __init__(self, scaler_type='standard', handle_outliers=True):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard', 'robust')
            handle_outliers: Whether to handle outliers
        """
        self.scaler_type = scaler_type
        self.handle_outliers = handle_outliers
        self.scaler = None
        self.label_encoder = {}
        self.imputer = None
        self.feature_names = None
        
    def _create_scaler(self):
        """Create appropriate scaler."""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            return StandardScaler()
    
    def handle_missing_values(self, df: pd.DataFrame, strategy='mean') -> pd.DataFrame:
        """
        Handle missing values in DataFrame.
        
        Args:
            df: Input DataFrame
            strategy: Imputation strategy
            
        Returns:
            DataFrame: DataFrame with imputed values
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if self.imputer is None:
            self.imputer = SimpleImputer(strategy=strategy)
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.imputer.transform(df[numeric_cols])
            
        return df
    
    def remove_outliers(self, df: pd.DataFrame, columns=None, method='iqr') -> pd.DataFrame:
        """
        Remove outliers from DataFrame.
        
        Args:
            df: Input DataFrame
            columns: Columns to process (None for all numeric columns)
            method: Method for outlier detection ('iqr', 'zscore')
            
        Returns:
            DataFrame: DataFrame with outliers removed
        """
        if not self.handle_outliers:
            return df
            
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        df_cleaned = df.copy()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & 
                                       (df_cleaned[col] <= upper_bound)]
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df_cleaned = df_cleaned[z_scores < 3]
        
        return df_cleaned
    
    def encode_categorical(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            columns: List of categorical column names
            
        Returns:
            DataFrame: DataFrame with encoded categories
        """
        df_encoded = df.copy()
        
        for col in columns:
            if col not in self.label_encoder:
                self.label_encoder[col] = LabelEncoder()
                df_encoded[col] = self.label_encoder[col].fit_transform(df[col].astype(str))
            else:
                df_encoded[col] = self.label_encoder[col].transform(df[col].astype(str))
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, fit=False) -> pd.DataFrame:
        """
        Scale features.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler
            
        Returns:
            DataFrame: Scaled DataFrame
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_scaled = df.copy()
        
        if fit:
            self.scaler = self._create_scaler()
            self.feature_names = numeric_cols.tolist()
            df_scaled[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            df_scaled[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df_scaled
    
    def preprocess_data(self, df: pd.DataFrame, 
                       target_column: str = None,
                       fit: bool = True) -> tuple:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            fit: Whether to fit transformers
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        df_processed = df.copy()
        
        # Handle missing values
        df_processed = self.handle_missing_values(df_processed)
        
        # Separate features and target
        if target_column:
            X = df_processed.drop(columns=[target_column])
            y = df_processed[target_column]
        else:
            X = df_processed
            y = None
        
        # Scale features
        X = self.scale_features(X, fit=fit)
        
        return X, y
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor to disk."""
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoder,
            'imputer': self.imputer,
            'feature_names': self.feature_names
        }
        joblib.dump(preprocessor_data, filepath)
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor from disk."""
        preprocessor_data = joblib.load(filepath)
        self.scaler = preprocessor_data['scaler']
        self.label_encoder = preprocessor_data['label_encoders']
        self.imputer = preprocessor_data['imputer']
        self.feature_names = preprocessor_data['feature_names']

