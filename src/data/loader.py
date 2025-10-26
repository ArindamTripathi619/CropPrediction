"""
Data loading utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_crop_data(file_path: str) -> pd.DataFrame:
    """
    Load crop recommendation dataset.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded crop data: {len(df)} samples, {len(df.columns)} features")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None


def load_fertilizer_data(file_path: str) -> pd.DataFrame:
    """
    Load fertilizer prediction dataset.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded fertilizer data: {len(df)} samples, {len(df.columns)} features")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None


def load_yield_data(file_path: str) -> pd.DataFrame:
    """
    Load yield prediction dataset.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded yield data: {len(df)} samples, {len(df.columns)} features")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None


def split_data(X: pd.DataFrame, y: pd.Series, 
               train_ratio: float = 0.7, 
               val_ratio: float = 0.15,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                  pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        train_ratio: Proportion of training data
        val_ratio: Proportion of validation data
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # First split: separate out training set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=random_state
    )
    
    # Second split: divide remaining data into validation and test
    test_ratio = val_ratio / (1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio, random_state=random_state
    )
    
    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Args:
        df: DataFrame
        
    Returns:
        dict: Dataset information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'basic_stats': df.describe().to_dict()
    }
    return info

