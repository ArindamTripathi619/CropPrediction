"""
Configuration management utilities.
"""

import yaml
from pathlib import Path


def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_config():
    """
    Get current configuration.
    
    Returns:
        dict: Current configuration
    """
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    return load_config(config_path)

