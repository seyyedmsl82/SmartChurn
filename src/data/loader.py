"""
Data ingestion module for loading customer churn data
"""
import os
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, Union
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataLoader:
    """
    Handles loading of customer churn data from various sources
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize DataLoader with optional configuration
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.raw_data_dir = self.data_dir / 'raw'
        self.processed_data_dir = self.data_dir / 'processed'
        
        # Create directories if they don't exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_from_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame containing the data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            raise
    
    def load_from_url(self, url: str, save_local: bool = True) -> pd.DataFrame:
        """
        Load data from a URL (e.g., GitHub raw data)
        
        Args:
            url: URL to the dataset
            save_local: Whether to save a local copy
            
        Returns:
            DataFrame containing the data
        """
        logger.info(f"Loading data from URL: {url}")
        
        try:
            df = pd.read_csv(url)
            logger.info(f"Successfully loaded {len(df)} rows from URL")
            
            if save_local:
                # Save a local copy
                local_path = self.raw_data_dir / 'telco_churn.csv'
                df.to_csv(local_path, index=False)
                logger.info(f"Saved local copy to {local_path}")
            
            return df
        except Exception as e:
            logger.error(f"Failed to load data from URL: {e}")
            raise
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save processed data to the processed directory
        
        Args:
            df: DataFrame to save
            filename: Name of the file
        """
        file_path = self.processed_data_dir / filename
        df.to_csv(file_path, index=False)
        logger.info(f"Saved processed data to {file_path}")
    
    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with dataset statistics
        """
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
        }
    
    def log_dataset_info(self, df: pd.DataFrame, name: str = "Dataset") -> None:
        """
        Log dataset information for tracking
        
        Args:
            df: DataFrame to analyze
            name: Name of the dataset
        """
        info = self.get_dataset_info(df)
        logger.info(f"\n{name} Information:")
        logger.info(f"  Shape: {info['shape']}")
        logger.info(f"  Memory: {info['memory_usage']:.2f} MB")
        logger.info(f"  Columns ({len(info['columns'])}): {info['columns']}")
        
        # Log missing values if any
        missing_cols = {k: v for k, v in info['missing_values'].items() if v > 0}
        if missing_cols:
            logger.warning(f"  Missing values found: {missing_cols}")
        else:
            logger.info("  No missing values found")
            