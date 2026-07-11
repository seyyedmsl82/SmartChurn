"""
Data preprocessing pipeline for production
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger

from src.data.preprocessor import DataPreprocessor
from src.data.validator import DataValidator


class DataPreprocessingPipeline:
    """
    Complete data preprocessing pipeline for production
    """
    
    def __init__(self, 
                 raw_data_path: str = 'data/raw/telco_churn.csv',
                 processed_data_path: str = 'data/processed/telco_churn_processed.csv',
                 preprocessor_path: str = 'models/preprocessor.pkl',
                 validation_report_path: str = 'reports/data_validation_report.json'):
        """
        Initialize the preprocessing pipeline
        
        Args:
            raw_data_path: Path to raw data file
            processed_data_path: Path to save processed data
            preprocessor_path: Path to save/load preprocessor
            validation_report_path: Path to save validation report
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.validation_report_path = Path(validation_report_path)
        
        # Create directories
        self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        self.validation_report_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.preprocessor = None
        self.validation_result = None
        self.processing_metadata = {}
    
    def load_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV
        
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {self.raw_data_path}")
        
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data not found at {self.raw_data_path}")
        
        self.df = pd.read_csv(self.raw_data_path)
        logger.info(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        
        # Store metadata
        self.processing_metadata['raw_data'] = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'file_size_mb': self.raw_data_path.stat().st_size / (1024 * 1024)
        }
        
        return self.df
    
    def validate_data(self) -> dict:
        """
        Validate the raw data
        
        Returns:
            Validation results
        """
        logger.info("Validating data...")
        
        validator = DataValidator()
        self.validation_result = validator.validate(self.df)
        
        # Create validation report
        report = {
            'timestamp': datetime.now().isoformat(),
            'is_valid': self.validation_result.is_valid,
            'errors': self.validation_result.errors,
            'warnings': self.validation_result.warnings,
            'error_count': len(self.validation_result.errors),
            'warning_count': len(self.validation_result.warnings),
            'schema_used': validator.schema
        }
        
        # Save validation report
        with open(self.validation_report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {self.validation_report_path}")
        
        if not self.validation_result.is_valid:
            logger.warning(f"Data validation found {len(self.validation_result.errors)} errors")
            for error in self.validation_result.errors[:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")
        else:
            logger.success("Data validation passed!")
        
        self.processing_metadata['validation'] = report
        
        return report
    
    def preprocess_data(self, fit: bool = True) -> pd.DataFrame:
        """
        Preprocess the data
        
        Args:
            fit: Whether to fit the preprocessor (True for training, False for inference)
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data...")
        
        # Initialize or load preprocessor
        if fit:
            # Create new preprocessor for training
            self.preprocessor = DataPreprocessor()
            logger.info("Created new preprocessor")
        else:
            # Load existing preprocessor for inference
            if self.preprocessor_path.exists():
                self.preprocessor = DataPreprocessor.load(str(self.preprocessor_path))
                logger.info(f"Loaded preprocessor from {self.preprocessor_path}")
            else:
                raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")
        
        # Separate features and target
        if 'Churn' in self.df.columns:
            y = self.df['Churn']
            X = self.df.drop(columns=['Churn'])
            has_target = True
        else:
            X = self.df
            y = None
            has_target = False
            logger.warning("No target column found - treating as inference data")
        
        # Preprocess
        if fit:
            X_transformed = self.preprocessor.fit_transform(X, y)
            logger.info("Fitted and transformed data")
            
            # Save preprocessor for production use
            self.preprocessor.save(str(self.preprocessor_path))
            logger.info(f"Preprocessor saved to {self.preprocessor_path}")
        else:
            X_transformed = self.preprocessor.transform(X)
            logger.info("Transformed data using existing preprocessor")
        
        # Store metadata
        self.processing_metadata['preprocessing'] = {
            'fit': fit,
            'original_features': X.shape[1],
            'transformed_features': X_transformed.shape[1],
            'has_target': has_target,
            'preprocessor_path': str(self.preprocessor_path)
        }
        
        # Convert to DataFrame if needed
        if isinstance(X_transformed, np.ndarray):
            # Get feature names from preprocessor
            try:
                feature_names = self.preprocessor.get_feature_names()
                X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
            except:
                X_transformed_df = pd.DataFrame(X_transformed)
        else:
            X_transformed_df = X_transformed
        
        logger.info(f"Preprocessed shape: {X_transformed_df.shape}")
        logger.info(f"Features: {X_transformed_df.shape[1]}")
        
        return X_transformed_df
    
    def save_processed_data(self, X_transformed: pd.DataFrame, y: pd.Series = None):
        """
        Save processed data to CSV
        
        Args:
            X_transformed: Preprocessed features
            y: Target variable (optional)
        """
        logger.info(f"Saving processed data to {self.processed_data_path}")
        
        # Combine features and target if available
        if y is not None:
            # Convert y to DataFrame with proper index
            if isinstance(y, pd.Series):
                y_df = y.reset_index(drop=True)
            else:
                y_df = pd.Series(y, name='Churn')
            
            # Combine
            processed_df = pd.concat([X_transformed, y_df], axis=1)
        else:
            processed_df = X_transformed
        
        # Save to CSV
        processed_df.to_csv(self.processed_data_path, index=False)
        
        logger.info(f"Saved {len(processed_df)} rows to {self.processed_data_path}")
        
        # Store metadata
        self.processing_metadata['processed_data'] = {
            'shape': processed_df.shape,
            'columns': processed_df.columns.tolist(),
            'file_size_mb': self.processed_data_path.stat().st_size / (1024 * 1024),
            'has_target': y is not None
        }
    
    def save_metadata(self):
        """Save processing metadata"""
        metadata_path = self.processed_data_path.parent / 'processing_metadata.json'
        
        self.processing_metadata['timestamp'] = datetime.now().isoformat()
        self.processing_metadata['total_steps'] = len(self.processing_metadata) - 1  # exclude timestamp
        
        with open(metadata_path, 'w') as f:
            json.dump(self.processing_metadata, f, indent=2, default=str)
        
        logger.info(f"Processing metadata saved to {metadata_path}")
    
    def run_full_pipeline(self, fit: bool = True) -> tuple:
        """
        Run the complete preprocessing pipeline
        
        Args:
            fit: Whether to fit the preprocessor
            
        Returns:
            Tuple of (X_transformed, y)
        """
        logger.info("=" * 60)
        logger.info("STARTING DATA PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Validate data
        self.validate_data()
        
        # Step 3: Preprocess data
        X_transformed = self.preprocess_data(fit=fit)
        
        # Step 4: Get target (if available)
        y = self.df['Churn'] if 'Churn' in self.df.columns else None
        
        # Step 5: Save processed data
        self.save_processed_data(X_transformed, y)
        
        # Step 6: Save metadata
        self.save_metadata()
        
        logger.info("=" * 60)
        logger.success("DATA PREPROCESSING PIPELINE COMPLETED!")
        logger.info("=" * 60)
        
        return X_transformed, y
