"""
Production data preprocessing script
Processes raw data and saves processed data for training
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger

from src.data.pipeline import DataPreprocessingPipeline


def main():
    """
    Main entry point for preprocessing script
    """
    # Configuration
    config = {
        'raw_data_path': 'data/raw/telco_churn.csv',
        'processed_data_path': 'data/processed/telco_churn_processed.csv',
        'preprocessor_path': 'models/preprocessor.pkl',
        'validation_report_path': 'reports/data_validation_report.json',
        'fit': True  # Set to False for inference
    }
    
    # Create pipeline and run
    pipeline = DataPreprocessingPipeline(
        raw_data_path=config['raw_data_path'],
        processed_data_path=config['processed_data_path'],
        preprocessor_path=config['preprocessor_path'],
        validation_report_path=config['validation_report_path']
    )
    
    try:
        X_transformed, y = pipeline.run_full_pipeline(fit=config['fit'])
        
        # Print summary
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Original data: {pipeline.df.shape}")
        print(f"Processed data: {X_transformed.shape}")
        print(f"Features: {X_transformed.shape[1]}")
        print(f"Preprocessor: {pipeline.preprocessor_path}")
        print(f"Validation: {'PASSED' if pipeline.validation_result.is_valid else 'HAS WARNINGS'}")
        print(f"Output: {pipeline.processed_data_path}")
        print("=" * 60)
        
        return X_transformed, y
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    X_transformed, y = main()
