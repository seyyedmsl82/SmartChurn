"""
Script to download the Telco Customer Churn dataset
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from loguru import logger

def main():
    """Main function to download the dataset"""
    logger.info("Starting dataset download...")
    
    # Initialize data loader
    loader = DataLoader()
    
    # URL for Telco Customer Churn dataset
    url = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    # Load data from URL
    df = loader.load_from_url(url, save_local=True)
    
    # Log dataset info
    loader.log_dataset_info(df, "Telco Customer Churn")
    
    # Save a processed copy
    loader.save_processed_data(df, "telco_churn_processed.csv")
    
    logger.info("Dataset download completed successfully!")
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

if __name__ == "__main__":
    main()