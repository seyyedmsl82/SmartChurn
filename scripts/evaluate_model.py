"""
Model evaluation script
"""
import sys
from pathlib import Path

import json

from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.models.registry import ModelRegistry
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.utils import prepare_features, generate_evaluation_report
from src.utils import NumpyEncoder, validate_required_files


def main():
    """Run model evaluation"""
    REPORT_PATH = Path("reports/evaluation_report.json")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_PATH = Path("models/selected_features.json")
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
  
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)
    
    logger.info("\n[0/5] Checking the required files...")
    validate_required_files()

    logger.info("All required files are present.")
    
    # Load data
    logger.info("\n[1/5] Loading data...")
    loader = DataLoader()
    df = loader.load_from_csv('data/processed/telco_churn_processed.csv')
    
    # Load preprocessor and model
    logger.info("\n[2/5] Loading preprocessor and model...")
    
    # Load best model
    registry = ModelRegistry()
    model, info = registry.load_model(version='latest')
    
    logger.info(f"   Loaded model: {info['model_name']} (v{info['version']})")
    
    # Prepare data
    logger.info("\n[3/5] Preparing data...")

    with Path(FEATURES_PATH).open() as f:
        selected_features = json.load(f)
    X_selected, y = prepare_features(df, selected_features)
    
    # Make predictions
    logger.info("\n[4/5] Making predictions...")
    
    y_pred = model.predict(X_selected)
    
    try:
        y_proba = model.predict_proba(X_selected)[:, 1]
        has_proba = True
    except:
        y_proba = None
        has_proba = False
    
    # Evaluate
    logger.info("\n[5/5] Evaluating model...")
    
    evaluator = ModelEvaluator(y, y_pred, y_proba)
    report = generate_evaluation_report(evaluator, has_proba, info)
    
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    
    logger.info("Evaluation complete! Report saved to reports/evaluation_report.json")
    
    return evaluator, report

if __name__ == "__main__":
    evaluator, report = main()
