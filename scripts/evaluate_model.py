"""
Model evaluation script
"""
import sys
from pathlib import Path

import pandas as pd
import json

from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor, handle_missing_values
from src.features.engineering import AdvancedFeatureEngineer
from src.models.registry import ModelRegistry
from src.evaluation.metrics import ModelEvaluator
from src.utils import encode_target, NumpyEncoder


def main():
    """Run model evaluation"""
    
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)
    
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
    
    y_raw = df['Churn']
    X = df.drop(columns=['Churn'])

    with Path("models/selected_features.json").open("r") as f:
        selected_features = json.load(f)

    print(selected_features)
    y, label_encoder = encode_target(y_raw)
    engineer = AdvancedFeatureEngineer()
    X_engineered = engineer.transform(X)

    # Keep only those that actually exist
    available_features = [
        f for f in selected_features
        if f in X_engineered.columns
    ]

    X_selected = X_engineered[available_features]

    # Handle missing values
    X_selected = handle_missing_values(X_selected, strategy="mean")

    # One-hot encode categorical columns
    categorical_cols = X_selected.select_dtypes(
        include=["object", "category"]
    ).columns

    X_selected = pd.get_dummies(
        X_selected,
        columns=categorical_cols,
        drop_first=True
    )
    
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
    
    # Print summary
    print(evaluator.get_summary())
    print("\nClassification Report:")
    print(evaluator.get_classification_report())
    
    # Find optimal threshold
    if has_proba:
        optimal_threshold, threshold_metrics = evaluator.find_optimal_threshold(metric='f1')
        logger.info(f"\nOptimal threshold for F1: {optimal_threshold:.3f}")
        logger.info(f"  F1 at threshold: {threshold_metrics['f1']:.4f}")
        logger.info(f"  Cost reduction: ${evaluator.metrics['total_cost'] - threshold_metrics['cost']:,.2f}")
    
    # Save evaluation report
    logger.info("\nSaving evaluation report...")
    
    report = {
        'metrics': evaluator.get_metrics(include_all=False),
        'classification_report': evaluator.get_classification_report(),
        'model_info': info,
        'optimal_threshold': optimal_threshold if has_proba else None,
        'threshold_metrics': threshold_metrics if has_proba else None
    }
    
    with open("reports/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    
    logger.info("Evaluation complete! Report saved to reports/evaluation_report.json")
    
    return evaluator, report

if __name__ == "__main__":
    evaluator, report = main()
