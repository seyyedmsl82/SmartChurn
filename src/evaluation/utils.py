import pandas as pd

from loguru import logger
from src.evaluation.metrics import ModelEvaluator
from src.data.preprocessor import handle_missing_values
from src.features.engineering import AdvancedFeatureEngineer
from src.utils import encode_target


def prepare_features(df: pd.DataFrame, selected: list) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare evaluation features."""

    y_raw = df["Churn"]
    X = df.drop(columns="Churn")

    y, _ = encode_target(y_raw)

    engineer = AdvancedFeatureEngineer()
    X = engineer.transform(X)

    X = X[[c for c in selected if c in X.columns]]

    X = handle_missing_values(X, strategy="mean")

    categorical = X.select_dtypes(
        include=["object", "category"]
    ).columns

    X = pd.get_dummies(
        X,
        columns=categorical,
        drop_first=True
    )

    return X, y

def generate_evaluation_report(evaluator: ModelEvaluator, has_proba: bool, info: dict) -> dict:
    """Generate evaluation report."""
    # Print summary
    logger.info(f"\n{evaluator.get_summary()}")
    logger.info(
        f"\nClassification Report:\n{evaluator.get_classification_report()}"
    )
    
    # Find optimal threshold
    if has_proba:
        optimal_threshold, threshold_metrics = evaluator.find_optimal_threshold(metric='f1')
        logger.info(f"\nOptimal threshold for F1: {optimal_threshold:.3f}")
        logger.info(f"  F1 at threshold: {threshold_metrics['f1']:.4f}")
        logger.info(f"  Cost reduction: ${evaluator.metrics['total_cost'] - threshold_metrics['cost']:,.2f}")
    
    report = {
        'metrics': evaluator.get_metrics(include_all=False),
        'classification_report': evaluator.get_classification_report(),
        'model_info': info,
        'optimal_threshold': optimal_threshold if has_proba else None,
        'threshold_metrics': threshold_metrics if has_proba else None
    }
    return report