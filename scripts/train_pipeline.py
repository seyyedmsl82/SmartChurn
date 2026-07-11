"""
Main training pipeline script with optional feature selection
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger

from src.data.loader import DataLoader
from src.data.preprocessor import handle_missing_values
from src.features.engineering import AdvancedFeatureEngineer
from src.models.registry import ModelRegistry
from src.models.train import ModelTrainer
from src.utils import encode_target, load_selected_features, run_feature_selection


def main():
    """Run the complete training pipeline"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train churn prediction model')
    parser.add_argument('--no-feature-selection', action='store_true',
                       help='Skip feature selection (use all features)')
    parser.add_argument('--use-existing-selection', action='store_true',
                       help='Use existing selected features from file')
    parser.add_argument('--n-features', type=int, default=30,
                       help='Number of features to select (default: 30)')
    parser.add_argument('--selection-method', type=str, default='combined',
                       choices=['filter', 'wrapper', 'embedded', 'combined'],
                       help='Feature selection method (default: combined)')
    parser.add_argument('--config', type=str, default='config/model_configs/default.yaml',
                       help='Model configuration file path')
    parser.add_argument('--impute-strategy', type=str, default='mean',
                       choices=['mean', 'median', 'most_frequent', 'constant'],
                       help='Imputation strategy for missing values (default: mean)')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Log configuration
    logger.info("\nConfiguration:")
    logger.info(f"  Feature Selection: {'Disabled' if args.no_feature_selection else 'Enabled'}")
    if not args.no_feature_selection:
        logger.info(f"  Use Existing: {args.use_existing_selection}")
        logger.info(f"  N Features: {args.n_features}")
        logger.info(f"  Selection Method: {args.selection_method}")
    logger.info(f"  Impute Strategy: {args.impute_strategy}")
    
    # 1. Load data
    logger.info("\n[1/6] Loading data...")
    loader = DataLoader()
    
    try:
        df = loader.load_from_csv('data/processed/telco_churn_processed.csv')
        logger.info(f"   Loaded processed data: {len(df)} rows")
    except:
        logger.warning("Processed data not found, loading raw data...")
        df = loader.load_from_csv('data/raw/telco_churn.csv')
        logger.info(f"   Loaded raw data: {len(df)} rows")
    
    # 2. Separate features and target
    logger.info("\n[2/6] Separating features and target...")
    y_raw = df['Churn']
    X = df.drop(columns=['Churn'])
    logger.info(f"   Features: {X.shape[1]}, Target: {y_raw.nunique()} classes")
    
    # 3. Encode target
    logger.info("\n[3/6] Encoding target...")
    y, label_encoder = encode_target(y_raw)
    
    # 4. Feature engineering
    logger.info("\n[4/6] Feature engineering...")
    engineer = AdvancedFeatureEngineer()
    X_engineered = engineer.transform(X)
    logger.info(f"   Original features: {X.shape[1]}")
    logger.info(f"   Engineered features: {X_engineered.shape[1]}")
    
    # Handle missing values in engineered features
    logger.info("\n   Handling missing values in engineered features...")
    X_engineered = handle_missing_values(X_engineered, strategy=args.impute_strategy)
    
    # 5. Feature selection (optional)
    logger.info("\n[5/6] Feature selection...")
    
    use_feature_selection = not args.no_feature_selection
    
    if use_feature_selection:
        if args.use_existing_selection:
            # Use existing selected features
            selected_features, X_selected = load_selected_features(X_engineered)
            if selected_features is None:
                # No existing features found, run selection
                logger.info("   No existing features found, running selection...")
                selected_features, X_selected = run_feature_selection(
                    X_engineered, y,
                    n_features=args.n_features,
                    method=args.selection_method
                )
        else:
            # Run new feature selection
            selected_features, X_selected = run_feature_selection(
                X_engineered, y,
                n_features=args.n_features,
                method=args.selection_method
            )
    else:
        # Skip feature selection - use all features
        logger.info("   Feature selection disabled - using all features")
        X_selected = X_engineered
        selected_features = X_engineered.columns.tolist()
        
        # Save all features as "selected" for consistency
        with open('models/selected_features.json', 'w') as f:
            json.dump(selected_features, f)
        logger.info(f"   Using all {len(selected_features)} features")
    
    # Handle missing values in selected features
    logger.info("\n   Final check for missing values...")
    X_selected = handle_missing_values(X_selected, strategy=args.impute_strategy)
    
    logger.info(f"   Final features: {X_selected.shape[1]}")
    
    # 6. Train/Validation split and Model training
    logger.info("\n[6/6] Training models...")
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_selected, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    logger.info(f"   Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Check for categorical columns in training data
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        logger.info(f"   Encoding {len(categorical_cols)} categorical columns...")
        X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
        X_val = pd.get_dummies(X_val, columns=categorical_cols, drop_first=True)
        
        # Align columns between train and val
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
        logger.info(f"   After encoding: {X_train.shape[1]} features")
    
    # Final check for NaN values
    if X_train.isnull().any().any():
        logger.warning("   NaN values still present! Applying final imputation...")
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)
    
    # Train models
    trainer = ModelTrainer(
        config_path=args.config,
        experiment_name='churn_prediction'
    )
    
    # Train all models
    results = trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # Log results
    logger.info("\n   Training Results:")
    for model_name, metrics in results.items():
        logger.info(f"   {model_name}: F1={metrics['val_f1']:.4f}, ROC-AUC={metrics.get('val_roc_auc', 0):.4f}")
    
    # Save best model
    logger.info("\n   Saving models...")
    best_model = trainer.best_model
    best_name = trainer.best_model_name
    
    trainer.save_model(best_model, best_name)
    
    # Save label encoder with the model
    label_encoder_path = Path('models/label_encoder.pkl')
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    logger.info(f"   Label encoder saved to {label_encoder_path}")
    
    # Save to registry
    registry = ModelRegistry()
    version = registry.register_model(
        model=best_model,
        model_name=best_name,
        metrics=results[best_name],
        description=f"Best model from training pipeline - F1: {results[best_name]['val_f1']:.4f}"
    )
    logger.info(f"   Model registered as version {version}")
    
    # Save results
    with open('models/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save training metadata
    training_metadata = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_samples': len(df),
        'n_features_original': X.shape[1],
        'n_features_engineered': X_engineered.shape[1],
        'n_features_selected': X_selected.shape[1],
        'feature_selection_used': use_feature_selection,
        'impute_strategy': args.impute_strategy,
        'target_classes': label_encoder.classes_.tolist(),
        'best_model': best_name,
        'best_metrics': results[best_name],
        'model_version': version,
        'command_line_args': vars(args)
    }
    
    with open('models/training_metadata.json', 'w') as f:
        json.dump(training_metadata, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.success("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Best Model: {best_name}")
    logger.info(f"F1 Score: {results[best_name]['val_f1']:.4f}")
    logger.info(f"Model Version: {version}")
    logger.info(f"Features Used: {X_selected.shape[1]}")
    logger.info("=" * 60)
    
    return trainer, results

if __name__ == "__main__":
    trainer, results = main()
