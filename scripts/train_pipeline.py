"""
Main training pipeline script with optional feature selection
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.features.engineering import AdvancedFeatureEngineer
from src.features.selection import FeatureSelector
from src.models.train import ModelTrainer
from src.models.registry import ModelRegistry
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def encode_target(y: pd.Series) -> tuple:
    """
    Encode target variable to binary values
    
    Args:
        y: Target Series with string values
        
    Returns:
        Tuple of (encoded_y, label_encoder)
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    logger.info(f"   Target encoded: {le.classes_} -> {list(range(len(le.classes_)))}")
    return y_encoded, le

def handle_missing_values(X: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Handle missing values in features
    
    Args:
        X: Feature DataFrame
        strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
        
    Returns:
        DataFrame with missing values handled
    """
    # Check for missing values
    missing_cols = X.columns[X.isnull().any()].tolist()
    
    if not missing_cols:
        logger.info("   No missing values found")
        return X
    
    logger.info(f"   Found missing values in {len(missing_cols)} columns: {missing_cols[:5]}...")
    
    # Separate numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    X_clean = X.copy()
    
    # Impute numeric columns
    if len(numeric_cols) > 0:
        numeric_missing = [col for col in numeric_cols if col in missing_cols]
        if numeric_missing:
            logger.info(f"   Imputing {len(numeric_missing)} numeric columns with {strategy}")
            imputer = SimpleImputer(strategy=strategy)
            X_clean[numeric_cols] = imputer.fit_transform(X_clean[numeric_cols])
    
    # Impute categorical columns
    if len(categorical_cols) > 0:
        categorical_missing = [col for col in categorical_cols if col in missing_cols]
        if categorical_missing:
            logger.info(f"   Imputing {len(categorical_missing)} categorical columns with 'most_frequent'")
            imputer = SimpleImputer(strategy='most_frequent')
            X_clean[categorical_cols] = imputer.fit_transform(X_clean[categorical_cols])
    
    return X_clean

def map_selected_features(selected_encoded: list, original_columns: list) -> list:
    """
    Map selected encoded feature names back to original feature names
    
    Args:
        selected_encoded: List of encoded feature names from selection
        original_columns: List of original column names
        
    Returns:
        List of original feature names that were selected
    """
    # Clean up encoded names to get original names
    mapped_features = []
    
    for feature in selected_encoded:
        # Check if this feature exists in original columns
        if feature in original_columns:
            mapped_features.append(feature)
        else:
            # Try to find the original feature by checking prefixes
            found = False
            for orig_col in original_columns:
                # Check if the encoded feature starts with the original column name
                if feature.startswith(orig_col + '_'):
                    if orig_col not in mapped_features:
                        mapped_features.append(orig_col)
                        found = True
                        break
                # Check for exact match without underscore
                elif feature == orig_col:
                    if orig_col not in mapped_features:
                        mapped_features.append(orig_col)
                        found = True
                        break
            
            if not found:
                # If not found, try to match by removing common suffixes
                for orig_col in original_columns:
                    if orig_col in feature or feature in orig_col:
                        if orig_col not in mapped_features:
                            mapped_features.append(orig_col)
                            found = True
                            break
                
                if not found:
                    logger.warning(f"   Could not map feature: {feature}")
    
    return mapped_features

def run_feature_selection(X_engineered: pd.DataFrame, y: np.ndarray, 
                          n_features: int = 30, 
                          method: str = 'combined') -> tuple:
    """
    Run feature selection and return selected features
    
    Args:
        X_engineered: Engineered features
        y: Target variable
        n_features: Number of features to select
        method: Selection method
        
    Returns:
        Tuple of (selected_features, X_selected)
    """
    logger.info(f"   Running feature selection (n_features={n_features}, method='{method}')...")
    
    # Handle missing values before selection
    X_clean = handle_missing_values(X_engineered)
    
    # Encode categorical for selection
    X_encoded = pd.get_dummies(X_clean, drop_first=True)
    X_encoded = X_encoded.fillna(0)
    
    logger.info(f"   Encoded features: {X_encoded.shape[1]}")
    
    # Run feature selection
    selector = FeatureSelector(method=method, n_features=n_features)
    selected_encoded = selector.select_features(X_encoded, y)
    
    # Map back to original feature names
    selected_features = map_selected_features(selected_encoded, X_engineered.columns.tolist())
    
    # Remove duplicates while preserving order
    selected_features = list(dict.fromkeys(selected_features))
    
    logger.info(f"   Mapped to {len(selected_features)} original features")
    
    # Save selected features
    with open('models/selected_features.json', 'w') as f:
        json.dump(selected_features, f)
    logger.info(f"   Selected {len(selected_features)} features")
    logger.info(f"   Selected features: {selected_features[:5]}...")
    
    # Filter to selected features
    available_features = [f for f in selected_features if f in X_engineered.columns]
    X_selected = X_engineered[available_features]
    
    return selected_features, X_selected

def load_selected_features(X_engineered: pd.DataFrame) -> tuple:
    """
    Load previously selected features from file
    
    Args:
        X_engineered: Engineered features
        
    Returns:
        Tuple of (selected_features, X_selected)
    """
    selected_features_path = Path('models/selected_features.json')
    
    if not selected_features_path.exists():
        logger.warning("   No selected features file found")
        return None, X_engineered
    
    logger.info("   Loading existing selected features...")
    with open(selected_features_path, 'r') as f:
        selected_features = json.load(f)
    logger.info(f"   Loaded {len(selected_features)} selected features")
    
    # Filter X_engineered to only selected features
    available_features = [f for f in selected_features if f in X_engineered.columns]
    
    if not available_features:
        logger.warning("   No selected features found in data, using all features")
        return None, X_engineered
    
    if len(available_features) < len(selected_features):
        logger.warning(f"   Only {len(available_features)} of {len(selected_features)} features available")
    
    X_selected = X_engineered[available_features]
    logger.info(f"   Using {len(available_features)} available selected features")
    
    return available_features, X_selected

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