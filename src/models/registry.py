"""
Model Registry for versioning and managing ML models
"""
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


class ModelRegistry:
    """
    Model registry for versioning, storing, and managing ML models
    
    Features:
    - Versioned model storage
    - Metadata tracking (metrics, parameters, training date)
    - Model promotion (staging -> production)
    - Rollback capabilities
    - Model search and retrieval
    """
    
    def __init__(self, registry_dir: str = 'models/registry'):
        """
        Initialize the model registry
        
        Args:
            registry_dir: Directory to store models and metadata
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.registry_dir / 'registry_metadata.json'
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """
        Load registry metadata from file
        
        Returns:
            Dictionary containing registry metadata
        """
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            # Initialize empty registry
            return {
                'models': [],
                'latest_version': None,
                'production_version': None,
                'staging_version': None,
                'total_models': 0
            }
    
    def _save_metadata(self) -> None:
        """Save registry metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def register_model(self, 
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: Optional[str] = None,
        stage: str = 'staging'
    ) -> str:
        """
        Register a new model in the registry
        
        Args:
            model: Trained model object
            model_name: Name of the model (e.g., 'xgboost', 'random_forest')
            metrics: Dictionary of evaluation metrics
            description: Optional description of the model
            tags: Optional list of tags
            version: Optional version string (auto-generated if not provided)
            stage: Stage to place model ('staging' or 'production')
            
        Returns:
            Version string of the registered model
        """
        logger.info(f"Registering model: {model_name}")
        
        # Generate version if not provided
        if version is None:
            version = self._generate_version(model_name)
        
        # Create model directory
        model_dir = self.registry_dir / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save model info
        model_info = {
            'version': version,
            'model_name': model_name,
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'description': description,
            'tags': tags or [],
            'model_path': str(model_path),
            'file_size_mb': model_path.stat().st_size / (1024 * 1024),
            'parameters': self._get_model_params(model)
        }
        
        # Save metadata file for this model
        info_path = model_dir / 'model_info.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Update registry metadata
        self.metadata['models'].append(model_info)
        self.metadata['latest_version'] = version
        self.metadata['total_models'] += 1
        
        # Set stage
        if stage == 'production':
            self.metadata['production_version'] = version
        elif stage == 'staging':
            self.metadata['staging_version'] = version
        
        self._save_metadata()
        
        logger.success(f"Model {version} registered successfully!")
        logger.info(f"  Name: {model_name}")
        logger.info(f"  Stage: {stage}")
        logger.info(f"  Path: {model_path}")
        logger.info(f"  Metrics: {metrics}")
        
        return version
    
    def _generate_version(self, model_name: str) -> str:
        """
        Generate a version string
        
        Args:
            model_name: Name of the model
            
        Returns:
            Version string (e.g., 'xgboost_v1.0.0')
        """
        # Count existing versions of this model
        versions = [m['version'] for m in self.metadata['models']
                    if m['model_name'] == model_name]
        
        if not versions:
            # First version
            return f"{model_name}_v1.0.0"
        
        # Find latest version number
        version_numbers = []
        for v in versions:
            # Extract version number (e.g., 'v1.0.0' -> [1, 0, 0])
            try:
                parts = v.split('_v')[1].split('.')
                if len(parts) == 3:
                    version_numbers.append([int(x) for x in parts])
            except:
                continue
        
        if not version_numbers:
            return f"{model_name}_v1.0.0"
        
        # Get latest version
        latest = max(version_numbers)
        # Increment patch version
        new_version = f"v{latest[0]}.{latest[1]}.{latest[2] + 1}"
        
        return f"{model_name}_{new_version}"
    
    def _get_model_params(self, model: Any) -> Dict:
        """
        Extract model parameters
        
        Args:
            model: Trained model
            
        Returns:
            Dictionary of model parameters
        """
        if hasattr(model, 'get_params'):
            params = model.get_params()
            # Convert non-serializable objects
            for key, value in params.items():
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    params[key] = str(value)
            return params
        return {}
    
    def load_model(self, 
        version: Optional[str] = None, 
        stage: Optional[str] = None
    ) -> Tuple[Any, Dict]:
        """
        Load a model from the registry
        
        Args:
            version: Specific version to load (if None, uses stage or latest)
            stage: Stage to load from ('production', 'staging', or None)
            
        Returns:
            Tuple of (model, model_info)
        """
        # Determine which version to load
        if version is not None:
            # Load specific version
            model_info = self.get_model_info(version)
            if model_info is None:
                raise ValueError(f"Version {version} not found in registry")
        
        elif stage is not None:
            # Load by stage
            if stage == 'production' and self.metadata['production_version']:
                version = self.metadata['production_version']
            elif stage == 'staging' and self.metadata['staging_version']:
                version = self.metadata['staging_version']
            else:
                raise ValueError(f"No model found in {stage} stage")
            
            model_info = self.get_model_info(version)
        
        else:
            # Load latest version
            version = self.metadata['latest_version']
            if version is None:
                raise ValueError("No models found in registry")
            model_info = self.get_model_info(version)
        
        # Load model
        model_path = Path(model_info['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Loaded model {version}")
        logger.info(f"  Name: {model_info['model_name']}")
        logger.info(f"  Stage: {model_info['stage']}")
        
        return model, model_info
    
    def get_model_info(self, version: str) -> Optional[Dict]:
        """
        Get information about a specific model version
        
        Args:
            version: Model version string
            
        Returns:
            Dictionary with model information, or None if not found
        """
        for model in self.metadata['models']:
            if model['version'] == version:
                return model
        return None
    
    def list_models(self, 
        model_name: Optional[str] = None, 
        stage: Optional[str] = None
    ) -> List[Dict]:
        """
        List all models in the registry
        
        Args:
            model_name: Filter by model name
            stage: Filter by stage ('production', 'staging')
            
        Returns:
            List of model information dictionaries
        """
        models = self.metadata['models']
        
        if model_name is not None:
            models = [m for m in models if m['model_name'] == model_name]
        
        if stage is not None:
            models = [m for m in models if m['stage'] == stage]
        
        return models
    
    def promote_model(self, 
        version: str, 
        target_stage: str = 'production'
    ) -> None:
        """
        Promote a model to a different stage
        
        Args:
            version: Model version to promote
            target_stage: Stage to promote to ('production' or 'staging')
        """
        model_info = self.get_model_info(version)
        if model_info is None:
            raise ValueError(f"Version {version} not found in registry")
        
        # Update stage
        old_stage = model_info['stage']
        model_info['stage'] = target_stage
        
        # Update registry
        if target_stage == 'production':
            self.metadata['production_version'] = version
            # Remove from staging if it was there
            if self.metadata.get('staging_version') == version:
                self.metadata['staging_version'] = None
        elif target_stage == 'staging':
            self.metadata['staging_version'] = version
        
        # Update model info file
        model_dir = self.registry_dir / version
        info_path = model_dir / 'model_info.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        self._save_metadata()
        
        logger.success(f"Model {version} promoted from '{old_stage}' to '{target_stage}'")
    
    def rollback(self, stage: str = 'production') -> str:
        """
        Rollback to previous version in a stage
        
        Args:
            stage: Stage to rollback ('production' or 'staging')
            
        Returns:
            Version of the rolled-back model
        """
        if stage == 'production':
            current_version = self.metadata['production_version']
            if current_version is None:
                raise ValueError("No model in production to rollback")
            
            # Find previous version
            models = self.list_models(stage='production')
            if len(models) <= 1:
                raise ValueError("No previous version to rollback to")
            
            # Get the second most recent model
            sorted_models = sorted(models, key=lambda x: x['timestamp'], reverse=True)
            rollback_version = sorted_models[1]['version']
            
            # Promote to production
            self.promote_model(rollback_version, 'production')
            
            logger.info(f"Rolled back from {current_version} to {rollback_version}")
            return rollback_version
        
        else:
            raise ValueError(f"Rollback not implemented for stage: {stage}")
    
    def delete_model(self, version: str, force: bool = False) -> None:
        """
        Delete a model from the registry
        
        Args:
            version: Model version to delete
            force: Force deletion even if in production
        """
        model_info = self.get_model_info(version)
        if model_info is None:
            raise ValueError(f"Version {version} not found in registry")
        
        # Check if in production
        if model_info['stage'] == 'production' and not force:
            raise ValueError(f"Cannot delete production model {version}. Use force=True to override.")
        
        # Remove from registry
        self.metadata['models'] = [m for m in self.metadata['models'] if m['version'] != version]
        
        # Update latest version
        if self.metadata['latest_version'] == version:
            if self.metadata['models']:
                self.metadata['latest_version'] = max(
                    self.metadata['models'],
                    key=lambda x: x['timestamp']
                )['version']
            else:
                self.metadata['latest_version'] = None
        
        # Update production/staging
        if self.metadata.get('production_version') == version:
            self.metadata['production_version'] = None
        if self.metadata.get('staging_version') == version:
            self.metadata['staging_version'] = None
        
        # Delete model files
        model_dir = self.registry_dir / version
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
        
        self.metadata['total_models'] -= 1
        self._save_metadata()
        
        logger.info(f"Deleted model {version}")
    
    def compare_models(self, version1: str, version2: str) -> Dict:
        """
        Compare two models in the registry
        
        Args:
            version1: First model version
            version2: Second model version
            
        Returns:
            Dictionary with comparison results
        """
        model1_info = self.get_model_info(version1)
        model2_info = self.get_model_info(version2)
        
        if model1_info is None or model2_info is None:
            raise ValueError("One or both models not found")
        
        # Compare metrics
        metrics1 = model1_info.get('metrics', {})
        metrics2 = model2_info.get('metrics', {})
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'model1_name': model1_info.get('model_name'),
            'model2_name': model2_info.get('model_name'),
            'timestamp1': model1_info.get('timestamp'),
            'timestamp2': model2_info.get('timestamp'),
            'stage1': model1_info.get('stage'),
            'stage2': model2_info.get('stage'),
            'metrics': {}
        }
        
        # Compare each metric
        all_metrics = set(metrics1.keys()) | set(metrics2.keys())
        for metric in all_metrics:
            val1 = metrics1.get(metric)
            val2 = metrics2.get(metric)
            
            if val1 is not None and val2 is not None:
                difference = val1 - val2
                percent_change = (difference / abs(val2)) * 100 if val2 != 0 else float('inf')
                
                comparison['metrics'][metric] = {
                    version1: val1,
                    version2: val2,
                    'difference': difference,
                    'percent_change': percent_change,
                    'better': version1 if val1 > val2 else version2
                }
        
        logger.info(f"Comparison between {version1} and {version2}")
        for metric, values in comparison['metrics'].items():
            logger.info(f"  {metric}: {values[version1]:.4f} vs {values[version2]:.4f} ({values['percent_change']:+.2f}%)")
        
        return comparison
    
    def get_best_model(self, metric: str = 'f1') -> Optional[Dict]:
        """
        Get the best performing model based on a metric
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Model information of the best model
        """
        best_model = None
        best_score = float('-inf')
        
        for model in self.metadata['models']:
            score = model.get('metrics', {}).get(metric)
            if score is not None and score > best_score:
                best_score = score
                best_model = model
        
        if best_model:
            logger.info(f"Best model: {best_model['version']} ({metric}: {best_score:.4f})")
        
        return best_model
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get a summary of all models in the registry
        
        Returns:
            DataFrame with model summary
        """
        if not self.metadata['models']:
            return pd.DataFrame()
        
        summary_data = []
        for model in self.metadata['models']:
            row = {
                'version': model['version'],
                'model_name': model['model_name'],
                'stage': model['stage'],
                'timestamp': model['timestamp'][:10],  # Date only
                'file_size_mb': model.get('file_size_mb', 0),
                'description': model.get('description', '')[:30]
            }
            # Add metrics
            for metric, value in model.get('metrics', {}).items():
                row[f'metric_{metric}'] = round(value, 4)
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        return df.sort_values('timestamp', ascending=False)
    
    def print_summary(self) -> None:
        """Print a summary of the registry"""
        print("\n" + "=" * 70)
        print("MODEL REGISTRY SUMMARY")
        print("=" * 70)
        
        print(f"\nTotal Models: {self.metadata['total_models']}")
        print(f"Latest Version: {self.metadata['latest_version']}")
        print(f"Production Version: {self.metadata['production_version'] or 'None'}")
        print(f"Staging Version: {self.metadata['staging_version'] or 'None'}")
        
        # Group by model name
        models = self.metadata['models']
        if models:
            print("\nModels by Name:")
            model_names = {}
            for model in models:
                name = model['model_name']
                if name not in model_names:
                    model_names[name] = []
                model_names[name].append(model['version'])
            
            for name, versions in model_names.items():
                print(f"  {name}: {len(versions)} versions")
                print(f"    Latest: {versions[-1]}")
                if any(self.metadata['production_version'] == v for v in versions):
                    print(f"    Production: {self.metadata['production_version']}")
        
        print("\n" + "=" * 70)
    
    def export_registry_report(
        self, 
        output_path: str = 'reports/model_registry_report.md'
    ) -> None:
        """
        Export a markdown report of the registry
        
        Args:
            output_path: Path to save the report
        """
        df = self.get_model_summary()
        
        report = f"""# Model Registry Report

Generated: {datetime.now().isoformat()}

## Summary

- **Total Models**: {self.metadata['total_models']}
- **Latest Version**: {self.metadata['latest_version']}
- **Production Version**: {self.metadata['production_version'] or 'None'}
- **Staging Version**: {self.metadata['staging_version'] or 'None'}

## All Models

{df.to_markdown(index=False)}

## Best Performing Models

| Metric | Best Version | Score |
|--------|-------------|-------|
"""
        
        # Add best models for key metrics
        for metric in ['f1', 'accuracy', 'precision', 'recall']:
            best = self.get_best_model(metric)
            if best:
                report += f"| {metric} | {best['version']} | {best['metrics'].get(metric, 'N/A')} |\n"
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Registry report saved to {output_path}")


# Convenience functions for common operations

def get_production_model(registry_dir: str = 'models/registry') -> Tuple[Any, Dict]:
    """
    Load the current production model
    
    Args:
        registry_dir: Registry directory
        
    Returns:
        Tuple of (model, model_info)
    """
    registry = ModelRegistry(registry_dir)
    return registry.load_model(stage='production')


def get_staging_model(registry_dir: str = 'models/registry') -> Tuple[Any, Dict]:
    """
    Load the current staging model
    
    Args:
        registry_dir: Registry directory
        
    Returns:
        Tuple of (model, model_info)
    """
    registry = ModelRegistry(registry_dir)
    return registry.load_model(stage='staging')


def get_latest_model(registry_dir: str = 'models/registry') -> Tuple[Any, Dict]:
    """
    Load the latest model
    
    Args:
        registry_dir: Registry directory
        
    Returns:
        Tuple of (model, model_info)
    """
    registry = ModelRegistry(registry_dir)
    return registry.load_model()
