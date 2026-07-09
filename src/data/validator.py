"""
Data validation module for ensuring data quality and schema compliance
"""
from pathlib import Path
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """Add an error to the validation result"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the validation result"""
        self.warnings.append(warning)

class DataValidator:
    """
    Validates data quality and schema compliance
    """
    
    def __init__(self, schema_path: Optional[str] = None, strict: bool = False):
        """
        Initialize the validator with a schema
        
        Args:
            schema_path: Path to schema YAML file
            strict: If True, fail on data type mismatches. If False, try to convert.
        """
        self.schema = self._load_schema(schema_path) if schema_path else self._default_schema()
        self.nullable_columns = self.schema.get('nullable_columns', [])
        self.strict = strict
    
    def _default_schema(self) -> Dict[str, Any]:
        """Default schema for Telco Churn dataset"""
        return {
            'required_columns': [
                'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
            ],
            'dtypes': {
                'customerID': 'object',
                'gender': 'object',
                'SeniorCitizen': 'int64',
                'Partner': 'object',
                'Dependents': 'object',
                'tenure': 'int64',
                'PhoneService': 'object',
                'MultipleLines': 'object',
                'InternetService': 'object',
                'OnlineSecurity': 'object',
                'OnlineBackup': 'object',
                'DeviceProtection': 'object',
                'TechSupport': 'object',
                'StreamingTV': 'object',
                'StreamingMovies': 'object',
                'Contract': 'object',
                'PaperlessBilling': 'object',
                'PaymentMethod': 'object',
                'MonthlyCharges': 'float64',
                'TotalCharges': 'float64',
                'Churn': 'object'
            },
            'categorical_columns': [
                'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod', 'Churn'
            ],
            'numeric_columns': [
                'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'
            ],
            'ranges': {
                'tenure': (0, 72),
                'SeniorCitizen': (0, 1),
                'MonthlyCharges': (0, 200),
                'TotalCharges': (0, 10000)
            },
            'categorical_values': {
                'gender': ['Female', 'Male'],
                'Partner': ['Yes', 'No'],
                'Dependents': ['Yes', 'No'],
                'PhoneService': ['Yes', 'No'],
                'MultipleLines': ['Yes', 'No', 'No phone service'],
                'InternetService': ['DSL', 'Fiber optic', 'No'],
                'OnlineSecurity': ['Yes', 'No', 'No internet service'],
                'OnlineBackup': ['Yes', 'No', 'No internet service'],
                'DeviceProtection': ['Yes', 'No', 'No internet service'],
                'TechSupport': ['Yes', 'No', 'No internet service'],
                'StreamingTV': ['Yes', 'No', 'No internet service'],
                'StreamingMovies': ['Yes', 'No', 'No internet service'],
                'Contract': ['Month-to-month', 'One year', 'Two year'],
                'PaperlessBilling': ['Yes', 'No'],
                'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
                'Churn': ['Yes', 'No']
            },
            'nullable_columns': ['TotalCharges'],  # Some TotalCharges might be missing
            'max_missing_rate': 0.1,  # Max 10% missing allowed
            'min_rows': 100,
            'max_rows': 1000000
        }
    
    def _load_schema(self, schema_path: str) -> Dict[str, Any]:
        """Load schema from YAML file"""
        with open(schema_path, 'r') as f:
            schema = yaml.safe_load(f)
        logger.info(f"Loaded schema from {schema_path}")
        return schema
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Run all validations on the DataFrame
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with all validation outcomes
        """
        result = ValidationResult(is_valid=True)
        
        # Run all validations
        self._validate_required_columns(df, result)
        self._validate_dtypes(df, result)
        self._validate_row_count(df, result)
        self._validate_missing_values(df, result)
        self._validate_categorical_values(df, result)
        self._validate_numeric_ranges(df, result)
        self._validate_unique_ids(df, result)
        
        # Log summary
        if result.is_valid:
            logger.success("Data validation passed!")
        else:
            logger.error(f"Data validation failed with {len(result.errors)} errors")
            for error in result.errors:
                logger.error(f"  - {error}")
        
        if result.warnings:
            logger.warning(f"{len(result.warnings)} warnings:")
            for warning in result.warnings:
                logger.warning(f"  - {warning}")
        
        return result
    
    def _validate_required_columns(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check that all required columns are present"""
        missing = set(self.schema['required_columns']) - set(df.columns)
        if missing:
            result.add_error(f"Missing required columns: {missing}")
    
    def _validate_dtypes(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check column data types with lenient conversion"""
        for col, expected_dtype in self.schema['dtypes'].items():
            if col not in df.columns:
                continue
                
            actual_dtype = str(df[col].dtype)
            
            # If strict mode, fail on type mismatch
            if self.strict:
                if expected_dtype not in actual_dtype:
                    result.add_error(f"Column '{col}' expected dtype {expected_dtype}, got {actual_dtype}")
                continue
            
            # Lenient mode: try to convert
            if expected_dtype in ['float64', 'int64']:
                # For numeric columns, try to convert strings to numbers
                try:
                    # Handle empty strings and spaces
                    cleaned = df[col].astype(str).str.strip().replace('', np.nan)
                    converted = pd.to_numeric(cleaned, errors='coerce')
                    
                    # Check if conversion lost too many values
                    loss_ratio = converted.isna().sum() / len(df)
                    if loss_ratio > 0.05:  # More than 5% loss
                        result.add_warning(
                            f"Column '{col}' had {loss_ratio:.1%} values that couldn't be converted to {expected_dtype}"
                        )
                    
                    # Update the dataframe in place for later validations
                    df[col] = converted
                    
                except Exception as e:
                    if not self.strict:
                        result.add_warning(f"Column '{col}' couldn't be converted to {expected_dtype}: {e}")
    
    def _validate_row_count(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check that the dataset has an appropriate number of rows"""
        n_rows = len(df)
        if n_rows < self.schema['min_rows']:
            result.add_error(f"Dataset has {n_rows} rows, minimum required is {self.schema['min_rows']}")
        if n_rows > self.schema['max_rows']:
            result.add_warning(f"Dataset has {n_rows} rows, which exceeds the expected maximum of {self.schema['max_rows']}")
    
    def _validate_missing_values(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for missing values"""
        missing_counts = df.isnull().sum()
        
        # Check if any missing values in non-nullable columns
        for col, count in missing_counts.items():
            if count > 0 and col not in self.schema['nullable_columns']:
                result.add_error(f"Column '{col}' has {count} missing values but is not nullable")
        
        # Calculate overall missing rate
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isnull().sum().sum()
        missing_rate = total_missing / total_cells if total_cells > 0 else 0
        
        if missing_rate > self.schema['max_missing_rate']:
            result.add_error(f"Overall missing rate {missing_rate:.2%} exceeds maximum {self.schema['max_missing_rate']:.0%}")
        elif missing_rate > 0:
            result.add_warning(f"Dataset has {missing_rate:.2%} missing values overall")
    
    def _validate_categorical_values(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check that categorical columns contain only valid values"""
        for col, expected_values in self.schema['categorical_values'].items():
            if col in df.columns and col not in self.nullable_columns:
                unique_values = set(df[col].dropna().unique())
                unexpected = unique_values - set(expected_values)
                if unexpected:
                    result.add_warning(f"Column '{col}' has unexpected values: {unexpected}")
    
    def _validate_numeric_ranges(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check that numeric columns fall within expected ranges"""
        for col, (min_val, max_val) in self.schema['ranges'].items():
            if col not in df.columns:
                continue
                
            # Skip if column is not numeric (already handled in dtype validation)
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            # Skip if all values are NaN
            if df[col].isna().all():
                result.add_warning(f"Column '{col}' has all missing values, cannot validate ranges")
                continue
            
            # Check min
            min_actual = df[col].min()
            if min_actual < min_val:
                result.add_error(f"Column '{col}' has minimum value {min_actual} below {min_val}")
            
            # Check max
            max_actual = df[col].max()
            if max_actual > max_val:
                result.add_error(f"Column '{col}' has maximum value {max_actual} above {max_val}")
    
    def _validate_unique_ids(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check that customerID is unique"""
        if 'customerID' in df.columns:
            unique_count = df['customerID'].nunique()
            total_count = len(df)
            if unique_count < total_count:
                result.add_error(f"customerID has {total_count - unique_count} duplicate values")\
                