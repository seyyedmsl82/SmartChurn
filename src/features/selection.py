"""
Feature selection module for churn prediction
"""
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (RFECV, SelectFromModel,
                                       SelectKBest, f_classif,
                                       mutual_info_classif)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class FeatureSelector:
    """
    Comprehensive feature selection with multiple methods
    """
    
    def __init__(self, 
        n_features: Optional[int] = None,
        method: str = 'combined',
        random_state: int = 42
    ):
        """
        Initialize feature selector
        
        Args:
            n_features: Number of features to select
            method: Selection method ('filter', 'wrapper', 'embedded', 'combined')
            random_state: Random seed for reproducibility
        """
        self.n_features = n_features
        self.method = method
        self.random_state = random_state
        self.selected_features = None
        self.feature_importance = {}
    
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Select features using specified method
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Starting feature selection using {self.method} method")
        
        if self.method == 'filter':
            selected = self._filter_method(X, y)
        elif self.method == 'wrapper':
            selected = self._wrapper_method(X, y)
        elif self.method == 'embedded':
            selected = self._embedded_method(X, y)
        else:  # combined
            selected = self._combined_method(X, y)
        
        self.selected_features = selected
        logger.info(f"Selected {len(selected)} features: {selected[:5]}...")
        
        return selected
    
    def _filter_method(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Filter method using statistical tests"""
        # Use both F-test and mutual information
        selector_f = SelectKBest(f_classif, k=self.n_features or min(20, X.shape[1]))
        selector_f.fit(X, y)
        f_scores = selector_f.scores_
        
        selector_mi = SelectKBest(mutual_info_classif, k=self.n_features or min(20, X.shape[1]))
        selector_mi.fit(X, y)
        mi_scores = selector_mi.scores_
        
        # Combine scores
        combined_scores = (f_scores + mi_scores) / 2
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': combined_scores
        }).sort_values('score', ascending=False)
        
        self.feature_importance['filter'] = feature_scores.to_dict('records')
        
        # Select top features
        n_select = self.n_features or min(20, len(X.columns))
        selected = feature_scores.head(n_select)['feature'].tolist()
        
        return selected
    
    def _wrapper_method(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Wrapper method using Recursive Feature Elimination"""
        # Scale features for linear models
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use logistic regression as base estimator
        estimator = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            class_weight='balanced'
        )
        
        # RFE with cross-validation
        n_select = self.n_features or min(20, X.shape[1])
        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=min(5, X.shape[0] // 10),  # Adjust CV based on data size
            min_features_to_select=min(n_select, X.shape[1]),
            n_jobs=-1
        )
        rfecv.fit(X_scaled, y)
        
        # Get selected features
        selected = X.columns[rfecv.support_].tolist()
        
        # Store importance
        self.feature_importance['wrapper'] = {
            'ranking': dict(zip(X.columns, rfecv.ranking_)),
            'selected': selected
        }
        
        return selected
    
    def _embedded_method(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Embedded method using feature importance from tree models"""
        # Use Random Forest for feature importance
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            class_weight='balanced'
        )
        rf.fit(X, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select from model with threshold
        selector = SelectFromModel(
            rf,
            threshold='median',
            max_features=self.n_features
        )
        selector.fit(X, y)
        
        selected = X.columns[selector.get_support()].tolist()
        
        self.feature_importance['embedded'] = {
            'importance': importance.to_dict('records'),
            'threshold': selector.threshold_,
            'selected': selected
        }
        
        return selected
    
    def _combined_method(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Combine multiple feature selection methods"""
        # Get selections from each method
        try:
            filter_selected = self._filter_method(X, y)
        except:
            filter_selected = []
            logger.warning("Filter method failed, continuing...")
        
        try:
            wrapper_selected = self._wrapper_method(X, y)
        except:
            wrapper_selected = []
            logger.warning("Wrapper method failed, continuing...")
        
        try:
            embedded_selected = self._embedded_method(X, y)
        except:
            embedded_selected = []
            logger.warning("Embedded method failed, continuing...")
        
        # If all methods failed, return top features from Random Forest
        if not filter_selected and not wrapper_selected and not embedded_selected:
            logger.warning("All selection methods failed. Using Random Forest importance.")
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            rf.fit(X, y)
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            n_select = self.n_features or min(20, len(X.columns))
            selected = importance.head(n_select)['feature'].tolist()
            
            self.feature_importance['combined'] = {
                'method': 'fallback_random_forest',
                'selected': selected
            }
            
            return selected
        
        # Combine selections
        all_selected = list(set(filter_selected + wrapper_selected + embedded_selected))
        
        # If combined list is empty, use embedded selection
        if not all_selected:
            all_selected = embedded_selected if embedded_selected else filter_selected
        
        # Count selections for features that appear in multiple methods
        selection_counts = {}
        for feature in all_selected:
            count = 0
            if feature in filter_selected:
                count += 1
            if feature in wrapper_selected:
                count += 1
            if feature in embedded_selected:
                count += 1
            selection_counts[feature] = count
        
        # Sort by count (features selected by more methods first)
        sorted_features = sorted(selection_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Get consensus features (selected by at least 2 methods)
        consensus_features = [f for f, c in sorted_features if c >= 2]
        
        # If no consensus, use all features sorted by selection count
        if not consensus_features:
            consensus_features = [f for f, _ in sorted_features]
        
        # Limit to n_features
        if self.n_features and len(consensus_features) > self.n_features:
            # Use importance from embedded method for ranking
            try:
                rf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    class_weight='balanced'
                )
                rf.fit(X[consensus_features], y)
                
                importance = pd.DataFrame({
                    'feature': consensus_features,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=False)
                
                consensus_features = importance.head(self.n_features)['feature'].tolist()
            except:
                # Fallback: take first n_features
                consensus_features = consensus_features[:self.n_features]
        
        self.feature_importance['combined'] = {
            'selection_counts': selection_counts,
            'selected': consensus_features,
            'all_candidates': all_selected
        }
        
        return consensus_features
    
    def plot_feature_importance(
        self, 
        X: pd.DataFrame, y: pd.Series, 
        top_n: int = 20, save_path: Optional[str] = None
    ):
        """Plot feature importance"""
        # Train a Random Forest for importance
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            class_weight='balanced'
        )
        rf.fit(X, y)
        
        # Get importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(importance['feature'], importance['importance'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (_, row) in enumerate(importance.iterrows()):
            ax.text(row['importance'] + 0.001, i, f'{row["importance"]:.3f}', 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def analyze_correlations(self, X: pd.DataFrame, threshold: float = 0.8):
        """Analyze feature correlations and identify highly correlated pairs"""
        # Select only numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]
        
        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric columns for correlation analysis")
            return []
        
        corr_matrix = X_numeric.corr().abs()
        
        # Find highly correlated pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_pairs = []
        for col in upper_tri.columns:
            high_corr = upper_tri[col][upper_tri[col] > threshold]
            for idx, value in high_corr.items():
                high_corr_pairs.append((col, idx, value))
        
        if high_corr_pairs:
            logger.info(f"Found {len(high_corr_pairs)} highly correlated pairs")
            for pair in high_corr_pairs[:5]:
                logger.info(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")
        
        return high_corr_pairs
    
    def get_recommendations(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Get feature selection recommendations"""
        # Run all selection methods
        try:
            filter_selected = self._filter_method(X, y)
        except:
            filter_selected = []
        
        try:
            wrapper_selected = self._wrapper_method(X, y)
        except:
            wrapper_selected = []
        
        try:
            embedded_selected = self._embedded_method(X, y)
        except:
            embedded_selected = []
        
        # Combined selection
        combined_selected = list(set(
            filter_selected + wrapper_selected + embedded_selected
        ))
        
        # Count frequencies
        all_selected = filter_selected + wrapper_selected + embedded_selected
        freq = pd.Series(all_selected).value_counts() if all_selected else pd.Series()
        
        recommendations = {
            'filter_method': {
                'selected': filter_selected,
                'n_features': len(filter_selected)
            },
            'wrapper_method': {
                'selected': wrapper_selected,
                'n_features': len(wrapper_selected)
            },
            'embedded_method': {
                'selected': embedded_selected,
                'n_features': len(embedded_selected)
            },
            'combined': {
                'selected': combined_selected,
                'n_features': len(combined_selected)
            },
            'selection_frequency': freq.head(20).to_dict(),
            'recommendation': combined_selected if len(combined_selected) <= 20 
                            else freq.head(20).index.tolist() if not freq.empty else combined_selected[:20]
        }
        
        return recommendations


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
