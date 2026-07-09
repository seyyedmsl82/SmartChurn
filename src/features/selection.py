"""
Feature selection module for churn prediction
"""
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (RFE, RFECV, SelectFromModel,
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
                 random_state: int = 42):
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
        selector_f = SelectKBest(f_classif, k=self.n_features or 20)
        selector_f.fit(X, y)
        f_scores = selector_f.scores_
        
        selector_mi = SelectKBest(mutual_info_classif, k=self.n_features or 20)
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
            cv=5,
            min_features_to_select=n_select,
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
        filter_selected = self._filter_method(X, y)
        wrapper_selected = self._wrapper_method(X, y)
        embedded_selected = self._embedded_method(X, y)
        
        # Find features selected by at least 2 methods
        all_selected = set(filter_selected + wrapper_selected + embedded_selected)
        
        # Count selections
        selection_counts = pd.Series(
            filter_selected + wrapper_selected + embedded_selected
        ).value_counts()
        
        # Features selected by at least 2 methods
        consensus_features = selection_counts[selection_counts >= 2].index.tolist()
        
        if not consensus_features:
            # Fallback: use top features from each method
            consensus_features = all_selected
        
        # Limit to n_features
        if self.n_features and len(consensus_features) > self.n_features:
            # Use importance from embedded method for ranking
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
        
        self.feature_importance['combined'] = {
            'selection_counts': selection_counts.to_dict(),
            'selected': consensus_features
        }
        
        return consensus_features
    
    def plot_feature_importance(self, X: pd.DataFrame, y: pd.Series, 
                                top_n: int = 20, save_path: Optional[str] = None):
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
        }).sort_values('importance', ascending=True).tail(top_n)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(importance['feature'], importance['importance'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def analyze_correlations(self, X: pd.DataFrame, threshold: float = 0.8):
        """Analyze feature correlations and identify highly correlated pairs"""
        corr_matrix = X.corr().abs()
        
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
        filter_selected = self._filter_method(X, y)
        wrapper_selected = self._wrapper_method(X, y)
        embedded_selected = self._embedded_method(X, y)
        
        # Combined selection
        combined_selected = list(set(
            filter_selected + wrapper_selected + embedded_selected
        ))
        
        # Count frequencies
        all_selected = filter_selected + wrapper_selected + embedded_selected
        freq = pd.Series(all_selected).value_counts()
        
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
                            else freq.head(20).index.tolist()
        }
        
        return recommendations
