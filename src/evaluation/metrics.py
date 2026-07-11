"""
Comprehensive model evaluation metrics
"""
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)


class ModelEvaluator:
    """
    Comprehensive model evaluation with business metrics
    """
    
    def __init__(self, 
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 y_proba: np.ndarray = None):
        """
        Initialize evaluator
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        
        self.metrics = {}
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate all metrics"""
        
        # Basic metrics
        self.metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        self.metrics['precision'] = precision_score(self.y_true, self.y_pred)
        self.metrics['recall'] = recall_score(self.y_true, self.y_pred)
        self.metrics['f1'] = f1_score(self.y_true, self.y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        self.metrics['confusion_matrix'] = cm.tolist()
        
        # Extract confusion matrix components
        tn, fp, fn, tp = cm.ravel()
        self.metrics['true_negatives'] = int(tn)
        self.metrics['false_positives'] = int(fp)
        self.metrics['false_negatives'] = int(fn)
        self.metrics['true_positives'] = int(tp)
        
        # Business metrics for churn
        self.metrics['churn_detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        self.metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Cost-based metrics
        # Assume: False Negative (missing a churner) costs $500
        # False Positive (unnecessary retention offer) costs $50
        self.metrics['fn_cost'] = 500
        self.metrics['fp_cost'] = 50
        
        total_cost = (fn * 500 + fp * 50)
        self.metrics['total_cost'] = total_cost
        
        # Cost per customer
        total_customers = len(self.y_true)
        self.metrics['cost_per_customer'] = total_cost / total_customers
        
        # If probabilities are available
        if self.y_proba is not None:
            self.metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_proba)
            self.metrics['pr_auc'] = average_precision_score(self.y_true, self.y_proba)
            
            # ROC curve points
            fpr, tpr, thresholds = roc_curve(self.y_true, self.y_proba)
            self.metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist()
            }
            
            # Precision-Recall curve points
            precision, recall, pr_thresholds = precision_recall_curve(
                self.y_true, self.y_proba
            )
            self.metrics['pr_curve'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
    
    def get_metrics(self, include_all: bool = False) -> Dict[str, Any]:
        """Get calculated metrics"""
        if include_all:
            return self.metrics
        
        # Return only key metrics
        key_metrics = {
            'accuracy': self.metrics['accuracy'],
            'precision': self.metrics['precision'],
            'recall': self.metrics['recall'],
            'f1': self.metrics['f1'],
            'churn_detection_rate': self.metrics['churn_detection_rate'],
            'false_alarm_rate': self.metrics['false_alarm_rate'],
            'cost_per_customer': self.metrics.get('cost_per_customer'),
            'total_cost': self.metrics.get('total_cost')
        }
        
        if 'roc_auc' in self.metrics:
            key_metrics['roc_auc'] = self.metrics['roc_auc']
            key_metrics['pr_auc'] = self.metrics['pr_auc']
        
        return key_metrics
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        summary = f"""
        ========================================
        MODEL PERFORMANCE SUMMARY
        ========================================
        
        Key Metrics:
        ------------
        Accuracy:              {self.metrics['accuracy']:.4f}
        Precision:             {self.metrics['precision']:.4f}
        Recall (Churn):        {self.metrics['recall']:.4f}
        F1 Score:              {self.metrics['f1']:.4f}
        
        Business Metrics:
        -----------------
        Churn Detection Rate:  {self.metrics['churn_detection_rate']:.2%}
        False Alarm Rate:      {self.metrics['false_alarm_rate']:.2%}
        
        Confusion Matrix:
        -----------------
        True Negatives:  {self.metrics['true_negatives']}
        False Positives: {self.metrics['false_positives']}
        False Negatives: {self.metrics['false_negatives']}
        True Positives:  {self.metrics['true_positives']}
        
        Costs (Estimated):
        -----------------
        FN Cost:         ${self.metrics['fn_cost']} each
        FP Cost:         ${self.metrics['fp_cost']} each
        Total Cost:      ${self.metrics['total_cost']:,.2f}
        Cost/Customer:   ${self.metrics['cost_per_customer']:.2f}
        """
        
        if 'roc_auc' in self.metrics:
            summary += f"""
        Probabilistic Metrics:
        ----------------------
        ROC-AUC:             {self.metrics['roc_auc']:.4f}
        PR-AUC:              {self.metrics['pr_auc']:.4f}
        """
        
        summary += "=" * 40
        return summary
    
    def get_classification_report(self) -> str:
        """Get detailed classification report"""
        return classification_report(
            self.y_true, 
            self.y_pred,
            target_names=['Not Churn', 'Churn']
        )
    
    def find_optimal_threshold(self, metric: str = 'f1') -> Tuple[float, Dict]:
        """
        Find optimal threshold for probability predictions
        
        Args:
            metric: Metric to optimize ('f1', 'precision', 'recall', 'cost')
            
        Returns:
            Optimal threshold and metrics at that threshold
        """
        if self.y_proba is None:
            raise ValueError("Predictions probabilities required for threshold tuning")
        
        thresholds = np.linspace(0.1, 0.9, 50)
        best_threshold = 0.5
        best_score = 0
        best_metrics = {}
        
        for threshold in thresholds:
            y_pred_thresh = (self.y_proba >= threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(self.y_true, y_pred_thresh)
            recall = recall_score(self.y_true, y_pred_thresh)
            f1 = f1_score(self.y_true, y_pred_thresh)
            
            # Calculate cost
            cm = confusion_matrix(self.y_true, y_pred_thresh)
            tn, fp, fn, tp = cm.ravel()
            total_cost = fn * 500 + fp * 50
            
            if metric == 'f1' and f1 > best_score:
                best_score = f1
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'cost': total_cost
                }
            elif metric == 'precision' and precision > best_score:
                best_score = precision
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'cost': total_cost
                }
            elif metric == 'recall' and recall > best_score:
                best_score = recall
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'cost': total_cost
                }
            elif metric == 'cost' and total_cost < best_score:
                best_score = total_cost
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'cost': total_cost
                }
        
        return best_threshold, best_metrics
