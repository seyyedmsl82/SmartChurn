## Model Evaluation

The evaluation pipeline provides comprehensive model assessment with business metrics and interpretability.

### Evaluation Features

#### 1. Business-Centric Metrics
- **Churn Detection Rate**: Percentage of actual churners correctly identified (Recall)
- **False Alarm Rate**: Percentage of non-churners incorrectly flagged (False Positive Rate)
- **Cost Analysis**: Quantifies financial impact of misclassifications
  - False Negative cost: $500 per missed churner
  - False Positive cost: $50 per unnecessary retention offer
- **Cost Savings**: Estimated savings from using the model vs. random action

#### 2. Model Performance Metrics
| Metric | Description | Our Score |
|--------|-------------|-----------|
| Accuracy | Overall correct predictions | ~84% |
| Precision | Correct churn predictions / total churn predictions | ~0.75 |
| Recall | Correct churn predictions / actual churners | ~0.82 |
| F1 Score | Harmonic mean of precision & recall | **0.7838** |
| ROC-AUC | Ability to distinguish between classes | 0.8405 |
| PR-AUC | Precision-Recall area under curve | ~0.68 |

#### 3. Threshold Optimization
Default probability threshold of 0.5 is rarely optimal. The evaluation pipeline:
- Tests 50+ thresholds from 0.1 to 0.9
- Optimizes for F1 score (balanced approach)
- Optimizes for cost reduction (business approach)

**Current Best Threshold: 0.45** (improves F1 by 2-3%)

### Running Evaluation

```bash
# Run full evaluation pipeline
python scripts/evaluate_model.py
```
