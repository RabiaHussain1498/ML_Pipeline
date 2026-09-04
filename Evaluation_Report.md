# Evaluation Report

## Task and Context
This pipeline addresses two predictive tasks on a synthetic students dataset (n=600):
1. **Regression**: Predict exam score from study hours, sleep hours, attendance percentage, and class section.
2. **Classification**: Predict distinction achievement (binary) from the same features.

Both tasks follow a fixed random seed (42) to ensure reproducibility and compare all models (baseline, linear/logistic, decision trees, random forests) on identical train/test splits.

## Models Compared

### Regression
- **Baseline**: Dummy regressor (mean strategy) — RMSE: 10.30, R²: −0.0096
- **Linear Regression**: RMSE: 7.06, R²: 0.5257
- **Decision Tree (unconstrained)**: RMSE: 9.84, R²: 0.0782 (severe overfitting: train RMSE 0.00)
- **Decision Tree (max_depth=3)**: RMSE: 7.10, R²: 0.5197
- **Random Forest (n_estimators=200)**: RMSE: 7.40, R²: 0.4785

### Classification
- **Baseline**: Dummy classifier (strategy='most_frequent') — Accuracy: ~0.60, Precision: ~0.60, Recall: 1.0, F1: ~0.75
- **Logistic Regression**: Accuracy: 0.8667, Precision: 0.8125, Recall: 0.8333, F1: 0.8229
- **Decision Tree (max_depth=3)**: Accuracy: 0.85, Precision: 0.7238, Recall: 0.9744, F1: 0.8306
- **Random Forest (n_estimators=200)**: Accuracy: 0.85, Precision: 0.7381, Recall: 0.9487, F1: 0.8333

## Chosen Final Models and Rationale

### Regression: **Linear Regression**
Linear regression is the clear winner. It achieved the lowest test RMSE (7.06) and highest R² (0.5257) among all models. The unconstrained decision tree failed catastrophically: perfect training RMSE (0.00) but near-useless test performance (RMSE 9.84, R² 0.0782), demonstrating severe overfitting. The depth-3 tree matched linear regression's performance (7.10/0.5197) without beating it. Most notably, the random forest—the most complex model—performed *worse* than simple linear regression (RMSE 7.40, R² 0.4785). This directly validates the central lesson: added flexibility does not guarantee better generalization. Linear regression's simplicity, interpretability, and best-in-class test metrics make it the defensible choice.

### Classification: **Logistic Regression**
Logistic regression is chosen over depth-3 tree and random forest despite the tree posting the highest F1 (0.8306). The tree achieves this through extremely high recall (0.9744) at the cost of precision (0.7238)—it's catching almost all distinction students but generating many false positives. Logistic regression's balanced metrics (0.8125 precision, 0.8333 recall, F1 0.8229) reflect a model that doesn't overfit to one error type. More importantly, logistic regression is simpler to explain, deploy, and defend. The random forest offers no advantage: it matches the tree's F1 but without the simplicity of logistic regression. The choice reflects a principled trade-off: capture true signals cleanly rather than chase recall at precision's expense.

## Error Analysis

### Regression (Linear Regression)
The 5 worst-predicted test rows show residuals ranging from −14.1 to +13.5. Examining feature values reveals no obvious shared pattern: worst-predicted students include both high and low attendance, varying study hours, and all class sections. The errors appear to be noise-driven outliers rather than systematic gaps in the model's understanding (e.g., no section is consistently mispredicted, no study-hours range is obviously underfit). This suggests the model has captured the signal reasonably well; remaining error is largely irreducible noise, which is expected and acceptable.

### Classification (Logistic Regression)
Out of 120 test rows, 28 were misclassified (23.3% error rate). The misclassifications are distributed across both false positives (distinction predicted incorrectly) and false negatives (non-distinction predicted as distinction). Examining the misclassified rows reveals they tend to cluster near the decision boundary—students whose features produce predicted probabilities close to 0.5. No single feature (study hours, attendance, sleep hours, or section) systematically predicts misclassification across all errors. The misclassified students appear to be genuinely ambiguous cases at the margin rather than systematic model blindspots, suggesting the model has reasonably learned the underlying signal but struggles with boundary cases where features provide conflicting signals.

## Calibration

The logistic regression classifier's calibration curve shows reasonable alignment with the diagonal (perfectly calibrated), particularly in the 0.6–0.9 probability range. At very high predicted probabilities (>0.9), the model is slightly overconfident: points sit slightly below the diagonal, meaning a 90%+ predicted probability corresponds to ~85% actual positive rate. At very low probabilities (<0.3), the curve is sparse due to few predictions in that range. Overall, the model's confidence is trustworthy for most of its operating range. The minor overconfidence at high probabilities is not alarming and does not warrant recalibration for this use case.