# **Delhivery-system-Capstone-project-**
Test repo for capstone project

## Data findings (PART - A)

1. **Service Type Delay Rates**: Express shipments experience the highest delay frequency at a delay rate of `0.4897` (119 delayed out of 243 shipments), followed by Standard at `0.2056`, and Economy at `0.0857`.
2. **Hub Tier Progression**: The delay rate increases significantly across hub operational tiers, rising from `0.0400` in Tier 1 hubs to `0.2150` in Tier 2 hubs (and `0.4400` in Tier 3 hubs).
3. **Declared Value Distribution**: The median-filled declared value across the dataset has a mean of `1533.05` INR and a median of `1121.50` INR.
4. **Log Transformation Effect**: The natural log transformation of declared value (`log_declared_value`) reduces skewness, bringing the mean (`7.0280`) and median (`7.0220`) into close alignment on the filled column.
5. **Delivery Hours Outliers**: Using the 1.5 * IQR rule (Q1 = `43.26` hours, Q3 = `72.56` hours, bounds = `0.71` to `116.52` hours), exactly `1` delivery time outlier was identified in the historical dataset.

## Model results (regression) (PART - B)

1. **Baseline Performance**: The naive baseline model (predicting the training mean for all validation rows) achieved an $R^2$ of `-0.021` and a validation RMSE of `23.45` hours, proving that a simple mean fallback fails to capture any predictive variance in delivery hours.
2. **Validation Winner**: **Gradient Boosting** emerged as the winning model on validation, achieving the highest validation $R^2$ of `0.931` and a validation RMSE of `6.12` hours (outperforming Random Forest at `0.914` and Linear Regression at `0.917`).
3. **Tree-Depth Diagnostics**: 
   - **High Bias**: Very shallow trees (such as `max_depth=1`) suffer from high bias, yielding high errors on both training and validation sets with a tight train-val RMSE gap (`0.50`).
   - **High Variance**: Unconstrained deep trees (`max_depth=None`) suffer from severe overfitting (high variance), achieving a training RMSE of `0.00` while expanding the generalization gap to `7.70` (training error never rises as depth grows). The optimal hyperparameter choice balancing bias and variance was found at `max_depth=6` with a validation RMSE of `6.28`.
4. **Cross-Validation vs. Validation Ranking**: The 5-fold cross-validation ranking of untuned models slightly disagreed with the single-split validation ranking (with Linear Regression yielding a mean CV $R^2$ of `0.935` compared to Gradient Boosting's CV mean of `0.926`), demonstrating minor sensitivity to partition splits.
5. **Final Model Selection**: Gradient Boosting was selected as the final production model to ship due to its superior non-linear modeling capability and interaction handling, scoring a strong test set $R^2$ of `0.949` and test RMSE of `5.85` hours.
