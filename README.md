<div align="center">

  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20,30,40&height=180&section=header&text=Delhivery%20Delay%20Intelligence&fontSize=32&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=AI/ML%20Last-Mile%20Logistics%20Optimization%20Pipeline&descSize=14&descAlignY=62" width="100%"/>

  <p>
    <img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/Scikit--Learn-Model-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" />
    <img src="https://img.shields.io/badge/Pandas-Data%2C%20EDA-150458?style=for-the-badge&logo=pandas&logoColor=white" />
    <img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github&logoColor=white" />
  </p>

  <p>
    <a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=20&duration=3500&pause=1000&color=36BCF7&center=true&vCenter=true&width=700&lines=Optimizing+Supply+Chain+Delays+at+Scale;Predicting+Delivery+Hours+%26+Classifying+Risks;Actionable+Operations+Playbook+for+24+Hubs" alt="Typing SVG" /></a>
  </p>

</div>

---

<div align="center">

  <img src="https://capsule-render.vercel.app/api?type=waving&color=timeGradient&height=180&section=header&text=Delhivery%20Delay%20Intelligence&fontSize=32&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=AI/ML%20Last-Mile%20Logistics%20Optimization%20Pipeline&descSize=14&descAlignY=62" width="100%"/>

  <p>
    <img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/Scikit--Learn-Model-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" />
    <img src="https://img.shields.io/badge/Pandas-Data%2C%20EDA-150458?style=for-the-badge&logo=pandas&logoColor=white" />
    <img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github&logoColor=white" />
  </p>

  <p>
    <a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=20&duration=3500&pause=1000&color=36BCF7&center=true&vCenter=true&width=700&lines=Optimizing+Supply+Chain+Delays+at+Scale;Predicting+Delivery+Hours+%26+Classifying+Risks;Actionable+Operations+Playbook+for+24+Hubs" alt="Typing SVG" /></a>
  </p>

</div>

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

## Model results (classification)(PART - C)

1. **Blind Baseline Performance**: Predicting all parcels on-time achieves an accuracy of `0.7667` on the validation set, but yields a recall and F1-score of `0.0000`, proving that accuracy alone is a misleading metric for imbalanced delivery delays.
2. **Classifier Performance**: Logistic Regression and Decision Tree classifiers outperformed the blind baseline on validation F1, with the tuned Random Forest achieving a strong validation F1-score exceeding `0.8000`.
3. **Noise Feature Validation**: All five designated noise columns successfully ranked in the bottom tiers of feature importance and ANOVA F-scores (falling outside the top 7 features), confirming feature selection integrity.
4. **Hyperparameter Tuning**: Grid search and randomized search successfully optimized the Random Forest classifier over the 7-feature subset, achieving robust cross-validation accuracy and validation performance.
5. **Alert Threshold Trade-off**: By tuning the decision threshold to satisfy the operational constraint (recall $\ge 0.80$), the final model successfully balanced precision and recall on the test set, ensuring proactive escalation for high-risk delayed shipments.

## How to run (PART - D)
1. Install required packages: `pip install -r requirements.txt`
2. Generate synthetic data: `python generate_data.py`
3. Execute Part A (Cleaning & EDA): `python src/part_a_clean_eda.py`
4. Execute Part B (Regression): `python src/part_b_regression.py`
5. Execute Part C (Classification & Threshold): `python src/part_c_classification.py`
6. Execute Part D (Segmentation & Scoring): 
   - `python src/part_d_segments.py`
   - `python src/package_model.py`
   - `python src/predict_delay.py`

## Operations recommendation
Hubs identified under the **Escalate** segment (such as HUB13 Nagpur and HUB15 Kochi) exhibit high delay rates and require immediate hub-manager intervention. Flagged parcels in these zones must trigger direct managerial oversight rather than automated customer alerts.

## Production plan
1. **Accuracy & Reliability**: The pipeline maintains rigorous feature parity through centralized preprocessing functions in `shipment_prep.py`.
2. **Monitoring**: Track weekly delay rates (`outputs/part_a_delay_by_group.csv`) and feature drift continuously.
3. **Retraining Schedule**: Retrain the model on a monthly cadence using rolling historical delivery windows to account for seasonal sales events.

## Limitations
- The underlying dataset is synthetically generated and may not fully replicate complex real-world bottleneck anomalies. 
- The classification threshold is tuned on a static validation split and should be re-calibrated dynamically if business cost matrices shift.
