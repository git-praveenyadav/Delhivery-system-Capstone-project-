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

## 📌 Project Scenario
The regional operations analytics team of Delhivery faces three core challenges without an automated tracking pipeline:
1. **Delivery-Hours Estimation**: Point estimates required for each parcel at dispatch time.
2. **Delay Classification**: Predicting which parcels will miss their delivery promise.
3. **Hub Segmentation**: Classifying hubs into operational tiers (**Standard**, **Watch**, **Escalate**) to optimize resource allocation.

---

<details>
<summary><b>📊 1. Data Findings (Part A) - Click to Expand</b></summary>

* **Service Type Delay Rates**: Express shipments experience the highest delay frequency at a delay rate of `0.4897` (119 delayed out of 243 shipments), followed by Standard at `0.2056`, and Economy at `0.0857`.
* **Hub Tier Progression**: Delay rates escalate significantly across operational tiers, rising from `0.0400` in Tier 1 hubs to `0.2150` in Tier 2 hubs, and `0.4400` in Tier 3 hubs.
* **Declared Value Distribution**: The median-filled declared value across the dataset has a mean of `1533.05` INR and a median of `1121.50` INR.
* **Log Transformation Effect**: Natural log transformation of declared value (`log_declared_value`) successfully regularizes skewness, aligning the mean (`7.0280`) and median (`7.0220`).
* **Delivery Hours Outliers**: Applying the $1.5 \times \text{IQR}$ rule ($Q_1 = 43.26$ hours, $Q_3 = 72.56$ hours, valid bounds $= 0.71$ to $116.52$ hours), exactly `1` delivery time outlier was identified.

</details>

<details>
<summary><b>📈 2. Model Results: Regression (Part B) - Click to Expand</b></summary>

* **Baseline Performance**: The naive baseline model (predicting training mean) yielded an $R^2$ of `-0.021` and validation RMSE of `23.45` hours.
* **Validation Winner**: **Gradient Boosting** achieved the top validation performance with an $R^2$ of `0.931` and validation RMSE of `6.12` hours (beating Random Forest at `0.914` and Linear Regression at `0.917`).
* **Tree-Depth Diagnostics**: Shallow trees (`max_depth=1`) exhibit high bias (train-val RMSE gap of `0.50`), whereas unconstrained trees (`max_depth=None`) suffer severe overfitting (training RMSE `0.00`, generalization gap `7.70`). Optimal balance was found at `max_depth=6` (validation RMSE `6.28`).
* **Cross-Validation**: 5-fold cross-validation yielded a mean $R^2$ of `0.926` for Gradient Boosting and `0.935` for Linear Regression.
* **Final Test Performance**: The selected Gradient Boosting model scored a robust test set $R^2$ of `0.949` and test RMSE of `5.85` hours.

</details>

<details>
<summary><b>🎯 3. Model Results: Classification (Part C) - Click to Expand</b></summary>

* **Blind Baseline**: Predicting all shipments on-time yields an accuracy of `0.7667` but a recall/F1 of `0.0000`, proving accuracy alone is insufficient for imbalanced delay tracking.
* **Classifier Performance**: Logistic Regression and Decision Trees outperformed baseline F1, with the tuned Random Forest achieving a validation F1-score exceeding `0.8000`.
* **Noise Feature Validation**: All 5 designated noise columns ranked in the bottom tiers of feature importance and ANOVA F-scores (falling outside the top 7 features), validating feature selection integrity.
* **Hyperparameter Tuning**: Grid search and randomized search optimized the Random Forest over the 7-feature subset for maximum generalization.
* **Alert Threshold**: Tuned decision threshold satisfying operational constraints ($\text{recall} \ge 0.80$) effectively balanced precision and recall on the test set for proactive risk escalation.

</details>

---

## ⚙️ How to Run

Execute the pipeline sequentially from your terminal:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate raw synthetic datasets
python generate_data.py

# 3. Run Part A (Cleaning, EDA, & Feature Engineering)
python src/part_a_clean_eda.py

# 4. Run Part B (Delivery-Hours Regression)
python src/part_b_regression.py

# 5. Run Part C (Delay Classification & Threshold Tuning)
python src/part_c_classification.py

# 6. Run Part D (Hub Segmentation, Packaging & Scoring)
python src/part_d_segments.py
python src/package_model.py
python src/predict_delay.py
