import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

print("--- Running Part C: Delay Classifier and Alert Threshold ---")

# 1. Load the modelling table
df = pd.read_csv("data/modelling_table.csv")

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

X = df[FEATURES]
y = df["is_delayed"]

# Task 1: Stratified Split (840 / 180 / 180)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.5,
    random_state=42,
    stratify=y_train_full,
)


def get_metrics_row(y_true, y_pred, model_name):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
  acc = accuracy_score(y_true, y_pred)
  prec = precision_score(y_true, y_pred, zero_division=0)
  rec = recall_score(y_true, y_pred, zero_division=0)
  f1 = f1_score(y_true, y_pred, zero_division=0)
  return {
      "model": model_name,
      "TN": int(tn),
      "FP": int(fp),
      "FN": int(fn),
      "TP": int(tp),
      "accuracy": round(float(acc), 4),
      "precision": round(float(prec), 4),
      "recall": round(float(rec), 4),
      "F1": round(float(f1), 4),
  }


# Task 2: Blind Baseline and Three Classifiers
metrics_rows = []
val_preds_blind = np.zeros_like(y_val)
metrics_rows.append(
    get_metrics_row(y_val, val_preds_blind, "blind_all_on_time")
)

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42)),
])
lr_pipe.fit(X_train, y_train)
val_preds_lr = lr_pipe.predict(X_val)
metrics_rows.append(get_metrics_row(y_val, val_preds_lr, "logistic_regression"))

best_dt_f1 = -1
best_cd = 2
best_dt_preds = None
for cd in [2, 3, 4, 5, 6, 8, 10]:
  dt = DecisionTreeClassifier(max_depth=cd, random_state=42)
  dt.fit(X_train, y_train)
  preds = dt.predict(X_val)
  f1 = f1_score(y_val, preds, zero_division=0)
  if f1 > best_dt_f1:
    best_dt_f1 = f1
    best_cd = cd
    best_dt_preds = preds

metrics_rows.append(
    get_metrics_row(y_val, best_dt_preds, f"decision_tree_depth_{best_cd}")
)

rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
rf_default.fit(X_train, y_train)
val_preds_rf = rf_default.predict(X_val)
metrics_rows.append(
    get_metrics_row(y_val, val_preds_rf, "random_forest_default")
)

pd.DataFrame(metrics_rows).to_csv("outputs/part_c_metrics.csv", index=False)

# Task 3: Feature Scores & Noise Analysis
selector_all = SelectKBest(score_func=f_classif, k="all")
selector_all.fit(X_train, y_train)
f_scores = selector_all.scores_

rf_imp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_imp.fit(X_train, y_train)
importances = rf_imp.feature_importances_

feature_scores_data = []
for idx, feat in enumerate(FEATURES):
  feature_scores_data.append({
      "feature": feat,
      "f_score": round(float(f_scores[idx]), 4),
      "importance": round(float(importances[idx]), 4),
      "is_noise": feat in NOISE_COLUMNS,
  })

fs_df = pd.DataFrame(feature_scores_data)
fs_df["f_rank"] = (
    fs_df["f_score"].rank(ascending=False, method="min").astype(int)
)
fs_df = fs_df.sort_values("f_rank")[
    ["feature", "f_score", "f_rank", "importance", "is_noise"]
]
fs_df.to_csv("outputs/part_c_feature_scores.csv", index=False)

# Task 4 & 5: Feature Selection & Hyperparameter Tuning using Pipelines
pipe_default = Pipeline([
    ("select", SelectKBest(score_func=f_classif, k=7)),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
])
pipe_default.fit(X_train, y_train)
val_f1_default = f1_score(y_val, pipe_default.predict(X_val), zero_division=0)

param_grid = {
    "model__n_estimators": [50, 100, 150],
    "model__max_depth": [4, 6, 8],
    "model__min_samples_leaf": [1, 5, 10],
}

pipe_tune = Pipeline([
    ("select", SelectKBest(score_func=f_classif, k=7)),
    ("model", RandomForestClassifier(random_state=42)),
])

grid_search = GridSearchCV(
    estimator=pipe_tune, param_grid=param_grid, cv=5, scoring="f1"
)
grid_search.fit(X_train, y_train)

rand_search = RandomizedSearchCV(
    estimator=pipe_tune,
    param_distributions=param_grid,
    n_iter=4,
    cv=5,
    random_state=42,
    scoring="f1",
)
rand_search.fit(X_train, y_train)

val_f1_grid = f1_score(y_val, grid_search.predict(X_val), zero_division=0)
val_f1_rand = f1_score(y_val, rand_search.predict(X_val), zero_division=0)

tuning_data = [
    {
        "candidate": "default_forest",
        "best_cv_accuracy": round(
            float(grid_search.cv_results_["mean_test_score"].mean()), 4
        ),
        "val_f1": round(float(val_f1_default), 4),
    },
    {
        "candidate": "grid_search",
        "best_cv_accuracy": round(float(grid_search.best_score_), 4),
        "val_f1": round(float(val_f1_grid), 4),
    },
    {
        "candidate": "randomized_search",
        "best_cv_accuracy": round(float(rand_search.best_score_), 4),
        "val_f1": round(float(val_f1_rand), 4),
    },
]
pd.DataFrame(tuning_data).to_csv("outputs/part_c_tuning.csv", index=False)

best_candidate_idx = np.argmax([val_f1_default, val_f1_grid, val_f1_rand])
winning_pipelines = [pipe_default, grid_search, rand_search]
winning_names = ["default_forest", "grid_search", "randomized_search"]
final_pipeline = winning_pipelines[best_candidate_idx]
winning_name = winning_names[best_candidate_idx]

config_data = [{
    "selector": "SelectKBest(f_classif, k=7)",
    "model": winning_name,
    "parameters": str(final_pipeline.get_params()),
}]
pd.DataFrame(config_data).to_csv("outputs/part_c_final_config.csv", index=False)

# Task 6: Alert Threshold Optimization
val_probs = final_pipeline.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.1, 1.0, 0.1)
threshold_rows = []
chosen_threshold = 0.3
max_rec_at_req = -1

for th in thresholds:
  preds_th = (val_probs >= th).astype(int)
  rec = recall_score(y_val, preds_th, zero_division=0)
  prec = precision_score(y_val, preds_th, zero_division=0)
  f1 = f1_score(y_val, preds_th, zero_division=0)
  tn, fp, fn, tp = confusion_matrix(y_val, preds_th, labels=[0, 1]).ravel()
  acc = accuracy_score(y_val, preds_th)

  threshold_rows.append({
      "threshold": round(float(th), 1),
      "TN": int(tn),
      "FP": int(fp),
      "FN": int(fn),
      "TP": int(tp),
      "accuracy": round(float(acc), 4),
      "precision": round(float(prec), 4),
      "recall": round(float(rec), 4),
      "F1": round(float(f1), 4),
  })

  if rec >= 0.80 and rec > max_rec_at_req:
    max_rec_at_req = rec
    chosen_threshold = th

pd.DataFrame(threshold_rows).to_csv("outputs/part_c_thresholds.csv", index=False)

# Task 7: Final Test Evaluation at Chosen Threshold
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])
final_pipeline.fit(X_train_val, y_train_val)

test_probs = final_pipeline.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= chosen_threshold).astype(int)

test_tn, test_fp, test_fn, test_tp = confusion_matrix(
    y_test, test_preds, labels=[0, 1]
).ravel()
test_acc = accuracy_score(y_test, test_preds)
test_prec = precision_score(y_test, test_preds, zero_division=0)
test_rec = recall_score(y_test, test_preds, zero_division=0)
test_f1 = f1_score(y_test, test_preds, zero_division=0)

final_test_data = [{
    "threshold": round(float(chosen_threshold), 1),
    "TN": int(test_tn),
    "FP": int(test_fp),
    "FN": int(test_fn),
    "TP": int(test_tp),
    "accuracy": round(float(test_acc), 4),
    "precision": round(float(test_prec), 4),
    "recall": round(float(test_rec), 4),
    "F1": round(float(test_f1), 4),
}]
pd.DataFrame(final_test_data).to_csv(
    "outputs/part_c_final_test.csv", index=False
)

print("Part C execution complete successfully using Pipeline!")
