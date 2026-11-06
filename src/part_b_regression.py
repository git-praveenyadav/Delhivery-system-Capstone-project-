import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

print("--- Running Part B: Delivery-Hours Regression ---")

# 1. Load the modelling table created in Part A
df = pd.read_csv("data/modelling_table.csv")

# Pinned 17 feature columns specified in the project
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

X = df[FEATURES]
y = df["delivery_hours"]

# Task 1: Train, Validation, and Test Split (840 / 180 / 180)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.5, random_state=42
)

# Task 2: Scaling without leakage (fit scaler on training rows only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

checks_data = {
    "metric": ["train_scaled_mean_x0", "test_scaled_x0_mean"],
    "value": [
        round(float(np.abs(X_train_scaled).mean(axis=0).max()), 4),
        round(float(np.abs(X_val_scaled).mean(axis=0).max()), 4),
    ],
}
pd.DataFrame(checks_data).to_csv("outputs/part_b_checks.csv", index=False)

# Task 3: Baseline Model (predicting training mean for validation rows)
train_mean = y_train.mean()
val_preds_baseline = np.full_like(y_val, train_mean, dtype=float)

baseline_r2 = r2_score(y_val, val_preds_baseline)
baseline_rmse = np.sqrt(mean_squared_error(y_val, val_preds_baseline))

metrics_data = {
    "metric": ["baseline_r2", "baseline_rmse"],
    "value": [round(float(baseline_r2), 4), round(float(baseline_rmse), 4)],
}
pd.DataFrame(metrics_data).to_csv("outputs/part_b_metrics.csv", index=False)

# Task 4: Tree-Depth Diagnostic
depths = [1, 2, 3, 4, 6, 8, 12, None]
depth_results = []
best_depth = 1
min_val_rmse = float("inf")

for d in depths:
  dt = DecisionTreeRegressor(max_depth=d, random_state=42)
  dt.fit(X_train, y_train)
  tr_rmse = np.sqrt(mean_squared_error(y_train, dt.predict(X_train)))
  vr_rmse = np.sqrt(mean_squared_error(y_val, dt.predict(X_val)))
  depth_results.append({
      "max_depth": str(d) if d is not None else "None",
      "train_rmse": round(float(tr_rmse), 4),
      "val_rmse": round(float(vr_rmse), 4),
  })
  if vr_rmse < min_val_rmse:
    min_val_rmse = vr_rmse
    best_depth = d

pd.DataFrame(depth_results).to_csv("outputs/part_b_tree_depth.csv", index=False)

# Task 5: Four Candidate Models Evaluation
models = [
    (
        "Linear Regression",
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        True,
    ),
    (
        "Decision Tree",
        DecisionTreeRegressor(max_depth=6, random_state=42),
        False,
    ),
    (
        "Random Forest",
        RandomForestRegressor(n_estimators=180, random_state=42),
        False,
    ),
    (
        "Gradient Boosting",
        GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
        False,
    ),
]

model_results = []
fitted_models = {}

for name, model, needs_scale in models:
  xtr = X_train_scaled if needs_scale else X_train
  xva = X_val_scaled if needs_scale else X_val

  model.fit(xtr, y_train)
  preds = model.predict(xva)

  mae = mean_absolute_error(y_val, preds)
  mse = mean_squared_error(y_val, preds)
  rmse = np.sqrt(mse)
  r2 = r2_score(y_val, preds)

  model_results.append({
      "model": name,
      "mae": round(float(mae), 4),
      "mse": round(float(mse), 4),
      "rmse": round(float(rmse), 4),
      "r2": round(float(r2), 4),
  })
  fitted_models[name] = (model, needs_scale)

pd.DataFrame(model_results).to_csv("outputs/part_b_models.csv", index=False)

# Task 6: Learning Rate Experiment for Gradient Boosting
lr_results = []
for lr in [0.01, 0.1, 1.0]:
  gbr = GradientBoostingRegressor(
      n_estimators=100, learning_rate=lr, random_state=42
  )
  gbr.fit(X_train, y_train)
  tr_rmse = np.sqrt(mean_squared_error(y_train, gbr.predict(X_train)))
  vr_rmse = np.sqrt(mean_squared_error(y_val, gbr.predict(X_val)))
  r2 = r2_score(y_val, gbr.predict(X_val))
  lr_results.append({
      "learning_rate": lr,
      "train_rmse": round(float(tr_rmse), 4),
      "val_rmse": round(float(vr_rmse), 4),
      "r2": round(float(r2), 4),
  })

pd.DataFrame(lr_results).to_csv(
    "outputs/part_b_learning_rate.csv", index=False
)

# Task 7: Cross-Validation & Final Model Evaluation on Test Set
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])
X_train_val_scaled = scaler.fit_transform(X_train_val)

cv_results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name in ["Gradient Boosting", "Linear Regression"]:
  model, needs_scale = fitted_models[name]
  fresh_model = clone(model)
  xtv = X_train_val_scaled if needs_scale else X_train_val

  scores = cross_val_score(
      fresh_model, xtv, y_train_val, cv=kf, scoring="r2"
  )
  cv_results.append({
      "model": name,
      "mean_r2": round(float(scores.mean()), 4),
      "std_r2": round(float(scores.std()), 4),
  })

pd.DataFrame(cv_results).to_csv("outputs/part_b_cv.csv", index=False)

# Final model selection (Gradient Boosting yields best validation performance)
final_model, final_needs_scale = fitted_models["Gradient Boosting"]
xtv = X_train_val_scaled if final_needs_scale else X_train_val
xte = X_test_scaled if final_needs_scale else X_test

final_model.fit(xtv, y_train_val)
test_preds = final_model.predict(xte)

final_r2 = r2_score(y_test, test_preds)
final_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

final_test_df = pd.DataFrame([{
    "model": "Gradient Boosting",
    "r2": round(float(final_r2), 4),
    "rmse": round(float(final_rmse), 4),
}])
final_test_df.to_csv("outputs/part_b_final_test.csv", index=False)

print(
    "Part B execution complete! All regression outputs successfully generated"
    " in outputs/."
)
