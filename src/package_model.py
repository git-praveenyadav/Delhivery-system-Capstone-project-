import os
import joblib
import numpy as np
import pandas as pd
from shipment_prep import clean_shipments
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
print("--- Packaging Model and Artifacts ---")

df = pd.read_csv("data/modelling_table.csv")
raw_shipments = pd.read_csv("data/raw_shipments.csv")
_, medians_dict = clean_shipments(raw_shipments)

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
y = df["is_delayed"]

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

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

selector = SelectKBest(score_func=f_classif, k=7)
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
pipeline = Pipeline([("select", selector), ("model", rf)])
pipeline.fit(X_train_val, y_train_val)

model_path = "models/delay_model.pkl"
joblib.dump(pipeline, model_path)

pd.DataFrame({"threshold": [0.3]}).to_csv(
    "models/delay_threshold.csv", index=False
)
pd.DataFrame({"feature": FEATURES}).to_csv(
    "models/feature_order.csv", index=False
)
pd.DataFrame([medians_dict]).to_csv("models/impute_values.csv", index=False)

loaded_pipeline = joblib.load(model_path)
probs_original = pipeline.predict_proba(X_test)[:, 1]
probs_reloaded = loaded_pipeline.predict_proba(X_test)[:, 1]
max_diff = np.max(np.abs(probs_original - probs_reloaded))

reload_check_df = pd.DataFrame(
    {"metric": ["reload_max_abs_probability_diff"], "value": [float(max_diff)]}
)
reload_check_df.to_csv("outputs/part_d_reload_check.csv", index=False)

print(f"Model packaged successfully. Reload max abs diff: {max_diff}")
