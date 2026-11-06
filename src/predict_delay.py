import os
import joblib
import pandas as pd
from shipment_prep import clean_shipments, build_features

os.makedirs("outputs", exist_ok=True)
print("--- Scoring New Shipments ---")

# Load artifacts
pipeline = joblib.load("models/delay_model.pkl")
threshold_df = pd.read_csv("models/delay_threshold.csv")
threshold = float(threshold_df["threshold"].iloc[0])
feature_order_df = pd.read_csv("models/feature_order.csv")
features_list = feature_order_df["feature"].tolist()
impute_df = pd.read_csv("models/impute_values.csv")
medians_dict = impute_df.iloc[0].to_dict()

# Load raw new shipments and hubs
new_raw = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
hub_segments = pd.read_csv("outputs/hub_segments.csv")

# Clean and build features
clean_new, _ = clean_shipments(new_raw, medians=medians_dict)
feature_table = build_features(clean_new, hubs)

# Ensure exact feature order
X_new = feature_table[features_list]

# Predict probabilities
probs = pipeline.predict_proba(X_new)[:, 1]
feature_table["delay_probability"] = [round(float(p), 4) for p in probs]
feature_table["flagged"] = (feature_table["delay_probability"] >= threshold).astype(int)

# Merge segment info
feature_table = feature_table.merge(hub_segments[["hub_id", "segment"]], on="hub_id", how="left")

# Assign operational action rules
def determine_action(row):
    if row["flagged"] == 1:
        if row["segment"] == "Escalate":
            return "Escalate to hub manager"
        else:
            return "Proactive customer call"
    else:
        return "No action"

feature_table["action"] = feature_table.apply(determine_action, axis=1)

# Select final columns for output
output_columns = ["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]
actions_df = feature_table[output_columns]
actions_df.to_csv("outputs/new_shipment_actions.csv", index=False)

print("New shipment scoring complete! outputs/new_shipment_actions.csv generated.")
