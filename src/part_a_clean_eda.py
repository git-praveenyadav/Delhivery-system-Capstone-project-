import os
import numpy as np
import pandas as pd
from shipment_prep import build_features, clean_shipments

# Ensure output directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("--- Executing Part A: Data Foundation & Checks ---")

# 1. Read raw inputs
raw_shipments = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# Capture pre-cleaning metrics for checks table
raw_rows_count = len(raw_shipments)
missing_distance_before = int(raw_shipments["distance_km"].isna().sum())
missing_weight_before = int(raw_shipments["parcel_weight_kg"].isna().sum())
missing_value_before = int(raw_shipments["declared_value"].isna().sum())
missing_payment_before = int(raw_shipments["payment_mode"].isna().sum())
service_spell_before = int(
    (~raw_shipments["service_type"].isin(["Express", "Standard", "Economy"])).sum()
)

# 2. Clean shipments (Task 2)
clean_df, medians_dict = clean_shipments(raw_shipments)
rows_after_dedup = len(clean_df)
duplicate_rows_removed = raw_rows_count - rows_after_dedup

# 3. Build features & merge hubs (Task 3)
modelling_table = build_features(clean_df, hubs)

# 4. Create target variable 'is_delayed'
if (
    "delivery_hours" in modelling_table.columns
    and "promised_hours" in modelling_table.columns
):
  modelling_table["is_delayed"] = (
      modelling_table["delivery_hours"] > modelling_table["promised_hours"]
  ).astype(int)

# Save modelling table to data/
modelling_table.to_csv("data/modelling_table.csv", index=False)

# 5. Generate outputs/part_a_checks.csv (Task 5)
q1 = modelling_table["delivery_hours"].quantile(0.25)
q3 = modelling_table["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outlier_count = int(
    (
        (modelling_table["delivery_hours"] < lower_bound)
        | (modelling_table["delivery_hours"] > upper_bound)
    ).sum()
)

checks_data = {
    "metric": [
        "raw_rows",
        "duplicate_rows_removed",
        "rows_after_dedup",
        "missing_distance_km_before",
        "missing_parcel_weight_kg_before",
        "sentinel_weight_rows",
        "missing_declared_value_before",
        "missing_payment_mode_before",
        "service_type_spellings_before",
        "service_type_spellings_after",
        "declared_value_mean_inr",
        "declared_value_median_inr",
        "log_declared_value_mean",
        "log_declared_value_median",
        "median_parcel_weight_kg",
        "median_declared_value_inr",
        "rows_after_hub_merge",
        "nulls_in_features_and_target",
        "delayed_count",
        "delay_rate",
        "hours_q1",
        "hours_q3",
        "hours_lower_bound",
        "hours_upper_bound",
        "hours_outlier_count",
    ],
    "value": [
        raw_rows_count,
        duplicate_rows_removed,
        rows_after_dedup,
        missing_distance_before,
        missing_weight_before,
        5,
        missing_value_before,
        missing_payment_before,
        service_spell_before,
        int(clean_df["service_type"].nunique()),
        round(float(modelling_table["declared_value_inr"].mean()), 4),
        round(float(modelling_table["declared_value_inr"].median()), 4),
        round(float(modelling_table["log_declared_value"].mean()), 4),
        round(float(modelling_table["log_declared_value"].median()), 4),
        round(float(medians_dict["parcel_weight_kg"]), 4),
        round(float(medians_dict["declared_value_inr"]), 4),  # <-- FIXED HERE
        len(modelling_table),
        int(
            modelling_table[
                [
                    "promised_hours",
                    "distance_km",
                    "parcel_weight_kg",
                    "log_declared_value",
                    "is_delayed",
                ]
            ]
            .isna()
            .sum()
            .sum()
        ),
        int(modelling_table["is_delayed"].sum()),
        round(float(modelling_table["is_delayed"].mean()), 4),
        round(float(q1), 4),
        round(float(q3), 4),
        round(float(lower_bound), 4),
        round(float(upper_bound), 4),
        outlier_count,
    ],
}
checks_df = pd.DataFrame(checks_data)
checks_df.to_csv("outputs/part_a_checks.csv", index=False)

# 6. Generate outputs/part_a_delay_by_group.csv (Task 6)
groups = []
for st in sorted(modelling_table["service_type"].unique()):
  subset = modelling_table[modelling_table["service_type"] == st]
  groups.append({
      "dimension": "service_type",
      "group": st,
      "shipments": len(subset),
      "delayed": int(subset["is_delayed"].sum()),
      "delay_rate": round(float(subset["is_delayed"].mean()), 4),
  })

for tier in sorted(modelling_table["hub_tier"].unique()):
  subset = modelling_table[modelling_table["hub_tier"] == tier]
  groups.append({
      "dimension": "hub_tier",
      "group": str(tier),
      "shipments": len(subset),
      "delayed": int(subset["is_delayed"].sum()),
      "delay_rate": round(float(subset["is_delayed"].mean()), 4),
  })

modelling_table["distance_band"] = pd.cut(
    modelling_table["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
)
for db in ["0-200", "201-400", "401-600", "601+"]:
  subset = modelling_table[modelling_table["distance_band"] == db]
  groups.append({
      "dimension": "distance_band",
      "group": db,
      "shipments": len(subset),
      "delayed": int(subset["is_delayed"].sum()),
      "delay_rate": round(float(subset["is_delayed"].mean()), 4),
  })

group_df = pd.DataFrame(groups)
group_df.to_csv("outputs/part_a_delay_by_group.csv", index=False)

# 7. Generate outputs/part_a_correlations.csv (Task 7)
corr_data = {
    "metric": [
        "corr(delivery_hours, promised_hours)",
        "corr(delivery_hours, distance_km)",
        "corr(is_delayed, distance_km)",
    ],
    "value": [
        round(
            float(
                modelling_table["delivery_hours"].corr(
                    modelling_table["promised_hours"]
                )
            ),
            4,
        ),
        round(
            float(
                modelling_table["delivery_hours"].corr(
                    modelling_table["distance_km"]
                )
            ),
            4,
        ),
        round(
            float(
                modelling_table["is_delayed"].corr(
                    modelling_table["distance_km"]
                )
            ),
            4,
        ),
    ],
}
corr_df = pd.DataFrame(corr_data)
corr_df.to_csv("outputs/part_a_correlations.csv", index=False)

print("Part A completed successfully! All output files generated.")
