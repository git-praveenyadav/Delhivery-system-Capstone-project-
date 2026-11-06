import numpy as np
import pandas as pd


def clean_shipments(raw, medians=None):
  """Cleans the raw shipments dataframe following project requirements."""
  clean_df = raw.copy()

  # i. Drop exact duplicate rows
  clean_df = clean_df.drop_duplicates()

  # ii. service_type: strip spaces and convert to title case
  if "service_type" in clean_df.columns:
    clean_df["service_type"] = clean_df["service_type"].str.strip().str.title()

  # iii. payment_mode: replace missing values with "Unknown"
  if "payment_mode" in clean_df.columns:
    clean_df["payment_mode"] = clean_df["payment_mode"].fillna("Unknown")

  # iv. declared_value: clean currency and convert to float
  if "declared_value" in clean_df.columns:
    if clean_df["declared_value"].dtype == object:
      clean_df["declared_value_inr"] = (
          clean_df["declared_value"]
          .astype(str)
          .str.replace("₹", "", regex=False)
          .str.replace(",", "", regex=False)
          .astype(float)
      )
    else:
      clean_df["declared_value_inr"] = clean_df["declared_value"].astype(float)
    clean_df = clean_df.drop(columns=["declared_value"])

  # v. dispatch_time: convert to datetime
  if "dispatch_time" in clean_df.columns:
    clean_df["dispatch_time"] = pd.to_datetime(clean_df["dispatch_time"])

  # vi. parcel_weight_kg: values above 100 are scanner faults -> treat as missing
  if "parcel_weight_kg" in clean_df.columns:
    clean_df.loc[clean_df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = (
        np.nan
    )

  # vii. Handle missing values using medians
  cols_to_fill = ["distance_km", "parcel_weight_kg", "declared_value_inr"]

  if medians is None:
    medians = {}
    for col in cols_to_fill:
      if col in clean_df.columns:
        medians[col] = float(clean_df[col].median())
      else:
        medians[col] = 0.0
  else:
    medians = {k: float(v) for k, v in medians.items()}

  for col in cols_to_fill:
    if col in clean_df.columns and col in medians:
      clean_df[col] = clean_df[col].fillna(medians[col])

  return clean_df, medians


def build_features(clean, hubs):
  """Builds machine learning features by merging hubs and engineering columns."""
  df = clean.merge(hubs, on="hub_id", how="left")

  # promised_hours mapping based on service type specifications
  promised_hours_map = {"Express": 36, "Standard": 60, "Economy": 96}
  if "service_type" in df.columns:
    df["promised_hours"] = (
        df["service_type"].map(promised_hours_map).fillna(60).astype(int)
    )
  else:
    df["promised_hours"] = 60

  # is_cod: 1 when payment_mode is cod, else 0
  if "payment_mode" in df.columns:
    df["is_cod"] = (
        df["payment_mode"].astype(str).str.lower() == "cod"
    ).astype(int)
  else:
    df["is_cod"] = 0

  # is_weekend: 1 when dispatch day is Saturday or Sunday
  if "dispatch_time" in df.columns:
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
  else:
    df["is_weekend"] = 0

  # dispatch_hour: hour of dispatch timestamp as integer
  if "dispatch_time" in df.columns:
    df["dispatch_hour"] = df["dispatch_time"].dt.hour
  else:
    df["dispatch_hour"] = 0

  # log_declared_value: natural log of declared_value_inr
  if "declared_value_inr" in df.columns:
    df["log_declared_value"] = np.log(df["declared_value_inr"] + 1)
  else:
    df["log_declared_value"] = 0.0

  return df


