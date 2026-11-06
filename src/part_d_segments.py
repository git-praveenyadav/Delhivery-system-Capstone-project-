import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

os.makedirs("outputs", exist_ok=True)
print("--- Running Part D: Hub Profiles & K-Means Segmentation ---")

# 1. Load modelling table and hubs
df = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# Group by hub_id to build hub profiles
profile_df = df.groupby("hub_id").agg(
    shipments=("shipment_id", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_stops=("n_stops", "mean")
).reset_index()

# Merge with hub metadata
hub_full_profile = hubs.merge(profile_df, on="hub_id", how="left").sort_values("hub_id").reset_index(drop=True)

# Save hub profile output
hub_full_profile[[
    "hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity",
    "shipments", "mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_stops"
]].to_csv("outputs/hub_profile.csv", index=False)

# 2. K-Means Clustering & Separation Evidence (K = 2, 3, 4, 5)
cluster_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_stops"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(hub_full_profile[cluster_cols])

separation_rows = []
kmeans_models = {}

for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    kmeans_models[k] = (km, labels)
    
    # Calculate across-to-within Euclidean distance ratio over 276 pairs
    distances = cdist(X_scaled, X_scaled, metric="euclidean")
    within_dists = []
    across_dists = []
    
    n_hubs = len(X_scaled)
    for i in range(n_hubs):
        for j in range(i + 1, n_hubs):
            if labels[i] == labels[j]:
                within_dists.append(distances[i, j])
            else:
                across_dists.append(distances[i, j])
                
    mean_across = np.mean(across_dists) if across_dists else 1.0
    mean_within = np.mean(within_dists) if within_dists else 1.0
    ratio = mean_across / mean_within
    
    separation_rows.append({"k": k, "across_over_within": round(float(ratio), 4)})

pd.DataFrame(separation_rows).to_csv("outputs/part_d_separation.csv", index=False)

# 3. K = 3 Segment Mapping & Naming
k3_km, k3_labels = kmeans_models[3]
hub_full_profile["cluster"] = k3_labels

# Order clusters by mean delay_rate: lowest = Standard, middle = Watch, highest = Escalate
cluster_delay_means = hub_full_profile.groupby("cluster")["delay_rate"].mean().sort_values()
cluster_names = {}
segment_order = ["Standard", "Watch", "Escalate"]
for idx, (cluster_id, _) in enumerate(cluster_delay_means.items()):
    cluster_names[cluster_id] = segment_order[idx]

hub_full_profile["segment"] = hub_full_profile["cluster"].map(cluster_names)

segments_output = hub_full_profile[[
    "hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity", "cluster", "segment"
]]
segments_output.to_csv("outputs/hub_segments.csv", index=False)

# 4. Tier vs Segment Mismatches
tier_map = {1: "Standard", 2: "Watch", 3: "Escalate"}
hub_full_profile["expected_segment"] = hub_full_profile["hub_tier"].map(tier_map)
mismatches = hub_full_profile[hub_full_profile["segment"] != hub_full_profile["expected_segment"]]

mismatch_rows = []
for _, row in mismatches.iterrows():
    mismatch_rows.append({
        "mismatch_detail": f"HUB{row['hub_id']} {row['hub_city']} (Tier {row['hub_tier']}, {row['segment']}); Escalate" if row['segment']=='Escalate' else f"HUB{row['hub_id']} {row['hub_city']}"
    })

# Format exact string required for verification
mismatch_output = mismatches.copy()
mismatch_output["description"] = mismatch_output.apply(
    lambda r: f"HUB{r['hub_id'].replace('HUB','')} {r['hub_city']} (Tier {r['hub_tier']}, {r['segment']})", axis=1
)
# Save required mismatch table
mismatches[["hub_id", "hub_city", "region", "hub_tier", "segment"]].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

print("Part D segmentation and hub profiling completed successfully.")
