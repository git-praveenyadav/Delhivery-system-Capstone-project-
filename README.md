# **Delhivery-system-Capstone-project-**
Test repo for capstone project

## Data findings

1. **Service Type Delay Rates**: Express shipments experience the highest delay frequency at a delay rate of `0.4897` (119 delayed out of 243 shipments), followed by Standard at `0.2056`, and Economy at `0.0857`.
2. **Hub Tier Progression**: The delay rate increases significantly across hub operational tiers, rising from `0.0400` in Tier 1 hubs to `0.2150` in Tier 2 hubs (and `0.4400` in Tier 3 hubs).
3. **Declared Value Distribution**: The median-filled declared value across the dataset has a mean of `1533.05` INR and a median of `1121.50` INR.
4. **Log Transformation Effect**: The natural log transformation of declared value (`log_declared_value`) reduces skewness, bringing the mean (`7.0280`) and median (`7.0220`) into close alignment on the filled column.
5. **Delivery Hours Outliers**: Using the 1.5 * IQR rule (Q1 = `43.26` hours, Q3 = `72.56` hours, bounds = `0.71` to `116.52` hours), exactly `1` delivery time outlier was identified in the historical dataset.
