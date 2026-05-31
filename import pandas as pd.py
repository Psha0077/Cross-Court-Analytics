import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1) Load combined dataset
# =========================

COMBINED_CSV_PATH = Path(
    "/Users/pranav/Desktop/Dataprojects/Squash/Cross Court Analytics/Raw_data/csv_files/all_matches_combined.csv"
)

shots = pd.read_csv(COMBINED_CSV_PATH, low_memory=False)
print(f"Loaded {len(shots):,} rows from {COMBINED_CSV_PATH}")
print("Columns:", list(shots.columns), "\n")

# Make sure key columns are present
required_cols = [
    "match_id", "game_no", "rally_no", "shot_idx",
    "x", "y", "shot_type", "winner", "winMethod"
]
missing = [c for c in required_cols if c not in shots.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Ensure numerics for x, y, shot_idx
shots["x"] = pd.to_numeric(shots["x"], errors="coerce")
shots["y"] = pd.to_numeric(shots["y"], errors="coerce")
shots["shot_idx"] = pd.to_numeric(shots["shot_idx"], errors="coerce")

# Drop rows with no coordinates or no shot index
shots = shots.dropna(subset=["x", "y", "shot_idx"])
print(f"After cleaning, {len(shots):,} rows remain.\n")

# =========================
# 2) Get last shot per rally (one row/rally)
# =========================

group_keys = ["match_id", "game_no", "rally_no"]

shots_sorted = shots.sort_values(group_keys + ["shot_idx"])

last_shots = (
    shots_sorted
    .groupby(group_keys, as_index=False)
    .tail(1)
    .reset_index(drop=True)
)

print(f"Last-shot dataset: {len(last_shots):,} rallies")
print(last_shots.head(), "\n")

# ==========================================
# 3) Define quadrants from x, y
# Sign convention:
# X = 0 (Left wall), X = 1 (Right wall)
# Y = 0 (Front wall), Y = 1 (Back wall)
# ==========================================

def assign_quadrant(x, y, mid=0.5):
    if np.isnan(x) or np.isnan(y):
        return np.nan
    if x < mid and y < mid:
        return "front_left"
    elif x >= mid and y < mid:
        return "front_right"
    elif x < mid and y >= mid:
        return "back_left"
    else:
        return "back_right"

last_shots["quadrant"] = [
    assign_quadrant(x, y) for x, y in zip(last_shots["x"], last_shots["y"])
]

print("Quadrant counts (last shot per rally):")
print(last_shots["quadrant"].value_counts(dropna=False), "\n")

# ==========================================
# 4) Define result variable (from winMethod)
# ==========================================

last_shots["result"] = (
    last_shots["winMethod"]
    .astype(str)
    .str.strip()
    .replace({"nan": np.nan})
)

print("Top result types (winMethod) for last shots:")
print(last_shots["result"].value_counts(dropna=True).head(15), "\n")

# Rally-level dataset
rallies = last_shots[
    [
        "match_id", "game_no", "rally_no",
        "x", "y", "quadrant",
        "shot_type", "winner", "result",
        "team", "shot_idx"
    ]
].copy()

print("Rally-level dataset (preview):")
print(rallies.head())

OUT_RALLIES_PATH = COMBINED_CSV_PATH.parent / "rallies_last_shot.csv"
rallies.to_csv(OUT_RALLIES_PATH, index=False)
print(f"\n💾 Saved rally-level dataset to: {OUT_RALLIES_PATH}")

print("\n--- Summary of shot types (last shot per rally) ---")
print(last_shots["shot_type"].value_counts(dropna=False).head(20))

# ==========================================
# 5) Filter dataset to focus on key rally results
# ==========================================

valid_results = ["Hit a winner", "Unforced error", "Forced error"]
filtered = rallies[rallies["result"].isin(valid_results)].copy()

print(f"\nFiltered to {len(filtered):,} rallies with result in {valid_results}")
print("Result distribution:\n", filtered["result"].value_counts(), "\n")

# ==========================================
# 6) Clean shot_type (handle casing / missing)
# ==========================================

filtered["shot_type"] = (
    filtered["shot_type"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({"nan": np.nan, "none": np.nan})
)

print("Top 15 shot types in filtered dataset:")
print(filtered["shot_type"].value_counts(dropna=False).head(15), "\n")

# ==========================================
# 7) Build contingency table: shot_type × quadrant
# ==========================================

contingency = pd.crosstab(filtered["shot_type"], filtered["quadrant"])

print("Shot Type × Quadrant contingency table (counts):")
print(contingency, "\n")

row_props = contingency.div(contingency.sum(axis=1), axis=0)
print("Row-wise proportions (within each shot type, %):")
print((row_props * 100).round(1), "\n")

# ==========================================
# 8) Chi-square test of independence
# ==========================================

print("\n================ Chi-square test: Shot Type × Quadrant ================\n")

chi2, p, dof, expected = chi2_contingency(contingency)

print(f"Chi-square statistic : {chi2:,.2f}")
print(f"Degrees of freedom   : {dof}")
print(f"P-value              : {p:.8f}")

if p < 0.05:
    print("\n✅ Result: Significant association (reject H₀ — shot type depends on court area)")
else:
    print("\n❌ Result: Not significant (fail to reject H₀ — shot type independent of court area)")

print(f"\nTotal rallies tested : {contingency.to_numpy().sum():,}")
print(f"Contingency shape    : {contingency.shape}\n")

# ==========================================
# 9) Standardized residuals
# ==========================================

residuals = (contingency - expected) / np.sqrt(expected)
residuals_df = pd.DataFrame(residuals, index=contingency.index, columns=contingency.columns)

print("Standardized residuals (>|2| ≈ strong association):")
print(residuals_df.round(2), "\n")

# ==========================================
# 10) Optional: visualizations
# ==========================================

# If you're running in a headless terminal and plots annoy you,
# you can comment out everything below this line.

# Stacked bar for top 5 shot types
top_shots = filtered["shot_type"].value_counts().head(5).index
subset = row_props.loc[top_shots]

subset.plot(kind="bar", stacked=True, figsize=(10, 6))
plt.title("Distribution of Quadrants for Top 5 Shot Types\n(Rallies ending in Winner, Unforced, or Forced Error)")
plt.ylabel("Percentage within Shot Type (%)")
plt.xlabel("Shot Type")
plt.legend(title="Quadrant", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# Heatmap of residuals
plt.figure(figsize=(8, 6))
sns.heatmap(residuals_df, cmap="coolwarm", center=0, annot=False)
plt.title("Standardized Residuals: Shot Type × Quadrant\n(>|2| indicates stronger deviation from independence)")
plt.xlabel("Quadrant")
plt.ylabel("Shot Type")
plt.tight_layout()
plt.show()
