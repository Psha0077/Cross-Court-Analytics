import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# 1) Load combined dataset
# =========================

COMBINED_CSV_PATH = Path(
    "/Users/pranav/Desktop/Dataprojects/Squash/Cross Court Analytics/Raw_data/csv_files/all_matches_combined.csv"
)

shots = pd.read_csv(COMBINED_CSV_PATH)
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


# 2) Get last shot per rally (one row/rally)

group_keys = ["match_id", "game_no", "rally_no"]

# Sort so that tail(1) gives last shot in each rally
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
# Y = 0 (Back wall), Y = 1 (Front wall)
# ==========================================

def assign_quadrant(x, y, mid=0.5):
    if np.isnan(x) or np.isnan(y):
        return np.nan
    if x < mid and y < mid:
        return "back_left"
    elif x >= mid and y < mid:
        return "back_right"
    elif x < mid and y >= mid:
        return "front_left"
    else:
        return "front_right"

last_shots["quadrant"] = [
    assign_quadrant(x, y) for x, y in zip(last_shots["x"], last_shots["y"])
]

print("Quadrant counts (last shot per rally hit to):")
print(last_shots["quadrant"].value_counts(dropna=False), "\n")

# 4) Define result variable (from winMethod)

last_shots["result"] = (
    last_shots["winMethod"]
    .astype(str)
    .str.strip()
    .replace({"nan": np.nan})
)

print("Top result types (winMethod) for last shots:")
print(last_shots["result"].value_counts(dropna=True).head(15), "\n")

# Optional: keep only columns you care about going forward
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

# Optional: save rally-level dataset for later analysis
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

print(f"Filtered to {len(filtered):,} rallies with result in {valid_results}")
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

# Optional: check the most common shot types
print("Top 15 shot types in filtered dataset:")
print(filtered["shot_type"].value_counts(dropna=False).head(15), "\n")

# ==========================================
# 7) Build contingency table: shot_type × quadrant
# ==========================================

contingency = pd.crosstab(filtered["shot_type"], filtered["quadrant"])

# Show a readable preview
print("Shot Type × Quadrant contingency table (counts):")
print(contingency.head(20), "\n")

# Row-wise proportions (distribution of quadrants within each shot type)
row_props = contingency.div(contingency.sum(axis=1), axis=0)

print("Row-wise proportions (within each shot type):")
print((row_props * 100).round(1).head(20), "\n")

#Run uni-dimensional tests on 1 variable and multi-dimensional tests on combinations of variables.

# ==========================================
# 7) Chi-square test of independence
# ==========================================

# ==========================================
# 7a) Chi-square Goodness-of-Fit: Shot Type Significance
# ==========================================

print("\n================ Chi-square Goodness-of-Fit: Shot Type =================\n")

# Frequency table of observed shot types
shot_counts = filtered["shot_type"].value_counts(dropna=False)
print("Frequency of each shot type (last shots only):\n")
print(shot_counts, "\n")

# Define observed and expected frequencies
observed = shot_counts.values
shot_labels = shot_counts.index.tolist()

# H₀: all shot types equally likely → expected = mean count
expected = np.full_like(observed, observed.mean())

# Run Chi-square Goodness-of-Fit test
from scipy.stats import chisquare
chi2, p = chisquare(f_obs=observed, f_exp=expected)

# Report results
print(f"Chi-square statistic : {chi2:,.2f}")
print(f"Degrees of freedom   : {len(observed) - 1}") #5 shot types, so 4 dof
print(f"P-value              : {p:.8f}") #p=value is 0.2 since the shot types is 5, so expected probability = 0.8 (Of the Null Hypothesis)

# Interpret result
if p < 0.05:
    print("\n✅ Result: Significant difference — shot types are not equally likely.")
else:
    print("\n❌ Result: Not significant — shot types occur equally often.")


# Statistical significance on shots, then quadrants, then shots and quadrants. 

# ==========================================
# 7b) Chi-square Goodness-of-Fit: Quadrant Type Significance
# ==========================================

print("\n================ Chi-square Goodness-of-Fit: Quadrant =================\n")

# 1️⃣ Frequency table of observed quadrants
quad_counts = filtered["quadrant"].value_counts(dropna=False)
print("Frequency of each quadrant (last shots only):\n")
print(quad_counts, "\n")

from scipy.stats import chisquare
import numpy as np
import pandas as pd

# Frequency table
quad_counts = filtered["quadrant"].value_counts(dropna=False)
print("Frequency of each quadrant (last shots only):\n")
print(quad_counts, "\n")

observed = quad_counts.values
labels   = quad_counts.index.tolist()
n = observed.sum()
k = len(observed)

# Equal expectation under H0: each quadrant = n/k
expected = np.full(shape=observed.shape, fill_value=n / k, dtype=float)

# (Optional) sanity check
assert np.isclose(observed.sum(), expected.sum())

# Chi-square GOF
chi2, p = chisquare(f_obs=observed, f_exp=expected)

print(f"Chi-square statistic : {chi2:,.2f}")
print(f"Degrees of freedom   : {k - 1}")
print(f"P-value              : {p:.4f}") #It should print 0.75, Check this

if p < 0.05:
    print("\n✅ Result: Significant difference — some quadrants are used more often than others.")
else:
    print("\n❌ Result: Not significant — all quadrants are used equally often.")



# Standardized residuals
residuals = (observed - expected) / np.sqrt(expected)
res_df = pd.DataFrame({
    "quadrant": labels,
    "observed": observed,
    "expected": expected,
    "std_residual": residuals
}).sort_values("std_residual", ascending=False)
print("\nStandardized residuals by quadrant (>|2| ≈ significant):\n")
print(res_df.to_string(index=False))

# Effect size: Cramér’s V
cramers_v = np.sqrt(chi2 / (n * (k - 1)))
print(f"\nCramér’s V = {cramers_v:.3f} (0.1=weak, 0.3=moderate, 0.5=strong)")

# ==========================================
# 8) Chi-square Test of Independence: Shot Type × Quadrant
# ==========================================

print("\n================ Chi-square Test of Independence: Shot Type × Quadrant =================\n")

from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd

# Use the contingency table built earlier
print("Contingency table (shot_type × quadrant):\n")
print(contingency, "\n")

# Run Pearson’s Chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency)

print(f"Chi-square statistic : {chi2:,.2f}")
print(f"Degrees of freedom   : {dof}")
print(f"P-value              : {p:.8f}")

if p < 0.05:
    print("\n✅ Result: Significant association — shot type depends on quadrant.")
else:
    print("\n❌ Result: No significant association — shot type and quadrant independent.")

# ---- Post-hoc analysis ----
# Compute standardized residuals to find which combinations drive significance
residuals = (contingency - expected) / np.sqrt(expected)
residuals_df = residuals.stack().reset_index()
residuals_df.columns = ["shot_type", "quadrant", "std_residual"]
residuals_df["abs_resid"] = residuals_df["std_residual"].abs()

print("\nTop 10 most significant shot-type × quadrant combinations (|residual|):\n")
print(residuals_df.sort_values("abs_resid", ascending=False).head(10).to_string(index=False))

# ---- Effect size ----
n = contingency.to_numpy().sum()
r, c = contingency.shape
cramers_v = np.sqrt(chi2 / (n * (min(r, c) - 1)))
print(f"\nCramér’s V = {cramers_v:.3f} (0.1=weak, 0.3=moderate, 0.5=strong)\n")

# ==========================================
# 9) Analysis: Where last shots were taken from and shot type
# ==========================================

print("\n================ Shot Type × Quadrant of Origin ================\n")

# Create an 'origin_quadrant' column based on x,y (impact position)
def assign_origin_quadrant(x, y, mid=0.5):
    if np.isnan(x) or np.isnan(y):
        return np.nan
    if y < mid and x < mid:
        return "back_left"
    elif y < mid and x >= mid:
        return "back_right"
    elif y >= mid and x < mid:
        return "front_left"
    else:
        return "front_right"

filtered["origin_quadrant"] = [
    assign_origin_quadrant(x, y) for x, y in zip(filtered["x"], filtered["y"])
]

# Frequency table of origin quadrants
print("Frequency of Origin Quadrants (last shots only):\n")
print(filtered["origin_quadrant"].value_counts(dropna=False), "\n")

# Build contingency table: shot_type × origin_quadrant
origin_ct = pd.crosstab(filtered["shot_type"], filtered["origin_quadrant"])
print("Shot Type × Origin Quadrant contingency table (counts):\n")
print(origin_ct, "\n")

