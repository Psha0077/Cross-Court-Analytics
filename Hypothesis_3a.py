
# 3a. Do RIGHT-handed players hit certain shots from specific locations in the court? 

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency

# ==========================================
# 0) Load dataset
# ==========================================

COMBINED_CSV_PATH = Path(
    "/Users/pranav/Desktop/Dataprojects/Squash/Cross Court Analytics/Raw_data/csv_files/all_matches_combined.csv"
)

shots = pd.read_csv(COMBINED_CSV_PATH)
print(f"Loaded {len(shots):,} rows from {COMBINED_CSV_PATH}")
print("Columns:", list(shots.columns), "\n")

# ==========================================
# 1) List of known left-handed players
# ==========================================

lefties = [
    "Amanda Sobhy",
    "Amanda Sohby",
    "Balázs Farkas",
    "Nick Wall",
    "Raphael Kandra"
]

# Strip whitespace
shots["playerA"] = shots["playerA"].astype(str).str.strip()
shots["playerB"] = shots["playerB"].astype(str).str.strip()

# ==========================================
# 2) WHO hit each shot (hitter name)
# ==========================================

shots["team"] = pd.to_numeric(shots["team"], errors="coerce")

def get_hitter(row):
    if row["team"] == 0:
        return row["playerA"]
    elif row["team"] == 1:
        return row["playerB"]
    else:
        return np.nan

shots["hitter_name"] = shots.apply(get_hitter, axis=1)

# ==========================================
# 3) REMOVE left-handed players
#    (keep ONLY shots by right-handed players)
# ==========================================

right_shots = shots[~shots["hitter_name"].isin(lefties)].copy()

print(f"\nTotal shots in dataset         : {len(shots):,}")
print(f"Shots kept (RIGHT-HANDERS only): {len(right_shots):,}\n")

print("Preview of RIGHT-HANDED shots:")
print(
    right_shots[
        ["match_id", "game_no", "rally_no", "shot_idx",
         "hitter_name", "x", "y", "shot_type"]
    ].head()
)

# ==========================================
# 4) Ensure numeric coordinate columns
# ==========================================

right_shots["x"] = pd.to_numeric(right_shots["x"], errors="coerce")
right_shots["y"] = pd.to_numeric(right_shots["y"], errors="coerce")
right_shots["finalPositionX"] = pd.to_numeric(right_shots["finalPositionX"], errors="coerce")
right_shots["finalPositionY"] = pd.to_numeric(right_shots["finalPositionY"], errors="coerce")

# ==========================================
# 5) Quadrant assignment function
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

right_shots["from_quadrant"] = [
    assign_quadrant(xx, yy) for xx, yy in zip(right_shots["x"], right_shots["y"])
]

right_shots["to_quadrant"] = [
    assign_quadrant(xx, yy) for xx, yy in zip(right_shots["finalPositionX"], right_shots["finalPositionY"])
]

# Keep only valid quadrant records
right_q = right_shots.dropna(subset=["from_quadrant", "to_quadrant"]).copy()

print(f"\nRIGHT-HANDED shots with valid from/to quadrants: {len(right_q):,}\n")

# ==========================================
# 6) Clean shot type
# ==========================================

right_q["shot_type_clean"] = (
    right_q["shot_type"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({"nan": np.nan, "none": np.nan})
)

right_q = right_q.dropna(subset=["shot_type_clean"])

print("Top shot types for RIGHT-handers:")
print(right_q["shot_type_clean"].value_counts().head(10), "\n")

# ==========================================
# 7) Shot Type × FROM Quadrant × TO Quadrant (counts)
# ==========================================

order_quads = ["back_left", "back_right", "front_left", "front_right"]

right_origin_dest_ct = (
    right_q
    .groupby(["shot_type_clean", "from_quadrant", "to_quadrant"])
    .size()
    .unstack("to_quadrant", fill_value=0)
    .reindex(columns=order_quads, fill_value=0)
)

print("===== RIGHT-HANDERS — Shot Type × FROM Quadrant × TO Quadrant (counts) =====\n")
print(right_origin_dest_ct.to_string(), "\n")


# ==========================================
# 8) Chi-square significance tests by shot type
# ==========================================

print("\n================ RIGHT-HANDERS — SIGNIFICANCE BY SHOT TYPE ================\n")

order = ["back_left", "back_right", "front_left", "front_right"]

for shot in right_q["shot_type_clean"].unique():

    print(f"\n===== Shot Type (RIGHTIES): {shot.upper()} =====\n")

    subset = right_q[right_q["shot_type_clean"] == shot]

    ct = pd.crosstab(
        subset["from_quadrant"],
        subset["to_quadrant"]
    ).reindex(index=order, columns=order, fill_value=0)

    if ct.to_numpy().sum() == 0:
        print("No data for this shot type among right-handers.\n")
        continue

    chi2, p, dof, expected = chi2_contingency(ct)

    print(f"Chi-square statistic : {chi2:,.2f}")
    print(f"Degrees of freedom   : {dof}")
    print(f"P-value              : {p:.6f}\n")

    expected_df = pd.DataFrame(expected, index=order, columns=order)
    residuals = (ct - expected_df) / np.sqrt(expected_df)

    flat = residuals.stack().reset_index()
    flat.columns = ["from_quadrant", "to_quadrant", "std_residual"]
    flat["abs_resid"] = flat["std_residual"].abs()
    flat["direction"] = np.where(
        flat["std_residual"] > 0,
        "overrepresented",
        "underrepresented"
    )

    flat_sorted = flat.sort_values("abs_resid", ascending=False)

    for _, row in flat_sorted.iterrows():
        from_q = row["from_quadrant"]
        to_q   = row["to_quadrant"]
        z      = row["std_residual"]
        direc  = row["direction"]

        print(
            f"{shot:<7} {from_q:>10} → {to_q:<11} "
            f"residual={z:+.2f} ({direc})"
        )
