# 3b - Do all players hit certain types of shots from certain places in the court? 

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
# 4) Ensure coordinate columns are numeric
# ==========================================

right_shots["x"] = pd.to_numeric(right_shots["x"], errors="coerce")
right_shots["y"] = pd.to_numeric(right_shots["y"], errors="coerce")

# Detect destination coordinate columns (where ball is going)
if "finalPositionX" in right_shots.columns and "finalPositionY" in right_shots.columns:
    dest_x_col, dest_y_col = "finalPositionX", "finalPositionY"
elif "finalpositionx" in right_shots.columns and "finalpositiony" in right_shots.columns:
    dest_x_col, dest_y_col = "finalpositionx", "finalpositiony"
elif "x_end" in right_shots.columns and "y_end" in right_shots.columns:
    dest_x_col, dest_y_col = "x_end", "y_end"
elif "final_x" in right_shots.columns and "final_y" in right_shots.columns:
    dest_x_col, dest_y_col = "final_x", "final_y"
else:
    raise KeyError(
        "Destination coordinate columns not found. "
        "Expected one of: finalPositionX/finalPositionY, "
        "finalpositionx/finalpositiony, x_end/y_end, final_x/final_y."
    )

right_shots[dest_x_col] = pd.to_numeric(right_shots[dest_x_col], errors="coerce")
right_shots[dest_y_col] = pd.to_numeric(right_shots[dest_y_col], errors="coerce")

# ==========================================
# 5) Define quadrants (same convention as before)
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

# FROM quadrant = where the shot was hit from (impact location)
right_shots["from_quadrant"] = [
    assign_quadrant(xx, yy) for xx, yy in zip(right_shots["x"], right_shots["y"])
]

# TO quadrant = where the ball is going to (second bounce / error target)
right_shots["to_quadrant"] = [
    assign_quadrant(xx, yy) for xx, yy in zip(right_shots[dest_x_col], right_shots[dest_y_col])
]

# Keep rows with valid quadrants
right_q = right_shots.dropna(subset=["from_quadrant", "to_quadrant"]).copy()
print(f"\nRIGHT-HANDER shots with valid from/to quadrants: {len(right_q):,}\n")

# ==========================================
# 6) Clean shot_type → shot_type_clean
# ==========================================

right_q["shot_type_clean"] = (
    right_q["shot_type"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({"nan": np.nan, "none": np.nan})
)

right_q = right_q.dropna(subset=["shot_type_clean"])

print("Top shot types for right-handers:")
print(right_q["shot_type_clean"].value_counts().head(10), "\n")

# ==========================================
# 7) Prepare bootstrap of chi-square tests
# ==========================================

order_quads = ["back_left", "back_right", "front_left", "front_right"]

# Unique right-handed players in this filtered dataset
right_players = sorted(right_q["hitter_name"].dropna().unique())
print(f"Unique RIGHT-HANDED players in dataset: {len(right_players)}\n")

# Bootstrap settings
n_boot = 1000
sample_size = 7   # randomly pick 7 righties at a time

if len(right_players) < sample_size:
    raise ValueError(f"Not enough right-handed players to sample {sample_size} without replacement.")

np.random.seed(42)
boot_results = []

print(f"Starting bootstrap with {n_boot} iterations, {sample_size} righties per sample...\n")

# ==========================================
# 8) Bootstrap loop — CHI-SQUARE ONLY
# ==========================================

for b in range(1, n_boot + 1):

    # 1) Sample right-handed players
    sampled_players = np.random.choice(
        right_players, size=sample_size, replace=False
    )

    df_sample = right_q[right_q["hitter_name"].isin(sampled_players)].copy()

    # 2) Loop over shot types within this sample
    for shot in df_sample["shot_type_clean"].unique():
        sub = df_sample[df_sample["shot_type_clean"] == shot]

        # Build 4×4 FROM × TO contingency table
        ct_full = pd.crosstab(
            sub["from_quadrant"],
            sub["to_quadrant"]
        ).reindex(index=order_quads, columns=order_quads, fill_value=0)

        # Drop any all-zero rows/columns (no shots from/to that quadrant)
        ct = ct_full.loc[ct_full.sum(axis=1) > 0, ct_full.sum(axis=0) > 0]

        # Need at least 2×2 table to do chi-square
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue

        total_shots = ct.to_numpy().sum()
        if total_shots == 0:
            continue  # extra safety

        # Chi-square test of independence (now safe)
        chi2, p, dof, expected = chi2_contingency(ct)

        boot_results.append({
            "iter": b,
            "shot_type": shot,
            "n_players": sample_size,
            "n_shots": total_shots,
            "chi2": chi2,
            "p_value": p,
            "dof": dof
        })

    if b % 100 == 0:
        print(f"Completed {b} / {n_boot} iterations...")

# ==========================================
# 9) Collect and save results
# ==========================================

boot_df = pd.DataFrame(boot_results)

print("\nBOOTSTRAP RESULTS PREVIEW:\n")
print(boot_df.head())

print("\nSummary of chi-square values by shot type (right-hander samples):\n")
print(boot_df.groupby("shot_type")["chi2"].describe())

out_path = COMBINED_CSV_PATH.parent / "righties_bootstrap_1000iters_chi2_only.csv"
boot_df.to_csv(out_path, index=False)
print(f"\nSaved bootstrap chi-square results to:\n{out_path}")

# ============================================================
# 4) Compare LEFT-HANDERS' chi-square to RIGHT-HANDER bootstrap
# ============================================================

from scipy.stats import chi2_contingency

# --- Rebuild left-hander subset in the SAME script so it's self-contained ---

lefties = [
    "Amanda Sobhy",
    "Amanda Sohby",   # duplicate spelling to catch all rows
    "Balázs Farkas",
    "Nick Wall",
    "Raphael Kandra"
]

# Make sure player names are clean
shots["playerA"] = shots["playerA"].astype(str).str.strip()
shots["playerB"] = shots["playerB"].astype(str).str.strip()

# Who hit each shot?
shots["team"] = pd.to_numeric(shots["team"], errors="coerce")

def get_hitter(row):
    if row["team"] == 0:
        return row["playerA"]
    elif row["team"] == 1:
        return row["playerB"]
    else:
        return np.nan

shots["hitter_name"] = shots.apply(get_hitter, axis=1)

# Keep ONLY left-handed shots
left_shots = shots[shots["hitter_name"].isin(lefties)].copy()

# Ensure numeric coordinates
left_shots["x"]  = pd.to_numeric(left_shots["x"],  errors="coerce")
left_shots["y"]  = pd.to_numeric(left_shots["y"],  errors="coerce")
left_shots["finalPositionX"] = pd.to_numeric(left_shots["finalPositionX"], errors="coerce")
left_shots["finalPositionY"] = pd.to_numeric(left_shots["finalPositionY"], errors="coerce")

# Quadrant function (same as before)
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

# FROM = impact location, TO = second bounce / error target
left_shots["from_quadrant"] = [
    assign_quadrant(xx, yy) for xx, yy in zip(left_shots["x"], left_shots["y"])
]
left_shots["to_quadrant"] = [
    assign_quadrant(xx, yy) for xx, yy in zip(left_shots["finalPositionX"], left_shots["finalPositionY"])
]

left_q = left_shots.dropna(subset=["from_quadrant", "to_quadrant"]).copy()

# Clean shot type
left_q["shot_type_clean"] = (
    left_q["shot_type"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({"nan": np.nan, "none": np.nan})
)
left_q = left_q.dropna(subset=["shot_type_clean"])

# Order of quadrants
order_quads = ["back_left", "back_right", "front_left", "front_right"]

# --- 1) Compute chi-square for LEFTIES by shot type ---

left_chi2 = {}   # shot_type -> chi2
left_dof  = {}   # shot_type -> dof

for shot in left_q["shot_type_clean"].unique():
    subset = left_q[left_q["shot_type_clean"] == shot]

    ct = pd.crosstab(
        subset["from_quadrant"],
        subset["to_quadrant"]
    ).reindex(index=order_quads, columns=order_quads, fill_value=0)

    if ct.to_numpy().sum() == 0:
        continue  # edge case

    chi2, p, dof, expected = chi2_contingency(ct)
    left_chi2[shot] = chi2
    left_dof[shot]  = dof

print("\n=== LEFT-HANDERS: chi-square per shot type ===")
for shot, val in left_chi2.items():
    print(f"{shot:<7} χ² = {val:8.3f}, dof = {left_dof[shot]}")

# --- 2) Compare to RIGHT-HANDER bootstrap (boot_df must already exist) ---

# boot_df is assumed to have columns: iter, shot_type, n_players, n_shots, chi2, p_value, dof

rows = []
for shot, chi_left in left_chi2.items():
    # Right-hander chi-square values for this shot type
    right_vals = boot_df.loc[boot_df["shot_type"] == shot, "chi2"].values

    if right_vals.size == 0:
        continue  # no bootstrap data for this shot type

    # One-sided bootstrap p-value: proportion of righties with chi2 >= lefties
    p_boot = np.mean(right_vals >= chi_left)

    # Percentile rank of lefty χ² within righty distribution
    percentile = 100 * np.mean(right_vals <= chi_left)

    rows.append({
        "shot_type": shot,
        "left_chi2": chi_left,
        "right_mean_chi2": right_vals.mean(),
        "right_median_chi2": np.median(right_vals),
        "right_std_chi2": right_vals.std(ddof=1),
        "right_min_chi2": right_vals.min(),
        "right_max_chi2": right_vals.max(),
        "bootstrap_p_ge": p_boot,           # P(χ²_right >= χ²_left)
        "percentile_in_right": percentile   # where lefty sits vs righties
    })

compare_df = pd.DataFrame(rows)

print("\n=== COMPARISON: Left-handers vs Right-hander bootstrap ===\n")
print(compare_df.sort_values("shot_type").to_string(index=False))

# Lefties behave differently from righties only for some shot types. (lift rebound, volley - much more random than righties)