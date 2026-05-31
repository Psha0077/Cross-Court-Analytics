
# Are there certain shot types players play from and to certain parts of the court? 

import pandas as pd
import numpy as np
from pathlib import Path

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
# 1) Quadrant function (sign convention confirmed)
# X = 0 left → 1 right
# Y = 0 back → 1 front
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


# ==========================================
# 2) Ensure numeric columns
# ==========================================

shots["x"] = pd.to_numeric(shots["x"], errors="coerce")
shots["y"] = pd.to_numeric(shots["y"], errors="coerce")
shots["shot_idx"] = pd.to_numeric(shots["shot_idx"], errors="coerce")

shots = shots.dropna(subset=["x", "y", "shot_idx"])
print(f"After cleaning, {len(shots):,} rows remain.\n")


# ==========================================
# 3) Detect destination coordinate columns
# (winner second-bounce or error attempted bounce)
# ==========================================

if "finalPositionX" in shots.columns and "finalPositionY" in shots.columns:
    dest_x_col, dest_y_col = "finalPositionX", "finalPositionY"
elif "finalpositionx" in shots.columns and "finalpositiony" in shots.columns:
    dest_x_col, dest_y_col = "finalpositionx", "finalpositiony"
elif "x_end" in shots.columns and "y_end" in shots.columns:
    dest_x_col, dest_y_col = "x_end", "y_end"
elif "final_x" in shots.columns and "final_y" in shots.columns:
    dest_x_col, dest_y_col = "final_x", "final_y"
else:
    raise KeyError("Could not identify destination coordinate columns.")

# Ensure numeric destination coords
shots[dest_x_col] = pd.to_numeric(shots[dest_x_col], errors="coerce")
shots[dest_y_col] = pd.to_numeric(shots[dest_y_col], errors="coerce")


# ==========================================
# 4) Assign FROM and TO quadrants
# ==========================================

shots["from_quadrant"] = [assign_quadrant(xx, yy) for xx, yy in zip(shots["x"], shots["y"])]
shots["to_quadrant"]   = [assign_quadrant(xx, yy) for xx, yy in zip(shots[dest_x_col], shots[dest_y_col])]

shots_q = shots.dropna(subset=["from_quadrant", "to_quadrant"]).copy()
print(f"Using {len(shots_q):,} shots with valid from/to quadrants.\n")


# ==========================================
# 5) Basic FROM → TO transition table (4x4)
# ==========================================

order_quads = ["back_left", "back_right", "front_left", "front_right"]

transition_ct = pd.crosstab(
    shots_q["from_quadrant"],
    shots_q["to_quadrant"]
).reindex(index=order_quads, columns=order_quads, fill_value=0)

print("===== Counts: FROM quadrant → TO quadrant =====\n")
print(transition_ct.to_string(), "\n")


# ==========================================
# 6) Shot Type × FROM Quadrant × TO Quadrant
# ==========================================

shots_q["shot_type_clean"] = (
    shots_q["shot_type"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({"nan": np.nan, "none": np.nan})
)

shots_q_st = shots_q.dropna(subset=["shot_type_clean"]).copy()

shot_from_to_ct = (
    shots_q_st
    .groupby(["shot_type_clean", "from_quadrant", "to_quadrant"])
    .size()
    .unstack("to_quadrant", fill_value=0)
    .reindex(columns=order_quads, fill_value=0)
)

print("===== Shot Type × FROM Quadrant × TO Quadrant (counts) =====\n")
print(shot_from_to_ct.head(20).to_string(), "\n")

from scipy.stats import chi2_contingency
import numpy as np

significant_results = {}

for stype, df_sub in shots_q_st.groupby("shot_type_clean"):
    print(f"\n===== Shot Type: {stype.upper()} =====")
    
    # 4×4 quadrant table
    ct = pd.crosstab(df_sub["from_quadrant"], df_sub["to_quadrant"]).reindex(
        index=order_quads, columns=order_quads, fill_value=0
    )
    
    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(ct)
    print(f"Chi-square p-value: {p:.6f}")
    
    # Standardised residuals
    residuals = (ct - expected) / np.sqrt(expected)
    res_df = pd.DataFrame(residuals, index=order_quads, columns=order_quads)
    
    # Save results
    significant_results[stype] = res_df
    
order = ["back_left", "back_right", "front_left", "front_right"]

print("\n================ SIGNIFICANCE BY SHOT TYPE ================\n")

# Group by shot type
for shot in shots_q_st["shot_type_clean"].unique():

    print(f"\n===== Shot Type: {shot.upper()} =====\n")

    # 1. Build contingency table for this shot type only
    subset = shots_q_st[shots_q_st["shot_type_clean"] == shot]

    ct = pd.crosstab(
        subset["from_quadrant"],
        subset["to_quadrant"]
    ).reindex(index=order, columns=order, fill_value=0)

    # 2. Expected frequencies under independence
    total = ct.to_numpy().sum()
    row_totals = ct.sum(axis=1).values.reshape(-1, 1)
    col_totals = ct.sum(axis=0).values.reshape(1, -1)
    expected = (row_totals @ col_totals) / total
    expected_df = pd.DataFrame(expected, index=order, columns=order)

    # 3. Standardized residuals
    residuals = (ct - expected_df) / np.sqrt(expected_df)

    # 4. Flatten results
    flat = residuals.stack().reset_index()
    flat.columns = ["from_quadrant", "to_quadrant", "std_residual"]
    flat["abs_resid"] = flat["std_residual"].abs()
    flat["direction"] = np.where(
        flat["std_residual"] > 0,
        "overrepresented",
        "underrepresented"
    )

    # 5. Print ALL transitions sorted by significance
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


# ==========================================
# Hypothesis 2B:
# Can we identify common rally "playbook" sequences?
# ==========================================

import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 0) Load dataset
# ==========================================

COMBINED_CSV_PATH = Path(
    "/Users/pranav/Desktop/Dataprojects/Squash/Cross Court Analytics/Raw_data/csv_files/all_matches_combined.csv"
)

shots = pd.read_csv(COMBINED_CSV_PATH, low_memory=False)

print(f"Loaded {len(shots):,} rows")

df = shots.copy()

# ==========================================
# 1) Clean key columns
# ==========================================

df["shot_idx"] = pd.to_numeric(df["shot_idx"], errors="coerce")
df["team"] = pd.to_numeric(df["team"], errors="coerce")
df["winner"] = pd.to_numeric(df["winner"], errors="coerce")

df["shot_type_clean"] = (
    df["shot_type"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({"nan": np.nan, "none": np.nan})
)

df = df.dropna(
    subset=[
        "match_id",
        "game_no",
        "rally_no",
        "shot_idx",
        "team",
        "winner",
        "shot_type_clean"
    ]
)

# ==========================================
# 2) Sort shots in rally order
# ==========================================

df = df.sort_values(
    ["match_id", "game_no", "rally_no", "shot_idx"]
)

# ==========================================
# 3) Create 3-shot sequences inside each rally
# ==========================================

sequence_rows = []

group_cols = ["match_id", "game_no", "rally_no"]

for rally_key, rally in df.groupby(group_cols):
    rally = rally.sort_values("shot_idx").reset_index(drop=True)

    if len(rally) < 3:
        continue

    rally_winner = rally["winner"].iloc[-1]

    for i in range(len(rally) - 2):
        shot_1 = rally.iloc[i]
        shot_2 = rally.iloc[i + 1]
        shot_3 = rally.iloc[i + 2]

        sequence = (
            f"{shot_1['shot_type_clean']} → "
            f"{shot_2['shot_type_clean']} → "
            f"{shot_3['shot_type_clean']}"
        )

        final_shot_team = shot_3["team"]

        final_shot_team_won_rally = int(final_shot_team == rally_winner)

        sequence_rows.append({
            "match_id": rally_key[0],
            "game_no": rally_key[1],
            "rally_no": rally_key[2],
            "start_shot_idx": shot_1["shot_idx"],
            "sequence": sequence,
            "shot_1_team": shot_1["team"],
            "shot_2_team": shot_2["team"],
            "shot_3_team": shot_3["team"],
            "rally_winner": rally_winner,
            "final_shot_team_won_rally": final_shot_team_won_rally
        })

seq_df = pd.DataFrame(sequence_rows)

print(f"Created {len(seq_df):,} 3-shot sequences")

# ==========================================
# 4) Summarise playbook patterns
# ==========================================

playbook = (
    seq_df
    .groupby("sequence")
    .agg(
        frequency=("sequence", "count"),
        final_shot_team_win_rate=("final_shot_team_won_rally", "mean")
    )
    .reset_index()
)

playbook["final_shot_team_win_rate"] = (
    playbook["final_shot_team_win_rate"] * 100
).round(1)

playbook = playbook.sort_values(
    ["frequency", "final_shot_team_win_rate"],
    ascending=[False, False]
)

print("\nTop 20 most common 3-shot sequences:")
print(playbook.head(20).to_string(index=False))

# ==========================================
# 5) Filter for meaningful sequences
# ==========================================

min_frequency = 100

strong_playbook = playbook[
    playbook["frequency"] >= min_frequency
].sort_values(
    "final_shot_team_win_rate",
    ascending=False
)

print(f"\nTop high-performing sequences with frequency >= {min_frequency}:")
print(strong_playbook.head(20).to_string(index=False))

# ==========================================
# 6) Save output
# ==========================================

out_path = COMBINED_CSV_PATH.parent / "three_shot_playbook_sequences.csv"
playbook.to_csv(out_path, index=False)

print(f"\nSaved playbook output to:\n{out_path}")


# ==========================================
# Hypothesis 2B:
# What happens in the 2–3 shots BEFORE the rally-ending shot?
# Goal: identify pressure-building patterns before points end.
# ==========================================

import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 0) Load dataset
# ==========================================

COMBINED_CSV_PATH = Path(
    "/Users/pranav/Desktop/Dataprojects/Squash/Cross Court Analytics/Raw_data/csv_files/all_matches_combined.csv"
)

shots = pd.read_csv(COMBINED_CSV_PATH, low_memory=False)

print(f"Loaded {len(shots):,} rows")

df = shots.copy()

# ==========================================
# 1) Clean key columns
# ==========================================

df["shot_idx"] = pd.to_numeric(df["shot_idx"], errors="coerce")
df["team"] = pd.to_numeric(df["team"], errors="coerce")
df["winner"] = pd.to_numeric(df["winner"], errors="coerce")
df["x"] = pd.to_numeric(df["x"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df["finalPositionX"] = pd.to_numeric(df["finalPositionX"], errors="coerce")
df["finalPositionY"] = pd.to_numeric(df["finalPositionY"], errors="coerce")

df["shot_type_clean"] = (
    df["shot_type"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({"nan": np.nan, "none": np.nan})
)

df = df.dropna(
    subset=[
        "match_id",
        "game_no",
        "rally_no",
        "shot_idx",
        "team",
        "winner",
        "x",
        "y",
        "finalPositionX",
        "finalPositionY",
        "shot_type_clean"
    ]
)

# ==========================================
# 2) Define quadrants
# Note:
# x = 0 left wall, x = 1 right wall
# y = 0 front wall, y = 1 back wall
# ==========================================

def assign_quadrant(x, y, mid=0.5):
    if pd.isna(x) or pd.isna(y):
        return np.nan

    if x < mid and y < mid:
        return "front_left"
    elif x >= mid and y < mid:
        return "front_right"
    elif x < mid and y >= mid:
        return "back_left"
    else:
        return "back_right"


df["from_quadrant"] = [
    assign_quadrant(x, y)
    for x, y in zip(df["x"], df["y"])
]

df["to_quadrant"] = [
    assign_quadrant(x, y)
    for x, y in zip(df["finalPositionX"], df["finalPositionY"])
]

df = df.dropna(subset=["from_quadrant", "to_quadrant"])

# ==========================================
# 3) Define distance from T
# This measures how far the player was from the T
# when they hit the shot.
# ==========================================

T_X = 0.5
T_Y = 0.45

df["dist_to_T"] = np.sqrt(
    (df["x"] - T_X) ** 2 +
    (df["y"] - T_Y) ** 2
)

# ==========================================
# 4) Sort rally order
# ==========================================

df = df.sort_values(
    ["match_id", "game_no", "rally_no", "shot_idx"]
).reset_index(drop=True)

# ==========================================
# 5) Extract final 4 shots of each rally
# We want:
# - setup_3: three shots before final shot
# - setup_2: two shots before final shot
# - setup_1: one shot before final shot
# - final: rally-ending shot
# ==========================================

group_cols = ["match_id", "game_no", "rally_no"]

rally_rows = []

for rally_key, rally in df.groupby(group_cols, sort=False):
    rally = rally.sort_values("shot_idx").reset_index(drop=True)

    if len(rally) < 4:
        continue

    rally_winner = rally["winner"].iloc[-1]

    setup_3 = rally.iloc[-4]
    setup_2 = rally.iloc[-3]
    setup_1 = rally.iloc[-2]
    final = rally.iloc[-1]

    final_shot_team = final["team"]

    rally_rows.append({
        "match_id": rally_key[0],
        "game_no": rally_key[1],
        "rally_no": rally_key[2],
        "rally_length": len(rally),
        "rally_winner": rally_winner,

        # Shot type sequence before point ends
        "shot_sequence_last4": (
            f"{setup_3['shot_type_clean']} → "
            f"{setup_2['shot_type_clean']} → "
            f"{setup_1['shot_type_clean']} → "
            f"{final['shot_type_clean']}"
        ),

        "shot_sequence_setup3": (
            f"{setup_3['shot_type_clean']} → "
            f"{setup_2['shot_type_clean']} → "
            f"{setup_1['shot_type_clean']}"
        ),

        # Location sequence before point ends
        "from_sequence_last4": (
            f"{setup_3['from_quadrant']} → "
            f"{setup_2['from_quadrant']} → "
            f"{setup_1['from_quadrant']} → "
            f"{final['from_quadrant']}"
        ),

        "to_sequence_last4": (
            f"{setup_3['to_quadrant']} → "
            f"{setup_2['to_quadrant']} → "
            f"{setup_1['to_quadrant']} → "
            f"{final['to_quadrant']}"
        ),

        # T-distance pattern
        "setup_3_dist_to_T": setup_3["dist_to_T"],
        "setup_2_dist_to_T": setup_2["dist_to_T"],
        "setup_1_dist_to_T": setup_1["dist_to_T"],
        "final_dist_to_T": final["dist_to_T"],

        "avg_setup_dist_to_T": np.mean([
            setup_3["dist_to_T"],
            setup_2["dist_to_T"],
            setup_1["dist_to_T"]
        ]),

        "final_shot_team": final_shot_team,
        "final_shot_team_won": int(final_shot_team == rally_winner),

        # Was the rally winner the same player who hit the final shot?
        # Useful sanity check for winners vs errors.
        "winMethod": final["winMethod"]
    })

rally_end_df = pd.DataFrame(rally_rows)

print(f"Created rally-ending setup dataset: {len(rally_end_df):,} rallies")

# ==========================================
# 6) Most common final 4-shot patterns
# ==========================================

last4_summary = (
    rally_end_df
    .groupby("shot_sequence_last4")
    .agg(
        frequency=("shot_sequence_last4", "count"),
        final_shot_team_win_rate=("final_shot_team_won", "mean"),
        avg_setup_dist_to_T=("avg_setup_dist_to_T", "mean"),
        avg_rally_length=("rally_length", "mean")
    )
    .reset_index()
)

last4_summary["final_shot_team_win_rate"] = (
    last4_summary["final_shot_team_win_rate"] * 100
).round(1)

last4_summary = last4_summary.sort_values(
    ["frequency", "final_shot_team_win_rate"],
    ascending=[False, False]
)

print("\nTop 20 most common final 4-shot patterns:")
print(last4_summary.head(20).to_string(index=False))

# ==========================================
# 7) Most common 3-shot setup patterns
# This ignores the final shot and focuses only on the build-up.
# ==========================================

setup3_summary = (
    rally_end_df
    .groupby("shot_sequence_setup3")
    .agg(
        frequency=("shot_sequence_setup3", "count"),
        final_shot_team_win_rate=("final_shot_team_won", "mean"),
        avg_setup_dist_to_T=("avg_setup_dist_to_T", "mean"),
        avg_rally_length=("rally_length", "mean")
    )
    .reset_index()
)

setup3_summary["final_shot_team_win_rate"] = (
    setup3_summary["final_shot_team_win_rate"] * 100
).round(1)

setup3_summary = setup3_summary.sort_values(
    ["frequency", "final_shot_team_win_rate"],
    ascending=[False, False]
)

print("\nTop 20 most common 3-shot setup patterns:")
print(setup3_summary.head(20).to_string(index=False))

# ==========================================
# 8) Location-based setup patterns
# This is more useful than obvious shot-type patterns.
# ==========================================

location_summary = (
    rally_end_df
    .groupby("to_sequence_last4")
    .agg(
        frequency=("to_sequence_last4", "count"),
        final_shot_team_win_rate=("final_shot_team_won", "mean"),
        avg_setup_dist_to_T=("avg_setup_dist_to_T", "mean"),
        avg_rally_length=("rally_length", "mean")
    )
    .reset_index()
)

location_summary["final_shot_team_win_rate"] = (
    location_summary["final_shot_team_win_rate"] * 100
).round(1)

location_summary = location_summary.sort_values(
    ["frequency", "final_shot_team_win_rate"],
    ascending=[False, False]
)

print("\nTop 20 most common location patterns before rally ends:")
print(location_summary.head(20).to_string(index=False))

# ==========================================
# 9) High-performing but reasonably common patterns
# ==========================================

min_frequency = 100

high_performing_setups = setup3_summary[
    setup3_summary["frequency"] >= min_frequency
].sort_values(
    "final_shot_team_win_rate",
    ascending=False
)

print(f"\nHigh-performing 3-shot setups with frequency >= {min_frequency}:")
print(high_performing_setups.head(20).to_string(index=False))

high_performing_locations = location_summary[
    location_summary["frequency"] >= min_frequency
].sort_values(
    "final_shot_team_win_rate",
    ascending=False
)

print(f"\nHigh-performing location patterns with frequency >= {min_frequency}:")
print(high_performing_locations.head(20).to_string(index=False))

# ==========================================
# 10) Save outputs
# ==========================================

out_dir = COMBINED_CSV_PATH.parent

rally_end_df.to_csv(
    out_dir / "rally_end_setup_dataset.csv",
    index=False
)

last4_summary.to_csv(
    out_dir / "last4_shot_sequence_summary.csv",
    index=False
)

setup3_summary.to_csv(
    out_dir / "setup3_shot_sequence_summary.csv",
    index=False
)

location_summary.to_csv(
    out_dir / "location_sequence_summary.csv",
    index=False
)

print("\nSaved outputs:")
print(out_dir / "rally_end_setup_dataset.csv")
print(out_dir / "last4_shot_sequence_summary.csv")
print(out_dir / "setup3_shot_sequence_summary.csv")
print(out_dir / "location_sequence_summary.csv")