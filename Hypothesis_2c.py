# ==========================================
# Hypothesis 2C:
# How much movement does each player force the opponent to make?
#
# Core idea:
# For a shot by Player A, measure how far Player B had to move
# from Player B's previous shot location to Player B's next shot location.
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

df = df.dropna(
    subset=[
        "match_id",
        "game_no",
        "rally_no",
        "shot_idx",
        "team",
        "winner",
        "x",
        "y"
    ]
)

df = df[df["team"].isin([0, 1])]
df = df[df["winner"].isin([0, 1])]

print(f"Rows after basic cleaning: {len(df):,}")

# ==========================================
# 2) Remove duplicate shot rows
# ==========================================
# Your dataset appears to contain duplicated rows per shot.
# This step keeps one row per unique shot.

df = df.drop_duplicates(
    subset=[
        "match_id",
        "game_no",
        "rally_no",
        "shot_idx",
        "team",
        "x",
        "y"
    ]
).copy()

df = df.sort_values(
    ["match_id", "game_no", "rally_no", "shot_idx"]
).reset_index(drop=True)

print(f"Rows after removing duplicate shots: {len(df):,}")

# ==========================================
# 3) Convert normalised court distance to metres
# ==========================================

COURT_WIDTH_M = 6.4
COURT_LENGTH_M = 9.75

def court_distance_m(x1, y1, x2, y2):
    """
    Converts normalised coordinate movement into approximate metres.
    x is court width: 0 to 1
    y is court length: 0 to 1
    """
    return np.sqrt(
        ((x2 - x1) * COURT_WIDTH_M) ** 2 +
        ((y2 - y1) * COURT_LENGTH_M) ** 2
    )

# ==========================================
# 4) Get previous and next shot in each rally
# ==========================================
# For current shot:
# previous row = opponent's previous shot
# next row     = opponent's next shot
#
# If previous and next rows are by the same opponent,
# then the current shot forced that opponent to move from
# their previous shot location to their next shot location.

rally_cols = ["match_id", "game_no", "rally_no"]

df["prev_team"] = df.groupby(rally_cols)["team"].shift(1)
df["next_team"] = df.groupby(rally_cols)["team"].shift(-1)

df["prev_x"] = df.groupby(rally_cols)["x"].shift(1)
df["prev_y"] = df.groupby(rally_cols)["y"].shift(1)

df["next_x"] = df.groupby(rally_cols)["x"].shift(-1)
df["next_y"] = df.groupby(rally_cols)["y"].shift(-1)

df["valid_forced_movement"] = (
    (df["prev_team"] == df["next_team"]) &
    (df["prev_team"] != df["team"])
)

df["opponent_forced_movement_m"] = np.where(
    df["valid_forced_movement"],
    court_distance_m(
        df["prev_x"],
        df["prev_y"],
        df["next_x"],
        df["next_y"]
    ),
    np.nan
)

print("\nPreview of forced movement:")
print(
    df[
        [
            "match_id",
            "game_no",
            "rally_no",
            "shot_idx",
            "team",
            "winner",
            "prev_team",
            "next_team",
            "valid_forced_movement",
            "opponent_forced_movement_m"
        ]
    ].head(20)
)

print(
    "\nValid forced movement rows:",
    df["opponent_forced_movement_m"].notna().sum()
)

# ==========================================
# 5) Rally-level summary:
# average movement forced by each player per rally
# ==========================================

forced_by_player = (
    df.dropna(subset=["opponent_forced_movement_m"])
    .groupby(["match_id", "game_no", "rally_no", "team"])
    .agg(
        avg_opponent_forced_movement_m=("opponent_forced_movement_m", "mean"),
        median_opponent_forced_movement_m=("opponent_forced_movement_m", "median"),
        total_opponent_forced_movement_m=("opponent_forced_movement_m", "sum"),
        max_opponent_forced_movement_m=("opponent_forced_movement_m", "max"),
        n_valid_forcing_shots=("opponent_forced_movement_m", "count")
    )
    .reset_index()
)

# ==========================================
# 6) Get rally winner
# ==========================================

rally_winner = (
    df.groupby(["match_id", "game_no", "rally_no"])["winner"]
    .first()
    .reset_index()
)

forced_by_player = forced_by_player.merge(
    rally_winner,
    on=["match_id", "game_no", "rally_no"],
    how="left"
)

forced_by_player["player_won_rally"] = (
    forced_by_player["team"] == forced_by_player["winner"]
).astype(int)

print("\nForced movement by player preview:")
print(forced_by_player.head())

# ==========================================
# 7) Compare rally winners vs rally losers
# ==========================================

winner_loser_summary = (
    forced_by_player
    .groupby("player_won_rally")
    .agg(
        avg_forced_movement_m=("avg_opponent_forced_movement_m", "mean"),
        median_forced_movement_m=("avg_opponent_forced_movement_m", "median"),
        avg_total_forced_movement_m=("total_opponent_forced_movement_m", "mean"),
        avg_max_forced_movement_m=("max_opponent_forced_movement_m", "mean"),
        avg_valid_forcing_shots=("n_valid_forcing_shots", "mean"),
        n_player_rallies=("player_won_rally", "count")
    )
    .reset_index()
)

winner_loser_summary["player_won_rally"] = winner_loser_summary[
    "player_won_rally"
].map({
    0: "Lost rally",
    1: "Won rally"
})

print("\nWinner vs loser forced movement summary:")
print(winner_loser_summary.to_string(index=False))

# ==========================================
# 8) Rally-level dominance:
# did the player who forced more movement win?
# ==========================================

pivot = forced_by_player.pivot_table(
    index=["match_id", "game_no", "rally_no"],
    columns="team",
    values="avg_opponent_forced_movement_m"
).reset_index()

print("\nPivot columns before rename:")
print(pivot.columns)

pivot = pivot.rename(columns={
    0.0: "team0_avg_forced_movement_m",
    1.0: "team1_avg_forced_movement_m",
    0: "team0_avg_forced_movement_m",
    1: "team1_avg_forced_movement_m"
})

pivot = pivot.merge(
    rally_winner,
    on=["match_id", "game_no", "rally_no"],
    how="left"
)

required_cols = [
    "team0_avg_forced_movement_m",
    "team1_avg_forced_movement_m",
    "winner"
]

missing_cols = [col for col in required_cols if col not in pivot.columns]

if missing_cols:
    print("\nCould not calculate movement dominance.")
    print("Missing columns:", missing_cols)
    print("This usually means one team had no valid forced-movement rows.")
else:
    pivot = pivot.dropna(subset=required_cols)

    pivot["movement_dominant_team"] = np.where(
        pivot["team0_avg_forced_movement_m"] >
        pivot["team1_avg_forced_movement_m"],
        0,
        1
    )

    pivot["movement_dominant_player_won"] = (
        pivot["movement_dominant_team"] == pivot["winner"]
    ).astype(int)

    movement_dom_win_rate = (
        pivot["movement_dominant_player_won"].mean() * 100
    )

    print(
        f"\nMovement-dominant player won "
        f"{movement_dom_win_rate:.1f}% of rallies."
    )

    print("\nMovement dominance preview:")
    print(pivot.head())

# ==========================================
# 9) Optional: summary by win method
# ==========================================

rally_win_method = (
    df.groupby(["match_id", "game_no", "rally_no"])["winMethod"]
    .first()
    .reset_index()
)

forced_by_player = forced_by_player.merge(
    rally_win_method,
    on=["match_id", "game_no", "rally_no"],
    how="left"
)

win_method_summary = (
    forced_by_player
    .groupby(["winMethod", "player_won_rally"])
    .agg(
        avg_forced_movement_m=("avg_opponent_forced_movement_m", "mean"),
        n=("avg_opponent_forced_movement_m", "count")
    )
    .reset_index()
)

print("\nForced movement by win method:")
print(win_method_summary.head(20).to_string(index=False))

# ==========================================
# 10) Save outputs
# ==========================================

out_dir = COMBINED_CSV_PATH.parent

df.to_csv(
    out_dir / "shot_level_forced_movement.csv",
    index=False
)

forced_by_player.to_csv(
    out_dir / "forced_movement_by_player_rally.csv",
    index=False
)

if "pivot" in locals():
    pivot.to_csv(
        out_dir / "movement_dominance_by_rally.csv",
        index=False
    )

winner_loser_summary.to_csv(
    out_dir / "winner_loser_forced_movement_summary.csv",
    index=False
)

win_method_summary.to_csv(
    out_dir / "forced_movement_by_win_method.csv",
    index=False
)

print("\nSaved outputs:")
print(out_dir / "shot_level_forced_movement.csv")
print(out_dir / "forced_movement_by_player_rally.csv")
print(out_dir / "movement_dominance_by_rally.csv")
print(out_dir / "winner_loser_forced_movement_summary.csv")
print(out_dir / "forced_movement_by_win_method.csv")



"""
Hypothesis 2C Result:

Average opponent movement forced per shot was only marginally higher
for rally winners (3.38m) compared to rally losers (3.34m).

Additionally, the player who forced greater average movement
won only 50.8% of rallies.

This suggests that movement volume alone has weak predictive power
for rally outcome.

The findings imply that the effectiveness and context of movement
are likely more important than total movement distance.

In particular, movement that creates positional disadvantage,
recovery difficulty, or T-displacement may contribute more strongly
to pressure and rally success than movement quantity alone.
"""