# ==========================================
# Hypothesis 2D:
# Do certain TYPES of movement and positional disruption
# contribute more strongly to rally outcomes?
#
# Builds on Hypothesis 2C:
# Movement quantity alone was weak.
# Now we test movement quality.
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
        "shot_type_clean"
    ]
)

df = df[df["team"].isin([0, 1])]
df = df[df["winner"].isin([0, 1])]

print(f"Rows after basic cleaning: {len(df):,}")

# ==========================================
# 2) Remove duplicate shot rows
# ==========================================

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
# 3) Court distance in metres
# ==========================================

COURT_WIDTH_M = 6.4
COURT_LENGTH_M = 9.75

def court_distance_m(x1, y1, x2, y2):
    return np.sqrt(
        ((x2 - x1) * COURT_WIDTH_M) ** 2 +
        ((y2 - y1) * COURT_LENGTH_M) ** 2
    )

# ==========================================
# 4) Define quadrants / zones
# y convention from your data:
# y = 0 front wall, y = 1 back wall
# ==========================================

def court_zone(x, y):
    if x < 0.5 and y < 0.5:
        return "front_left"
    elif x >= 0.5 and y < 0.5:
        return "front_right"
    elif x < 0.5 and y >= 0.5:
        return "back_left"
    else:
        return "back_right"

df["from_zone"] = [
    court_zone(x, y)
    for x, y in zip(df["x"], df["y"])
]

# ==========================================
# 5) Previous and next opponent positions
# ==========================================
# For the current shot:
# previous row = opponent's previous shot position
# next row     = opponent's next shot position
#
# This lets us estimate how much the CURRENT shot
# forced the opponent to move.

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

df["forced_movement_m"] = np.where(
    df["valid_forced_movement"],
    court_distance_m(
        df["prev_x"],
        df["prev_y"],
        df["next_x"],
        df["next_y"]
    ),
    np.nan
)

# ==========================================
# 6) Movement components
# ==========================================
# These tell us WHAT TYPE of movement was forced:
# - lateral movement: left/right
# - depth movement: front/back
# - diagonal movement: both left/right and front/back

df["forced_dx_m"] = np.where(
    df["valid_forced_movement"],
    (df["next_x"] - df["prev_x"]) * COURT_WIDTH_M,
    np.nan
)

df["forced_dy_m"] = np.where(
    df["valid_forced_movement"],
    (df["next_y"] - df["prev_y"]) * COURT_LENGTH_M,
    np.nan
)

df["abs_forced_dx_m"] = df["forced_dx_m"].abs()
df["abs_forced_dy_m"] = df["forced_dy_m"].abs()

# ==========================================
# 7) Classify movement type
# ==========================================

def classify_movement(dx, dy, min_component=1.0):
    """
    dx = lateral movement in metres
    dy = front/back movement in metres

    Threshold prevents tiny movements being classified as meaningful.
    """
    if pd.isna(dx) or pd.isna(dy):
        return np.nan

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    if abs_dx < min_component and abs_dy < min_component:
        return "small_movement"

    if abs_dx >= min_component and abs_dy < min_component:
        return "lateral_only"

    if abs_dx < min_component and abs_dy >= min_component:
        return "front_back_only"

    if abs_dx >= min_component and abs_dy >= min_component:
        return "diagonal"

    return "unknown"

df["movement_type"] = [
    classify_movement(dx, dy)
    for dx, dy in zip(df["forced_dx_m"], df["forced_dy_m"])
]

# Direction detail
def classify_direction(dx, dy):
    if pd.isna(dx) or pd.isna(dy):
        return np.nan

    horizontal = "right" if dx > 0 else "left"
    vertical = "back" if dy > 0 else "front"

    return f"{vertical}_{horizontal}"

df["movement_direction"] = [
    classify_direction(dx, dy)
    for dx, dy in zip(df["forced_dx_m"], df["forced_dy_m"])
]

# ==========================================
# 8) Did the current shot hitter win the rally?
# ==========================================

df["shot_hitter_won_rally"] = (
    df["team"] == df["winner"]
).astype(int)

# ==========================================
# 9) Shot-level movement quality summary
# ==========================================

movement_type_summary = (
    df.dropna(subset=["forced_movement_m", "movement_type"])
    .groupby("movement_type")
    .agg(
        n_shots=("movement_type", "count"),
        avg_forced_movement_m=("forced_movement_m", "mean"),
        median_forced_movement_m=("forced_movement_m", "median"),
        avg_lateral_movement_m=("abs_forced_dx_m", "mean"),
        avg_front_back_movement_m=("abs_forced_dy_m", "mean"),
        shot_hitter_win_rate=("shot_hitter_won_rally", "mean")
    )
    .reset_index()
)

movement_type_summary["shot_hitter_win_rate"] = (
    movement_type_summary["shot_hitter_win_rate"] * 100
).round(1)

movement_type_summary = movement_type_summary.sort_values(
    "shot_hitter_win_rate",
    ascending=False
)

print("\nMovement type summary:")
print(movement_type_summary.to_string(index=False))

# ==========================================
# 10) Movement direction summary
# ==========================================

movement_direction_summary = (
    df.dropna(subset=["forced_movement_m", "movement_direction"])
    .groupby("movement_direction")
    .agg(
        n_shots=("movement_direction", "count"),
        avg_forced_movement_m=("forced_movement_m", "mean"),
        shot_hitter_win_rate=("shot_hitter_won_rally", "mean")
    )
    .reset_index()
)

movement_direction_summary["shot_hitter_win_rate"] = (
    movement_direction_summary["shot_hitter_win_rate"] * 100
).round(1)

movement_direction_summary = movement_direction_summary.sort_values(
    "shot_hitter_win_rate",
    ascending=False
)

print("\nMovement direction summary:")
print(movement_direction_summary.to_string(index=False))

# ==========================================
# 11) Movement type by shot type
# ==========================================

shot_movement_summary = (
    df.dropna(subset=["forced_movement_m", "movement_type"])
    .groupby(["shot_type_clean", "movement_type"])
    .agg(
        n_shots=("movement_type", "count"),
        avg_forced_movement_m=("forced_movement_m", "mean"),
        shot_hitter_win_rate=("shot_hitter_won_rally", "mean")
    )
    .reset_index()
)

shot_movement_summary["shot_hitter_win_rate"] = (
    shot_movement_summary["shot_hitter_win_rate"] * 100
).round(1)

shot_movement_summary = shot_movement_summary.sort_values(
    ["shot_hitter_win_rate", "n_shots"],
    ascending=[False, False]
)

print("\nShot type × movement type summary:")
print(shot_movement_summary.head(30).to_string(index=False))

# ==========================================
# 12) Rally-level summary:
# Which player forced more high-quality movement?
# ==========================================

# Define high-pressure movement.
# You can adjust this later.
df["high_pressure_movement"] = (
    (df["movement_type"].isin(["diagonal", "front_back_only"])) &
    (df["forced_movement_m"] >= 3.0)
).astype(int)

pressure_by_player = (
    df.dropna(subset=["forced_movement_m"])
    .groupby(["match_id", "game_no", "rally_no", "team"])
    .agg(
        avg_forced_movement_m=("forced_movement_m", "mean"),
        total_forced_movement_m=("forced_movement_m", "sum"),
        high_pressure_count=("high_pressure_movement", "sum"),
        avg_lateral_movement_m=("abs_forced_dx_m", "mean"),
        avg_front_back_movement_m=("abs_forced_dy_m", "mean"),
        n_forcing_shots=("forced_movement_m", "count")
    )
    .reset_index()
)

rally_winner = (
    df.groupby(["match_id", "game_no", "rally_no"])["winner"]
    .first()
    .reset_index()
)

pressure_by_player = pressure_by_player.merge(
    rally_winner,
    on=["match_id", "game_no", "rally_no"],
    how="left"
)

pressure_by_player["player_won_rally"] = (
    pressure_by_player["team"] == pressure_by_player["winner"]
).astype(int)

# ==========================================
# 13) Winner vs loser pressure summary
# ==========================================

winner_loser_pressure = (
    pressure_by_player
    .groupby("player_won_rally")
    .agg(
        avg_forced_movement_m=("avg_forced_movement_m", "mean"),
        avg_total_forced_movement_m=("total_forced_movement_m", "mean"),
        avg_high_pressure_count=("high_pressure_count", "mean"),
        avg_front_back_movement_m=("avg_front_back_movement_m", "mean"),
        avg_lateral_movement_m=("avg_lateral_movement_m", "mean"),
        n_player_rallies=("player_won_rally", "count")
    )
    .reset_index()
)

winner_loser_pressure["player_won_rally"] = winner_loser_pressure[
    "player_won_rally"
].map({
    0: "Lost rally",
    1: "Won rally"
})

print("\nWinner vs loser movement-quality summary:")
print(winner_loser_pressure.to_string(index=False))

# ==========================================
# 14) High-pressure dominant player win rate
# ==========================================

pressure_pivot = pressure_by_player.pivot_table(
    index=["match_id", "game_no", "rally_no"],
    columns="team",
    values="high_pressure_count"
).reset_index()

pressure_pivot = pressure_pivot.rename(columns={
    0.0: "team0_high_pressure_count",
    1.0: "team1_high_pressure_count",
    0: "team0_high_pressure_count",
    1: "team1_high_pressure_count"
})

pressure_pivot = pressure_pivot.merge(
    rally_winner,
    on=["match_id", "game_no", "rally_no"],
    how="left"
)

pressure_pivot = pressure_pivot.dropna(
    subset=[
        "team0_high_pressure_count",
        "team1_high_pressure_count",
        "winner"
    ]
)

pressure_pivot = pressure_pivot[
    pressure_pivot["team0_high_pressure_count"] !=
    pressure_pivot["team1_high_pressure_count"]
].copy()

pressure_pivot["high_pressure_dominant_team"] = np.where(
    pressure_pivot["team0_high_pressure_count"] >
    pressure_pivot["team1_high_pressure_count"],
    0,
    1
)

pressure_pivot["high_pressure_dominant_player_won"] = (
    pressure_pivot["high_pressure_dominant_team"] ==
    pressure_pivot["winner"]
).astype(int)

high_pressure_win_rate = (
    pressure_pivot["high_pressure_dominant_player_won"].mean() * 100
)

print(
    f"\nHigh-pressure-movement dominant player won "
    f"{high_pressure_win_rate:.1f}% of non-tied rallies."
)

print("\nHigh-pressure dominance preview:")
print(pressure_pivot.head())

# ==========================================
# 15) Save outputs
# ==========================================

out_dir = COMBINED_CSV_PATH.parent

df.to_csv(
    out_dir / "shot_level_movement_quality.csv",
    index=False
)

movement_type_summary.to_csv(
    out_dir / "movement_type_summary.csv",
    index=False
)

movement_direction_summary.to_csv(
    out_dir / "movement_direction_summary.csv",
    index=False
)

shot_movement_summary.to_csv(
    out_dir / "shot_type_movement_type_summary.csv",
    index=False
)

pressure_by_player.to_csv(
    out_dir / "pressure_by_player_rally.csv",
    index=False
)

winner_loser_pressure.to_csv(
    out_dir / "winner_loser_movement_quality_summary.csv",
    index=False
)

pressure_pivot.to_csv(
    out_dir / "high_pressure_dominance_by_rally.csv",
    index=False
)

print("\nSaved outputs:")
print(out_dir / "shot_level_movement_quality.csv")
print(out_dir / "movement_type_summary.csv")
print(out_dir / "movement_direction_summary.csv")
print(out_dir / "shot_type_movement_type_summary.csv")
print(out_dir / "pressure_by_player_rally.csv")
print(out_dir / "winner_loser_movement_quality_summary.csv")
print(out_dir / "high_pressure_dominance_by_rally.csv")