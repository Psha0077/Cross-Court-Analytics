# How does T-domination determine victory of points? 


# ==========================================
# 0) Load packages
# ==========================================

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ==========================================
# 1) Load dataset
# ==========================================

COMBINED_CSV_PATH = Path(
    "/Users/pranav/Desktop/Dataprojects/Squash/Cross Court Analytics/Raw_data/csv_files/all_matches_combined.csv"
)

shots = pd.read_csv(COMBINED_CSV_PATH, low_memory=False)

print(f"Loaded {len(shots):,} rows")
print("Columns:", list(shots.columns))

# ==========================================
# 2) Make a working copy
# ==========================================

df = shots.copy()

print(df.head())

# 3 Define the T zone. 
T_X = 0.5
T_Y = 0.45

# 4 Take the Euclidean distance of where the shot was hit from to the T

df["dist_to_T"] = np.sqrt(
    (df["x"] - T_X) ** 2 +
    (df["y"] - T_Y) ** 2
)

print(df[["x", "y", "dist_to_T"]].head())