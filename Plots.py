# ==========================================
# 0) Load packages
# ==========================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ==========================================
# 1) Load dataset
# ==========================================

COMBINED_CSV_PATH = Path(
    "/Users/pranav/Desktop/Dataprojects/Squash/Cross Court Analytics/Raw_data/csv_files/all_matches_combined.csv"
)

shots = pd.read_csv(COMBINED_CSV_PATH, low_memory=False)

print(f"Loaded {len(shots):,} rows")

# ==========================================
# 2) Create working dataframe
# ==========================================

df = shots.copy()

# Clean coordinates
df["finalPositionX"] = pd.to_numeric(
    df["finalPositionX"],
    errors="coerce"
)

df["finalPositionY"] = pd.to_numeric(
    df["finalPositionY"],
    errors="coerce"
)

df = df.dropna(
    subset=["finalPositionX", "finalPositionY"]
)

# ==========================================
# 3) Run clustering again
# ==========================================

coords = df[["finalPositionX", "finalPositionY"]]

kmeans = KMeans(
    n_clusters=6,
    random_state=42
)

df["cluster"] = kmeans.fit_predict(coords)

# Get cluster centres
centers = kmeans.cluster_centers_


# ==========================================
# LinkedIn-Ready Cluster Visualization
# ==========================================

import matplotlib.pyplot as plt

# ------------------------------------------
# 1) Sample data for cleaner plotting
# ------------------------------------------

df_sample = df.sample(50000, random_state=42)

# ------------------------------------------
# 2) Create figure
# ------------------------------------------

fig, ax = plt.subplots(figsize=(7, 12))

# ------------------------------------------
# 3) Plot clustered shot locations
# ------------------------------------------

scatter = ax.scatter(
    df_sample["finalPositionX"],
    df_sample["finalPositionY"],
    c=df_sample["cluster"],
    cmap="viridis",
    s=5,
    alpha=0.5
)

# ------------------------------------------
# 4) Plot cluster centres
# ------------------------------------------

ax.scatter(
    centers[:, 0],
    centers[:, 1],
    c="red",
    s=250,
    marker="X",
    edgecolors="black",
    linewidths=1.5,
    label="Cluster Centres"
)

# ------------------------------------------
# 5) Draw simple squash court outline
# ------------------------------------------

# Outer court boundary
court_x = [0, 1, 1, 0, 0]
court_y = [0, 0, 1, 1, 0]

ax.plot(court_x, court_y, color="white", linewidth=2)

# Middle line
ax.plot([0.5, 0.5], [0, 1], color="white", linestyle="--", linewidth=1)

# Short line approximation
ax.plot([0, 1], [0.45, 0.45], color="white", linestyle="--", linewidth=1)

# ------------------------------------------
# 6) Styling
# ------------------------------------------

fig.patch.set_facecolor("#111111")
ax.set_facecolor("#111111")

ax.set_title(
    "Professional Squash Shot Clusters",
    fontsize=18,
    color="white",
    pad=20
)

ax.text(
    0.5,
    1.03,
    "K-Means clustering using exact shot coordinates",
    fontsize=11,
    color="lightgray",
    ha="center",
    transform=ax.transAxes
)

ax.set_xlabel(
    "Court Width (Left → Right)",
    fontsize=12,
    color="white"
)

ax.set_ylabel(
    "Court Length (Front → Back)",
    fontsize=12,
    color="white"
)

ax.tick_params(colors="white")

# Remove top/right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Make remaining spines white
ax.spines["bottom"].set_color("white")
ax.spines["left"].set_color("white")

# Legend
legend = ax.legend(facecolor="#111111", edgecolor="white")
for text in legend.get_texts():
    text.set_color("white")

# ------------------------------------------
# 7) Tight layout + save
# ------------------------------------------

plt.tight_layout()

plt.savefig(
    "linkedin_cluster_visual.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)

plt.show()