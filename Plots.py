# ==========================================
# K-Means Clustering of Professional Squash Shots
# Creates:
# 1) elbow_method_clusters.png
# 2) linkedin_cluster_visual.png
# ==========================================

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

# Clean destination coordinates
df["finalPositionX"] = pd.to_numeric(
    df["finalPositionX"],
    errors="coerce"
)

df["finalPositionY"] = pd.to_numeric(
    df["finalPositionY"],
    errors="coerce"
)

# Keep only rows with valid shot destination coordinates
df = df.dropna(
    subset=["finalPositionX", "finalPositionY"]
)

print(f"Rows with valid final position coordinates: {len(df):,}")

# Coordinates used for clustering
coords = df[["finalPositionX", "finalPositionY"]]

# ==========================================
# 3) Elbow Method
# ==========================================
# Purpose:
# Test multiple values of k to understand how many clusters
# are reasonable for the shot landing locations.

inertia_values = []
k_values = range(1, 11)

for k in k_values:
    kmeans_test = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans_test.fit(coords)

    inertia_values.append(kmeans_test.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o",
    linewidth=2
)

plt.title(
    "Elbow Method for Shot Landing Clusters",
    fontsize=14
)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")

plt.xticks(list(k_values))
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "elbow_method_clusters.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved elbow method image as: elbow_method_clusters.png")

# ==========================================
# 4) Run final K-Means clustering
# ==========================================
# Based on the elbow method + squash court interpretability,
# we use 6 clusters:
# front-left, front-right, mid-left, mid-right,
# back-left, back-right.

kmeans = KMeans(
    n_clusters=6,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(coords)

# Get cluster centres
centers = kmeans.cluster_centers_

print("\nCluster centres:")
print(centers)

# ==========================================
# 5) Optional: Label clusters manually
# ==========================================
# These labels are based on your observed cluster centres.
# If the cluster numbers change, update these labels.

cluster_names = {
    0: "back_left_deep",
    1: "front_right",
    2: "mid_right",
    3: "front_left",
    4: "back_right_deep",
    5: "mid_left"
}

df["cluster_name"] = df["cluster"].map(cluster_names)

print("\nShot count by cluster:")
print(df["cluster_name"].value_counts())

# ==========================================
# 6) LinkedIn-Ready Cluster Visualization
# ==========================================

# Sample data for cleaner plotting
sample_size = min(len(df), 50000)

df_sample = df.sample(
    sample_size,
    random_state=42
)

# Create figure
fig, ax = plt.subplots(figsize=(7, 12))

# Plot clustered shot locations
scatter = ax.scatter(
    df_sample["finalPositionX"],
    df_sample["finalPositionY"],
    c=df_sample["cluster"],
    cmap="viridis",
    s=5,
    alpha=0.5
)

# Plot cluster centres
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

# ==========================================
# 7) Draw simple squash court outline
# ==========================================

# Outer court boundary
court_x = [0, 1, 1, 0, 0]
court_y = [0, 0, 1, 1, 0]

ax.plot(
    court_x,
    court_y,
    color="white",
    linewidth=2
)

# Centre line
ax.plot(
    [0.5, 0.5],
    [0, 1],
    color="white",
    linestyle="--",
    linewidth=1
)

# Short line approximation
ax.plot(
    [0, 1],
    [0.45, 0.45],
    color="white",
    linestyle="--",
    linewidth=1
)

# ==========================================
# 8) Styling
# ==========================================

fig.patch.set_facecolor("#111111")
ax.set_facecolor("#111111")

ax.set_title(
    "K-Means Clustering of Professional Squash Shots",
    fontsize=18,
    color="white",
    pad=20
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
legend = ax.legend(
    facecolor="#111111",
    edgecolor="white"
)

for text in legend.get_texts():
    text.set_color("white")

# ==========================================
# 9) Save cluster visual
# ==========================================

plt.tight_layout()

plt.savefig(
    "linkedin_cluster_visual.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)

plt.show()

print("Saved cluster visual as: linkedin_cluster_visual.png")