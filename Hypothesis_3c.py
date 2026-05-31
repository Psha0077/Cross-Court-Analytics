
# Do shots cluster into natural target zones on the court

# ==========================================
# 0) Load dataset
# ==========================================

import pandas as pd
from pathlib import Path

COMBINED_CSV_PATH = Path(
    "/Users/pranav/Desktop/Dataprojects/Squash/Cross Court Analytics/Raw_data/csv_files/all_matches_combined.csv"
)

shots = pd.read_csv(COMBINED_CSV_PATH)

print(f"Loaded {len(shots):,} rows")
df = shots.copy()

# Clean data first
df = shots.copy()

df["finalPositionX"] = pd.to_numeric(df["finalPositionX"], errors="coerce")
df["finalPositionY"] = pd.to_numeric(df["finalPositionY"], errors="coerce")

df = df.dropna(subset=["finalPositionX", "finalPositionY"])

coords = df[["finalPositionX", "finalPositionY"]]

# ==========================================
# 0 a) Use the elbow method to understand how many clusters are needed
# ==========================================
from sklearn.cluster import KMeans

inertia_values = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(coords)
    inertia_values.append(kmeans.inertia_)

import matplotlib.pyplot as plt

plt.plot(range(2, 11), inertia_values, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

# ==========================================
# 1) Run k-means clustering
# ==========================================

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=6, random_state=42)
df["cluster"] = kmeans.fit_predict(coords)

# ==========================================
# 2) Get Cluster Centres
# ==========================================

centers = kmeans.cluster_centers_
print(centers)

# ==========================================
# 3) Get Cluster Centres
# ==========================================

pd.crosstab(df["cluster"], df["shot_type"])

# ==========================================
# 4) which shot types belong to each cluster?
# ==========================================

cluster_shot_ct = pd.crosstab(df["cluster"], df["shot_type"])
print(cluster_shot_ct)

cluster_shot_pct = pd.crosstab(
    df["cluster"], 
    df["shot_type"], 
    normalize="index"
) * 100

print(cluster_shot_pct.round(1))

cluster_names = {
    0: "back_left_deep",
    1: "front_right",
    2: "mid_right",
    3: "front_left",
    4: "back_right_deep",
    5: "mid_left"
}

df["cluster_name"] = df["cluster"].map(cluster_names)

print(pd.crosstab(df["cluster_name"], df["shot_type"]))

# ==========================================
# 4) Plotting the clusters
# ==========================================

import matplotlib.pyplot as plt

# Optional: sample to speed things up (recommended)
df_sample = df.sample(50000, random_state=42)

plt.figure(figsize=(6, 10))

plt.scatter(
    df_sample["finalPositionX"],
    df_sample["finalPositionY"],
    c=df_sample["cluster"],
    cmap="viridis",
    s=5,
    alpha=0.6
)

# Plot cluster centres
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c="red",
    s=200,
    marker="X",
    label="Cluster Centres"
)

plt.title("Shot Landing Clusters (All Shots)")
plt.xlabel("Court Width (Left → Right)")
plt.ylabel("Court Length (Back → Front)")
plt.legend()
plt.show()

# ==========================================
# 4) Plotting the clusters by shot type
# ==========================================

import matplotlib.pyplot as plt

shot_types = df["shot_type"].dropna().unique()

n = len(shot_types)
cols = 3
rows = (n // cols) + (n % cols > 0)

fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
axes = axes.flatten()

for i, shot in enumerate(shot_types):
    ax = axes[i]

    subset = df[df["shot_type"] == shot]

    # Safe sampling
    sample_size = min(len(subset), 20000)
    subset = subset.sample(sample_size, random_state=42)

    ax.scatter(
        subset["finalPositionX"],
        subset["finalPositionY"],
        c=subset["cluster"],
        cmap="viridis",
        s=5,
        alpha=0.6
    )

    # Cluster centres
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c="red",
        s=100,
        marker="X"
    )

    ax.set_title(shot.capitalize())
    ax.set_xlabel("Left → Right")
    ax.set_ylabel("Back → Front")

# Remove empty plots (if any)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()