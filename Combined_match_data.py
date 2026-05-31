import pandas as pd
from pathlib import Path

# === 1. Define the folder path ===
folder_path = Path("/Users/pranav/Desktop/Dataprojects/Squash/Cross Court Analytics/Raw_data/csv_files")

# === 2. Get all CSV files in that folder ===
csv_files = sorted(folder_path.glob("*.csv"))
print(f"Found {len(csv_files)} CSV files in: {folder_path}")

# === 3. Read and combine all CSVs ===
dfs = []
for f in csv_files:
    try:
        df = pd.read_csv(f)
        df["source_file"] = f.name        # optional: helps trace origin match later
        dfs.append(df)
    except Exception as e:
        print(f"⚠️ Skipping {f.name} due to error: {e}")

if not dfs:
    raise RuntimeError("No valid CSV files found — check the folder path.")

shots_all = pd.concat(dfs, ignore_index=True)
print(f"✅ Combined DataFrame shape: {shots_all.shape}")
print(f"✅ Total rows combined: {len(shots_all):,}")

# === 4. Quick sanity check ===
print("\nColumns:", list(shots_all.columns))
print(shots_all.head())

# === 5. Optional: Save combined dataset ===
out_path = folder_path / "all_matches_combined.csv"
shots_all.to_csv(out_path, index=False)
print(f"\n💾 Saved combined dataset to: {out_path}")
