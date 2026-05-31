# inspect_xy_plane.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- EDIT PATH IF NEEDED ---
CSV_PATH = Path("/Users/pranav/Desktop/Dataprojects/Squash/Cross Court Analytics/Raw_data/csv_files/Ali Farag vs Fares Dessouky_match.shots.csv")

# ---------- helpers ----------
def find_col(cols, candidates):
    cols_low = {c.lower(): c for c in cols}
    # exact first
    for cand in candidates:
        for c in cols:
            if c.lower() == cand.lower():
                return c
    # contains
    for cand in candidates:
        for lc, orig in cols_low.items():
            if cand.lower() in lc:
                return orig
    return None

def describe01(s, name):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        print(f"{name}: no numeric data")
        return np.nan, np.nan, np.nan, np.nan, np.nan
    q = (s.min(), s.quantile(.25), s.quantile(.5), s.quantile(.75), s.max())
    print(f"{name}: min={q[0]:.3f}, q25={q[1]:.3f}, q50={q[2]:.3f}, q75={q[3]:.3f}, max={q[4]:.3f}")
    return q

# ---------- load ----------
df = pd.read_csv(CSV_PATH, engine="python")
print(f"\nLoaded {len(df):,} rows from:\n{CSV_PATH}\n")
print("Columns:", list(df.columns), "\n")

# detect columns
x_col       = find_col(df.columns, ["x","impactx","shot_x"])
y_col       = find_col(df.columns, ["y","impacty","shot_y"])
stype_col   = find_col(df.columns, ["shot_type","shottype","type"])
team_col    = find_col(df.columns, ["team","hitter","player_team"])
winner_col  = find_col(df.columns, ["winner","winner_team"])
shotidx_col = find_col(df.columns, ["shot_idx","shotindex","shotno","shot_number"])

if not x_col or not y_col:
    raise ValueError(f"Could not find x/y columns. Found x={x_col}, y={y_col}.")

print(f"Detected:\n  x={x_col}, y={y_col}\n  shot_type={stype_col}, team={team_col}, winner={winner_col}, shot_idx={shotidx_col}\n")

# numeric & clean
df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
df = df.dropna(subset=[x_col, y_col])

# ranges
describe01(df[x_col], "x")
describe01(df[y_col], "y")
print()

# infer Y convention via medians of typical shot types
stype = df[stype_col].astype(str).str.lower() if stype_col else pd.Series([""]*len(df))
def med(mask):
    arr = df.loc[mask, y_col]
    return float(arr.median()) if len(arr) else np.nan

med_rebound = med(stype.str.contains("rebound"))
med_volley  = med(stype.str.contains("volley"))
med_drive   = med(stype.str.contains("drive"))
med_lift    = med(stype.str.contains("lift"))
med_boast   = med(stype.str.contains("boast"))

print("Median y by shot type (NaN means not present):")
print(f"  rebound={med_rebound:.3f}  volley={med_volley:.3f}  drive={med_drive:.3f}  lift={med_lift:.3f}  boast={med_boast:.3f}")

votes = []
def vote(a,b):
    if not np.isnan(a) and not np.isnan(b):
        votes.append("y0=back" if a < b else "y0=front")
vote(med_rebound, med_volley)
vote(med_drive, med_volley)
vote(med_lift, med_volley)

conclusion_y = "y0=back" if votes.count("y0=back") >= votes.count("y0=front") else "y0=front"
print("Votes:", votes or "(no votes possible)")
print(">>> Inferred Y convention:", "0=Back, 1=Front" if conclusion_y=="y0=back" else "0=Front, 1=Back", "\n")

# choose a sample for plotting
sample = df.copy()
if len(sample) > 4000:
    sample = sample.sample(4000, random_state=42)

# color map for a few shot types (optional)
def color_for(st):
    st = (st or "").lower()
    if "volley" in st:   return "#1f77b4"
    if "rebound" in st:  return "#ff7f0e"
    if "boast" in st:    return "#2ca02c"
    if "lift" in st:     return "#d62728"
    if "drive" in st:    return "#9467bd"
    return "#7f7f7f"

colors = sample[stype_col].map(color_for) if stype_col else "#7f7f7f"

# ----------- PLOTS: unit-square only -----------
fig, axes = plt.subplots(1, 2, figsize=(10,5), squeeze=False)
ax1, ax2 = axes[0]

# left: raw orientation (0..1 y upward)
ax1.scatter(sample[x_col], sample[y_col], s=10, alpha=0.5, c=colors)
ax1.set_title("Unit square: raw (y up)")
ax1.set_xlim(0,1); ax1.set_ylim(0,1)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlabel("x"); ax1.set_ylabel("y")

# right: flipped y (back wall at top visual)
ax2.scatter(sample[x_col], 1.0 - sample[y_col], s=10, alpha=0.5, c=colors)
ax2.set_title("Unit square: flipped y (visual check)")
ax2.set_xlim(0,1); ax2.set_ylim(0,1)
ax2.set_aspect('equal', adjustable='box')
ax2.set_xlabel("x"); ax2.set_ylabel("1 - y")

plt.tight_layout()
plt.show()



def find_col(cols, candidates):
    # exact match first, then substring
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        for c in cols:
            if c.lower() == cand.lower(): return c
    for cand in candidates:
        for lc, orig in low.items():
            if cand.lower() in lc: return orig
    return None

df = pd.read_csv(CSV_PATH, engine="python")
x_col     = find_col(df.columns, ["x","impactx","shot_x"])
y_col     = find_col(df.columns, ["y","impacty","shot_y"])
stype_col = find_col(df.columns, ["shot_type","shottype","type"])

if not x_col or not y_col or not stype_col:
    raise ValueError(f"Missing key columns. Found x={x_col}, y={y_col}, shot_type={stype_col}")

# clean types
df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
df = df.dropna(subset=[x_col, y_col])

# quick peek at available shot types to confirm naming
print("Unique shot_type examples:", df[stype_col].dropna().astype(str).str.lower().value_counts().head(15), "\n")

# filter drops (adjust the pattern if your data uses a different label, e.g., 'drop shot')
drops = df[df[stype_col].astype(str).str.lower().str.contains(r"\bdrop\b", na=False)].copy()
n = len(drops)
print(f"Found {n} drop shots.")

if n == 0:
    print("No rows matched 'drop'. Try adjusting the pattern (e.g., 'drop shot' or check the list above).")
else:
    y = drops[y_col]
    print(f"Drop-shot y stats: min={y.min():.3f}, q25={y.quantile(.25):.3f}, median={y.median():.3f}, q75={y.quantile(.75):.3f}, max={y.max():.3f}")
    frac_front = (y >= 0.5).mean()
    print(f"Share of drop shots with y ≥ 0.5: {frac_front:.3%}")

    # Simple verdict
    if y.median() >= 0.5:
        print(">>> Verdict: Drops are closer to y=1 (front). This supports convention: y=0 back, y=1 front.")
    else:
        print(">>> Verdict: Drops are closer to y=0 (back). This suggests the opposite convention in this file.")

    # Optional: quick unit-square scatter for drops
    plt.figure(figsize=(5,5))
    plt.scatter(drops[x_col], drops[y_col], s=12, alpha=0.6)
    plt.title("Drop shots on unit square (x: left→right, y: back→front?)")
    plt.xlim(0,1); plt.ylim(0,1); plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("x"); plt.ylabel("y")
    plt.show()