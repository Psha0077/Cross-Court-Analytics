import json
from pathlib import Path
import pandas as pd
file_path = Path("/Users/pranav/Desktop/Dataprojects/Squash/Cross Court Analytics/Raw_data/farag_jsons_251023/Ali Farag vs Fares Dessouky_match.json")

with open(file_path, "r") as m1:
    data = json.load(m1)
print(data.keys())

#print("Type of 'matches':", type(data["matches"]))
#print("Contents of 'matches':", data["matches"])

matches_node = data.get("matches")                 # The top-level key that holds match objects
if matches_node is None:                           # If the file has no 'matches' at all, fail fast
    raise ValueError("No 'matches' key found in JSON.")

if isinstance(matches_node, list):                 # Normal case: already a list
    matches = matches_node
elif isinstance(matches_node, dict):               # Sometimes APIs wrap list under another key
    matches = matches_node.get("data") or matches_node.get("items") or [matches_node]
else:                                              # Any other shape is unexpected for this dataset
    raise TypeError(f"Unsupported 'matches' type: {type(matches_node).__name__}")

if not matches:                                    # If we still have zero matches, stop early
    raise ValueError("'matches' is empty.")

# ======= 4) Work with the FIRST match in the file (common case) =======
m = matches[0]                                     # Take first match object
meta = m.get("metadata", {})                       # Optional metadata dict (players, event, date)
playerA = meta.get("playerA") or meta.get("teamA") # Try multiple keys for A’s name
playerB = meta.get("playerB") or meta.get("teamB") # Try multiple keys for B’s name
event   = meta.get("event")                        # Event or tournament name (may be None)
date    = meta.get("date")                         # Match date (may be None)
match_id = meta.get("matchId") or file_path.stem   # Prefer explicit matchId, else use filename stem

# ======= 5) Collect rows here (one row per shot) =======
rows = []                                          # We’ll fill this list with dicts then DataFrame
def as_float(x):                                   # Convert to float or return None (cleaner tables)
    try:
        return float(x)
    except Exception:
        return None
# ======= 7) Helper to derive a single shot_type from flags =======
def derive_shot_type(shot_dict):                   # Decide shot_type based on provided booleans
    sd = shot_dict.get("shotData", {})             # Nested booleans live under 'shotData' in your sample
    # Priority: boast > volley > lift > rebound > drive (fallback)
    if sd.get("Boast"):   return "boast"
    if sd.get("Volley"):  return "volley"
    if sd.get("Lift"):    return "lift"
    if sd.get("Back"):    return "rebound"         # “Back” in your note = hit on rebound/back wall
    # If the file provides an explicit type, keep it; else default to drive
    explicit = shot_dict.get("shotType")
    return explicit if explicit else "drive"
# ======= 8) Iterate games → rallies → shots (or rallies directly) =======
def extract_metadata_from_matches_block(root: Dict[str, Any]) -> Dict[str, Any]:
    matches = root.get("matches", [])
    if not matches:
        return {}
    m = matches[0]
    event = m.get("tournament")

    date_raw = m.get("date")
    date_iso = None
    if date_raw:
        try:
            date_iso = datetime.strptime(date_raw, "%d/%m/%Y").date().isoformat()
        except Exception:
            date_iso = date_raw

    teams = m.get("teams", [])
    def player_name(team_idx: int):
        try:
            p = teams[team_idx][0]
            first = (p.get("firstName") or "").strip()
            last  = (p.get("lastName")  or "").strip()
            return f"{first} {last}".strip() or None
        except Exception:
            return None

    return {
        "playerA": player_name(0),
        "playerB": player_name(1),
        "event": event,
        "date": date_iso,
        "round": m.get("round"),
        "country": m.get("country"),
    }

def as_float(x):
    try:
        return float(x)
    except Exception:
        return None

def derive_shot_type(shot: Dict[str, Any]) -> str:
    sd = shot.get("shotData", {}) or {}
    if sd.get("Boast"):  return "boast"
    if sd.get("Volley"): return "volley"
    if sd.get("Lift"):   return "lift"
    if sd.get("Back"):   return "rebound"
    return shot.get("shotType") or "drive"

def is_rally(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("shots"), list)

def collect_rallies_with_game(node: Any, current_game_no: int | None = None,
                              out: List[Tuple[Dict[str, Any], int | None]] | None = None
                             ) -> List[Tuple[Dict[str, Any], int | None]]:
    if out is None:
        out = []
    if isinstance(node, dict):
        if "gameNumber" in node:
            try:
                current_game_no = int(node["gameNumber"])
            except Exception:
                current_game_no = node.get("gameNumber")
        if is_rally(node):
            out.append((node, current_game_no))
        for v in node.values():
            collect_rallies_with_game(v, current_game_no, out)
    elif isinstance(node, list):
        for item in node:
            collect_rallies_with_game(item, current_game_no, out)
    return out

# ---- get metadata ----
_md = extract_metadata_from_matches_block(data)
playerA  = _md.get("playerA")
playerB  = _md.get("playerB")
event    = _md.get("event")
date     = _md.get("date")
match_id = Path(file_path).stem

# ---- find rallies and build rows ----
rallies_with_game = collect_rallies_with_game(data)
print("Found rallies:", len(rallies_with_game))   # diagnostic so you see a non-zero count

rows: List[Dict[str, Any]] = []
for r_idx, (rally, game_no) in enumerate(rallies_with_game, start=1):
    teamAScore     = rally.get("teamAScore") or rally.get("teamAscore")
    teamBScore     = rally.get("teamBScore") or rally.get("teamBscore")
    winner         = rally.get("winner")
    winMethod      = rally.get("winMethod")
    finalPositionX = as_float(rally.get("finalPositionX"))
    finalPositionY = as_float(rally.get("finalPositionY"))

    for s_idx, shot in enumerate(rally.get("shots", [])):
        x = as_float(shot.get("x")); y = as_float(shot.get("y"))
        if x is None or y is None:
            sd = shot.get("shotData", {}) or {}
            x = as_float(sd.get("x")) if x is None else x
            y = as_float(sd.get("y")) if y is None else y

        rows.append({
            "match_id": match_id,
            "playerA": playerA,
            "playerB": playerB,
            "event": event,
            "date": date,
            "game_no": game_no,
            "rally_no": r_idx,
            "teamAScore": teamAScore,
            "teamBScore": teamBScore,
            "winner": winner,
            "winMethod": winMethod,
            "finalPositionX": finalPositionX,
            "finalPositionY": finalPositionY,
            "shot_idx": s_idx,
            "team": shot.get("team"),
            "timestamp": shot.get("timestamp"),
            "x": x,
            "y": y,
            "shot_type": derive_shot_type(shot),
        })

if not rows:
    raise RuntimeError("No rows were built. Check that rallies were detected (see 'Found rallies' print).")

# ======= STEP 9: build DataFrame + save =======
cols = [
    "match_id","playerA","playerB","event","date",
    "game_no","rally_no","teamAScore","teamBScore","winner","winMethod","finalPositionX","finalPositionY",
    "shot_idx","team","timestamp","x","y","shot_type"
]
shots_df = pd.DataFrame(rows, columns=cols)
print("Rows in shots_df:", len(shots_df))

# ======= 10) Quick sanity check and optional save =======
in_path = Path(file_path)                               # ensure we have a Path object even if file_path was a string
out_dir = Path("/Users/pranav/Desktop/Dataprojects/Squash/Cross Court Analytics/Raw_data/csv_files")
out_dir.mkdir(parents=True, exist_ok=True)              # create the output folder if it doesn't exist (safe to call repeatedly)

base_name = in_path.stem                                # take the input JSON's base name without extension
# Optional: uncomment the next line to remove spaces in filenames
# base_name = base_name.replace(" ", "_")

out_csv = out_dir / f"{base_name}.shots.csv"            # build the final CSV path: <out_dir>/<json_stem>.shots.csv

shots_df.to_csv(out_csv, index=False)                   # write the table with no row index
print(f"Rows: {len(shots_df)}")                         # quick confirmation of row count
print(f"Saved to: {out_csv}")                           # show exactly where it was saved

# ======= STEP 11: recursively process ALL JSONs in ALL subfolders =======
root_dir = Path("/Users/pranav/Desktop/Dataprojects/Squash/Cross Court Analytics/Raw_data")
output_dir = root_dir / "csv_files"
output_dir.mkdir(parents=True, exist_ok=True)

processed_keys = set()         # track unique match identifiers
processed, skipped, total_rows = 0, 0, 0

# Sort paths so results are deterministic (same kept copy each run)
all_jsons = sorted(root_dir.rglob("*.json"), key=lambda p: str(p).lower())

for json_path in all_jsons:
    if "csv_files" in json_path.parts:
        continue

    print(f"Processing: {json_path.relative_to(root_dir)}")

    try:
        # --- Load JSON ---
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # --- Extract metadata ---
        md = extract_metadata_from_matches_block(data)
        playerA  = md.get("playerA")
        playerB  = md.get("playerB")
        event    = md.get("event")
        date     = md.get("date")

        # --- Build unique match signature ---
        match_key = f"{event}_{date}_{playerA}_vs_{playerB}"
        match_key = match_key.replace(" ", "_").replace("/", "-")

        # --- Skip duplicates by logical match key ---
        if match_key in processed_keys:
            print(f"  ⚠️  Skipping duplicate match: {match_key}")
            skipped += 1
            continue
        processed_keys.add(match_key)

        # --- Extract rallies ---
        rallies_with_game = collect_rallies_with_game(data)
        rows = []
        for r_idx, (rally, game_no) in enumerate(rallies_with_game, start=1):
            teamAScore     = rally.get("teamAScore") or rally.get("teamAscore")
            teamBScore     = rally.get("teamBScore") or rally.get("teamBscore")
            winner         = rally.get("winner")
            winMethod      = rally.get("winMethod")
            finalPositionX = as_float(rally.get("finalPositionX"))
            finalPositionY = as_float(rally.get("finalPositionY"))

            for s_idx, shot in enumerate(rally.get("shots", [])):
                x = as_float(shot.get("x"))
                y = as_float(shot.get("y"))
                if x is None or y is None:
                    sd = shot.get("shotData", {}) or {}
                    x = as_float(sd.get("x")) if x is None else x
                    y = as_float(sd.get("y")) if y is None else y

                rows.append({
                    "match_id": match_key,
                    "playerA": playerA,
                    "playerB": playerB,
                    "event": event,
                    "date": date,
                    "game_no": game_no,
                    "rally_no": r_idx,
                    "teamAScore": teamAScore,
                    "teamBScore": teamBScore,
                    "winner": winner,
                    "winMethod": winMethod,
                    "finalPositionX": finalPositionX,
                    "finalPositionY": finalPositionY,
                    "shot_idx": s_idx,
                    "team": shot.get("team"),
                    "timestamp": shot.get("timestamp"),
                    "x": x,
                    "y": y,
                    "shot_type": derive_shot_type(shot),
                })

        if not rows:
            print(f"  ⚠️  No shots found — skipping {json_path.name}")
            continue

        # --- Save CSV using logical match_id (not folder name) ---
        cols = [
            "match_id","playerA","playerB","event","date",
            "game_no","rally_no","teamAScore","teamBScore","winner","winMethod",
            "finalPositionX","finalPositionY",
            "shot_idx","team","timestamp","x","y","shot_type"
        ]
        shots_df = pd.DataFrame(rows, columns=cols)
        out_csv = output_dir / f"{match_key}.csv"
        shots_df.to_csv(out_csv, index=False)

        print(f"  ✅ Saved {out_csv.name} ({len(shots_df)} rows)")
        processed += 1
        total_rows += len(shots_df)

    except Exception as e:
        print(f"  ❌ Error in {json_path.relative_to(root_dir)}: {e}")

print(f"\n🎯 Done. Kept {processed} unique match file(s), skipped {skipped} duplicate(s).")
print(f"   Wrote {total_rows:,} total row(s) to: {output_dir}")




