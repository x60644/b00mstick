"""
b00mstick — Phase 0 data pull
=============================
Pulls 2022-2025 NFL play-by-play via nfl_data_py (nflfastR data) and
extracts the three market datasets:

  data/first_drives.csv   one row per team-game: first offensive drive result
  data/punts.csv          one row per team-game: punt count (+ game totals)
  data/kicker_points.csv  one row per kicker-game: FG/XP makes and points
  data/games.csv          one row per game: coaches, roof, surface, temp, wind
  data/oc_template.csv    blank coordinator mapping for manual fill-in

Run:  pip install pandas pyarrow
      python phase0_pull.py

Note: reads nflverse play-by-play parquet releases directly (same data
nfl_data_py wraps) -- avoids the numpy<2.0 pin that breaks on Python 3.14.
"""

import os
import pandas as pd

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{yr}.parquet"

SEASONS = [2022, 2023, 2024, 2025, 2026]  # 2026 skipped gracefully until posted
OUT = "data"
os.makedirs(OUT, exist_ok=True)

COLS = [
    "game_id", "season", "week", "season_type", "game_date",
    "home_team", "away_team", "posteam", "defteam",
    "fixed_drive", "fixed_drive_result",
    "play_type", "punt_attempt", "punter_player_name",
    "field_goal_result", "kick_distance",
    "extra_point_result", "kicker_player_name", "kicker_player_id",
    "home_coach", "away_coach",
    "roof", "surface", "temp", "wind", "qtr",
    "spread_line", "total_line", "yardline_100",
]

# ---------------------------------------------------------------- pull
frames = []
for yr in SEASONS:
    print(f"Pulling {yr} play-by-play...")
    url = PBP_URL.format(yr=yr)
    try:
        df = pd.read_parquet(url, columns=COLS)
    except Exception:
        try:
            df = pd.read_parquet(url)
            df = df[[c for c in COLS if c in df.columns]]
        except Exception as e:
            print(f"  {yr} unavailable ({type(e).__name__}) -- skipping")
            continue
    frames.append(df)

pbp = pd.concat(frames, ignore_index=True)
reg = pbp[pbp["season_type"] == "REG"].copy()
print(f"\nTotal regular-season plays: {len(reg):,}")

# ---------------------------------------------------- first drive result
d = reg[reg["fixed_drive"].notna() & reg["posteam"].notna()]
drive_results = (
    d.groupby(["game_id", "season", "week", "posteam", "defteam", "fixed_drive"])
    ["fixed_drive_result"].first().reset_index()
)
idx = drive_results.groupby(["game_id", "posteam"])["fixed_drive"].idxmin()
first_drives = drive_results.loc[idx].copy()

RESULT_MAP = {"Touchdown": "TD", "Field goal": "FG", "Punt": "PUNT"}
first_drives["outcome"] = (
    first_drives["fixed_drive_result"].map(RESULT_MAP).fillna("OTHER")
)
first_drives["received_opening_kick"] = (first_drives["fixed_drive"] == 1).astype(int)
first_drives = first_drives.rename(columns={"fixed_drive_result": "raw_result"})
first_drives.to_csv(f"{OUT}/first_drives.csv", index=False)

# ---------------------------------------------------------------- punts
punts = reg[reg["punt_attempt"] == 1]
team_punts = (
    punts.groupby(["game_id", "season", "week", "posteam", "defteam"])
    .size().reset_index(name="punts")
)
# teams with zero punts in a game are missing above -- reindex from first_drives
all_team_games = first_drives[["game_id", "season", "week", "posteam", "defteam"]]
team_punts = all_team_games.merge(
    team_punts, on=["game_id", "season", "week", "posteam", "defteam"], how="left"
)
team_punts["punts"] = team_punts["punts"].fillna(0).astype(int)
team_punts["game_total_punts"] = team_punts.groupby("game_id")["punts"].transform("sum")
team_punts.to_csv(f"{OUT}/punts.csv", index=False)

# ---------------------------------------------------------- drive stats
dd = reg[reg["fixed_drive"].notna() & reg["posteam"].notna()]
drv = (
    dd.groupby(["game_id", "season", "week", "posteam", "defteam", "fixed_drive"])
    .agg(min_yl=("yardline_100", "min"), result=("fixed_drive_result", "first"))
    .reset_index()
)
drv["fg_range"] = (drv["min_yl"] <= 35).astype(int)
drv["td"] = (drv["result"] == "Touchdown").astype(int)
drv["stall"] = ((drv["fg_range"] == 1) & (drv["td"] == 0)).astype(int)
drive_stats = (
    drv.groupby(["game_id", "season", "week", "posteam", "defteam"])
    .agg(drives=("fixed_drive", "count"), fg_range_drives=("fg_range", "sum"),
         stalls=("stall", "sum"), off_tds=("td", "sum"))
    .reset_index()
)
drive_stats.to_csv(f"{OUT}/drive_stats.csv", index=False)

# -------------------------------------------------------- kicker points
# build from ALL attempts (made or missed) so 0-point kicker games exist
att = reg[
    reg["field_goal_result"].notna() | reg["extra_point_result"].notna()
].copy()
att["fg_made"] = (att["field_goal_result"] == "made").astype(int)
att["fg_att"] = att["field_goal_result"].notna().astype(int)
att["xp_made"] = (att["extra_point_result"] == "good").astype(int)
att["xp_att"] = att["extra_point_result"].notna().astype(int)
att["pts"] = att["fg_made"] * 3 + att["xp_made"]
att["made_dist"] = att["kick_distance"].where(att["fg_made"] == 1)
kicker_pts = (
    att.groupby(["game_id", "season", "week", "posteam", "kicker_player_name"])
    .agg(
        fg_made=("fg_made", "sum"), fg_att=("fg_att", "sum"),
        xp_made=("xp_made", "sum"), xp_att=("xp_att", "sum"),
        points=("pts", "sum"), long_fg=("made_dist", "max"),
    )
    .reset_index()
)
kicker_pts.to_csv(f"{OUT}/kicker_points.csv", index=False)

# ---------------------------------------------------------------- games
games = (
    reg.groupby("game_id")
    .agg(
        season=("season", "first"), week=("week", "first"),
        game_date=("game_date", "first"),
        home_team=("home_team", "first"), away_team=("away_team", "first"),
        home_coach=("home_coach", "first"), away_coach=("away_coach", "first"),
        roof=("roof", "first"), surface=("surface", "first"),
        temp=("temp", "first"), wind=("wind", "first"),
        spread_line=("spread_line", "first"), total_line=("total_line", "first"),
    )
    .reset_index()
)
games.to_csv(f"{OUT}/games.csv", index=False)

# ----------------------------------------------- coordinator template
teams = sorted(reg["posteam"].dropna().unique())
oc_rows = [
    {"season": yr, "team": t, "oc_name": "", "playcaller": "", "prior_team": "", "notes": ""}
    for yr in SEASONS + [2026] for t in teams
]
pd.DataFrame(oc_rows).to_csv(f"{OUT}/oc_template.csv", index=False)

# -------------------------------------------------------------- summary
print("\n===== SANITY CHECKS =====")
print("\nFirst-drive outcome distribution by season (%):")
print(
    first_drives.groupby("season")["outcome"]
    .value_counts(normalize=True).mul(100).round(1).unstack().fillna(0)
)
print("\nAvg punts per team-game by season:")
print(team_punts.groupby("season")["punts"].mean().round(2))
print("\nAvg game total punts by season:")
print(team_punts.drop_duplicates("game_id").groupby("season")["game_total_punts"].mean().round(2))
print("\nAvg kicker points per kicker-game by season:")
print(kicker_pts.groupby("season")["points"].mean().round(2))
print("\nRow counts:")
print(f"  first_drives:  {len(first_drives):,}")
print(f"  punts:         {len(team_punts):,}")
print(f"  kicker_points: {len(kicker_pts):,}")
print(f"  games:         {len(games):,}")
print("\nDone. CSVs written to ./data/")
