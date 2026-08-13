"""
b00mstick — score_week.py (production weekly scorer)
====================================================
The script the GitHub Actions crons run (and you can run locally today):

  1. refreshes data/ by running phase0_pull.py (2026 included once posted)
  2. loads the 2026 schedule (nflverse) and picks the upcoming Sunday slate
     (or a specific week:  python score_week.py 3)
  3. loads 2026 rosters for each team's kicker + punter (names + headshots)
  4. fetches wind forecasts (Open-Meteo, free, no key) for outdoor stadiums
  5. builds model state as of the current week via b00mstick_model
  6. writes output/week{W}_slate.json for the app

Manual DK lines are entered in the app; the JSON carries schedule lines as
pre-fill defaults plus linear sensitivities (tt_slope) so the frontend can
adjust means when you override lines, and pmf/dist arrays so it can price
any over/under client-side.
"""

import json
import os
import subprocess
import sys
import urllib.request
from datetime import date

import numpy as np
import pandas as pd

import b00mstick_model as bm

SCHED_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_2026.parquet"
SEASON = 2026

# approximate stadium coordinates for weather (outdoor accuracy needs only ~city level)
STADIUMS = {
    "ARI": (33.53, -112.26), "ATL": (33.76, -84.40), "BAL": (39.28, -76.62),
    "BUF": (42.77, -78.79), "CAR": (35.23, -80.85), "CHI": (41.86, -87.62),
    "CIN": (39.10, -84.52), "CLE": (41.51, -81.70), "DAL": (32.75, -97.09),
    "DEN": (39.74, -105.02), "DET": (42.34, -83.05), "GB": (44.50, -88.06),
    "HOU": (29.68, -95.41), "IND": (39.76, -86.16), "JAX": (30.32, -81.64),
    "KC": (39.05, -94.48), "LA": (33.95, -118.34), "LAC": (33.95, -118.34),
    "LV": (36.09, -115.18), "MIA": (25.96, -80.24), "MIN": (44.97, -93.26),
    "NE": (42.09, -71.26), "NO": (29.95, -90.08), "NYG": (40.81, -74.07),
    "NYJ": (40.81, -74.07), "PHI": (39.90, -75.17), "PIT": (40.45, -80.02),
    "SEA": (47.60, -122.33), "SF": (37.40, -121.97), "TB": (27.98, -82.50),
    "TEN": (36.17, -86.77), "WAS": (38.91, -76.86),
}

def refresh_data():
    print("Refreshing data via phase0_pull.py ...")
    subprocess.run([sys.executable, "phase0_pull.py"], check=True)

def load_schedule():
    sched = pd.read_csv(SCHED_URL)
    return sched[sched["season"] == SEASON].copy()

def pick_week(sched):
    if len(sys.argv) > 1:
        return int(sys.argv[1])
    upcoming = sched[pd.to_datetime(sched["gameday"]).dt.date >= date.today()]
    return int(upcoming["week"].min()) if len(upcoming) else int(sched["week"].max())

def load_specialists():
    """team -> {kicker: {...}, punter: {...}} from 2026 rosters."""
    try:
        ros = pd.read_parquet(ROSTER_URL)
    except Exception as e:
        print(f"  roster load failed ({type(e).__name__}); specialists blank")
        return {}
    hist = pd.read_csv("data/kicker_points.csv")
    career_att = hist.groupby("kicker_player_name")["fg_att"].sum().to_dict()
    def short(full):   # nflverse pbp style: 'J.Tucker'
        parts = str(full).split()
        return f"{parts[0][0]}.{parts[-1]}" if len(parts) >= 2 else str(full)
    out = {}
    for team, g in ros.groupby("team"):
        entry = {}
        for pos, key in (("K", "kicker"), ("P", "punter")):
            cands = g[g["position"] == pos]
            active = cands[cands["status"].astype(str).str.upper().str.startswith("ACT")]
            pool = active if len(active) else cands
            if not len(pool):
                continue
            pool = pool.assign(att=pool["full_name"].map(lambda n: career_att.get(short(n), 0)))
            row = pool.sort_values("att", ascending=False).iloc[0]
            entry[key] = {"name": short(row["full_name"]), "full_name": row["full_name"],
                          "headshot": row.get("headshot_url", "") or ""}
        out[team] = entry
    return out

def wind_forecast(team, gameday):
    """Max afternoon wind (mph) at the stadium; 0 on any failure or out of range."""
    lat, lon = STADIUMS.get(team, (None, None))
    if lat is None:
        return 0.0
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&hourly=wind_speed_10m&wind_speed_unit=mph&timezone=America%2FNew_York"
           f"&start_date={gameday}&end_date={gameday}")
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        hours = data["hourly"]["time"]
        speeds = data["hourly"]["wind_speed_10m"]
        aft = [s for t, s in zip(hours, speeds)
               if 13 <= int(t[11:13]) <= 19 and s is not None]
        return float(max(aft)) if aft else 0.0
    except Exception:
        return 0.0

def kicker_dist(state, mean):
    res = state["kk"]["residuals"]
    pts = np.clip(np.round(mean + res).astype(int), 0, None)
    vals, counts = np.unique(pts, return_counts=True)
    return {int(v): round(float(c) / len(pts), 4) for v, c in zip(vals, counts)}

def main():
    refresh_data()
    frames = bm.load_data("data")
    played = frames["fd"][frames["fd"]["season"] == SEASON]
    asof_week = int(played["week"].max()) + 1 if len(played) else 1
    state = bm.build_state(frames, SEASON, asof_week)
    sched = load_schedule()
    week = pick_week(sched)
    slate = sched[(sched["week"] == week) & (sched["weekday"] == "Sunday")].copy()
    print(f"Scoring {SEASON} week {week}: {len(slate)} Sunday games "
          f"(state as of week {asof_week})")
    specialists = load_specialists()

    oc = pd.read_csv("data/oc_map.csv")
    pc = {(r["season"], r["team"]): r["playcaller"] for _, r in oc.iterrows()}
    hc26 = {r["team"]: r["hc_2026"] for _, r in oc[oc["season"] == SEASON].iterrows()}

    s_pt = state["pt"]
    games_out = []
    for _, g in slate.iterrows():
        home, away = g["home_team"], g["away_team"]
        total = float(g["total_line"]) if pd.notna(g.get("total_line")) else s_pt["mean_tot"]
        spread = float(g["spread_line"]) if pd.notna(g.get("spread_line")) else 0.0
        roof = str(g.get("roof", "outdoors"))
        wind = wind_forecast(home, g["gameday"]) if roof == "outdoors" else 0.0
        wflag = 1 if wind >= 15 else 0
        tts = {home: (total + spread) / 2, away: (total - spread) / 2}

        drive, punts, kickers, punters = {}, {}, {}, {}
        for side, team, opp in (("home", home, away), ("away", away, home)):
            drive[side] = {k: round(v, 4) for k, v in
                           bm.predict_first_drive(state, team, opp, side == "home").items()}
            mu = bm.predict_punts(state, team, opp, total, wflag)
            punts[side] = {"mean": round(mu, 3), "tt_slope": round(float(s_pt["coef"][2]), 4)}
            spec = specialists.get(team, {})
            if "kicker" in spec:
                km = bm.predict_kicker(state, team, spec["kicker"]["name"], tts[team], wflag)
                kk = state["kk"]
                slope = 3 * kk["k_fgpct"].get(spec["kicker"]["name"], kk["lg_fgpct"]) \
                    * (0.95 if wflag else 1.0) * float(kk["c1"][3]) + \
                    kk["k_xppct"].get(spec["kicker"]["name"], kk["lg_xppct"]) * float(kk["c2"][2])
                kickers[side] = {**spec["kicker"], "mean": round(km, 3),
                                 "tt_base": round(tts[team], 2), "tt_slope": round(slope, 4),
                                 "dist": kicker_dist(state, km)}
            if "punter" in spec:
                punters[side] = spec["punter"]

        games_out.append({
            "game_id": g["game_id"], "gameday": g["gameday"],
            "gametime": str(g.get("gametime", "")), "home": home, "away": away,
            "roof": roof, "wind_mph": round(wind, 1), "wind_flag": wflag,
            "lines": {"spread_line": spread, "total_line": total,
                      "home_tt": round(tts[home], 2), "away_tt": round(tts[away], 2)},
            "drive": drive,
            "punts": {**punts, "game_mean": round(punts["home"]["mean"] + punts["away"]["mean"], 3)},
            "kickers": kickers, "punters": punters,
            "coaches": {"home_hc": hc26.get(home, ""), "away_hc": hc26.get(away, ""),
                        "home_pc": pc.get((SEASON, home), ""), "away_pc": pc.get((SEASON, away), ""),
                        "home_new_pc": pc.get((SEASON, home)) != pc.get((SEASON - 1, home)),
                        "away_new_pc": pc.get((SEASON, away)) != pc.get((SEASON - 1, away))},
        })

    os.makedirs("output", exist_ok=True)
    out = {"season": SEASON, "week": week, "asof_week": asof_week,
           "generated": pd.Timestamp.now().isoformat(), "games": games_out}
    path = f"output/week{week}_slate.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Wrote {path} ({len(games_out)} games)")

if __name__ == "__main__":
    main()
