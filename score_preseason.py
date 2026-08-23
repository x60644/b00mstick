"""
b00mstick — score_preseason.py (PRESEASON smoke test scorer)
============================================================
Dress-rehearsal variant of score_week.py for the 2026 preseason ONLY.
nflverse carries no preseason schedule or play-by-play, so the slate
comes from ESPN's public scoreboard API instead; the model state is the
same regular-season history Week 1 will use (built as of 2026 week 1).

  python score_preseason.py        # current preseason week per ESPN
  python score_preseason.py 3      # specific preseason week

Guardrails so this can never leak into the real season:
  - hard-exits on/after Sep 1, 2026 (CUTOFF below)
  - never touches data/ (no phase0_pull) and never writes week{N}_slate.json;
    output goes to output/pre{W}_slate.json + output/preseason_current.json
  - the slate is stamped mode="preseason" + an expires date; the app refuses
    to render an expired preseason slate and falls back to the regular flow
  - game lines are placeholders (no real preseason lines in nflverse):
    total 37, spread 0 — flagged lines.placeholder so the app labels them

Model caveat, on purpose: priors describe last season's starters running
real gameplans. Preseason output is for pipeline/GUI testing and paper
bets only — the app tags these bets [PRE] and keeps them out of the record.
"""

import json
import os
import sys
import urllib.request
from datetime import date, datetime, timedelta

import pandas as pd

import b00mstick_model as bm
import score_week as sw   # reuse STADIUMS, load_specialists, wind_forecast, kicker_dist

SEASON = 2026
CUTOFF = date(2026, 9, 1)          # hard stop: preseason mode dies here
EXPIRES = CUTOFF.isoformat()       # stamped into the slate for the app-side guard
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=1"
ESPN_ABBR = {"WSH": "WAS", "LAR": "LA"}   # ESPN -> nflverse team codes
PLACEHOLDER_TOTAL = 37.0           # typical preseason total; overridden in-app

def fetch_espn(week=None):
    url = ESPN_URL + (f"&week={week}" if week else "")
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read())

def parse_games(data):
    games = []
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        home = ESPN_ABBR.get(teams["home"]["team"]["abbreviation"],
                             teams["home"]["team"]["abbreviation"])
        away = ESPN_ABBR.get(teams["away"]["team"]["abbreviation"],
                             teams["away"]["team"]["abbreviation"])
        # ESPN dates are UTC ("2026-08-23T00:00Z"); August == EDT (UTC-4)
        utc = datetime.strptime(ev["date"][:16], "%Y-%m-%dT%H:%M")
        et = utc - timedelta(hours=4)
        indoor = bool(comp.get("venue", {}).get("indoor", False))
        games.append({"espn_id": ev["id"], "home": home, "away": away,
                      "gameday": et.date().isoformat(),
                      "gametime": et.strftime("%H:%M"),
                      "roof": "dome" if indoor else "outdoors"})
    return games

def main():
    if date.today() >= CUTOFF:
        sys.exit(f"Preseason window closed ({CUTOFF}) — use score_week.py. "
                 "This script never scores regular-season games.")

    week_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    data = fetch_espn(week_arg)
    week = week_arg or int(data.get("week", {}).get("number", 0))
    games = parse_games(data)
    if not games:
        sys.exit(f"No preseason games returned by ESPN for week {week}.")
    print(f"Scoring PRESEASON week {week}: {len(games)} games "
          f"(model state = regular-season history as of {SEASON} wk1)")

    frames = bm.load_data("data")
    state = bm.build_state(frames, SEASON, 1)
    specialists = sw.load_specialists()

    oc = pd.read_csv("data/oc_map.csv")
    pc = {(r["season"], r["team"]): r["playcaller"] for _, r in oc.iterrows()}
    hc26 = {r["team"]: r["hc_2026"] for _, r in oc[oc["season"] == SEASON].iterrows()}

    s_pt = state["pt"]
    total, spread = PLACEHOLDER_TOTAL, 0.0
    games_out = []
    for g in games:
        home, away = g["home"], g["away"]
        wind = sw.wind_forecast(home, g["gameday"]) if g["roof"] == "outdoors" else 0.0
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
                kk = state["kk"]   # same tt sensitivity formula as score_week.py
                slope = 3 * kk["k_fgpct"].get(spec["kicker"]["name"], kk["lg_fgpct"]) \
                    * (0.95 if wflag else 1.0) * float(kk["c1"][3]) + \
                    kk["k_xppct"].get(spec["kicker"]["name"], kk["lg_xppct"]) * float(kk["c2"][2])
                kickers[side] = {**spec["kicker"], "mean": round(km, 3),
                                 "tt_base": round(tts[team], 2), "tt_slope": round(slope, 4),
                                 "dist": sw.kicker_dist(state, km)}
            if "punter" in spec:
                punters[side] = spec["punter"]

        games_out.append({
            "game_id": f"PRE{week}_{away}_{home}", "espn_id": g["espn_id"],
            "gameday": g["gameday"], "gametime": g["gametime"],
            "home": home, "away": away,
            "roof": g["roof"], "wind_mph": round(wind, 1), "wind_flag": wflag,
            "lines": {"spread_line": spread, "total_line": total,
                      "home_tt": round(tts[home], 2), "away_tt": round(tts[away], 2),
                      "placeholder": True},
            "drive": drive,
            "punts": {**punts, "game_mean": round(punts["home"]["mean"] + punts["away"]["mean"], 3)},
            "kickers": kickers, "punters": punters,
            "coaches": {"home_hc": hc26.get(home, ""), "away_hc": hc26.get(away, ""),
                        "home_pc": pc.get((SEASON, home), ""), "away_pc": pc.get((SEASON, away), ""),
                        "home_new_pc": pc.get((SEASON, home)) != pc.get((SEASON - 1, home)),
                        "away_new_pc": pc.get((SEASON, away)) != pc.get((SEASON - 1, away))},
        })

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
            return None
        return o

    out = _clean({"season": SEASON, "week": week, "asof_week": 1,
                  "mode": "preseason", "expires": EXPIRES,
                  "generated": pd.Timestamp.now().isoformat(), "games": games_out})
    os.makedirs("output", exist_ok=True)
    for path in (f"output/pre{week}_slate.json", "output/preseason_current.json"):
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print(f"Wrote {path} ({len(games_out)} games)")

if __name__ == "__main__":
    main()
