"""
b00mstick — settle_week.py (Tuesday settlement cron)
====================================================
Auto-grades PENDING rows in Supabase b00mstick_bets from final nflverse
play-by-play. Outcome extraction is IMPORTED from phase0_pull.py (the
same functions that build the model's training data), so scoring and
settlement can never disagree about what happened on the field.

  python settle_week.py                    # settle season 2026, all weeks
  python settle_week.py --week 3           # just week 3
  python settle_week.py --season 2025 --week 18   # arbitrary target
  python settle_week.py --test             # synthetic self-test (see below)

Selections graded (exactly what the app writes):
  "{TEAM} 1st drive {TD|FG|PUNT|OTHER}"        market DRIVE  (no pushes)
  "game punts {over|under}"  + line            market PUNTS  (int lines push)
  "{K.Name} pts {over|under}" + line           market KICKS  (int lines push)

Rules: [PRE]-tagged bets are never touched (preseason paper bets, and
preseason games aren't in nflverse anyway). Unmatchable bets stay PENDING
and are listed. If the season's pbp parquet isn't posted yet, exit
gracefully with a note.

Synthetic test (--test): inserts 3 fake PENDING bets against known 2025
week 18 games — one expected WIN, one LOSS, one PUSH — runs the settler,
verifies all three grades, then deletes the fakes.
"""

import argparse
import json
import re
import sys
import urllib.request

import phase0_pull as p0
from score_week import SUPA_URL, SB_HDRS

BETS = f"{SUPA_URL}/rest/v1/b00mstick_bets"
DRIVE_RE = re.compile(r"^([A-Z]{2,3}) 1st drive (TD|FG|PUNT|OTHER)$")
PUNTS_RE = re.compile(r"^game punts (over|under)$")
KICKS_RE = re.compile(r"^(.+) pts (over|under)$")

# ---------------------------------------------------------------- supabase
def sb(method, url, body=None, prefer=None):
    hdrs = dict(SB_HDRS)
    hdrs["Content-Type"] = "application/json"
    if prefer:
        hdrs["Prefer"] = prefer
    req = urllib.request.Request(url, headers=hdrs, method=method,
                                 data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read()
        return json.loads(raw) if raw else None

# ----------------------------------------------------------------- actuals
def load_actuals(season, week=None):
    """Ground truth per game from nflverse pbp, or None if not posted yet."""
    print(f"Pulling {season} play-by-play...")
    pbp = p0.load_pbp_season(season)
    if pbp is None:
        return None
    reg = pbp[pbp["season_type"] == "REG"].copy()
    if week is not None:
        reg = reg[reg["week"] == week]
    if not len(reg):
        return None
    fd = p0.extract_first_drives(reg)
    pt = p0.extract_punts(reg, fd)
    kk = p0.extract_kicker_points(reg)
    return {
        "drive": {(r.game_id, r.posteam): r.outcome for r in fd.itertuples()},
        "game_punts": {r.game_id: int(r.game_total_punts) for r in pt.itertuples()},
        "kicker": {(r.game_id, r.kicker_player_name): int(r.points) for r in kk.itertuples()},
    }

# ----------------------------------------------------------------- grading
def grade_ou(actual, line, side):
    line = float(line)
    if float(line).is_integer() and actual == int(line):
        return "PUSH"
    won = actual > line if side == "over" else actual < line
    return "WIN" if won else "LOSS"

def grade_bet(b, act):
    """WIN/LOSS/PUSH, or None if this bet can't be matched to a result."""
    sel = str(b.get("selection") or "")
    gid = str(b.get("game_id") or "")
    if sel.startswith("[PRE]") or gid.startswith("PRE"):
        return "PRE"
    mkt = str(b.get("market") or "")
    if mkt == "DRIVE":
        m = DRIVE_RE.match(sel)
        if not m or (gid, m.group(1)) not in act["drive"]:
            return None
        return "WIN" if act["drive"][(gid, m.group(1))] == m.group(2) else "LOSS"
    if mkt == "PUNTS":
        m = PUNTS_RE.match(sel)
        if not m or gid not in act["game_punts"] or b.get("line") is None:
            return None
        return grade_ou(act["game_punts"][gid], b["line"], m.group(1))
    if mkt == "KICKS":
        m = KICKS_RE.match(sel)
        if not m or b.get("line") is None:
            return None
        key = (gid, m.group(1).strip())
        if key not in act["kicker"]:
            return None
        return grade_ou(act["kicker"][key], b["line"], m.group(2))
    return None

def settle(season, week=None):
    act = load_actuals(season, week)
    if act is None:
        print(f"No {season} pbp available yet"
              f"{f' for week {week}' if week else ''} — nothing to settle.")
        return 0
    url = f"{BETS}?select=*&result=eq.PENDING&season=eq.{season}"
    if week is not None:
        url += f"&week=eq.{week}"
    pending = sb("GET", url)
    print(f"{len(pending)} PENDING bets for {season}"
          f"{f' week {week}' if week else ''}")
    graded, skipped_pre, unmatched = 0, 0, []
    for b in pending:
        res = grade_bet(b, act)
        if res == "PRE":
            skipped_pre += 1
            continue
        if res is None:
            unmatched.append(f"  id={b['id']} {b.get('market')} \"{b.get('selection')}\" {b.get('game_id')}")
            continue
        sb("PATCH", f"{BETS}?id=eq.{b['id']}", {"result": res})
        print(f"  {res:5s}  {b.get('selection')}"
              f"{'' if b.get('line') is None else ' ' + str(b['line'])}  ({b.get('game_id')})")
        graded += 1
    print(f"Graded {graded}; skipped {skipped_pre} [PRE]; {len(unmatched)} left PENDING")
    if unmatched:
        print("Unmatched (left PENDING):")
        print("\n".join(unmatched))
    return graded

# ------------------------------------------------------------ synthetic test
def synthetic_test():
    """Insert 3 fakes vs known 2025 wk18 results (WIN/LOSS/PUSH), settle, verify, delete."""
    season, week = 2025, 18
    act = load_actuals(season, week)
    if act is None:
        sys.exit("TEST ABORT: 2025 wk18 pbp unavailable")
    (gid_d, team_d), out_d = next(iter(act["drive"].items()))
    wrong = next(o for o in ("TD", "FG", "PUNT", "OTHER") if o != out_d)
    gid_p, punts_p = next(iter(act["game_punts"].items()))
    (gid_k, name_k), pts_k = next(iter(act["kicker"].items()))
    fakes = [
        # drive bet on the WRONG outcome -> expect LOSS
        {"game_id": gid_d, "market": "DRIVE", "selection": f"{team_d} 1st drive {wrong}",
         "line": None, "expect": "LOSS"},
        # punts line == actual integer total -> expect PUSH
        {"game_id": gid_p, "market": "PUNTS", "selection": "game punts over",
         "line": float(punts_p), "expect": "PUSH"},
        # kicker over at actual-0.5 -> expect WIN
        {"game_id": gid_k, "market": "KICKS", "selection": f"{name_k} pts over",
         "line": pts_k - 0.5, "expect": "WIN"},
    ]
    ids, expects = [], {}
    for f in fakes:
        row = {k: f[k] for k in ("game_id", "market", "selection", "line")}
        row.update({"your_odds": -110, "model_prob": 0.5, "fair_prob_at_bet": 0.5,
                    "tier": "TEST", "season": season, "week": week})
        made = sb("POST", BETS, [row], prefer="return=representation")[0]
        ids.append(made["id"])
        expects[made["id"]] = f["expect"]
        print(f"  inserted fake: {f['selection']} {f['line']} -> expect {f['expect']}")
    try:
        settle(season, week)
        rows = sb("GET", f"{BETS}?select=id,result&id=in.({','.join(str(i) for i in ids)})")
        got = {r["id"]: r["result"] for r in rows}
        fails = [f"id={i}: expected {expects[i]}, got {got.get(i)}"
                 for i in ids if got.get(i) != expects[i]]
    finally:
        for i in ids:
            try:
                sb("DELETE", f"{BETS}?id=eq.{i}")
            except Exception as e:
                print(f"  WARNING: could not delete fake bet id={i} ({e}) — remove manually")
        left = sb("GET", f"{BETS}?select=id&id=in.({','.join(str(i) for i in ids)})")
        print(f"  fakes deleted ({len(left)} remaining)" if not left else
              f"  WARNING: {len(left)} fake rows still present")
    if fails:
        print("SYNTHETIC TEST FAIL:\n  " + "\n  ".join(fails))
        sys.exit(1)
    print("SYNTHETIC TEST PASS: LOSS/PUSH/WIN all graded correctly, fakes removed")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--test", action="store_true", help="synthetic 2025 wk18 self-test")
    args = ap.parse_args()
    if args.test:
        synthetic_test()
    else:
        settle(args.season, args.week)
