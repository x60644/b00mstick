"""
b00mstick — audit_week.py (weekly model audit)
==============================================
Scores every archived weekly slate against what actually happened, so the
AUDIT tab can show whether the model was right — independent of whether any
bet was logged. Runs in the Tuesday cron after settlement.

  python audit_week.py            # audit season 2026, all played weeks

For each archived output/week{N}_slate.json whose games have final pbp:

  PUNTS   per game:   model game_mean  vs actual total punts  vs naive
  KICKS   per kicker: model proj pts   vs actual points       vs naive
  DRIVE   per team:   model P(actual outcome) vs naive P      (log loss)

"Naive" is the no-model baseline the backtest gates used: the league
trailing average (all training data before that week), recomputed per week
from the data/ CSVs. A prediction "hits" when it lands closer to the actual
than naive did — beating naive consistently is the whole claim.

Outcome extraction is IMPORTED from phase0_pull.py (same functions that
build training data and grade bets), so the audit can never disagree with
scoring or settlement about what happened on the field.

Writes output/audit.json. Weeks with no completed games are skipped;
projected kickers with no recorded attempts (injury/swap/zero-attempt
games) are shown but excluded from error stats.
"""

import glob
import json
import math
import os
import re

import pandas as pd

import phase0_pull as p0

SEASON = 2026
OUT = "output/audit.json"
EPS = 1e-6  # log-loss clamp


def trailing(df, week):
    """Training rows strictly before SEASON week `week` (prior seasons + earlier weeks)."""
    return df[(df["season"] < SEASON) | ((df["season"] == SEASON) & (df["week"] < week))]


def mae(rows, key):
    v = [abs(r[key] - r["actual"]) for r in rows if r.get("actual") is not None]
    return round(sum(v) / len(v), 3) if v else None


def main():
    # ---- archived slates -------------------------------------------------
    slates = {}
    for f in glob.glob("output/week*_slate.json"):
        m = re.search(r"week(\d+)_slate\.json$", f)
        if not m:
            continue
        with open(f) as fh:
            sl = json.load(fh)
        if sl.get("season") == SEASON:
            slates[int(m.group(1))] = sl
    if not slates:
        print("No archived slates found — nothing to audit.")
        return

    # ---- actuals from final pbp -----------------------------------------
    print(f"Pulling {SEASON} play-by-play...")
    pbp = p0.load_pbp_season(SEASON)
    if pbp is None:
        print(f"No {SEASON} pbp available yet — nothing to audit.")
        return
    reg = pbp[pbp["season_type"] == "REG"].copy()
    fd = p0.extract_first_drives(reg)
    pt = p0.extract_punts(reg, fd)
    kk = p0.extract_kicker_points(reg)
    act_drive = {(r.game_id, r.posteam): r.outcome for r in fd.itertuples()}
    act_punts = {r.game_id: int(r.game_total_punts) for r in pt.itertuples()}
    act_kick = {(r.game_id, r.kicker_player_name): int(r.points) for r in kk.itertuples()}

    # ---- naive baselines from training CSVs ------------------------------
    pcsv = pd.read_csv("data/punts.csv").drop_duplicates("game_id")
    kcsv = pd.read_csv("data/kicker_points.csv")
    fcsv = pd.read_csv("data/first_drives.csv")

    weeks = []
    for wknum in sorted(slates):
        sl = slates[wknum]
        naive_p = round(float(trailing(pcsv, wknum)["game_total_punts"].mean()), 2)
        naive_k = round(float(trailing(kcsv, wknum)["points"].mean()), 2)
        fdist = trailing(fcsv, wknum)["outcome"].value_counts(normalize=True).to_dict()

        prow, krow, dmod, dnaive, dn = [], [], 0.0, 0.0, 0
        for g in sl["games"]:
            gid = g["game_id"]
            mu = f'{g["away"]} @ {g["home"]}'
            if gid in act_punts:
                prow.append({"game": mu, "model": g["punts"]["game_mean"],
                             "actual": act_punts[gid], "naive": naive_p})
            for side in ("away", "home"):
                team = g[side]
                k = (g.get("kickers") or {}).get(side)
                if k and gid in act_punts:  # game played
                    krow.append({"game": mu, "team": team, "kicker": k["name"],
                                 "model": k["mean"],
                                 "actual": act_kick.get((gid, k["name"])),
                                 "naive": naive_k})
                out = act_drive.get((gid, team))
                if out:
                    dmod += -math.log(max(EPS, g["drive"][side].get(out) or 0))
                    dnaive += -math.log(max(EPS, fdist.get(out, 0)))
                    dn += 1

        if not prow:  # week not played yet
            continue
        weeks.append({
            "week": wknum,
            "punts": {"rows": prow, "model_mae": mae(prow, "model"),
                      "naive_mae": mae(prow, "naive")},
            "kicks": {"rows": krow, "model_mae": mae(krow, "model"),
                      "naive_mae": mae(krow, "naive")},
            "drive": {"n": dn,
                      "model_logloss": round(dmod / dn, 4) if dn else None,
                      "naive_logloss": round(dnaive / dn, 4) if dn else None},
        })
        print(f"week {wknum}: punts MAE {weeks[-1]['punts']['model_mae']} "
              f"(naive {weeks[-1]['punts']['naive_mae']}) · "
              f"kicks MAE {weeks[-1]['kicks']['model_mae']} "
              f"(naive {weeks[-1]['kicks']['naive_mae']}) · "
              f"drive LL {weeks[-1]['drive']['model_logloss']} "
              f"(naive {weeks[-1]['drive']['naive_logloss']})")

    os.makedirs("output", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"season": SEASON, "generated": pd.Timestamp.now().isoformat(),
                   "weeks": weeks}, f, indent=1)
    print(f"Wrote {OUT} ({len(weeks)} audited weeks)")


if __name__ == "__main__":
    main()
