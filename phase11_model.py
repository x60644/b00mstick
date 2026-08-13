"""
b00mstick — Phase 1.1: playcaller lineage + HC punt tendency
============================================================
Adds two coach factors and judges them the pre-agreed way:

  FIRST DRIVE + lineage: each playcaller gets a fingerprint built from
  every game he called (any team). A team's offense ratio becomes a
  geometric blend: team^(1-B) * playcaller^B. Blend B is tuned ONLY on
  2024 changed-playcaller games; 2025 is untouched until final scoring.
  B=0 reproduces the phase 1 baseline exactly, so the comparison is clean.
  Verdict is rendered on CHANGED-playcaller teams' 2025 games, not the
  full slate (changes touch ~1/3 of teams; the rest would drown the signal).

  PUNTS + HC tendency: each head coach gets a shrunk punt-tendency ratio
  from every game he coached (follows the coach across teams). Added as
  a regression feature; lift measured with-vs-without on the same folds.

Requires data/oc_map.csv (spot-checked). Run: python phase11_model.py
"""

import numpy as np
import pandas as pd

DATA = "data"
TARGET = 2025
CAL = 2024
DECAY = 0.5
OUTCOMES = ["TD", "FG", "PUNT", "OTHER"]
FD_K, FD_TAU = 20, 0.5          # locked from phase 1 calibration
PC_K = 16                        # playcaller fingerprint shrinkage

fd = pd.read_csv(f"{DATA}/first_drives.csv")
pt = pd.read_csv(f"{DATA}/punts.csv")
gm = pd.read_csv(f"{DATA}/games.csv")
oc = pd.read_csv(f"{DATA}/oc_map.csv")

gm["wind"] = pd.to_numeric(gm["wind"], errors="coerce").fillna(0)
gm["wind_flag"] = ((gm["wind"] >= 15) & (gm["roof"] == "outdoors")).astype(int)
gm["total_line"] = pd.to_numeric(gm["total_line"], errors="coerce")
gm["total_line"] = gm["total_line"].fillna(gm["total_line"].mean())

gcols = ["game_id", "home_team", "home_coach", "away_coach", "wind_flag", "total_line"]
fd = fd.merge(gm[gcols], on="game_id")
pt = pt.merge(gm[gcols], on="game_id")
for df in (fd, pt):
    df["is_home"] = (df["posteam"] == df["home_team"]).astype(int)
    df["coach"] = np.where(df["is_home"] == 1, df["home_coach"], df["away_coach"])

pc_map = {(r["season"], r["team"]): r["playcaller"] for _, r in oc.iterrows()}
fd["pc"] = fd.apply(lambda r: pc_map.get((r["season"], r["posteam"]),
                                         f"UNK_{r['posteam']}_{r['season']}"), axis=1)

def changed_teams(season):
    return {t for t in oc[oc["season"] == season]["team"]
            if pc_map.get((season, t)) != pc_map.get((season - 1, t))}

def wts(df, asof):
    return DECAY ** (asof - df["season"]).clip(lower=0)

def shrunk(vals, w, league, K):
    return (np.sum(vals * w) + K * league) / (np.sum(w) + K)

# ---------------------------------------------------------- first drive
def fd_tables(train, asof):
    t = train.assign(w=wts(train, asof))
    league = {o: np.average((t["outcome"] == o), weights=t["w"]) for o in OUTCOMES}
    base = {}
    for r in (0, 1):
        s = t[t["received_opening_kick"] == r]
        base[r] = {o: np.average((s["outcome"] == o), weights=s["w"]) for o in OUTCOMES}
    off = {tm: {o: shrunk((g["outcome"] == o).values, g["w"].values, league[o], FD_K) / league[o]
                for o in OUTCOMES} for tm, g in t.groupby("posteam")}
    dfn = {tm: {o: shrunk((g["outcome"] == o).values, g["w"].values, league[o], FD_K) / league[o]
                for o in OUTCOMES} for tm, g in t.groupby("defteam")}
    pcr = {p: {o: shrunk((g["outcome"] == o).values, g["w"].values, league[o], PC_K) / league[o]
               for o in OUTCOMES} for p, g in t.groupby("pc")}
    hm = t[t["is_home"] == 1]
    home = {o: np.average((hm["outcome"] == o), weights=hm["w"]) / league[o] for o in OUTCOMES}
    return base, off, dfn, pcr, home, league

def fd_walk(target, B):
    prior = fd[fd["season"] < target]
    tgt = fd[fd["season"] == target]
    ones = {o: 1.0 for o in OUTCOMES}
    rows = []
    for wk in sorted(tgt["week"].unique()):
        train = pd.concat([prior, tgt[tgt["week"] < wk]])
        base, off, dfn, pcr, home, league = fd_tables(train, target)
        for _, g in tgt[tgt["week"] == wk].iterrows():
            o_t, p_t = off.get(g["posteam"], ones), pcr.get(g["pc"], ones)
            d_t = dfn.get(g["defteam"], ones)
            ps = []
            for r in (0, 1):
                p = np.array([
                    base[r][o]
                    * (o_t[o] ** (1 - B) * p_t[o] ** B) ** (1.0 * FD_TAU)
                    * d_t[o] ** (0.7 * FD_TAU)
                    * (home[o] ** FD_TAU if g["is_home"] else 1.0)
                    for o in OUTCOMES])
                ps.append(p / p.sum())
            pred = 0.5 * ps[0] + 0.5 * ps[1]
            rows.append({"posteam": g["posteam"],
                         "ll": -np.log(pred[OUTCOMES.index(g["outcome"])] + 1e-12)})
    return pd.DataFrame(rows)

print("Tuning lineage blend B on 2024 changed-playcaller games...")
ch24 = changed_teams(CAL)
print(f"  2024 changed teams ({len(ch24)}): {sorted(ch24)}")
grid = {}
for B in (0.0, 0.2, 0.4, 0.6, 0.8):
    r = fd_walk(CAL, B)
    grid[B] = r[r["posteam"].isin(ch24)]["ll"].mean()
    print(f"  B={B:.1f}  changed-team log loss: {grid[B]:.4f}")
BEST_B = min(grid, key=grid.get)
print(f"  chosen B={BEST_B}")

ch25 = changed_teams(TARGET)
print(f"\n2025 changed teams ({len(ch25)}): {sorted(ch25)}")
r0 = fd_walk(TARGET, 0.0)
r1 = fd_walk(TARGET, BEST_B)
c0, c1 = r0[r0["posteam"].isin(ch25)]["ll"].mean(), r1[r1["posteam"].isin(ch25)]["ll"].mean()
print("\n===== FIRST DRIVE LINEAGE — 2025 verdict =====")
print(f"Changed-team log loss — baseline: {c0:.4f}   lineage: {c1:.4f}   "
      f"{'LINEAGE WINS' if c1 < c0 else '** NO LIFT — LINEAGE DIES **'}")
print(f"All-games log loss    — baseline: {r0['ll'].mean():.4f}   lineage: {r1['ll'].mean():.4f}")

# ---------------------------------------------------------------- punts
def punts_walk(target, use_hc, K=12):
    prior = pt[pt["season"] < target]
    tgt = pt[pt["season"] == target]
    rows = []
    for wk in sorted(tgt["week"].unique()):
        train = pd.concat([prior, tgt[tgt["week"] < wk]]).copy()
        train["w"] = wts(train, target)
        lg = np.average(train["punts"], weights=train["w"])
        mean_tot = np.average(train["total_line"], weights=train["w"])
        team_r = {tm: shrunk(g["punts"].values, g["w"].values, lg, K) / lg for tm, g in train.groupby("posteam")}
        opp_r = {tm: shrunk(g["punts"].values, g["w"].values, lg, K) / lg for tm, g in train.groupby("defteam")}
        hc_r = {c: shrunk(g["punts"].values, g["w"].values, lg, K) / lg for c, g in train.groupby("coach")}
        base_tr = lg * train["posteam"].map(team_r).fillna(1) * train["defteam"].map(opp_r).fillna(1) ** 0.7
        cols = [np.ones(len(train)), base_tr, train["total_line"] - mean_tot, train["wind_flag"]]
        if use_hc:
            cols.append(train["coach"].map(hc_r).fillna(1) - 1)
        X = np.column_stack(cols)
        sw = np.sqrt(train["w"].values)[:, None]
        coef, *_ = np.linalg.lstsq(X * sw, train["punts"].values * sw.flatten(), rcond=None)
        for _, g in tgt[tgt["week"] == wk].iterrows():
            b = lg * team_r.get(g["posteam"], 1) * opp_r.get(g["defteam"], 1) ** 0.7
            x = [1, b, g["total_line"] - mean_tot, g["wind_flag"]]
            if use_hc:
                x.append(hc_r.get(g["coach"], 1) - 1)
            rows.append({"game_id": g["game_id"], "coach": g["coach"],
                         "pred": max(float(np.array(x) @ coef), 0.3), "actual": g["punts"]})
    return pd.DataFrame(rows)

print("\n===== PUNTS + HC TENDENCY — 2025 =====")
p0 = punts_walk(TARGET, use_hc=False)
p1 = punts_walk(TARGET, use_hc=True)
m0 = (p0["pred"] - p0["actual"]).abs().mean()
m1 = (p1["pred"] - p1["actual"]).abs().mean()
print(f"Team punts MAE — without HC: {m0:.3f}   with HC: {m1:.3f}   "
      f"{'HC FACTOR WINS' if m1 < m0 else '** NO LIFT — HC FACTOR DIES **'}")
g0 = p0.groupby("game_id").agg(pred=("pred", "sum"), actual=("actual", "first" if False else "sum"))
g1 = p1.groupby("game_id").agg(pred=("pred", "sum"), actual=("actual", "sum"))
g0a = p0.groupby("game_id").agg(pred=("pred", "sum"), actual=("actual", "sum"))
print(f"Game total MAE — without HC: {(g0a['pred']-g0a['actual']).abs().mean():.3f}   "
      f"with HC: {(g1['pred']-g1['actual']).abs().mean():.3f}")

print("\nDone. Winners go into the shared scoring module; losers are dropped.")
