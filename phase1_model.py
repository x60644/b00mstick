"""
b00mstick — Phase 1 structured model + 2025 walk-forward backtest
=================================================================
Baseline v1 (no playcaller lineage yet — that lands in v1.1 so we can
measure its lift against this run).

Reads the phase 0 CSVs from ./data and:
  1. builds recency-weighted, shrunk factor tables
  2. calibrates shrinkage K + sharpening tau on 2024 (using 2022-23 priors)
  3. walk-forward backtests 2025 week by week (each week predicted using
     only data available before that week -- mirrors production)
  4. prints the beat-naive-or-die report

Run:  python phase1_model.py
"""

import numpy as np
import pandas as pd

DATA = "data"
TARGET = 2025          # backtest season
CAL = 2024             # calibration season
DECAY = 0.5            # per-season recency decay
OUTCOMES = ["TD", "FG", "PUNT", "OTHER"]

# ------------------------------------------------------------------ load
fd = pd.read_csv(f"{DATA}/first_drives.csv")
pt = pd.read_csv(f"{DATA}/punts.csv")
kk = pd.read_csv(f"{DATA}/kicker_points.csv")
gm = pd.read_csv(f"{DATA}/games.csv")

gm["wind"] = pd.to_numeric(gm["wind"], errors="coerce").fillna(0)
gm["outdoor"] = (gm["roof"] == "outdoors").astype(int)
gm["wind_flag"] = ((gm["wind"] >= 15) & (gm["outdoor"] == 1)).astype(int)
gm["total_line"] = pd.to_numeric(gm["total_line"], errors="coerce")
gm["spread_line"] = pd.to_numeric(gm["spread_line"], errors="coerce")
gm["total_line"] = gm["total_line"].fillna(gm["total_line"].mean())
gm["spread_line"] = gm["spread_line"].fillna(0)

gcols = ["game_id", "home_team", "wind_flag", "total_line", "spread_line"]
for df in (fd, pt, kk):
    df.drop(columns=[c for c in gcols[1:] if c in df.columns], inplace=True, errors="ignore")

fd = fd.merge(gm[gcols], on="game_id")
pt = pt.merge(gm[gcols], on="game_id")
kk = kk.merge(gm[gcols], on="game_id")

for df in (fd, pt, kk):
    df["is_home"] = (df["posteam"] == df["home_team"]).astype(int)
    # derived DK-style team total: spread_line positive = home favored
    df["team_total"] = np.where(
        df["is_home"] == 1,
        (df["total_line"] + df["spread_line"]) / 2,
        (df["total_line"] - df["spread_line"]) / 2,
    )

def wts(df, asof_season):
    """Recency weight per row as of a target season."""
    return DECAY ** (asof_season - df["season"]).clip(lower=0)

def timeslice(df, season, week):
    """All rows strictly before (season, week)."""
    return df[(df["season"] < season) | ((df["season"] == season) & (df["week"] < week))]

def shrunk_rate(values, weights, league, K):
    return (np.sum(values * weights) + K * league) / (np.sum(weights) + K)

# =================================================================
# MARKET 1 — FIRST DRIVE RESULT
# =================================================================
def fd_tables(train, asof, K):
    w = wts(train, asof)
    t = train.assign(w=w)
    league = {o: np.average((t["outcome"] == o), weights=t["w"]) for o in OUTCOMES}
    base = {}
    for r in (0, 1):
        sub = t[t["received_opening_kick"] == r]
        base[r] = {o: np.average((sub["outcome"] == o), weights=sub["w"]) for o in OUTCOMES}
    off, dfn = {}, {}
    for team, gtt in t.groupby("posteam"):
        off[team] = {o: shrunk_rate((gtt["outcome"] == o).values, gtt["w"].values, league[o], K) / league[o] for o in OUTCOMES}
    for team, gtt in t.groupby("defteam"):
        dfn[team] = {o: shrunk_rate((gtt["outcome"] == o).values, gtt["w"].values, league[o], K) / league[o] for o in OUTCOMES}
    hm = t[t["is_home"] == 1]
    home = {o: np.average((hm["outcome"] == o), weights=hm["w"]) / league[o] for o in OUTCOMES}
    return base, off, dfn, home

def fd_predict(base, off, dfn, home, team, opp, is_home, tau):
    ps = []
    for r in (0, 1):
        p = np.array([
            base[r][o]
            * off.get(team, {o: 1 for o in OUTCOMES})[o] ** (1.0 * tau)
            * dfn.get(opp, {o: 1 for o in OUTCOMES})[o] ** (0.7 * tau)
            * (home[o] ** tau if is_home else 1.0)
            for o in OUTCOMES
        ])
        ps.append(p / p.sum())
    return 0.5 * ps[0] + 0.5 * ps[1]   # coin-toss blend

def fd_walk(target, K, tau):
    prior = fd[fd["season"] < target]
    tgt = fd[fd["season"] == target]
    preds, actuals, naive = [], [], []
    for wk in sorted(tgt["week"].unique()):
        train = pd.concat([prior, tgt[tgt["week"] < wk]])
        base, off, dfn, home = fd_tables(train, target, K)
        w_league = wts(train, target)
        lg = np.array([np.average((train["outcome"] == o), weights=w_league) for o in OUTCOMES])
        for _, g in tgt[tgt["week"] == wk].iterrows():
            preds.append(fd_predict(base, off, dfn, home, g["posteam"], g["defteam"], g["is_home"], tau))
            naive.append(lg / lg.sum())
            actuals.append(OUTCOMES.index(g["outcome"]))
    P, N, y = np.array(preds), np.array(naive), np.array(actuals)
    eps = 1e-12
    ll_m = -np.mean(np.log(P[np.arange(len(y)), y] + eps))
    ll_n = -np.mean(np.log(N[np.arange(len(y)), y] + eps))
    return ll_m, ll_n, P, y

print("Calibrating first-drive model on 2024...")
best = None
for K in (8, 12, 20):
    for tau in (0.5, 0.75, 1.0, 1.25):
        ll, _, _, _ = fd_walk(CAL, K, tau)
        if best is None or ll < best[0]:
            best = (ll, K, tau)
FD_K, FD_TAU = best[1], best[2]
print(f"  chosen K={FD_K}, tau={FD_TAU} (2024 log loss {best[0]:.4f})")

ll_m, ll_n, P, y = fd_walk(TARGET, FD_K, FD_TAU)
print("\n===== FIRST DRIVE — 2025 walk-forward =====")
print(f"Model log loss: {ll_m:.4f}   Naive (league avg): {ll_n:.4f}   "
      f"{'BEATS NAIVE' if ll_m < ll_n else '** DOES NOT BEAT NAIVE **'}")
print("\nCalibration (predicted prob bucket vs actual rate, all outcomes pooled):")
flat_p = P.flatten()
flat_hit = np.zeros_like(P)
flat_hit[np.arange(len(y)), y] = 1
flat_hit = flat_hit.flatten()
buckets = pd.cut(flat_p, [0, .1, .2, .3, .4, .5, 1.0])
cal = pd.DataFrame({"pred": flat_p, "hit": flat_hit, "b": buckets}).groupby("b", observed=True).agg(
    mean_pred=("pred", "mean"), actual=("hit", "mean"), n=("hit", "size"))
print(cal.round(3).to_string())

# =================================================================
# MARKET 2 — PUNTS
# =================================================================
def punts_walk(target, K=12):
    prior = pt[pt["season"] < target]
    tgt = pt[pt["season"] == target]
    rows = []
    for wk in sorted(tgt["week"].unique()):
        train = pd.concat([prior, tgt[tgt["week"] < wk]]).copy()
        train["w"] = wts(train, target)
        lg = np.average(train["punts"], weights=train["w"])
        mean_tot = np.average(train["total_line"], weights=train["w"])
        team_r = {tm: shrunk_rate(g["punts"].values, g["w"].values, lg, K) / lg
                  for tm, g in train.groupby("posteam")}
        opp_r = {tm: shrunk_rate(g["punts"].values, g["w"].values, lg, K) / lg
                 for tm, g in train.groupby("defteam")}
        # fit residual coefficients on train: punts ~ base + c*(total-mean) + d*wind
        base_tr = lg * train["posteam"].map(team_r).fillna(1) * train["defteam"].map(opp_r).fillna(1) ** 0.7
        X = np.column_stack([np.ones(len(train)), base_tr,
                             train["total_line"] - mean_tot, train["wind_flag"]])
        coef, *_ = np.linalg.lstsq(X * np.sqrt(train["w"].values)[:, None],
                                   train["punts"].values * np.sqrt(train["w"].values), rcond=None)
        for _, g in tgt[tgt["week"] == wk].iterrows():
            b = lg * team_r.get(g["posteam"], 1) * opp_r.get(g["defteam"], 1) ** 0.7
            pred = coef[0] + coef[1] * b + coef[2] * (g["total_line"] - mean_tot) + coef[3] * g["wind_flag"]
            rows.append({"game_id": g["game_id"], "posteam": g["posteam"],
                         "pred": max(pred, 0.3), "naive": lg, "actual": g["punts"]})
    return pd.DataFrame(rows)

print("\n===== PUNTS — 2025 walk-forward =====")
pr = punts_walk(TARGET)
mae_m = (pr["pred"] - pr["actual"]).abs().mean()
mae_n = (pr["naive"] - pr["actual"]).abs().mean()
print(f"Team punts MAE — model: {mae_m:.3f}   naive: {mae_n:.3f}   "
      f"{'BEATS NAIVE' if mae_m < mae_n else '** DOES NOT BEAT NAIVE **'}")
gt = pr.groupby("game_id").agg(pred=("pred", "sum"), naive=("naive", "sum"), actual=("actual", "sum"))
mae_gm = (gt["pred"] - gt["actual"]).abs().mean()
mae_gn = (gt["naive"] - gt["actual"]).abs().mean()
print(f"Game total MAE  — model: {mae_gm:.3f}   naive: {mae_gn:.3f}")
from math import lgamma
def pois_ll(mu, k):
    return k * np.log(mu) - mu - np.array([lgamma(x + 1) for x in k])
ll_pm = -np.mean(pois_ll(gt["pred"].values, gt["actual"].values))
ll_pn = -np.mean(pois_ll(gt["naive"].values, gt["actual"].values))
print(f"Game total Poisson log loss — model: {ll_pm:.4f}   naive: {ll_pn:.4f}")

# =================================================================
# MARKET 3 — KICKER POINTS (v2: two-stage attempts decomposition)
# =================================================================
ds = pd.read_csv(f"{DATA}/drive_stats.csv")
kk = kk.merge(ds[["game_id", "posteam", "stalls", "off_tds"]],
              on=["game_id", "posteam"], how="left")
kk[["stalls", "off_tds"]] = kk[["stalls", "off_tds"]].fillna(0)

def kicker_walk(target, K=8, KP=15):
    prior = kk[kk["season"] < target]
    tgt = kk[kk["season"] == target]
    rows = []
    for wk in sorted(tgt["week"].unique()):
        train = pd.concat([prior, tgt[tgt["week"] < wk]]).copy()
        train["w"] = wts(train, target)
        lg_pts = np.average(train["points"], weights=train["w"])
        lg_fga = np.average(train["fg_att"], weights=train["w"])
        lg_xpa = np.average(train["xp_att"], weights=train["w"])
        lg_stall = np.average(train["stalls"], weights=train["w"])
        lg_fgpct = np.sum(train["fg_made"] * train["w"]) / max(np.sum(train["fg_att"] * train["w"]), 1)
        lg_xppct = np.sum(train["xp_made"] * train["w"]) / max(np.sum(train["xp_att"] * train["w"]), 1)
        t_fga = {t: shrunk_rate(g["fg_att"].values, g["w"].values, lg_fga, K) for t, g in train.groupby("posteam")}
        t_xpa = {t: shrunk_rate(g["xp_att"].values, g["w"].values, lg_xpa, K) for t, g in train.groupby("posteam")}
        t_stall = {t: shrunk_rate(g["stalls"].values, g["w"].values, lg_stall, K) for t, g in train.groupby("posteam")}
        # kicker make rates shrunk with pseudo-ATTEMPTS
        k_fgpct, k_xppct = {}, {}
        for kr, g in train.groupby("kicker_player_name"):
            k_fgpct[kr] = (np.sum(g["fg_made"] * g["w"]) + KP * lg_fgpct) / (np.sum(g["fg_att"] * g["w"]) + KP)
            k_xppct[kr] = (np.sum(g["xp_made"] * g["w"]) + KP * lg_xppct) / (np.sum(g["xp_att"] * g["w"]) + KP)
        sw = np.sqrt(train["w"].values)[:, None]
        # stage 1: FG attempts ~ team fga history, stall rate, vegas total, wind
        X1 = np.column_stack([np.ones(len(train)),
                              train["posteam"].map(t_fga).fillna(lg_fga),
                              train["posteam"].map(t_stall).fillna(lg_stall),
                              train["team_total"], train["wind_flag"]])
        c1, *_ = np.linalg.lstsq(X1 * sw, train["fg_att"].values * sw.flatten(), rcond=None)
        # stage 2: XP attempts ~ team xpa history, vegas total
        X2 = np.column_stack([np.ones(len(train)),
                              train["posteam"].map(t_xpa).fillna(lg_xpa),
                              train["team_total"]])
        c2, *_ = np.linalg.lstsq(X2 * sw, train["xp_att"].values * sw.flatten(), rcond=None)
        for _, g in tgt[tgt["week"] == wk].iterrows():
            fga = max(float(np.array([1, t_fga.get(g["posteam"], lg_fga),
                                      t_stall.get(g["posteam"], lg_stall),
                                      g["team_total"], g["wind_flag"]]) @ c1), 0)
            xpa = max(float(np.array([1, t_xpa.get(g["posteam"], lg_xpa),
                                      g["team_total"]]) @ c2), 0)
            fgp = k_fgpct.get(g["kicker_player_name"], lg_fgpct)
            xpp = k_xppct.get(g["kicker_player_name"], lg_xppct)
            wind_pct = 0.95 if g["wind_flag"] == 1 else 1.0   # misses rise in wind
            pred = 3 * fga * fgp * wind_pct + xpa * xpp
            rows.append({"pred": pred, "naive": lg_pts, "actual": g["points"],
                         "pred_fga": fga, "actual_fga": g["fg_att"]})
    return pd.DataFrame(rows)

print("\n===== KICKER POINTS v2 — 2025 walk-forward =====")
kr = kicker_walk(TARGET)
mae_km = (kr["pred"] - kr["actual"]).abs().mean()
mae_kn = (kr["naive"] - kr["actual"]).abs().mean()
print(f"Kicker pts MAE — model: {mae_km:.3f}   naive: {mae_kn:.3f}   "
      f"{'BEATS NAIVE' if mae_km < mae_kn else '** DOES NOT BEAT NAIVE **'}")
mae_fga = (kr["pred_fga"] - kr["actual_fga"]).abs().mean()
mae_fgan = (kr["actual_fga"].mean() - kr["actual_fga"]).abs().mean()
print(f"FG attempts MAE — model: {mae_fga:.3f}   naive: {mae_fgan:.3f}")
res = kr["actual"] - kr["pred"]
print(f"Residual mean: {res.mean():+.3f} (bias)   std: {res.std():.3f}")
for q in (0.5, 0.8):
    lo, hi = res.quantile((1 - q) / 2), res.quantile(1 - (1 - q) / 2)
    cov = ((res >= lo) & (res <= hi)).mean()
    print(f"{int(q*100)}% interval coverage: {cov:.1%}")

print("\nDone. If any market fails to beat naive, its factors get rebuilt before GUI work.")
