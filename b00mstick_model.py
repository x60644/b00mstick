"""
b00mstick_model.py — shared scoring module (single source of truth)
===================================================================
The locked, backtest-validated models: phase 1 first drive (no lineage),
phase 1 punts (no HC factor), kicker v2 two-stage. The weekly cron and
any future backtest import THIS file, so production and validation can
never drift.

Public API:
  frames = load_data("data")
  state  = build_state(frames, asof_season=2026, asof_week=1)
  predict_first_drive(state, team, opp, is_home) -> {TD,FG,PUNT,OTHER: p}
  predict_punts(state, team, opp, total_line, wind_flag) -> mean
  poisson_probs(mean, line) -> (p_over, p_under, p_push)
  predict_kicker(state, team, kicker, team_total, wind_flag) -> mean
  kicker_probs(state, mean, line) -> (p_over, p_under, p_push)
  american_to_prob(odds), devig_two_way(a, b), devig_multiway({...})
  classify_edge(model_p, fair_p, market) -> BET / VALUE / PASS

Self-test:  python b00mstick_model.py
"""

import numpy as np
import pandas as pd
from math import exp, lgamma

DECAY = 0.5
OUTCOMES = ["TD", "FG", "PUNT", "OTHER"]
FD_K, FD_TAU = 20, 0.5
PUNT_K, KICK_K, KICK_KP = 12, 8, 15
# stingy tiering: edge in probability points vs devigged fair prob
THRESHOLDS = {
    "drive":  {"BET": 0.050, "VALUE": 0.025},
    "punts":  {"BET": 0.045, "VALUE": 0.022},
    "kicker": {"BET": 0.045, "VALUE": 0.022},
}

# ------------------------------------------------------------------ data
def load_data(data_dir="data"):
    fd = pd.read_csv(f"{data_dir}/first_drives.csv")
    pt = pd.read_csv(f"{data_dir}/punts.csv")
    kk = pd.read_csv(f"{data_dir}/kicker_points.csv")
    ds = pd.read_csv(f"{data_dir}/drive_stats.csv")
    gm = pd.read_csv(f"{data_dir}/games.csv")
    gm["wind"] = pd.to_numeric(gm["wind"], errors="coerce").fillna(0)
    gm["wind_flag"] = ((gm["wind"] >= 15) & (gm["roof"] == "outdoors")).astype(int)
    for c in ("total_line", "spread_line"):
        gm[c] = pd.to_numeric(gm[c], errors="coerce")
    gm["total_line"] = gm["total_line"].fillna(gm["total_line"].mean())
    gm["spread_line"] = gm["spread_line"].fillna(0)
    kk = kk.merge(ds[["game_id", "posteam", "stalls", "off_tds"]],
                  on=["game_id", "posteam"], how="left")
    kk[["stalls", "off_tds"]] = kk[["stalls", "off_tds"]].fillna(0)
    gcols = ["game_id", "home_team", "wind_flag", "total_line", "spread_line"]
    out = {}
    for name, df in (("fd", fd), ("pt", pt), ("kk", kk)):
        df = df.merge(gm[gcols], on="game_id")
        df["is_home"] = (df["posteam"] == df["home_team"]).astype(int)
        df["team_total"] = np.where(df["is_home"] == 1,
                                    (df["total_line"] + df["spread_line"]) / 2,
                                    (df["total_line"] - df["spread_line"]) / 2)
        out[name] = df
    out["gm"] = gm
    return out

def _wts(df, asof_season):
    return DECAY ** (asof_season - df["season"]).clip(lower=0)

def _cut(df, season, week):
    return df[(df["season"] < season) | ((df["season"] == season) & (df["week"] < week))]

def _shrunk(vals, w, league, K):
    return (np.sum(vals * w) + K * league) / (np.sum(w) + K)

# ----------------------------------------------------------------- state
def build_state(frames, asof_season, asof_week):
    st = {"asof": (asof_season, asof_week)}

    # ---- first drive
    t = _cut(frames["fd"], asof_season, asof_week).assign(w=lambda d: _wts(d, asof_season))
    league = {o: np.average((t["outcome"] == o), weights=t["w"]) for o in OUTCOMES}
    base = {}
    for r in (0, 1):
        s = t[t["received_opening_kick"] == r]
        base[r] = {o: np.average((s["outcome"] == o), weights=s["w"]) for o in OUTCOMES}
    st["fd"] = {
        "base": base,
        "off": {tm: {o: _shrunk((g["outcome"] == o).values, g["w"].values, league[o], FD_K) / league[o]
                     for o in OUTCOMES} for tm, g in t.groupby("posteam")},
        "def": {tm: {o: _shrunk((g["outcome"] == o).values, g["w"].values, league[o], FD_K) / league[o]
                     for o in OUTCOMES} for tm, g in t.groupby("defteam")},
        "home": {o: np.average((t[t["is_home"] == 1]["outcome"] == o),
                               weights=t[t["is_home"] == 1]["w"]) / league[o] for o in OUTCOMES},
    }

    # ---- punts
    p = _cut(frames["pt"], asof_season, asof_week).assign(w=lambda d: _wts(d, asof_season))
    lg = np.average(p["punts"], weights=p["w"])
    mean_tot = np.average(p["total_line"], weights=p["w"])
    team_r = {tm: _shrunk(g["punts"].values, g["w"].values, lg, PUNT_K) / lg for tm, g in p.groupby("posteam")}
    opp_r = {tm: _shrunk(g["punts"].values, g["w"].values, lg, PUNT_K) / lg for tm, g in p.groupby("defteam")}
    base_tr = lg * p["posteam"].map(team_r).fillna(1) * p["defteam"].map(opp_r).fillna(1) ** 0.7
    X = np.column_stack([np.ones(len(p)), base_tr, p["total_line"] - mean_tot, p["wind_flag"]])
    sw = np.sqrt(p["w"].values)[:, None]
    coef, *_ = np.linalg.lstsq(X * sw, p["punts"].values * sw.flatten(), rcond=None)
    st["pt"] = {"lg": lg, "mean_tot": mean_tot, "team": team_r, "opp": opp_r, "coef": coef}

    # ---- kicker (two-stage)
    k = _cut(frames["kk"], asof_season, asof_week).assign(w=lambda d: _wts(d, asof_season))
    lg_pts = np.average(k["points"], weights=k["w"])
    lg_fga = np.average(k["fg_att"], weights=k["w"])
    lg_xpa = np.average(k["xp_att"], weights=k["w"])
    lg_stall = np.average(k["stalls"], weights=k["w"])
    lg_fgpct = np.sum(k["fg_made"] * k["w"]) / max(np.sum(k["fg_att"] * k["w"]), 1)
    lg_xppct = np.sum(k["xp_made"] * k["w"]) / max(np.sum(k["xp_att"] * k["w"]), 1)
    t_fga = {t2: _shrunk(g["fg_att"].values, g["w"].values, lg_fga, KICK_K) for t2, g in k.groupby("posteam")}
    t_xpa = {t2: _shrunk(g["xp_att"].values, g["w"].values, lg_xpa, KICK_K) for t2, g in k.groupby("posteam")}
    t_stall = {t2: _shrunk(g["stalls"].values, g["w"].values, lg_stall, KICK_K) for t2, g in k.groupby("posteam")}
    k_fgpct, k_xppct = {}, {}
    for kr, g in k.groupby("kicker_player_name"):
        k_fgpct[kr] = (np.sum(g["fg_made"] * g["w"]) + KICK_KP * lg_fgpct) / (np.sum(g["fg_att"] * g["w"]) + KICK_KP)
        k_xppct[kr] = (np.sum(g["xp_made"] * g["w"]) + KICK_KP * lg_xppct) / (np.sum(g["xp_att"] * g["w"]) + KICK_KP)
    X1 = np.column_stack([np.ones(len(k)), k["posteam"].map(t_fga).fillna(lg_fga),
                          k["posteam"].map(t_stall).fillna(lg_stall), k["team_total"], k["wind_flag"]])
    c1, *_ = np.linalg.lstsq(X1 * sw[:len(k)] if len(sw) == len(k) else X1 * np.sqrt(k["w"].values)[:, None],
                             k["points"].values * 0 + k["fg_att"].values * np.sqrt(k["w"].values), rcond=None)
    swk = np.sqrt(k["w"].values)[:, None]
    c1, *_ = np.linalg.lstsq(X1 * swk, k["fg_att"].values * swk.flatten(), rcond=None)
    X2 = np.column_stack([np.ones(len(k)), k["posteam"].map(t_xpa).fillna(lg_xpa), k["team_total"]])
    c2, *_ = np.linalg.lstsq(X2 * swk, k["xp_att"].values * swk.flatten(), rcond=None)
    st["kk"] = {"lg_pts": lg_pts, "lg_fga": lg_fga, "lg_xpa": lg_xpa, "lg_stall": lg_stall,
                "lg_fgpct": lg_fgpct, "lg_xppct": lg_xppct,
                "t_fga": t_fga, "t_xpa": t_xpa, "t_stall": t_stall,
                "k_fgpct": k_fgpct, "k_xppct": k_xppct, "c1": c1, "c2": c2}
    # in-sample residuals -> empirical distribution for O/U pricing
    preds = []
    for _, g in k.iterrows():
        preds.append(predict_kicker({"kk": st["kk"]}, g["posteam"], g["kicker_player_name"],
                                    g["team_total"], g["wind_flag"]))
    st["kk"]["residuals"] = np.sort(k["points"].values - np.array(preds))
    return st

# ------------------------------------------------------------- predictors
def predict_first_drive(state, team, opp, is_home):
    s = state["fd"]
    ones = {o: 1.0 for o in OUTCOMES}
    o_t, d_t = s["off"].get(team, ones), s["def"].get(opp, ones)
    ps = []
    for r in (0, 1):
        p = np.array([s["base"][r][o] * o_t[o] ** FD_TAU * d_t[o] ** (0.7 * FD_TAU)
                      * (s["home"][o] ** FD_TAU if is_home else 1.0) for o in OUTCOMES])
        ps.append(p / p.sum())
    blend = 0.5 * ps[0] + 0.5 * ps[1]
    return dict(zip(OUTCOMES, blend))

def predict_punts(state, team, opp, total_line, wind_flag):
    s = state["pt"]
    b = s["lg"] * s["team"].get(team, 1) * s["opp"].get(opp, 1) ** 0.7
    x = np.array([1, b, total_line - s["mean_tot"], wind_flag])
    return max(float(x @ s["coef"]), 0.3)

def poisson_probs(mean, line):
    """(p_over, p_under, p_push) for a punts-style count line."""
    kmax = max(int(mean + 10 * np.sqrt(mean)), int(line) + 10)
    pmf = np.array([exp(k * np.log(mean) - mean - lgamma(k + 1)) for k in range(kmax + 1)])
    pmf = pmf / pmf.sum()
    if float(line).is_integer():
        L = int(line)
        return float(pmf[L + 1:].sum()), float(pmf[:L].sum()), float(pmf[L])
    return float(pmf[int(np.ceil(line)):].sum()), float(pmf[:int(np.ceil(line))].sum()), 0.0

def predict_kicker(state, team, kicker, team_total, wind_flag):
    s = state["kk"]
    fga = max(float(np.array([1, s["t_fga"].get(team, s["lg_fga"]),
                              s["t_stall"].get(team, s["lg_stall"]), team_total, wind_flag]) @ s["c1"]), 0)
    xpa = max(float(np.array([1, s["t_xpa"].get(team, s["lg_xpa"]), team_total]) @ s["c2"]), 0)
    fgp = s["k_fgpct"].get(kicker, s["lg_fgpct"])
    xpp = s["k_xppct"].get(kicker, s["lg_xppct"])
    wind_pct = 0.95 if wind_flag else 1.0
    return 3 * fga * fgp * wind_pct + xpa * xpp

def kicker_probs(state, mean, line):
    """(p_over, p_under, p_push) from empirical residual distribution."""
    res = state["kk"]["residuals"]
    outcomes = np.round(mean + res)          # integer point outcomes
    p_over = float(np.mean(outcomes > line))
    p_under = float(np.mean(outcomes < line))
    p_push = float(np.mean(outcomes == line)) if float(line).is_integer() else 0.0
    return p_over, p_under, p_push

# ------------------------------------------------------------------ odds
def american_to_prob(odds):
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)

def devig_two_way(odds_a, odds_b):
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    return pa / (pa + pb), pb / (pa + pb)

def devig_multiway(odds_dict):
    ps = {k: american_to_prob(v) for k, v in odds_dict.items()}
    tot = sum(ps.values())
    return {k: v / tot for k, v in ps.items()}

def classify_edge(model_p, fair_p, market):
    edge = model_p - fair_p
    th = THRESHOLDS[market]
    if edge >= th["BET"]:
        return "BET", edge
    if edge >= th["VALUE"]:
        return "VALUE", edge
    return "PASS", edge

# -------------------------------------------------------------- self-test
if __name__ == "__main__":
    print("b00mstick model self-test (state as of 2026 week 1)...")
    frames = load_data("data")
    state = build_state(frames, 2026, 1)
    fd_p = predict_first_drive(state, "PHI", "DAL", is_home=True)
    print("PHI first drive vs DAL (home):", {k: round(v, 3) for k, v in fd_p.items()})
    dv = devig_multiway({"TD": 260, "FG": 450, "PUNT": 130, "OTHER": 400})
    tier, edge = classify_edge(fd_p["PUNT"], dv["PUNT"], "drive")
    print(f"  sample DK drive line devig PUNT fair={dv['PUNT']:.3f} -> {tier} ({edge:+.3f})")
    mu = predict_punts(state, "PHI", "DAL", total_line=47.5, wind_flag=0)
    po, pu, pp = poisson_probs(mu * 2, 8.5)   # crude game proxy for smoke test
    print(f"PHI expected punts: {mu:.2f}   P(game over 8.5) ~ {po:.3f}")
    some_kicker = frames["kk"][frames["kk"]["season"] == 2025]["kicker_player_name"].mode()[0]
    mk = predict_kicker(state, "PHI", some_kicker, team_total=25.5, wind_flag=0)
    ko, ku, kp = kicker_probs(state, mk, 8.5)
    print(f"Sample kicker ({some_kicker}) expected pts: {mk:.2f}   P(over 8.5)={ko:.3f} push={kp:.3f}")
    print("Self-test complete.")
