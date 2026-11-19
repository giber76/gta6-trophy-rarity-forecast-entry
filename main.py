"""gta6-trophy-rarity-forecast-01: a refresh-aware forecaster for GTA VI PS5 trophy
earn rates on Exophase (Python 3.13, standard library only).

The same program as main.ts, number for number. See main.ts for the full
description of the model: the history is a step function (frozen between
Exophase's refreshes), so all the signal is in the jumps. Per trophy it
estimates, by least squares over the refreshes, the arrival earn rate s_i and
the progression hazard g_i from dE_i = s_i * dP + g_i * (non-earners) * dt, and
the players tracked at the target from a saturating fit. With no refresh in
the history it falls back to a prior (first refresh multiplies the tracked
players several-fold; new players hold each trophy at a small fraction of the
current rate). The platinum is capped at the lowest of the other trophies'
medians. Spread sigma = 0.35 + 0.15 * horizon_days (+0.30 with no refresh
signal), in logit units.

Reads every <id>.in in the input directory and writes <id>.out to the output
directory: {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.
Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
G0 = 5
F = 0.07
G_PRIOR = 0.05
SIGMA_BASE = 0.35
SIGMA_PER_DAY = 0.15
SIGMA_FLAT = 0.3
P_INF_MULTS = [1.5, 2, 3, 5, 8, 12, 20, 40]
P_KS = [0.05, 0.1, 0.2, 0.4, 0.7, 1.2, 2.0, 3.5]


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(l):
    return 100 / (1 + math.exp(-l))


def round4(x):
    return round(x * 10000) / 10000


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def parse_reads(case):
    reads = []
    for h in case["history"]:
        P = h["players_tracked"]
        E = [(r * P / 100.0) if r is not None else 0.0 for r in h["rates"]]
        reads.append({"at": parse_time(h["at"]), "P": P, "E": E})
    return reads


def refresh_events(reads):
    """One event per refresh; dt is the time since the last distinct state."""
    ev = []
    prev = reads[0]
    for b in reads[1:]:
        if b["P"] != prev["P"]:
            ev.append({
                "dP": b["P"] - prev["P"],
                "dt": max((b["at"] - prev["at"]) / 86400.0, 1e-9),
                "Pprev": prev["P"], "Eprev": prev["E"], "Ecur": b["E"],
            })
            prev = b
    return ev


def fit_s_g(ev, i):
    """Least squares for dE_i = s*dP + g*(non-earners)*dt. None if <2 refreshes."""
    if len(ev) < 2:
        return None
    S11 = S12 = S22 = T1 = T2 = 0.0
    for e in ev:
        x1 = float(e["dP"])
        x2 = max(e["Pprev"] - e["Eprev"][i], 0.0) * e["dt"]
        y = e["Ecur"][i] - e["Eprev"][i]
        S11 += x1 * x1; S12 += x1 * x2; S22 += x2 * x2; T1 += x1 * y; T2 += x2 * y
    det = S11 * S22 - S12 * S12
    if abs(det) < 1e-12:
        return None
    s = (T1 * S22 - T2 * S12) / det
    g = (S11 * T2 - S12 * T1) / det
    return {"s": min(1.0, max(0.0, s)), "g": min(0.5, max(0.0, g))}


def fit_P(reads, horizon_days):
    """Saturating P(t) fit; P at (last time + horizon), never below last count."""
    pts = []
    for r in reads:
        if not pts or r["P"] != pts[-1][1]:
            pts.append((r["at"], r["P"]))
    if len(pts) < 3:
        return None
    t0 = pts[0][0]
    P_start = pts[0][1]
    P_last = pts[-1][1]
    xs = [(t - t0) / 86400.0 for t, _ in pts]
    ys = [math.log(p) for _, p in pts]
    best = None
    for mult in P_INF_MULTS:
        P_inf = P_last * mult
        for k in P_KS:
            err = 0.0
            for x, y in zip(xs, ys):
                pred = P_inf - (P_inf - P_start) * math.exp(-k * x)
                if pred <= 0:
                    err = 1e18
                    break
                err += (math.log(pred) - y) ** 2
            if best is None or err < best[0]:
                best = (err, P_inf, k)
    if best is None:
        return None
    _, P_inf, k = best
    t_last = xs[-1]
    P_T = P_inf - (P_inf - P_start) * math.exp(-k * (t_last + horizon_days))
    return max(P_T, P_last * 1.02)


def forecast_case(case):
    reads = parse_reads(case)
    if not reads:
        raise ValueError(f"{case['id']} has no history")
    horizon_days = case["horizon_minutes"] / 1440
    last = reads[-1]
    P0 = last["P"]
    ev = refresh_events(reads)

    P_T = fit_P(reads, horizon_days)
    if P_T is None or P_T <= P0:
        P_T = P0 * G0 if not ev else P0 * (1 + 0.5 * horizon_days)
    P_T = max(P_T, P0)

    medians = []
    for i, t in enumerate(case["trophies"]):
        E0 = last["E"][i]
        r0 = E0 / P0 * 100.0
        fg = fit_s_g(ev, i)
        if fg is not None:
            s, g = fg["s"], fg["g"]
        elif r0 >= 100:
            s, g = 1.0, 0.0
        else:
            s, g = F * r0 / 100.0, G_PRIOR
        E_new = max(P_T - P0, 0.0) * s
        E_prog = max(P0 - E0, 0.0) * (1 - math.exp(-g * horizon_days))
        medians.append(100.0 * (E0 + E_new + E_prog) / P_T)

    plat = next((i for i, t in enumerate(case["trophies"]) if t["tier"] == "platinum"), None)
    if plat is not None:
        cap = min(m for j, m in enumerate(medians) if j != plat)
        if medians[plat] > cap:
            medians[plat] = cap

    sigma = SIGMA_BASE + SIGMA_PER_DAY * horizon_days
    if len(ev) < 2:
        sigma += SIGMA_FLAT

    out = {}
    for i, t in enumerate(case["trophies"]):
        l = logit(medians[i])
        out[t["id"]] = [round4(inv_logit(l + z * sigma)) for z in Z]
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        cid = fname[: -len(".in")]
        with open(os.path.join(INPUT_DIR, fname), encoding="utf8") as fh:
            case = json.load(fh)
        if len(case["levels"]) != len(LEVELS) or any(
            t != LEVELS[j] for j, t in enumerate(case["levels"])
        ):
            raise ValueError(f"{fname} asks for levels {case['levels']}, not {LEVELS}")
        f = forecast_case(case)
        trophies = [{"id": t["id"], "q": f[t["id"]]} for t in case["trophies"]]
        with open(os.path.join(OUTPUT_DIR, f"{cid}.out"), "w", encoding="utf8") as fh:
            fh.write(json.dumps({"trophies": trophies}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
