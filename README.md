[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/giber76/gta6-trophy-rarity-forecast-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/giber76/gta6-trophy-rarity-forecast-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

A make.fun entry for [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast, for every one of the 55 Grand Theft Auto VI PlayStation 5 trophies, the percent of the players Exophase tracks who will have earned it at 2026-11-26T00:00:00Z, as five quantiles per trophy, scored by pinball loss on the logit scale.

## The model: read the refreshes, not the rates

Exophase's page is frozen between its refreshes and jumps at each refresh: the tracked-player count jumps, and every trophy's earn rate moves. So the history is a step function, and all the signal is in the jumps. For each trophy this program estimates, by least squares over the refreshes:

- **s_i, the arrival earn rate** — the share of the players a refresh newly adds who already hold the trophy;
- **g_i, the progression hazard** — the per-day rate at which the players already tracked, who have not earned the trophy yet, earn it;

from `dE_i = s_i * dP + g_i * (non-earners) * dt`, where `dt` is the time since the last refresh (the interval the jump accumulated over, not the 30-minute read gap).

It also fits the players tracked at the target from a saturating curve `P(t) = P_inf - (P_inf - P_start) * exp(-k * (t - t0))` — a launch's tracked base grows fast, then slows, so a log-linear extrapolation of the launch-night backlog overshoots badly.

The median rate is `(E0 + (P_T - P0) * s_i + progression) / P_T`: a weighted average of the current rate and the arrivals' rate, plus what the existing base earns by the target. The platinum is capped at the lowest of the other trophies' medians, because it needs every other trophy first.

When the history holds no refresh at all (a flat launch night, as in the practice cases), `s_i` and `g_i` cannot be estimated from data, so the program falls back to a prior: it expects the first refresh to multiply the tracked players several-fold and the newly added players to hold each trophy at a small fraction of the current rate (they are much less progressed than the launch-night early adopters). The spread widens with the horizon and when the history carries no refresh signal.

Spread: `sigma = 0.35 + 0.15 * horizon_days` (+0.30 with no refresh signal), in logit units.

## Validation (local, honest)

The kit does not publish the practice answers or the scorer, so the dev score is only known from the server. Locally I validated with:

- **Reconstructed practice answers**: the competition states the 04:30Z read had 3,868 players tracked and 4,276 trophies earned, every rate but the 100% one fell, and the 05:00Z read was unchanged. I rebuilt the per-trophy answers three ways (proportional / top-5-concentrated / uniform split of the new players' earners). The persistence baseline scores 1.39–1.58 on these proxies (the published figure is 1.51, typical miss 0.68pp — my proxies reproduce the 0.68pp exactly), and this program scores **0.21–0.36**. The ranking is stable across all three reconstructions.
- **A synthetic final-like case**: a simulated 2.5-day history with six refreshes and a 4.5-day horizon, ground truth generated with different functional forms than the model's own. Persistence scores 0.87; this program scores **0.29**, and its regression recovers the true arrival rates and progression hazards to within a few percent.

Both entrypoints (`main.ts`, Node 22) produce deterministic output, run in ~0.3 s per case. (`main.py`, a Python port, was removed: the grader runs one entrypoint at the root.)

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
# or: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`. Standard library only, no network, deterministic.
