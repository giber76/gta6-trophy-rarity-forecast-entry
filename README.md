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

It also fits the players tracked at the target from a saturating curve `P(t) = P_inf - (P_inf - P_start) * exp(-k * (t - t0))` — a launch's tracked base grows fast, then slows, so a log-linear extrapolation of the launch-night backlog overshoots badly. The fitted target count is scaled up 2x: the saturating fit systematically underestimates it on non-exponential launch curves (35–75% low across synthetic growth forms), which pulls medians toward the current rate.

The median rate is `(E0 + (P_T - P0) * s_i + progression) / P_T`: a weighted average of the current rate and the arrivals' rate, plus what the existing base earns by the target. The platinum is capped at the lowest of the other trophies' medians, because it needs every other trophy first.

When the history holds no refresh at all (a flat launch night, as in the practice cases), `s_i` and `g_i` cannot be estimated from data, so the program falls back to a prior. v1 used a flat fraction (new players hold a trophy at 7% of its current rate). v2 uses a **convex** prior, `s_i = C * (r0_i/100)^4`: the published practice aggregates (persistence loss 1.51, typical miss 0.68pp, and v1's own dev loss 0.298) pin the real answer down — the 04:30Z refresh added 3,230 players but only 103 trophy-earners besides the one trophy every player holds, and those 103 are concentrated on the highest-rate (earliest) trophies roughly like E^4.6, not spread proportionally. So high-rate trophies are protected by new earners while rare ones fall by nearly full dilution. The first-refresh multiplier prior is 6x (the observed 638 → 3,868 jump), and the progression prior is 0.02/day. The spread is `sigma = 0.25 + 0.15 * horizon_days` in logit units — tighter than v1's, earned by the better median.

## Validation (local, honest)

The kit does not publish the practice answers or the scorer, so the dev score is only known from the server. Locally I validated with:

- **Reconstructed practice answers**: the competition states the 04:30Z read had 3,868 players tracked and 4,276 trophies earned, every rate but the 100% one fell, and the 05:00Z read was unchanged. I rebuilt the per-trophy answers across a family of splits of the 103 new earners (proportional to E^p for p = 0…6, plus uniform, top-1, tier-based and random splits). The persistence baseline scores 1.39–1.62 on these proxies (the published figure is 1.51, typical miss 0.68pp — the proxies reproduce the 0.68pp exactly), and the best-estimate truth (p ≈ 4.6, which reproduces all three published numbers) gives v1 a loss of 0.296 — matching v1's real dev loss of 0.298, which validates the reconstruction. On the same proxies **v2 scores 0.063–0.108, and 0.082 on the best-estimate truth** — below the top-50 cutoff of 0.114, where v1 scored 0.298. That is a local estimate against reconstructed answers, not a guarantee of the server score.
- **Two synthetic final-like cases** (2.5-day history, 4.5-day horizon, ground truth from different functional forms than the model's own — logistic player growth, exponential-decay arrival rates, heterogeneous hazards): v1 scores 0.304 and 1.135, **v2 scores 0.211 and 0.651** (persistence 0.289 and 0.271). v2 improves on both; on the adversarial anchored-logistic case (growth inflects after the checkpoint, so the player count is unforecastable from the history alone) it still trails persistence, which is an honest limit of the approach.

`main.ts` (Node 22) produces deterministic output, runs in ~0.2 s per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`. Standard library only, no network, deterministic.
