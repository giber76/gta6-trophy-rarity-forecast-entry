// gta6-trophy-rarity-forecast-01: a refresh-aware forecaster for GTA VI PS5 trophy
// earn rates on Exophase (Node 22, standard library only).
//
// The page the house reads is frozen between Exophase's refreshes and jumps at
// each refresh: the tracked-player count P jumps, and every trophy's earn rate
// moves. So the history is a step function, and ALL the signal is in the jumps.
// This program reads the jumps and forecasts each trophy's rate at the target
// from three quantities it estimates from the case's own history:
//
//   s_i  the arrival earn rate: the share of the players a refresh newly adds
//        who already hold trophy i. Estimated per trophy by least squares over
//        the refreshes, from dE_i = s_i * dP + g_i * (non-earners) * dt.
//   g_i  the progression hazard: the per-day rate at which the players already
//        tracked, who have not earned trophy i yet, earn it. Same regression.
//   P_T  the players tracked at the target, from a saturating fit
//        P(t) = P_inf - (P_inf - P_start) * exp(-k * (t - t0)) to the distinct
//        player counts (a launch's tracked base grows fast, then slows; a
//        log-linear extrapolation of the launch-night backlog overshoots).
//
// The median rate is then (E0 + (P_T - P0) * s_i + progression) / P_T: a weighted
// average of the current rate and the arrivals' rate, plus what the existing
// base earns by the target. The platinum is capped at the lowest of the other
// trophies' medians, because it needs every other trophy first.
//
// When the history holds no refresh at all (a flat launch night, as in the
// practice cases), s_i and g_i cannot be estimated from data, so the program
// falls back to a prior: it expects the first refresh to multiply the tracked
// players several-fold and the newly added players to hold each trophy at a
// small fraction of the current rate (they are much less progressed than the
// launch-night early adopters). The spread widens with the horizon and when
// the history carries no refresh signal, because both make the median less sure.
//
// Score: pinball loss on the logit scale over five quantiles per trophy.
// The spread is sigma = 0.35 + 0.15 * horizon_days (+0.30 with no refresh
// signal), in logit units.
//
// Reads every <id>.in in the input directory and writes <id>.out to the output
// directory: {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}, one row per
// trophy, each q five finite numbers from 0 to 100, non-decreasing.
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** First-refresh prior: how many times the tracked players multiply when the
 *  page, flat since launch, first refreshes (launch backlog, several-fold). */
const G0 = 5;
/** Arrival-rate prior: a newly added player holds a trophy at this fraction of
 *  the current rate (new players are much less progressed than the early base). */
const F = 0.07;
/** Progression prior (per day) when no refresh can estimate it. */
const G_PRIOR = 0.05;
/** Spread (logit): base + per day of horizon, plus a flat-history penalty. */
const SIGMA_BASE = 0.35;
const SIGMA_PER_DAY = 0.15;
const SIGMA_FLAT = 0.3;
/** P-fit grid: P_inf as a multiple of the last player count, and k per day. */
const P_INF_MULTS = [1.5, 2, 3, 5, 8, 12, 20, 40];
const P_KS = [0.05, 0.1, 0.2, 0.4, 0.7, 1.2, 2.0, 3.5];

interface Trophy { id: number; tier: string }
interface Read { slot: string; at: string; players_tracked: number; trophies_earned: number; rates: Array<number | null> }
interface Case {
  id: string; at: string; horizon_minutes: number; target: string;
  levels: number[]; trophies: Trophy[]; history: Read[];
}

interface ParsedRead { atMs: number; P: number; E: number[] }
interface Refresh { dP: number; dt: number; Pprev: number; Eprev: number[]; Ecur: number[] }

const clamp = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => Math.log(clamp(pct) / (100 - clamp(pct)));
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** History reads as (time, players, earners per trophy). */
function parseReads(c: Case): ParsedRead[] {
  return c.history.map((h) => ({
    atMs: Date.parse(h.at),
    P: h.players_tracked,
    E: h.rates.map((r) => (r === null ? 0 : (r * h.players_tracked) / 100)),
  }));
}

/** One event per refresh: a read whose player count differs from the last
 *  distinct state. dt is the time since that distinct state, the interval the
 *  jump accumulated over (the page is frozen between refreshes). */
function refreshEvents(reads: ParsedRead[]): Refresh[] {
  const ev: Refresh[] = [];
  let prev = reads[0];
  for (let j = 1; j < reads.length; j++) {
    const b = reads[j] as ParsedRead;
    if (b.P !== prev.P) {
      ev.push({
        dP: b.P - prev.P,
        dt: Math.max((b.atMs - prev.atMs) / 86_400_000, 1e-9),
        Pprev: prev.P,
        Eprev: prev.E,
        Ecur: b.E,
      });
      prev = b;
    }
  }
  return ev;
}

/** Least squares for dE_i = s * dP + g * (non-earners) * dt over the refreshes.
 *  Returns null when fewer than two refreshes (unidentifiable). */
function fitSG(ev: Refresh[], i: number): { s: number; g: number } | null {
  if (ev.length < 2) return null;
  let S11 = 0, S12 = 0, S22 = 0, T1 = 0, T2 = 0;
  for (const e of ev) {
    const x1 = e.dP;
    const x2 = Math.max(e.Pprev - e.Eprev[i]!, 0) * e.dt;
    const y = e.Ecur[i]! - e.Eprev[i]!;
    S11 += x1 * x1; S12 += x1 * x2; S22 += x2 * x2; T1 += x1 * y; T2 += x2 * y;
  }
  const det = S11 * S22 - S12 * S12;
  if (Math.abs(det) < 1e-12) return null;
  const s = (T1 * S22 - T2 * S12) / det;
  const g = (S11 * T2 - S12 * T1) / det;
  return { s: Math.min(1, Math.max(0, s)), g: Math.min(0.5, Math.max(0, g)) };
}

/** Saturating fit P(t) = P_inf - (P_inf - P_start) * exp(-k * (t - t0)) to the
 *  distinct player counts; returns P at (last time + horizon days), never
 *  below the last count. Null when too few distinct states to fit. */
function fitP(reads: ParsedRead[], horizonDays: number): number | null {
  const pts: Array<[number, number]> = [];
  for (const r of reads) {
    if (pts.length === 0 || r.P !== pts[pts.length - 1]![1]) pts.push([r.atMs, r.P]);
  }
  if (pts.length < 3) return null;
  const t0 = pts[0]![0];
  const Pstart = pts[0]![1];
  const Plast = pts[pts.length - 1]![1];
  const xs = pts.map(([t]) => (t - t0) / 86_400_000);
  const ys = pts.map(([, p]) => Math.log(p));
  let best: { err: number; Pinf: number; k: number } | null = null;
  for (const mult of P_INF_MULTS) {
    const Pinf = Plast * mult;
    for (const k of P_KS) {
      let err = 0;
      for (let j = 0; j < xs.length; j++) {
        const pred = Pinf - (Pinf - Pstart) * Math.exp(-k * xs[j]!);
        if (pred <= 0) { err = 1e18; break; }
        err += (Math.log(pred) - ys[j]!) ** 2;
      }
      if (best === null || err < best.err) best = { err, Pinf, k };
    }
  }
  if (best === null) return null;
  const tLast = xs[xs.length - 1]!;
  const PT = best.Pinf - (best.Pinf - Pstart) * Math.exp(-best.k * (tLast + horizonDays));
  return Math.max(PT, Plast * 1.02);
}

/** The forecast for one case: {trophy id: [q05, q25, q50, q75, q95]}. */
function forecastCase(c: Case): Record<number, number[]> {
  const reads = parseReads(c);
  if (reads.length === 0) throw new Error(`${c.id} has no history`);
  const horizonDays = c.horizon_minutes / 1440;
  const last = reads[reads.length - 1] as ParsedRead;
  const P0 = last.P;
  const ev = refreshEvents(reads);

  let PT = fitP(reads, horizonDays);
  if (PT === null || PT <= P0) PT = ev.length === 0 ? P0 * G0 : P0 * (1 + 0.5 * horizonDays);
  PT = Math.max(PT, P0);

  const medians: number[] = c.trophies.map((t, i) => {
    const E0 = last.E[i] ?? 0;
    const r0 = (E0 / P0) * 100;
    let s: number;
    let g: number;
    const fg = fitSG(ev, i);
    if (fg !== null) {
      s = fg.s; g = fg.g;
    } else if (r0 >= 100) {
      s = 1; g = 0;
    } else {
      s = (F * r0) / 100; g = G_PRIOR;
    }
    const Enew = Math.max(PT - P0, 0) * s;
    const Eprog = Math.max(P0 - E0, 0) * (1 - Math.exp(-g * horizonDays));
    return (100 * (E0 + Enew + Eprog)) / PT;
  });

  // The platinum needs every other trophy first: it cannot exceed the lowest
  // of their medians.
  const plat = c.trophies.findIndex((t) => t.tier === 'platinum');
  if (plat >= 0) {
    const cap = Math.min(...medians.filter((_, j) => j !== plat));
    if (medians[plat]! > cap) medians[plat] = cap;
  }

  let sigma = SIGMA_BASE + SIGMA_PER_DAY * horizonDays;
  if (ev.length < 2) sigma += SIGMA_FLAT;

  const out: Record<number, number[]> = {};
  c.trophies.forEach((t, i) => {
    const l = logit(medians[i] ?? 0);
    out[t.id] = Z.map((z) => round4(invLogit(l + z * sigma)));
  });
  return out;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  const f = forecastCase(c);
  const trophies = c.trophies.map((t) => ({ id: t.id, q: f[t.id] }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
