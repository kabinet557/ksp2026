#!/usr/bin/env python3
"""
detect_outbursts_v5.py
----------------------
Outburst detector for irregular X-ray binary light curves.

v5 change: event detection is now point-level hysteresis thresholding
(two-threshold "seed and grow", as in Canny edge detection) instead of
Bayesian-Blocks-block gating. The previous version segmented each light
curve with Bayesian Blocks and then discarded any whole block whose own
local mean fell under the low threshold *before* the merge step ran, so a
single smoothly declining outburst could be sliced into 2-3 spurious
"outbursts" purely because of where a BB changepoint or a few noisy points
landed for that particular noise draw -- the same underlying signal would
come out differently every run. Hysteresis keeps a contiguous elevated span
together no matter how a segmentation step might have chopped it up, it
lets the whole span's pooled significance (not one lucky block) decide
whether the event is real, and it does not require the O(N^2) Bayesian
Blocks fit inside the bootstrap loop, which made FAP calibration slow.

Keeps the same baseline estimation, bootstrap false-alarm calibration, and
periodicity checks as before.
"""

import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle

warnings.filterwarnings("ignore")

MAXI_SUFFIXES = ("_maxi_lc.csv", "_maxi.csv")
ZTF_SUFFIXES = ("_ztf.csv",)

REBIN_CADENCE_THRESHOLD_DAYS = 0.5
REBIN_TARGET_DAYS = 1.0


# ----------------------------------------------------------------------
# Filename / instrument parsing
# ----------------------------------------------------------------------

def parse_filename(fpath):
    base = os.path.basename(fpath)

    instrument, stem = None, None
    for suf in MAXI_SUFFIXES:
        if base.lower().endswith(suf):
            stem, instrument = base[: -len(suf)], "maxi"
            break
    if instrument is None:
        for suf in ZTF_SUFFIXES:
            if base.lower().endswith(suf):
                stem, instrument = base[: -len(suf)], "ztf"
                break
    if instrument is None:
        stem = os.path.splitext(base)[0]
        instrument = "ztf" if "ztf" in stem.lower() else "maxi"

    m = re.search(r"(J\d{4,6})([pm])(\d{2,4})", stem, flags=re.IGNORECASE)
    if m:
        sign = "+" if m.group(2).lower() == "p" else "-"
        source = stem[: m.start()].rstrip("_") + " " + m.group(1) + sign + m.group(3)
        source = source.strip().replace("_", " ")
    else:
        source = stem.replace("_", " ").strip()

    return source, instrument, stem


# ----------------------------------------------------------------------
# Loading + cleaning light curves
# ----------------------------------------------------------------------

def _get_col(df, *candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _rebin_daily(mjd, val, err):
    day = np.floor(mjd).astype(int)
    out_mjd, out_val, out_err = [], [], []
    for d in np.unique(day):
        sel = day == d
        w = 1.0 / np.clip(err[sel], 1e-8, None) ** 2
        wsum = w.sum()
        out_mjd.append(d + 0.5)
        out_val.append(np.sum(w * val[sel]) / wsum)
        out_err.append(np.sqrt(1.0 / wsum))
    return np.array(out_mjd), np.array(out_val), np.array(out_err)


def _dedupe_exact(mjd, val, err):
    if len(mjd) == len(np.unique(mjd)):
        return mjd, val, err
    df = pd.DataFrame({"mjd": mjd, "val": val, "err": err})
    w = 1.0 / df["err"].clip(lower=1e-8) ** 2
    df["w"] = w
    df["wv"] = w * df["val"]
    g = df.groupby("mjd").agg(wsum=("w", "sum"), wv=("wv", "sum"))
    out_val = g["wv"] / g["wsum"]
    out_err = np.sqrt(1.0 / g["wsum"])
    return g.index.to_numpy(float), out_val.to_numpy(float), out_err.to_numpy(float)


def load_maxi(fpath):
    df = pd.read_csv(fpath)
    mjd_c = _get_col(df, "mjd")
    flux_c = _get_col(df, "flux_2_20")
    err_c = _get_col(df, "error_2_20", "err_2_20")
    if mjd_c is None or flux_c is None or err_c is None:
        raise ValueError(f"Could not find mjd/flux_2_20/error columns in {fpath}")

    d = df[[mjd_c, flux_c, err_c]].copy()
    d.columns = ["mjd", "flux", "err"]
    d = d.dropna()
    d = d[d["err"] > 0]
    d = d.sort_values("mjd")
    mjd, val, err = d["mjd"].to_numpy(float), d["flux"].to_numpy(float), d["err"].to_numpy(float)

    if len(mjd) > 1 and np.median(np.diff(mjd)) < REBIN_CADENCE_THRESHOLD_DAYS:
        mjd, val, err = _rebin_daily(mjd, val, err)
    else:
        mjd, val, err = _dedupe_exact(mjd, val, err)

    return {"2-20keV": (mjd, val, err)}


def load_ztf(fpath):
    df = pd.read_csv(fpath)
    mjd_c = _get_col(df, "mjd")
    mag_c = _get_col(df, "mag")
    magerr_c = _get_col(df, "magerr")
    flag_c = _get_col(df, "catflags")
    filt_c = _get_col(df, "filtercode")
    if None in (mjd_c, mag_c, magerr_c, filt_c):
        raise ValueError(f"Could not find required ZTF columns in {fpath}")

    cols = [mjd_c, mag_c, magerr_c, filt_c] + ([flag_c] if flag_c else [])
    d = df[cols].copy()
    d.columns = ["mjd", "mag", "magerr", "filt"] + (["flag"] if flag_c else [])
    d = d.dropna(subset=["mjd", "mag", "magerr"])
    if "flag" in d.columns:
        d = d[d["flag"] == 0]
    d = d[d["magerr"] > 0]

    out = {}
    for band, sub in d.groupby("filt"):
        sub = sub.sort_values("mjd")
        mjd = sub["mjd"].to_numpy(float)
        mag = sub["mag"].to_numpy(float)
        magerr = sub["magerr"].to_numpy(float)
        if len(mjd) < 10:
            continue
        flux = 10 ** (-0.4 * mag)
        err = flux * np.log(10) * 0.4 * magerr
        mjd, flux, err = _dedupe_exact(mjd, flux, err)
        out[band] = (mjd, flux, err)
    return out


# ----------------------------------------------------------------------
# Robust baseline
# ----------------------------------------------------------------------

def robust_baseline_global(val, err, clip_sigma=3.0, n_iter=5):
    """Iteratively sigma-clipped median + scaled MAD over the full curve."""
    v = val.copy()
    mask = np.ones(len(v), dtype=bool)
    for _ in range(n_iter):
        if mask.sum() < 5:
            break
        med = np.median(v[mask])
        mad = np.median(np.abs(v[mask] - med))
        scatter = max(1.4826 * mad, np.median(err[mask]))
        new_mask = (np.abs(v - med) <= clip_sigma * scatter) | (v < med)
        if new_mask.sum() == mask.sum():
            mask = new_mask
            break
        mask = new_mask
    med = np.median(v[mask])
    mad = np.median(np.abs(v[mask] - med))
    scatter = max(1.4826 * mad, np.median(err[mask]))
    return med, scatter, mask


def robust_baseline_local(mjd, val, err, window_days, clip_sigma=3.0, n_iter=5, min_local_n=15):
    """
    Local baseline/scatter from a sliding window, with fallback to global.
    """
    n = len(mjd)
    _, _, global_mask = robust_baseline_global(val, err, clip_sigma, n_iter)

    base_val_local = np.empty(n)
    base_sigma_local = np.empty(n)
    half = window_days / 2.0

    gv = np.median(val[global_mask])
    gmad = np.median(np.abs(val[global_mask] - gv))
    gs = max(1.4826 * gmad, np.median(err[global_mask]))

    order = np.argsort(mjd)
    mjd_sorted = mjd[order]
    for idx in range(n):
        i = order[idx]
        lo, hi = mjd[i] - half, mjd[i] + half
        lo_idx = np.searchsorted(mjd_sorted, lo, side="left")
        hi_idx = np.searchsorted(mjd_sorted, hi, side="right")
        window_idx = order[lo_idx:hi_idx]
        sel = global_mask[window_idx]
        if sel.sum() >= min_local_n:
            lv = val[window_idx][sel]
            le = err[window_idx][sel]
            med = np.median(lv)
            mad = np.median(np.abs(lv - med))
            scatter = max(1.4826 * mad, np.median(le))
            base_val_local[i] = med
            base_sigma_local[i] = scatter
        else:
            base_val_local[i] = gv
            base_sigma_local[i] = gs

    return base_val_local, base_sigma_local, global_mask


# ----------------------------------------------------------------------
# Bayesian Blocks event finder
# ----------------------------------------------------------------------

def _find_true_runs(mask):
    runs = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _merge_runs_by_gap(runs, mjd, min_separation_days):
    if not runs:
        return runs
    merged = [list(runs[0])]
    for i0, i1 in runs[1:]:
        if mjd[i0] - mjd[merged[-1][1]] < min_separation_days:
            merged[-1][1] = i1
        else:
            merged.append([i0, i1])
    return [tuple(r) for r in merged]


def _weighted_mean_and_err(x, err):
    w = 1.0 / np.clip(err, 1e-8, None) ** 2
    wsum = w.sum()
    if wsum <= 0:
        return float(np.mean(x)), float(np.std(x, ddof=1) if len(x) > 1 else 0.0)
    mean = np.sum(w * x) / wsum
    mean_err = np.sqrt(1.0 / wsum)
    return float(mean), float(mean_err)


def _connected_runs_with_gap_limit(mjd, active_mask, max_gap_days):
    """
    Group indices where `active_mask` is True into maximal runs, breaking a
    run only when the *time* gap between two consecutive qualifying points
    exceeds max_gap_days. Points that fail active_mask are simply skipped
    over (not treated as a break) as long as the next qualifying point isn't
    too far away in time -- a single noisy point briefly dipping under the
    floor should not fragment an otherwise continuous outburst, the same
    way a short run of genuinely missing data shouldn't either.
    """
    idx = np.where(active_mask)[0]
    if idx.size == 0:
        return []
    runs = []
    current = [int(idx[0])]
    for i in idx[1:]:
        if mjd[i] - mjd[current[-1]] > max_gap_days:
            runs.append(current)
            current = [int(i)]
        else:
            current.append(int(i))
    runs.append(current)
    return runs


def detect_outbursts_blocks(
    mjd,
    val,
    err,
    base_val,
    base_sigma,
    bb_p0=0.01,           # unused, kept only for CLI/back-compat
    nsigma_enter=5.0,
    nsigma_exit=2.5,
    min_ratio=1.5,
    min_points=4,
    min_duration_days=1.0,
    cadence_duration_factor=3.0,
    min_separation_days=2.0,
    require_peak_robust=True,
    min_seed_points=2,
    max_gap_factor=5.0,
):
    """
    Detect outbursts using point-level hysteresis thresholding (as in Canny
    edge detection): a point qualifies as a hysteresis "seed" if it clears
    the high bar (nsigma_enter). Starting from any seed, the candidate event
    is grown outward, in time order, through neighboring points as long as
    they clear the low bar (nsigma_exit) and are not separated from their
    neighbor by a real data gap wider than max_gap_factor x local cadence.

    This replaces an earlier Bayesian-Blocks-based version. BB re-segments
    the curve differently for every noise realization, and blocks whose own
    *local* mean dipped under nsigma_exit were discarded outright before the
    merge step ever saw them -- so a single smoothly decaying outburst could
    be sliced into 2-3 spurious "outbursts" (or have its rise clipped off)
    purely because of where a few noisy points happened to fall, with no
    change to the underlying signal. Hysteresis on individual points is
    immune to that failure mode: contiguous elevated points always stay
    together in one span regardless of how a segmentation algorithm might
    have chosen to chop them up, and it is also ~two orders of magnitude
    cheaper than re-running Bayesian Blocks (an O(N^2) fit) in every
    bootstrap draw.
    """
    n = len(mjd)
    if n == 0:
        return []

    base_val_arr = np.broadcast_to(base_val, n).astype(float)
    base_sigma_arr = np.broadcast_to(base_sigma, n).astype(float)
    if np.any(base_sigma_arr <= 0):
        return []

    order = np.argsort(mjd)
    mjd = mjd[order]
    val = val[order]
    err = err[order]
    base_val_arr = base_val_arr[order]
    base_sigma_arr = base_sigma_arr[order]

    cadence = float(np.median(np.diff(mjd))) if n > 1 else 1.0
    min_dur = max(min_duration_days, cadence_duration_factor * cadence)
    max_gap_days = max(max_gap_factor * cadence, 2.0 * min_separation_days)

    point_z = (val - base_val_arr) / base_sigma_arr
    amp_ok = np.where(
        base_val_arr > 0,
        val > min_ratio * base_val_arr,
        (val - base_val_arr) > min_ratio * base_sigma_arr,
    )

    seed_mask = (point_z > nsigma_enter) & amp_ok       # high bar
    weak_mask = (point_z > nsigma_exit) & amp_ok         # low bar (hysteresis floor)

    # Grow every weak-threshold run that contains at least min_seed_points
    # seed points into a candidate span.
    runs = _connected_runs_with_gap_limit(mjd, weak_mask, max_gap_days)
    candidates = [r for r in runs if seed_mask[r].sum() >= min_seed_points]

    # Merge candidate spans that are still close together (e.g. a brief real
    # dip back toward baseline that doesn't fully reset between two flares).
    candidates = sorted(candidates, key=lambda idx: mjd[idx[0]])
    merged = []
    for idx in candidates:
        if merged and (mjd[idx[0]] - mjd[merged[-1][-1]]) < min_separation_days:
            merged[-1] = merged[-1] + idx
        else:
            merged.append(list(idx))

    results = []
    for idx in merged:
        idx = np.array(sorted(idx))
        if idx.size < min_points:
            continue

        block_val = val[idx]
        block_err = err[idx]
        block_base = base_val_arr[idx]
        block_base_sigma = base_sigma_arr[idx]
        block_point_z = point_z[idx]

        mjd_start, mjd_end = float(mjd[idx[0]]), float(mjd[idx[-1]])
        duration = mjd_end - mjd_start
        if duration < min_dur:
            continue

        mean_val, mean_err = _weighted_mean_and_err(block_val, block_err)
        mean_base = float(np.mean(block_base))
        mean_base_sigma = float(np.median(block_base_sigma))
        # Standard error of the pooled mean (shrinks as ~1/sqrt(N), as it
        # should for a span with more points) combined in quadrature with
        # the baseline's own uncertainty, which is a shared systematic for
        # the whole span and does NOT shrink with N. A single point's error
        # is deliberately *not* used as a floor here -- that would defeat
        # the point of pooling multiple points into one significance test.
        pooled_sigma = float(np.hypot(mean_err, mean_base_sigma))
        block_z = (mean_val - mean_base) / pooled_sigma

        # Stricter than before in the way that matters: the old code only
        # required ONE point above nsigma_enter to seed a whole block. Here
        # a span must already contain >= min_seed_points independent points
        # clearing the high bar (enforced when building `merged`, re-checked
        # here for safety), and the pooled span mean must clear the low bar.
        # Requiring the *pooled mean* to also clear the high bar would
        # incorrectly penalize short-but-real events: pooled significance is
        # capped by baseline uncertainty (a fixed systematic) and does not
        # grow much with just a handful of points.
        if block_z <= nsigma_exit:
            continue
        if (block_point_z > nsigma_enter).sum() < min_seed_points:
            continue

        peak_local = int(np.argmax(block_val))
        if require_peak_robust and idx.size > 1:
            trimmed = np.delete(block_point_z, peak_local)
            if trimmed.size == 0 or not np.any(trimmed > nsigma_exit):
                continue

        results.append(
            dict(
                mjd_start=mjd_start,
                mjd_end=mjd_end,
                duration_days=float(duration),
                n_points=int(idx.size),
                peak_value=float(block_val[peak_local]),
                peak_mjd=float(mjd[idx[peak_local]]),
                baseline_value=mean_base,
                significance_sigma=float(np.max(block_point_z)),
                mean_significance_sigma=float(np.mean(block_point_z)),
                block_z=float(block_z),
            )
        )

    return results


# ----------------------------------------------------------------------
# Monte Carlo / moving-block bootstrap false-alarm calibration
# ----------------------------------------------------------------------

def monte_carlo_fap(
    mjd,
    val,
    err,
    base_val,
    base_sigma,
    baseline_mask,
    detect_kwargs,
    n_boot=150,
    seed=42,
    block_length=None,
):
    """
    Resample quiescent residuals in contiguous chunks, rerun the same detector,
    and use the synthetic nulls to estimate the false-alarm probability.
    """
    rng = np.random.default_rng(seed)
    base_val_arr = np.broadcast_to(base_val, len(mjd)).astype(float)

    quiescent_resid = (val - base_val_arr)[baseline_mask]
    n_q = quiescent_resid.size
    if n_q < 10:
        return None

    n = len(mjd)
    if block_length is None:
        block_length = max(3, int(detect_kwargs.get("min_points", 4)))
    block_length = min(block_length, n_q)

    max_sig_boot = np.zeros(n_boot, dtype=float)
    n_outbursts_boot = np.zeros(n_boot, dtype=int)

    for k in range(n_boot):
        pieces, total = [], 0
        while total < n:
            start = rng.integers(0, n_q - block_length + 1)
            pieces.append(quiescent_resid[start : start + block_length])
            total += block_length
        resid_draw = np.concatenate(pieces)[:n]
        val_null = base_val_arr + resid_draw
        obs_null = detect_outbursts_blocks(mjd, val_null, err, base_val, base_sigma, **detect_kwargs)
        n_outbursts_boot[k] = len(obs_null)
        max_sig_boot[k] = max((o["significance_sigma"] for o in obs_null), default=0.0)

    return dict(max_sig_boot=max_sig_boot, n_outbursts_boot=n_outbursts_boot)


def empirical_fap(sig, mc_result):
    if mc_result is None:
        return None
    return float(np.mean(mc_result["max_sig_boot"] >= sig))


# ----------------------------------------------------------------------
# Periodicity
# ----------------------------------------------------------------------

def flag_periodicity(
    outbursts,
    mjd,
    val,
    err,
    min_n=5,
    cv_threshold=0.30,
    fap_threshold=0.01,
    edge_guard_frac=0.02,
):
    """
    Identify roughly periodic recurrence in the outburst start times.
    """
    if len(outbursts) < min_n:
        return False, None, None, None

    starts = np.array(sorted(o["mjd_start"] for o in outbursts))
    gaps = np.diff(starts)
    spacing_cv = float(np.std(gaps) / np.mean(gaps)) if np.mean(gaps) > 0 and len(gaps) > 1 else None

    best_period, best_fap = None, None
    try:
        span = float(mjd.max() - mjd.min())
        cadence = float(np.median(np.diff(np.sort(mjd)))) if len(mjd) > 1 else 1.0
        min_period = max(2.0, 5.0 * cadence)
        max_period = span / 2.0

        if max_period > min_period:
            ls = LombScargle(mjd, val, err)
            freq, power = ls.autopower(
                minimum_frequency=1.0 / max_period,
                maximum_frequency=1.0 / min_period,
            )
            i = int(np.argmax(power))
            if len(freq) > 1:
                frac = (freq[i] - freq.min()) / (freq.max() - freq.min())
            else:
                frac = 0.5
            if edge_guard_frac < frac < (1 - edge_guard_frac):
                best_period = float(1.0 / freq[i])
                best_fap = float(ls.false_alarm_probability(power[i], method="baluev"))
    except Exception:
        pass

    spacing_regular = spacing_cv is not None and spacing_cv < cv_threshold
    ls_matches = (
        best_period is not None
        and best_fap is not None
        and best_fap < fap_threshold
        and len(gaps) > 0
        and abs(best_period - np.median(gaps)) / np.median(gaps) < 0.25
    )
    return bool(spacing_regular or ls_matches), best_period, best_fap, spacing_cv


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def process_file(fpath, args):
    source, instrument, stem = parse_filename(fpath)
    rows = []
    try:
        bands = load_maxi(fpath) if instrument == "maxi" else load_ztf(fpath)
    except Exception as exc:
        print(f"  [skip] {os.path.basename(fpath)}: {exc}")
        return rows

    detect_kwargs = dict(
        nsigma_enter=args.nsigma_enter,
        nsigma_exit=args.nsigma_exit,
        min_ratio=args.min_ratio,
        min_points=args.min_points,
        min_duration_days=args.min_duration_days,
        cadence_duration_factor=args.cadence_duration_factor,
        min_separation_days=args.min_separation_days,
        require_peak_robust=not args.no_peak_robust_check,
        min_seed_points=args.min_seed_points,
        max_gap_factor=args.max_gap_factor,
    )

    for band, (mjd, val, err) in bands.items():
        if len(mjd) < max(10, 2 * args.min_points):
            continue

        if args.rolling_baseline_days:
            base_val, base_sigma, mask = robust_baseline_local(
                mjd, val, err, window_days=args.rolling_baseline_days
            )
        else:
            base_val, base_sigma, mask = robust_baseline_global(val, err)

        outbursts = detect_outbursts_blocks(mjd, val, err, base_val, base_sigma, **detect_kwargs)

        mc_result = None
        if not args.skip_fap and outbursts:
            mc_result = monte_carlo_fap(
                mjd, val, err, base_val, base_sigma, mask, detect_kwargs, n_boot=args.n_bootstrap
            )

        candidate_periodic, best_period, best_fap_period, spacing_cv = flag_periodicity(
            outbursts,
            mjd,
            val,
            err,
            fap_threshold=args.fap_threshold,
        )

        for i, ob in enumerate(outbursts, start=1):
            fap = empirical_fap(ob["significance_sigma"], mc_result) if mc_result else None
            rows.append(
                dict(
                    source=source,
                    file_stem=stem,
                    instrument=instrument,
                    band=band,
                    outburst_id=i,
                    mjd_start=round(ob["mjd_start"], 4),
                    mjd_end=round(ob["mjd_end"], 4),
                    duration_days=round(ob["duration_days"], 4),
                    n_points=ob["n_points"],
                    peak_value=round(ob["peak_value"], 6),
                    peak_mjd=round(ob["peak_mjd"], 4),
                    baseline_value=round(ob["baseline_value"], 6),
                    significance_sigma=round(ob["significance_sigma"], 2),
                    fap_empirical=round(fap, 4) if fap is not None else None,
                    candidate_periodic=candidate_periodic,
                    best_period_days=round(best_period, 3) if best_period else None,
                    period_fap=f"{best_fap_period:.2e}" if best_fap_period else None,
                    spacing_cv=round(spacing_cv, 3) if spacing_cv else None,
                )
            )
    return rows


def main():
    ap = argparse.ArgumentParser(description="Detect XRB outbursts (Bayesian Blocks version).")
    ap.add_argument("--data-dir", default="Total_Data")
    ap.add_argument("--out", default="outbursts_v4.csv")

    ap.add_argument("--nsigma-enter", type=float, default=5.0,
                    help="Sigma above baseline a point must clear to seed an outburst (hysteresis high bar).")
    ap.add_argument("--nsigma-exit", type=float, default=2.5,
                    help="Lower sigma bar a point needs to clear to stay part of an already-seeded outburst "
                         "(hysteresis low bar / floor).")
    ap.add_argument("--min-ratio", type=float, default=1.5)
    ap.add_argument("--min-points", type=int, default=4,
                    help="Minimum data points required in an outburst span.")
    ap.add_argument("--min-seed-points", type=int, default=2,
                    help="Minimum number of independent points above nsigma-enter required within a span "
                         "before it is considered a real outburst rather than a single-point noise spike.")
    ap.add_argument("--min-duration-days", type=float, default=1.0)
    ap.add_argument("--cadence-duration-factor", type=float, default=3.0,
                    help="Event duration must also be >= this many x the light curve's median cadence.")
    ap.add_argument("--max-gap-factor", type=float, default=5.0,
                    help="Hysteresis growth will not bridge a real data gap wider than this many x the "
                         "light curve's median cadence (stops a span from jumping across missing data).")
    ap.add_argument("--min-separation-days", type=float, default=2.0,
                    help="Candidate spans closer together than this (after gap-limited growth) are merged "
                         "into one outburst.")
    ap.add_argument("--no-peak-robust-check", action="store_true",
                    help="Disable the single-outlier-point robustness check.")
    ap.add_argument("--rolling-baseline-days", type=float, default=None,
                    help="Use a local rolling baseline window (days) instead of one global baseline.")
    ap.add_argument("--n-bootstrap", type=int, default=30,
                    help="Number of Monte Carlo null realizations per light curve.")
    ap.add_argument("--skip-fap", action="store_true",
                    help="Skip Monte Carlo FAP calibration.")
    ap.add_argument("--fap-threshold", type=float, default=0.05,
                    help="Periodicity false-alarm threshold.")

    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if not files:
        print(f"No CSV files found in {args.data_dir}")
        sys.exit(1)

    print(f"Found {len(files)} light curve files in '{args.data_dir}'.\n")

    all_rows = []
    for i, fpath in enumerate(files, 1):
        source, instrument, stem = parse_filename(fpath)
        print(f"[{i}/{len(files)}] {os.path.basename(fpath)}  ->  source='{source}', instrument={instrument}")
        rows = process_file(fpath, args)
        n_periodic = sum(1 for r in rows if r["candidate_periodic"])
        flag = f" ({n_periodic} flagged candidate_periodic)" if n_periodic else ""
        print(f"    found {len(rows)} outburst(s){flag}")
        all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows, columns=[
        "source", "file_stem", "instrument", "band", "outburst_id",
        "mjd_start", "mjd_end", "duration_days", "n_points",
        "peak_value", "peak_mjd", "baseline_value", "significance_sigma",
        "fap_empirical", "candidate_periodic", "best_period_days",
        "period_fap", "spacing_cv",
    ])
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {len(out_df)} outburst rows across {len(files)} files -> {args.out}")
    if not args.skip_fap:
        print(f"Tip: inspect fap_empirical before filtering. A common cut is fap_empirical < {args.fap_threshold}.")


if __name__ == "__main__":
    main()