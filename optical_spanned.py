"""
simulate_optical_reprocessing.py
=================================

Forward-simulates the optical (ZTF/ATLAS) light curve of an X-ray binary
outburst from its observed X-ray (MAXI) light curve, using the classical
disc-reprocessing shape law:

    optical_flux = xray_flux ** beta   (dimensionless shape, unit amplitude)

beta is estimated once per compact-object class (BH/NS) from quasi-simultaneous
MAXI/ZTF pairs using a *within-source* (fixed-effects) weighted log-log fit:
each source's own log-log points are mean-centered before pooling, so that
source-to-source differences in distance/extinction/intrinsic brightness
cannot bias the pooled slope -- only the common flux-flux relation survives
the pooling step. This replaces an earlier version that fit a free baseline
and amplitude jointly with beta (highly degenerate, and prone to reporting
a beta so small that noise near the MAXI detection floor got amplified into
a near-binary on/off signal in the forward-simulated curve).

This script supports both ZTF and ATLAS optical input light curves.
"""

import os
import re
import sys
import glob
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path imports & setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fred_spanned as fm  # noqa: E402

SHORTLISTED_DIR = fm.SHORTLISTED_DIR
FRED_DIR        = fm.FRED_DIR
PLOT_DIR        = fm.PLOT_DIR

CLASSIFICATION_CSV = r"C:\Users\abhin\Documents\ksp2026\classification_output\classification_results.csv"

SUMMARY_CSV = os.path.join(FRED_DIR, "norris_tophat_outburst_summary.csv")
XRAY_BAND   = "2-20 keV"   # main MAXI energy band

BETA_LOG_PATH   = os.path.join(FRED_DIR, "optical_reprocessing_beta_fit.log")
BETA_CSV_PATH   = os.path.join(FRED_DIR, "optical_reprocessing_beta_fit.csv")

BETA_BASELINE      = 0.5   # fallback reprocessing exponent (van Paradijs & McClintock 1994)
MATCH_TOL_DAYS     = 1.0   # quasi-simultaneous tolerance window (days)
MIN_PAIRS_FOR_FIT  = 5     # minimum *centered* pairs required for a class fit
MIN_PAIRS_PER_SOURCE = 2   # a source needs >=2 quasi-simultaneous pairs to
                            # contribute to the within-source (fixed-effects) fit;
                            # single-pair sources carry no within-source slope info
MIN_XRAY_SNR       = 3.0   # Minimum SNR for MAXI detection (FX / FX_err >= 3.0),
                            # used BOTH when collecting fit pairs and when
                            # forward-simulating (see simulate_outburst_optical)
CLIP_SIGMA         = 3.0   # one-pass sigma-clip on log-residuals before the
                            # final slope fit, to limit the influence of a
                            # handful of mismatched/misclassified sources
BETA_FIT_PHASE     = "rise"  # "all" | "rise" | "decay" -- restrict the beta
                            # fit to the rise phase of each outburst as a
                            # cheap hard-state proxy (reprocessing is
                            # expected to hold mainly in the hard state;
                            # pooling the whole span, including the soft-
                            # state-heavy decay/tail, dilutes a real slope
                            # toward zero -- see fit_beta_fixed_effects results)
SNR_TAPER_WIDTH    = 1.5   # width (in SNR units) of the smooth roll-off used
                            # in simulate_outburst_optical, replacing a hard
                            # step at MIN_XRAY_SNR with a logistic taper so
                            # the forward-simulated curve doesn't jump
                            # vertically every time a point crosses the floor
BASELINE_PERCENTILE = 10.0 # percentile of a source's FULL optical light curve
                            # used as its quiescent baseline (approximates the
                            # non-varying companion-star continuum), subtracted
                            # off before fitting/simulating the reprocessing
                            # excess -- see collect_quasi_simultaneous_pairs
MAX_ABS_BETA       = 3.0   # sanity ceiling on |beta|. A physically-motivated
                            # reprocessing index is expected to sit roughly in
                            # 0-1 (rarely up to ~2); a fit landing outside
                            # +/-3 almost always means too few sources/pairs
                            # left the weighted-slope estimator ill-conditioned
                            # (tiny denominator -> huge, meaningless slope)
                            # rather than a real measurement -- treat as a
                            # failed fit and fall back to BETA_BASELINE
VIEW_PAD_DAYS       = fm.VIEW_PAD_DAYS

# MAXI count conversion factor (erg/cm^2/s per ph/s/cm^2)
MAXI_CONV = 2.4e-8 / 3.3


# ---------------------------------------------------------------------------
# Pre-Simulation Cleanup
# ---------------------------------------------------------------------------
def cleanup_previous_simulation_files():
    """
    Deletes previously generated fit logs, summary CSVs, simulation output CSVs,
    and output plot images before running a new simulation pass.
    """
    print("\n[0] Cleaning up previous simulation files...")
    for path in [BETA_LOG_PATH, BETA_CSV_PATH]:
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"  [deleted] {os.path.basename(path)}")
            except Exception as e:
                print(f"  [WARN] Failed to delete {path}: {e}")

    if os.path.exists(PLOT_DIR):
        patterns = ["*_optical_simulation.csv", "*_optical_predicted.png", "*_optical_comparison.png"]
        for pattern in patterns:
            for fp in glob.glob(os.path.join(PLOT_DIR, "*", pattern)):
                try:
                    os.remove(fp)
                    print(f"  [deleted] {os.path.basename(fp)}")
                except Exception as e:
                    print(f"  [WARN] Failed to delete {fp}: {e}")


# ---------------------------------------------------------------------------
# Unit conversions & helper functions
# ---------------------------------------------------------------------------
def mag_to_flux(mag, mag_err):
    """Convert AB magnitude to flux density in mJy."""
    flux = 3631.0 * (10.0 ** (-0.4 * mag)) * 1000.0  # mJy
    flux_err = flux * (mag_err / 1.0857)
    return flux, flux_err


def _find_energy_band_cols(df, band_tag="2_20"):
    cols = {c.lower(): c for c in df.columns}
    flux_col = next((orig for lc, orig in cols.items() if "flux" in lc and band_tag in lc), None)
    err_col = next((orig for lc, orig in cols.items() if "err" in lc and band_tag in lc), None)
    return flux_col, err_col


def standardize_maxi_columns(df, band_tag="2_20"):
    df = df.copy()
    flux_col, err_col = _find_energy_band_cols(df, band_tag)
    if flux_col is None:
        raise KeyError(f"Could not find a 2-20 keV flux column in MAXI file. Columns: {df.columns.tolist()}")
    if err_col is None:
        raise KeyError(f"Could not find a 2-20 keV error column in MAXI file. Columns: {df.columns.tolist()}")
    if flux_col != "flux_2_20":
        df["flux_2_20"] = df[flux_col]
    if err_col != "err_2_20":
        df["err_2_20"] = df[err_col]
    return df


def add_xray_flux_columns(xray_win):
    xray_win = xray_win.copy()
    xray_win["X_FLUX"] = xray_win["flux_2_20"] * MAXI_CONV
    xray_win["X_FLUX_ERR"] = xray_win["err_2_20"] * MAXI_CONV
    return xray_win


# ---------------------------------------------------------------------------
# Optical File Discovery
# ---------------------------------------------------------------------------
def find_optical_files(directory):
    patterns = [
        os.path.join(directory, "*_atlas_lc.csv"),
        os.path.join(directory, "*_atlas.csv"),
        os.path.join(directory, "*_ztf_lc.csv"),
        os.path.join(directory, "*_ztf.csv"),
        os.path.join(directory, "*_opt_lc.csv"),
        os.path.join(directory, "*_opt.csv"),
        os.path.join(directory, "*_optical.csv"),
    ]
    found = {}
    for pat in patterns:
        for fp in glob.glob(pat):
            base = os.path.splitext(os.path.basename(fp))[0]
            key = re.sub(r"(_atlas_lc|_atlas|_ztf_lc|_ztf|_opt_lc|_opt|_optical)$", "", base, flags=re.IGNORECASE)
            if key not in found:
                found[key] = fp
    return found


def _first_matching_col(columns, keywords):
    cols_lower = {c.lower(): c for c in columns}
    for kw in keywords:
        for lc, orig in cols_lower.items():
            if kw in lc:
                return orig
    return None


def load_optical_lightcurve(fp):
    df = pd.read_csv(fp)
    # Normalize ATLAS/ZTF column names (e.g. ###MJD -> MJD)
    df.columns = [c.strip().lstrip('#') for c in df.columns]
    mjd_col = _first_matching_col(df.columns, ["mjd", "time", "jd"])
    mag_col = _first_matching_col(df.columns, ["magpsf", "mag", "m_app", "magnitude"])
    err_col = _first_matching_col(df.columns, ["sigmapsf", "magerr", "dmag", "err_mag", "err"])
    filt_col = _first_matching_col(df.columns, ["filter", "fid", "band", "flt", "passband", "f"])

    if mjd_col is None or mag_col is None:
        return None

    out = pd.DataFrame()
    out["MJD"] = pd.to_numeric(df[mjd_col], errors="coerce")
    out["MAG"] = pd.to_numeric(df[mag_col], errors="coerce")
    out["MAGERR"] = pd.to_numeric(df[err_col], errors="coerce") if err_col else 0.05
    out["FILTER"] = df[filt_col].astype(str).str.strip().str.lower() if filt_col else "unknown"
    out = out.dropna(subset=["MJD", "MAG"])
    return out


# ---------------------------------------------------------------------------
# Source Name Matching & Classification Lookup
# ---------------------------------------------------------------------------
def canonical_source_key(name):
    s = str(name).strip()
    s = re.sub(r"(?<=\d)m(?=\d)", "-", s)
    s = re.sub(r"(?<=\d)p(?=\d)", "+", s)
    s = re.sub(r"[\s_\-+]+", "", s).upper()
    return s


def build_canonical_lookup(files_dict):
    return {canonical_source_key(k): (k, fp) for k, fp in files_dict.items()}


def load_classification(path):
    if not os.path.exists(path):
        print(f"[WARN] Classification CSV not found at: {path}")
        return {}

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    name_col = _first_matching_col(df.columns, ["source", "name"])
    obj_col = _first_matching_col(df.columns, ["compact", "bh_ns", "object_type", "co_type"])
    type_col = _first_matching_col(df.columns, ["src_type", "xrb_type", "type"])

    if name_col is None:
        return {}

    lookup = {}
    for _, row in df.iterrows():
        raw_name = row[name_col]
        if pd.isna(raw_name):
            continue

        co_label = "UNKNOWN"
        if obj_col is not None and pd.notna(row[obj_col]):
            v = str(row[obj_col]).upper()
            if "BH" in v or "BLACK" in v:
                co_label = "BH"
            elif "NS" in v or "NEUTRON" in v:
                co_label = "NS"

        type_label = str(row[type_col]).upper() if type_col and pd.notna(row[type_col]) else "UNKNOWN"

        lookup[canonical_source_key(raw_name)] = {
            "raw_name": raw_name,
            "compact_object": co_label,
            "src_type": type_label,
        }
    return lookup


def classify_source(source_key, lookup):
    entry = lookup.get(canonical_source_key(source_key))
    if entry is None:
        return "UNKNOWN", "UNKNOWN"
    return entry["compact_object"], entry["src_type"]


# ---------------------------------------------------------------------------
# Reprocessing Model & Beta Fitting
# ---------------------------------------------------------------------------
def _reprocessing_shape(x_flux, beta):
    """
    Dimensionless disc-reprocessing shape law, unit amplitude:
        F_opt = F_X ** beta
    Deliberately has no free baseline/amplitude: those were degenerate with
    beta in the old 3-parameter fit and are what let beta drift to values
    (e.g. ~0.08) that then made the forward simulation hypersensitive to
    noise near the X-ray detection floor.
    """
    return np.power(np.maximum(x_flux, 0.0), beta)


def _per_source_centered_points(source_key, xray_vals, optical_vals, optical_err):
    """
    Mean-center one source's log10(X-ray flux) and log10(optical flux) on its
    own means. This is the "fixed-effects" trick: differencing out each
    source's average level removes that source's distance/extinction/
    intrinsic-brightness normalization entirely, leaving only how its optical
    flux co-varies with its own X-ray flux from point to point. Pooling these
    centered residuals across many sources then estimates a single common
    slope (beta) that isn't biased by which sources happen to be brighter or
    closer -- exactly the contamination flagged as a likely culprit for the
    earlier anomalous class-level fits.

    Returns None if the source has fewer than MIN_PAIRS_PER_SOURCE valid
    pairs (a single point carries no within-source slope information).
    """
    xray_vals = np.asarray(xray_vals, dtype=float)
    optical_vals = np.asarray(optical_vals, dtype=float)
    optical_err = np.asarray(optical_err, dtype=float) if optical_err is not None else None

    valid = (xray_vals > 0) & (optical_vals > 0) & np.isfinite(xray_vals) & np.isfinite(optical_vals)
    if optical_err is not None:
        valid &= np.isfinite(optical_err)
    xray_vals = xray_vals[valid]
    optical_vals = optical_vals[valid]
    optical_err = optical_err[valid] if optical_err is not None else None

    if len(xray_vals) < MIN_PAIRS_PER_SOURCE:
        return None

    logx = np.log10(xray_vals)
    logy = np.log10(optical_vals)

    # Propagate fractional optical flux error into log-space sigma; fall back
    # to unweighted (sigma=1) if errors are missing/non-positive.
    if optical_err is not None and np.all(optical_err > 0):
        sigma_logy = optical_err / (optical_vals * np.log(10.0))
        sigma_logy = np.clip(sigma_logy, 1e-3, None)
    else:
        sigma_logy = np.ones_like(logy)

    logx_c = logx - np.average(logx, weights=1.0 / sigma_logy**2)
    logy_c = logy - np.average(logy, weights=1.0 / sigma_logy**2)
    return logx_c, logy_c, sigma_logy


def _weighted_slope_through_origin(x, y, sigma):
    """Weighted least squares slope for y = beta * x (no intercept)."""
    w = 1.0 / sigma**2
    sxx = np.sum(w * x * x)
    if sxx <= 0:
        return None
    beta = np.sum(w * x * y) / sxx
    beta_err = np.sqrt(1.0 / sxx)
    resid = y - beta * x
    scatter_dex = float(np.sqrt(np.average(resid**2, weights=w))) if len(x) > 1 else float("nan")
    return beta, beta_err, scatter_dex


def fit_beta_fixed_effects(source_groups, min_pairs_per_source=MIN_PAIRS_PER_SOURCE,
                            min_total_points=MIN_PAIRS_FOR_FIT, clip_sigma=CLIP_SIGMA):
    """
    Estimate a single class-level beta from a list of (source_key, xray_vals,
    optical_vals, optical_err) tuples, one entry per source in that
    compact-object class.

    Steps:
      1. Mean-center each source's log-log points on its own mean (removes
         per-source normalization -- see _per_source_centered_points).
      2. Pool all centered points across sources.
      3. Fit a single weighted slope through the origin (beta).
      4. Do one pass of clip_sigma-based clipping on the residuals and refit,
         to limit the leverage of a handful of outlier pairs (relevant given
         how few BH sources are confidently classified -- Table 2).
    """
    logx_all, logy_all, sig_all, src_ids = [], [], [], []
    n_sources_total = len(source_groups)
    n_pairs_total = 0

    for source_key, xr, opt, opterr in source_groups:
        n_pairs_total += len(xr)
        centered = _per_source_centered_points(source_key, xr, opt, opterr)
        if centered is None:
            continue
        lx, ly, sg = centered
        logx_all.append(lx)
        logy_all.append(ly)
        sig_all.append(sg)
        src_ids.append(np.full(len(lx), source_key))

    if not logx_all:
        return None

    x = np.concatenate(logx_all)
    y = np.concatenate(logy_all)
    s = np.concatenate(sig_all)
    n_sources_used = len({sk for group in src_ids for sk in group}) if len(src_ids) else 0

    if len(x) < min_total_points:
        return None

    fit0 = _weighted_slope_through_origin(x, y, s)
    if fit0 is None:
        return None
    beta0, _, _ = fit0

    # One-pass sigma clip on residuals, then refit.
    resid = (y - beta0 * x) / s
    keep = np.abs(resid) <= clip_sigma
    if keep.sum() >= min_total_points and keep.sum() < len(x):
        x, y, s = x[keep], y[keep], s[keep]

    fit1 = _weighted_slope_through_origin(x, y, s)
    if fit1 is None:
        return None
    beta, beta_err, scatter_dex = fit1

    if abs(beta) > MAX_ABS_BETA:
        print(f"  [WARN] Rejecting ill-conditioned fit: |beta|={abs(beta):.3f} exceeds "
              f"the sanity ceiling of {MAX_ABS_BETA} ({n_sources_used} source(s), "
              f"{len(x)} centered pairs used). This is a numerically unstable slope "
              f"estimate, not a physical measurement -- almost always caused by too "
              f"few sources/points after phase restriction and centering, which "
              f"leaves the weighted-slope denominator tiny. Falling back instead of "
              f"reporting a meaningless beta.")
        return None

    return {
        "beta": float(beta),
        "beta_err": float(beta_err),
        "scatter_dex": scatter_dex,
        "n_pairs": int(len(x)),          # pairs actually used in the final fit
        "n_pairs_total": int(n_pairs_total),  # pairs found before per-source/clip filtering
        "n_sources_used": int(n_sources_used),
        "n_sources_total": int(n_sources_total),
    }


def collect_quasi_simultaneous_pairs(source_key, spans, tol_days=MATCH_TOL_DAYS,
                                      min_snr=MIN_XRAY_SNR, phase="all",
                                      baseline_percentile=BASELINE_PERCENTILE):
    """
    Collect quasi-simultaneous X-ray/optical pairs, with an SNR floor to
    exclude noise-floor points.

    Before matching, a per-source optical *baseline* is estimated as the
    `baseline_percentile`-th percentile of that source's ENTIRE optical
    light curve (not just the outburst spans) -- an approximation to the
    non-varying companion-star continuum. This baseline is subtracted from
    every matched optical point, and only the positive *excess* flux is kept.

    This matters because the raw optical flux is dominated by that constant
    continuum: e.g. a source's r-band flux can sit at essentially the same
    level at outburst peak and deep in quiescence, with the X-ray-driven
    reprocessing signal only a small fraction of the total. Fitting
    log(raw optical flux) against log(X-ray flux) forces the recovered slope
    toward zero almost by construction, since the dominant (non-varying) term
    swamps the part that's actually supposed to be measured. Subtracting the
    baseline first means the fit measures the reprocessing excess, not the
    sum of "constant star light" + "reprocessing".

    `phase` restricts which part of each outburst span contributes pairs:
      - "all":   every point in [t_start, t_end]
      - "rise":  only points from t_start up to the X-ray peak within the span
      - "decay": only points from the X-ray peak to t_end
    (see BETA_FIT_PHASE for why "rise" is used as a hard-state proxy).
    """
    maxi_files = fm.find_maxi_files(SHORTLISTED_DIR)
    opt_files = find_optical_files(SHORTLISTED_DIR)
    opt_lookup = build_canonical_lookup(opt_files)

    if source_key not in maxi_files:
        return np.array([]), np.array([]), np.array([])
    opt_entry = opt_lookup.get(canonical_source_key(source_key))
    if opt_entry is None:
        return np.array([]), np.array([]), np.array([])
    opt_fp = opt_entry[1]

    xdf = pd.read_csv(maxi_files[source_key])
    xdf.columns = [c.strip().lower() for c in xdf.columns]
    if "mjd" not in xdf.columns:
        return np.array([]), np.array([]), np.array([])
    try:
        xdf = standardize_maxi_columns(xdf)
    except KeyError as exc:
        print(f"  [WARN] {source_key}: {exc}")
        return np.array([]), np.array([]), np.array([])
    xdf = add_xray_flux_columns(xdf)

    odf = load_optical_lightcurve(opt_fp)
    if odf is None or odf.empty:
        return np.array([]), np.array([]), np.array([])
    odf["OPT_FLUX"], odf["OPT_FLUX_ERR"] = mag_to_flux(odf["MAG"].values, odf["MAGERR"].values)

    # Per-source quiescent baseline, from the FULL light curve (more robust
    # than estimating it only from the points that happen to fall in an
    # outburst span).
    baseline = float(np.percentile(odf["OPT_FLUX"].values, baseline_percentile))

    x_out, o_out, oe_out = [], [], []
    for (t_start, t_end) in spans:
        xw = xdf[(xdf["mjd"] >= t_start) & (xdf["mjd"] <= t_end)]
        ow = odf[(odf["MJD"] >= t_start) & (odf["MJD"] <= t_end)]
        if xw.empty or ow.empty:
            continue

        if phase != "all":
            peak_mjd = xw.loc[xw["X_FLUX"].idxmax(), "mjd"]
            if phase == "rise":
                xw = xw[xw["mjd"] <= peak_mjd]
            elif phase == "decay":
                xw = xw[xw["mjd"] >= peak_mjd]
            if xw.empty:
                continue

        for _, orow in ow.iterrows():
            dt = np.abs(xw["mjd"].values - orow["MJD"])
            j = int(np.argmin(dt))
            if dt[j] <= tol_days:
                xf = xw["X_FLUX"].values[j]
                xerr = xw["X_FLUX_ERR"].values[j]
                opt_excess = orow["OPT_FLUX"] - baseline

                # Filter out negative/low-SNR MAXI data, and require the
                # baseline-subtracted optical excess to still be positive.
                snr = xf / max(xerr, 1e-30)
                if xf > 0 and snr >= min_snr and opt_excess > 0:
                    x_out.append(xf)
                    o_out.append(opt_excess)
                    oe_out.append(orow["OPT_FLUX_ERR"])
    return np.array(x_out), np.array(o_out), np.array(oe_out)


# ---------------------------------------------------------------------------
# Forward Simulation of Optical Reprocessing
# ---------------------------------------------------------------------------
def simulate_outburst_optical(mjd, xray_flux, xray_err, beta, snr_floor=MIN_XRAY_SNR,
                               taper_width=SNR_TAPER_WIDTH):
    """
    Simulate optical light curve from X-ray light curve using the fitted
    reprocessing shape:
        optical_flux = xray_flux ** beta   (unit amplitude, dimensionless shape)

    Rather than a hard cut at `snr_floor` (flux -> exactly 0 below it, full
    value above it), a logistic taper is applied in SNR space:
        weight(snr) = 1 / (1 + exp(-(snr - snr_floor) / taper_width))
    weight -> 0 well below the floor, -> 1 well above it, and passes smoothly
    through 0.5 at the floor itself. This keeps the earlier fix (background
    noise fluctuating around zero can't be raised to a fractional power and
    amplified into an on/off spike) while removing the vertical jump that a
    hard cutoff leaves behind whenever a real light curve dips in and out of
    detectability -- which is common, and was still visible as a stepwise
    look in the regenerated plots even after the original sawtooth bug was
    fixed.
    """
    xray_flux = np.asarray(xray_flux, dtype=float)
    xray_err = np.asarray(xray_err, dtype=float)
    peak_idx = int(np.argmax(xray_flux))

    snr = np.divide(xray_flux, np.maximum(xray_err, 1e-30))
    weight = 1.0 / (1.0 + np.exp(-(snr - snr_floor) / taper_width))
    flux_clean = weight * np.maximum(xray_flux, 0.0)
    optical_flux = _reprocessing_shape(flux_clean, beta)

    phase = np.where(np.arange(len(xray_flux)) <= peak_idx, "rise", "decay")
    return optical_flux, phase, peak_idx


# ---------------------------------------------------------------------------
# Plotting & Visualization
# ---------------------------------------------------------------------------
def plot_predicted_only(source_key, band_label, outburst_idx, mjd, xray_flux, xray_err,
                         optical_flux, beta, co_label, t_start, t_end, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.errorbar(mjd, xray_flux, yerr=xray_err, fmt="o-", ms=3, lw=0.8,
                 color="steelblue", ecolor="lightsteelblue", alpha=0.85,
                 label="MAXI X-ray (2-20 keV)")
    ax1.axvspan(t_start, t_end, color="steelblue", alpha=0.08)
    ax1.set_ylabel(r"X-ray flux [erg cm$^{-2}$ s$^{-1}$]")
    ax1.set_title(f"{source_key}  ({co_label})  --  Outburst #{outburst_idx}: X-ray light curve", fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.2)

    ax2.plot(mjd, optical_flux, "--", color="darkred", lw=2.0,
              label=f"Simulated optical (DIM reprocessing, " r"$\beta$" f"={beta:.3f})")
    ax2.set_xlabel("MJD")
    ax2.set_ylabel("Simulated optical flux (unit-amplitude shape)")
    ax2.set_title("Predicted optical light curve (forward simulation)", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(source_key, band_label, outburst_idx, mjd, xray_flux, xray_err,
                     optical_flux, beta, co_label, t_start, t_end, opt_window, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.errorbar(mjd, xray_flux, yerr=xray_err, fmt="o-", ms=3, lw=0.8,
                 color="steelblue", ecolor="lightsteelblue", alpha=0.85,
                 label="MAXI X-ray (2-20 keV)")
    ax1.axvspan(t_start, t_end, color="steelblue", alpha=0.08)
    ax1.set_ylabel(r"X-ray flux [erg cm$^{-2}$ s$^{-1}$]")
    ax1.set_title(f"{source_key}  ({co_label})  --  Outburst #{outburst_idx}: X-ray outburst", fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.2)

    ax2.plot(mjd, optical_flux, "--", color="black", lw=2.0, zorder=3,
              label=f"Simulated optical (" r"$\beta$" f"={beta:.3f})")
    ax2.set_xlabel("MJD")
    ax2.set_ylabel("Simulated optical flux (unit-amplitude shape)", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.grid(alpha=0.2)

    if opt_window is not None and not opt_window.empty:
        ax2b = ax2.twinx()
        for filt, grp in opt_window.groupby("FILTER"):
            filt_lc = str(filt).lower().strip()
            if "cyan" in filt_lc or filt_lc == "c":
                color, band_lbl = "#00b4d8", "ATLAS c"
            elif "orange" in filt_lc or filt_lc == "o":
                color, band_lbl = "#f77f00", "ATLAS o"
            elif "g" in filt_lc:
                color, band_lbl = "#2ca02c", "g"
            elif "r" in filt_lc:
                color, band_lbl = "#d62728", "r"
            elif "i" in filt_lc:
                color, band_lbl = "#9467bd", "i"
            else:
                color, band_lbl = "#ff7f00", str(filt)

            ax2b.errorbar(grp["MJD"], grp["OPT_FLUX"], yerr=grp["OPT_FLUX_ERR"],
                          fmt="s", ms=6, color=color, ecolor=color,
                          markeredgecolor="black", markeredgewidth=0.4,
                          alpha=0.9, elinewidth=1.0, zorder=4,
                          label=f"Observed optical {band_lbl}-band")
        ax2b.set_ylabel("Observed optical flux [mJy]", color="dimgray")
        ax2b.tick_params(axis="y", labelcolor="dimgray")

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
        ax2.set_title("Simulated vs. observed optical light curve", fontsize=11)
    else:
        ax2.legend(fontsize=8, loc="upper right")
        ax2.set_title("Simulated optical light curve (no optical coverage in this outburst window)", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def load_outburst_spans():
    if os.path.exists(SUMMARY_CSV):
        df = pd.read_csv(SUMMARY_CSV)
    else:
        rows = []
        for source_dir in glob.glob(os.path.join(PLOT_DIR, "*")):
            source_key = os.path.basename(source_dir)
            src_csv = os.path.join(source_dir, f"{source_key}_norris_tophat_outbursts.csv")
            if os.path.exists(src_csv):
                rows.append(pd.read_csv(src_csv))
        if not rows:
            print(f"[ERROR] No outburst summary found at {SUMMARY_CSV}.")
            return {}
        df = pd.concat(rows, ignore_index=True)

    df = df[df["band"] == XRAY_BAND]
    spans_by_source = {}
    for _, row in df.iterrows():
        spans_by_source.setdefault(row["source"], []).append(
            (float(row["span_start_mjd"]), float(row["span_end_mjd"]), int(row["outburst_idx"]))
        )
    return spans_by_source


def main():
    print("=" * 72)
    print("Optical reprocessing (DIM) forward simulation")
    print("=" * 72)

    # Clean up previous simulation output files
    cleanup_previous_simulation_files()

    lookup = load_classification(CLASSIFICATION_CSV)
    spans_by_source = load_outburst_spans()
    if not spans_by_source:
        return

    maxi_files = fm.find_maxi_files(SHORTLISTED_DIR)
    opt_files = find_optical_files(SHORTLISTED_DIR)
    opt_lookup = build_canonical_lookup(opt_files)

    # -----------------------------------------------------------------
    # Step 1: Globally fit beta_BH and beta_NS across high-SNR pairs
    # -----------------------------------------------------------------
    print(f"\n[1/2] Fitting global beta_BH / beta_NS from quasi-simultaneous high-SNR "
          f"'{BETA_FIT_PHASE}'-phase pairs (within-source / fixed-effects log-log fit)...")
    pooled = {"BH": [], "NS": []}
    for source_key, spans in spans_by_source.items():
        co_label, _ = classify_source(source_key, lookup)
        if co_label not in ("BH", "NS"):
            continue
        xr, opt, opterr = collect_quasi_simultaneous_pairs(
            source_key, [(s[0], s[1]) for s in spans], phase=BETA_FIT_PHASE
        )
        if len(xr):
            pooled[co_label].append((source_key, xr, opt, opterr))

    beta_results = {}
    for co_label in ("BH", "NS"):
        fit = fit_beta_fixed_effects(pooled[co_label])
        n_pairs_total = sum(len(xr) for _, xr, _, _ in pooled[co_label])
        if fit is None:
            print(f"  [{co_label}] Insufficient within-source pairs after centering "
                  f"({n_pairs_total} raw pairs across {len(pooled[co_label])} sources, "
                  f"need >= {MIN_PAIRS_FOR_FIT} centered pairs from sources with "
                  f">= {MIN_PAIRS_PER_SOURCE} points each); falling back to baseline beta = {BETA_BASELINE}")
            beta_results[co_label] = {
                "beta": BETA_BASELINE, "beta_err": float("nan"), "scatter_dex": float("nan"),
                "n_pairs": 0, "n_pairs_total": n_pairs_total,
                "n_sources_used": 0, "n_sources_total": len(pooled[co_label]),
            }
        else:
            print(f"  [{co_label}] beta = {fit['beta']:.3f} +/- {fit['beta_err']:.3f} "
                  f"(scatter = {fit['scatter_dex']:.3f} dex, "
                  f"{fit['n_pairs']}/{fit['n_pairs_total']} pairs used, "
                  f"{fit['n_sources_used']}/{fit['n_sources_total']} sources contributing)")
            beta_results[co_label] = fit
    beta_results["UNKNOWN"] = {
        "beta": BETA_BASELINE, "beta_err": float("nan"), "scatter_dex": float("nan"),
        "n_pairs": 0, "n_pairs_total": 0, "n_sources_used": 0, "n_sources_total": 0,
    }

    beta_df = pd.DataFrame([
        {
            "compact_object": k,
            "beta": v["beta"],
            "beta_err": v["beta_err"],
            "scatter_dex": v["scatter_dex"],
            "n_pairs_used": v["n_pairs"],
            "n_pairs_total": v["n_pairs_total"],
            "n_sources_used": v["n_sources_used"],
            "n_sources_total": v["n_sources_total"],
        }
        for k, v in beta_results.items()
    ])
    beta_df.to_csv(BETA_CSV_PATH, index=False)
    with open(BETA_LOG_PATH, "w", encoding="utf-8") as fh:
        fh.write(f"Optical reprocessing beta fit -- {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC\n")
        fh.write(f"Model: optical_flux = xray_flux ** beta (unit amplitude, dimensionless shape)\n")
        fh.write(f"Pairs restricted to outburst phase: '{BETA_FIT_PHASE}' "
                 f"(hard-state proxy -- see collect_quasi_simultaneous_pairs docstring)\n")
        fh.write(f"Simulation SNR floor uses a logistic taper (width={SNR_TAPER_WIDTH}), "
                 f"not a hard step\n")
        fh.write(f"Fit method: within-source (fixed-effects) weighted log-log slope, "
                 f"{CLIP_SIGMA}-sigma clipped once\n")
        fh.write(f"Fallback value: beta = {BETA_BASELINE} (van Paradijs & McClintock 1994)\n")
        fh.write(f"Match tolerance: {MATCH_TOL_DAYS} day(s), Min X-ray SNR: {MIN_XRAY_SNR}, "
                 f"Min pairs/source: {MIN_PAIRS_PER_SOURCE}\n\n")
        fh.write(beta_df.to_string(index=False))
        fh.write("\n")
    print(f"  [saved] {BETA_CSV_PATH}")
    print(f"  [saved] {BETA_LOG_PATH}")

    # -----------------------------------------------------------------
    # Step 2: Forward-simulate & plot each outburst
    # -----------------------------------------------------------------
    print("\n[2/2] Forward-simulating optical light curves per outburst...")
    for source_key, spans in sorted(spans_by_source.items()):
        if source_key not in maxi_files:
            print(f"  [skip] {source_key}: no MAXI file found")
            continue

        co_label, src_type = classify_source(source_key, lookup)
        fit_info = beta_results.get(co_label, beta_results["UNKNOWN"])
        beta = fit_info["beta"]

        xdf = pd.read_csv(maxi_files[source_key])
        xdf.columns = [c.strip().lower() for c in xdf.columns]
        if "mjd" not in xdf.columns:
            print(f"  [skip] {source_key}: MAXI file missing mjd column")
            continue
        try:
            xdf = standardize_maxi_columns(xdf)
        except KeyError as exc:
            print(f"  [skip] {source_key}: {exc}")
            continue
        xdf = add_xray_flux_columns(xdf)

        odf = None
        opt_entry = opt_lookup.get(canonical_source_key(source_key))
        if opt_entry is not None:
            odf = load_optical_lightcurve(opt_entry[1])
            if odf is not None and not odf.empty:
                odf["OPT_FLUX"], odf["OPT_FLUX_ERR"] = mag_to_flux(odf["MAG"].values, odf["MAGERR"].values)

        source_dir = os.path.join(PLOT_DIR, source_key)
        os.makedirs(source_dir, exist_ok=True)

        for (t_start, t_end, outburst_idx) in spans:
            view_lo, view_hi = t_start - VIEW_PAD_DAYS, t_end + VIEW_PAD_DAYS
            win = xdf[(xdf["mjd"] >= view_lo) & (xdf["mjd"] <= view_hi)].sort_values("mjd")
            xray_win = win[(win["mjd"] >= t_start) & (win["mjd"] <= t_end)]
            if len(xray_win) < 3:
                print(f"  [skip] {source_key} outburst #{outburst_idx}: too few X-ray points in span")
                continue

            mjd = xray_win["mjd"].values
            xray_flux = xray_win["X_FLUX"].values
            xray_err = xray_win["X_FLUX_ERR"].values

            optical_flux, phase, peak_idx = simulate_outburst_optical(mjd, xray_flux, xray_err, beta)

            # --- Export simulation output CSV ---
            sim_df = pd.DataFrame({
                "mjd": mjd,
                "x_flux_erg_cm2_s": xray_flux,
                "x_flux_err_erg_cm2_s": xray_err,
                "predicted_optical_flux_shape": optical_flux,  # unit-amplitude shape, not mJy
                "phase": phase,
            })
            sim_csv_path = os.path.join(source_dir, f"{source_key}_outburst{outburst_idx}_optical_simulation.csv")
            sim_df.to_csv(sim_csv_path, index=False)

            # --- Generate Plot 1: Prediction plot ---
            pred_plot_path = os.path.join(source_dir, f"{source_key}_outburst{outburst_idx}_optical_predicted.png")
            plot_predicted_only(source_key, XRAY_BAND, outburst_idx, mjd, xray_flux, xray_err,
                                 optical_flux, beta, co_label, t_start, t_end, pred_plot_path)

            # --- Generate Plot 2: Simulated vs Observed comparison plot ---
            opt_window = None
            if odf is not None and not odf.empty:
                opt_window = odf[(odf["MJD"] >= t_start) & (odf["MJD"] <= t_end)]
            comp_plot_path = os.path.join(source_dir, f"{source_key}_outburst{outburst_idx}_optical_comparison.png")
            plot_comparison(source_key, XRAY_BAND, outburst_idx, mjd, xray_flux, xray_err,
                             optical_flux, beta, co_label, t_start, t_end, opt_window, comp_plot_path)

            print(f"  [done] {source_key} outburst #{outburst_idx} ({co_label}, beta={beta:.3f}) -> "
                  f"{os.path.basename(pred_plot_path)}, {os.path.basename(comp_plot_path)}")

    print(f"\nCompleted processing. Output directories:\n  {PLOT_DIR}\\<source>\\\n"
          f"  {FRED_DIR}\\ ({os.path.basename(BETA_CSV_PATH)}, {os.path.basename(BETA_LOG_PATH)})")


if __name__ == "__main__":
    main()