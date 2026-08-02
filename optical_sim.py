"""
simulate_optical_reprocessing.py
=================================

Forward-simulates the optical (ZTF-band) light curve of an X-ray binary
outburst from its observed X-ray (MAXI) light curve, using a simple
inward-propagating Disc Instability Model (DIM) / X-ray reprocessing
picture:

    optical_flux = 1.0 * (xray_flux ** beta)

This is the classical van Paradijs & McClintock (1994) reprocessing
relation F_opt ~ F_X^beta, which Russell et al. (2006, "Global OIR-X-ray
correlations...", Russell-XBs2.tex) find holds with beta ~ 0.5-0.6 for
both BHXBs and NSXBs (their global hard-state fits give 0.61+/-0.02 for
BHXBs and 0.63+/-0.04 for NSXBs), and which Russell et al. (2010,
nov11.tex, the 4U 1957+11 paper) find observationally as F_opt ~ F_X^0.5
for a single reprocessing-dominated LMXB. beta ~ 0.5 is the value
predicted for simple X-ray-heated-disc reprocessing (L_opt ~ T^2 ~
L_X^0.5), while a somewhat steeper index (beta ~ 0.7) is expected if
optical synchrotron jet emission contributes (more often relevant for
BHXBs). This is why beta is kept as a free, per-source-class parameter
here rather than hard-coded at the textbook value of 0.5: we use the
project's own BH and NS samples (from `classification_results.csv`) to
empirically re-establish beta_BH and beta_NS from quasi-simultaneous
MAXI/ZTF pairs, in the same spirit as Russell et al.

IMPORTANT — what is and is not "fit" here
------------------------------------------
1) beta_BH and beta_NS ARE fit, once, globally, by pooling all
   quasi-simultaneous (|dt| <= MATCH_TOL_DAYS) MAXI-vs-ZTF flux pairs
   across every outburst of every BH-classified / NS-classified source.
   This mirrors what Russell et al. do with their whole sample.
2) The per-outburst *simulated* optical curves are then produced by
   forward-evaluating optical_flux = xray_flux**beta on the observed
   X-ray light curve ALONE, using the beta appropriate to that source's
   compact-object class. The simulation is never re-fit to that
   source's own ZTF points -- the real ZTF data for a given outburst is
   only ever *overlaid* on top of the simulation for visual comparison,
   never used to tune it. That would be circular ("cheating").

Units
-----
Per project convention we skip full CGS bookkeeping. X-ray fluxes are
expressed in "Crab-relative" units: MAXI 2-20 keV instrument counts are
converted to approximate erg/cm^2/s using a single Crab-calibrated
conversion factor (maxi_conv = 2.4e-8 / 3.3), the same shortcut used in
`fred_model.py`'s upstream light-curve products. ZTF AB magnitudes are
converted to mJy flux densities. Because the reprocessing law is
evaluated with a fixed amplitude of 1.0 (no free normalization -- only
beta floats), the resulting "simulated optical flux" is a dimensionless
DIM light-curve *shape*, not a physically normalized mJy flux, so it is
plotted on its own scaled axis (never forced onto the ZTF mJy axis by
rescaling against the real data, which would smuggle a fit back in).

Data locations
--------------
This script imports `fred_model.py` and reuses its path constants
(`SHORTLISTED_DIR`, `FRED_DIR`, `PLOT_DIR`) and its `find_maxi_files`
helper, since that is where the outburst MJD spans and MAXI light
curves already live:

  * Outburst MJD spans + Norris-tophat fit parameters:
      FRED_DIR/norris_tophat_outburst_summary.csv        (master, all sources)
      PLOT_DIR/<source>/<source>_norris_tophat_outbursts.csv (per source)
    -> columns used here: source, band, outburst_idx,
       span_start_mjd, span_end_mjd  (we use band == "2-20 keV")

  * Raw MAXI light curves: SHORTLISTED_DIR/<source>_maxi(_lc).csv
  * Raw ZTF light curves:  SHORTLISTED_DIR/<source>_ztf(_lc).csv

  * Source classification (BH/NS, LMXB/HMXB):
      classification_output/classification_results.csv

All outputs of THIS script (plots + per-outburst simulation CSVs + the
global beta fit) are written to their own OPTICAL_DIR tree
(Optical_Sim/<source>/...), separate from fred_model's Fit_Fred outputs
that they read spans from.
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
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Pull in fred_model's path constants + MAXI file finder so the outburst
# MJD spans / light curves are read from exactly the same place fred_model
# wrote/reads them.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fred_model as fm  # noqa: E402

SHORTLISTED_DIR = fm.SHORTLISTED_DIR
FRED_DIR        = fm.FRED_DIR   # read-only: where fred_model's Norris-tophat spans/fits live (Fit_Fred)
PLOT_DIR        = fm.PLOT_DIR   # read-only: fred_model's per-source plot/csv folder, under FRED_DIR

# All of THIS script's outputs (per-outburst simulation plots/CSVs + the
# global beta fit) go in their own tree instead of being mixed into
# fred_model's Fit_Fred output.
OPTICAL_DIR = r"C:\Users\abhin\Documents\ksp2026\Optical_Sim"

CLASSIFICATION_CSV = r"C:\Users\abhin\Documents\ksp2026\classification_output\classification_results.csv"

SUMMARY_CSV = os.path.join(FRED_DIR, "norris_tophat_outburst_summary.csv")
XRAY_BAND   = "2-20 keV"   # band whose catalog-derived span was fit in fred_model

BETA_LOG_PATH   = os.path.join(OPTICAL_DIR, "optical_reprocessing_beta_fit.log")
BETA_CSV_PATH   = os.path.join(OPTICAL_DIR, "optical_reprocessing_beta_fit.csv")

BETA_BASELINE     = 0.5   # standard reprocessing exponent (van Paradijs & McClintock 1994)
MATCH_TOL_DAYS     = 1.0   # "quasi-simultaneous" tolerance, per Russell et al. (Sec. 2, Russell-XBs2.tex)
MIN_PAIRS_FOR_FIT  = 5     # need at least this many quasi-simultaneous pairs to trust a class fit
VIEW_PAD_DAYS       = fm.VIEW_PAD_DAYS  # reuse fred_model's plot padding convention

# maxi_conv (erg/cm^2/s per ph/s/cm^2), Crab-relative shortcut -- exactly as specified.
MAXI_CONV = 2.4e-8 / 3.3


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
def mag_to_flux(mag, mag_err):
    flux = 3631 * 10 ** (-0.4 * mag) * 1000  # mJy
    flux_err = flux * (mag_err / 1.0857)
    return flux, flux_err


def _find_energy_band_cols(df, band_tag="2_20"):
    """
    Locate the flux/error column pair for a given energy band (default
    2-20 keV), tolerant of naming variants seen across different MAXI
    export scripts -- e.g. 'err_2_20' in some files vs 'error_2_20' in
    others. Matching is done on lowercased column names, requiring both
    the band tag (e.g. '2_20') and a 'flux'/'err' substring.
    """
    cols = {c.lower(): c for c in df.columns}
    flux_col = next((orig for lc, orig in cols.items() if "flux" in lc and band_tag in lc), None)
    err_col = next((orig for lc, orig in cols.items() if "err" in lc and band_tag in lc), None)
    return flux_col, err_col


def standardize_maxi_columns(df, band_tag="2_20"):
    """
    Ensure the dataframe has canonical 'flux_2_20' / 'err_2_20' columns
    regardless of which naming convention the source CSV used, so every
    downstream function can rely on those exact names.
    """
    df = df.copy()
    flux_col, err_col = _find_energy_band_cols(df, band_tag)
    if flux_col is None:
        raise KeyError(
            f"Could not find a 2-20 keV flux column in MAXI file "
            f"(columns present: {df.columns.tolist()})"
        )
    if err_col is None:
        raise KeyError(
            f"Could not find a 2-20 keV error column in MAXI file "
            f"(looked for 'err'/'error' + '{band_tag}'; "
            f"columns present: {df.columns.tolist()})"
        )
    if flux_col != "flux_2_20":
        df["flux_2_20"] = df[flux_col]
    if err_col != "err_2_20":
        df["err_2_20"] = df[err_col]
    return df


def add_xray_flux_columns(xray_win):
    """Convert MAXI counts -> Crab-relative erg/cm^2/s flux, in place."""
    xray_win = xray_win.copy()
    xray_win["X_FLUX"] = xray_win["flux_2_20"] * MAXI_CONV
    xray_win["X_FLUX_ERR"] = xray_win["err_2_20"] * MAXI_CONV
    return xray_win


# ---------------------------------------------------------------------------
# File discovery (ZTF companion to fred_model.find_maxi_files)
# ---------------------------------------------------------------------------
def find_ztf_files(directory):
    patterns = [
        os.path.join(directory, "*_ztf_lc.csv"),
        os.path.join(directory, "*_ztf.csv"),
        os.path.join(directory, "*_ztf_lc"),
        os.path.join(directory, "*_ztf"),
    ]
    found = {}
    for pat in patterns:
        for fp in glob.glob(pat):
            base = os.path.splitext(os.path.basename(fp))[0]
            key = re.sub(r"(_ztf_lc|_ztf)$", "", base)
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


def load_ztf_lightcurve(fp):
    """
    Load a ZTF CSV and return a standardized dataframe with columns
    MJD, MAG, MAGERR, FILTER (FILTER may be 'unknown' if not present).
    Column names in the raw file are auto-detected since ZTF export
    conventions vary (mjd/MJD, mag/magpsf, magerr/sigmapsf, filter/fid/band).
    """
    df = pd.read_csv(fp)
    mjd_col = _first_matching_col(df.columns, ["mjd"])
    mag_col = _first_matching_col(df.columns, ["mag"])
    err_col = _first_matching_col(df.columns, ["magerr", "sigmapsf", "err"])
    filt_col = _first_matching_col(df.columns, ["filter", "fid", "band"])

    if mjd_col is None or mag_col is None:
        return None

    out = pd.DataFrame()
    out["MJD"] = pd.to_numeric(df[mjd_col], errors="coerce")
    out["MAG"] = pd.to_numeric(df[mag_col], errors="coerce")
    out["MAGERR"] = pd.to_numeric(df[err_col], errors="coerce") if err_col else 0.05
    out["FILTER"] = df[filt_col].astype(str) if filt_col else "unknown"
    out = out.dropna(subset=["MJD", "MAG"])
    return out


# ---------------------------------------------------------------------------
# Classification lookup (BH / NS per source)
# ---------------------------------------------------------------------------
def canonical_source_key(name):
    """
    Canonicalize a source name for matching across MAXI filenames, ZTF
    filenames, and the classification CSV, which don't always agree on
    formatting. Two normalizations are applied:

      1) IAU-style filesystem-safe sign encoding: some pipelines write
         a literal 'm'/'p' in place of '-'/'+' around a digit run in
         J-names (e.g. 'MAXI_J1848m024' vs 'MAXI J1848-024'), since '-'
         and '+' can be awkward in filenames. Digit-sandwiched 'm'/'p'
         are converted back to '-'/'+' before stripping punctuation, so
         both spellings collapse to the same key.
      2) All whitespace/underscores/dashes/pluses are then stripped and
         the result is uppercased, so remaining formatting differences
         (spaces vs underscores, presence/absence of the sign once
         restored) don't cause a mismatch either.
    """
    s = str(name).strip()
    s = re.sub(r"(?<=\d)m(?=\d)", "-", s)
    s = re.sub(r"(?<=\d)p(?=\d)", "+", s)
    s = re.sub(r"[\s_\-+]+", "", s).upper()
    return s


def build_canonical_lookup(files_dict):
    """files_dict: {raw_key: filepath} -> {canonical_key: (raw_key, filepath)}"""
    return {canonical_source_key(k): (k, fp) for k, fp in files_dict.items()}


def load_classification(path):
    """
    Returns dict: normalized_source_name -> {"raw_name", "compact_object",
    "src_type"} where compact_object is one of "BH", "NS", or "UNKNOWN"
    (auto-detected from whatever label the classifier used, e.g. "BH",
    "Black Hole", "NS", "Neutron Star").
    """
    if not os.path.exists(path):
        print(f"[WARN] classification CSV not found at:\n  {path}")
        return {}

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    name_col = _first_matching_col(df.columns, ["source", "name"])
    obj_col = _first_matching_col(df.columns, ["compact", "bh_ns", "object_type", "co_type"])
    type_col = _first_matching_col(df.columns, ["src_type", "xrb_type", "type"])

    if name_col is None:
        print(f"[WARN] could not find a source-name column in {path}. "
              f"Columns found: {df.columns.tolist()}")
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
# Reprocessing model + beta fitting
# ---------------------------------------------------------------------------
def reprocessing_model(x_flux, beta):
    """optical_flux = 1.0 * xray_flux**beta -- fixed unit amplitude, beta free.
    Used ONLY for the forward simulation (Step 2), never for fitting."""
    return 1.0 * np.power(np.clip(x_flux, 1e-30, None), beta)


def _normalized_power_law(x_flux, log_n, beta):
    """n * xray_flux**beta, fit in log-log space. The real MAXI (~1e-7 erg
    cm^-2 s^-1) and ZTF (~mJy) flux scales differ by many orders of
    magnitude, so recovering beta from real data requires a free
    normalization n (exactly as van Paradijs & McClintock / Russell et al.
    always include a normalization alongside their fitted slope). Only
    beta is kept afterwards -- n is discarded, since the forward
    simulation uses the fixed unit amplitude specified by the model."""
    return log_n + beta * np.log10(np.clip(x_flux, 1e-30, None))


def fit_beta(xray_vals, optical_vals, optical_err=None, p0=BETA_BASELINE):
    """Fit beta (the reprocessing slope) from real quasi-simultaneous
    MAXI/ZTF pairs via a log-log power-law fit with a free normalization.
    Only the slope (beta) is returned -- the fitted normalization is
    discarded, since Step 2's forward simulation always uses the
    spec-fixed unit amplitude (optical_flux = 1.0 * xray_flux**beta)."""
    xray_vals = np.asarray(xray_vals, dtype=float)
    optical_vals = np.asarray(optical_vals, dtype=float)
    if len(xray_vals) < MIN_PAIRS_FOR_FIT:
        return None

    log_y = np.log10(optical_vals)

    # propagate fractional flux errors into log-space sigma for weighting
    sigma = None
    if optical_err is not None:
        frac_err = np.asarray(optical_err, dtype=float) / optical_vals
        sigma = frac_err / np.log(10.0)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma,
                          np.nanmedian(sigma[np.isfinite(sigma) & (sigma > 0)])
                          if np.any(np.isfinite(sigma) & (sigma > 0)) else 1.0)

    try:
        popt, pcov = curve_fit(
            _normalized_power_law, xray_vals, log_y,
            p0=[np.nanmedian(log_y), p0], sigma=sigma, absolute_sigma=sigma is not None,
            maxfev=20000,
        )
        beta_err = float(np.sqrt(pcov[1, 1])) if np.isfinite(pcov[1, 1]) else float("nan")
        return {"beta": float(popt[1]), "beta_err": beta_err, "n_pairs": len(xray_vals)}
    except Exception as exc:
        print(f"  [WARN] beta fit failed: {exc}")
        return None


def collect_quasi_simultaneous_pairs(source_key, spans, tol_days=MATCH_TOL_DAYS):
    """
    For one source, over its list of (t_start, t_end) outburst spans,
    match MAXI X-ray points to ZTF optical points within tol_days and
    return arrays of (xray_flux, optical_flux_mJy, optical_flux_err_mJy).
    """
    maxi_files = fm.find_maxi_files(SHORTLISTED_DIR)
    ztf_files = find_ztf_files(SHORTLISTED_DIR)
    ztf_lookup = build_canonical_lookup(ztf_files)

    if source_key not in maxi_files:
        return np.array([]), np.array([]), np.array([])
    ztf_entry = ztf_lookup.get(canonical_source_key(source_key))
    if ztf_entry is None:
        return np.array([]), np.array([]), np.array([])
    ztf_fp = ztf_entry[1]

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

    zdf = load_ztf_lightcurve(ztf_fp)
    if zdf is None or zdf.empty:
        return np.array([]), np.array([]), np.array([])
    zdf["OPT_FLUX"], zdf["OPT_FLUX_ERR"] = mag_to_flux(zdf["MAG"].values, zdf["MAGERR"].values)

    x_out, o_out, oe_out = [], [], []
    for (t_start, t_end) in spans:
        xw = xdf[(xdf["mjd"] >= t_start) & (xdf["mjd"] <= t_end)]
        zw = zdf[(zdf["MJD"] >= t_start) & (zdf["MJD"] <= t_end)]
        if xw.empty or zw.empty:
            continue
        for _, zrow in zw.iterrows():
            dt = np.abs(xw["mjd"].values - zrow["MJD"])
            j = int(np.argmin(dt))
            if dt[j] <= tol_days:
                xf = xw["X_FLUX"].values[j]
                if xf > 0 and zrow["OPT_FLUX"] > 0:
                    x_out.append(xf)
                    o_out.append(zrow["OPT_FLUX"])
                    oe_out.append(zrow["OPT_FLUX_ERR"])
    return np.array(x_out), np.array(o_out), np.array(oe_out)


# ---------------------------------------------------------------------------
# Forward simulation of one outburst
# ---------------------------------------------------------------------------
def simulate_outburst_optical(mjd, xray_flux, beta):
    """
    Pure forward simulation: optical_flux = 1.0 * xray_flux**beta,
    evaluated on the observed X-ray curve alone. The rise/decay split is
    kept explicit (peak-referenced) because the reprocessing law was
    specified for the decay phase; the same functional form is applied
    on the rise since no separate rise-phase law was given and the DIM
    reprocessing picture (disc reprocessing X-rays into optical) applies
    continuously, not only after peak.
    """
    xray_flux = np.asarray(xray_flux, dtype=float)
    peak_idx = int(np.argmax(xray_flux))

    optical_flux = np.zeros_like(xray_flux, dtype=float)
    rise_phase = slice(0, peak_idx + 1)
    decay_phase = slice(peak_idx + 1, len(xray_flux))

    optical_flux[rise_phase] = 1.0 * np.power(np.clip(xray_flux[rise_phase], 0, None), beta)
    optical_flux[decay_phase] = 1.0 * np.power(np.clip(xray_flux[decay_phase], 0, None), beta)

    phase = np.where(np.arange(len(xray_flux)) <= peak_idx, "rise", "decay")
    return optical_flux, phase, peak_idx


# ---------------------------------------------------------------------------
# Plotting (styled after the mentor's reference figures: stacked 2-panel,
# X-ray on top in steelblue/navy, optical below with its own axis/color,
# FRED-style titles + light grid)
# ---------------------------------------------------------------------------
def plot_predicted_only(source_key, band_label, outburst_idx, mjd, xray_flux, xray_err,
                         optical_flux, beta, co_label, t_start, t_end, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.errorbar(mjd, xray_flux, yerr=xray_err, fmt="o-", ms=3, lw=0.8,
                 color="steelblue", ecolor="lightsteelblue", alpha=0.85,
                 label="MAXI X-ray (2-20 keV)")
    ax1.axvspan(t_start, t_end, color="steelblue", alpha=0.08)
    ax1.set_ylabel(r"X-ray flux [erg cm$^{-2}$ s$^{-1}$] (Crab-relative)")
    ax1.set_title(f"{source_key}  ({co_label})  --  Outburst #{outburst_idx}: X-ray light curve", fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.2)

    ax2.plot(mjd, optical_flux, "--", color="darkred", lw=2.0,
              label=f"Simulated optical (DIM reprocessing, " r"$\beta$" f"={beta:.3f})")
    ax2.set_xlabel("MJD")
    ax2.set_ylabel(r"Simulated optical flux [arb. units, $\propto F_X^{\beta}$]")
    ax2.set_title("Predicted optical light curve (forward simulation only, no ZTF data used)", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(source_key, band_label, outburst_idx, mjd, xray_flux, xray_err,
                     optical_flux, beta, co_label, t_start, t_end, ztf_window, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.errorbar(mjd, xray_flux, yerr=xray_err, fmt="o-", ms=3, lw=0.8,
                 color="steelblue", ecolor="lightsteelblue", alpha=0.85,
                 label="MAXI X-ray (2-20 keV)")
    ax1.axvspan(t_start, t_end, color="steelblue", alpha=0.08)
    ax1.set_ylabel(r"X-ray flux [erg cm$^{-2}$ s$^{-1}$] (Crab-relative)")
    ax1.set_title(f"{source_key}  ({co_label})  --  Outburst #{outburst_idx}: X-ray outburst", fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.2)

    ax2.plot(mjd, optical_flux, "--", color="black", lw=2.0, zorder=3,
              label=f"Simulated optical (" r"$\beta$" f"={beta:.3f})")
    ax2.set_xlabel("MJD")
    ax2.set_ylabel(r"Simulated optical flux [arb. units]", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.grid(alpha=0.2)

    if ztf_window is not None and not ztf_window.empty:
        ax2b = ax2.twinx()
        # High-contrast, colorblind-friendlier palette. Matched by substring
        # (not first-character) since ZTF filtercodes are typically 'zg'/
        # 'zr'/'zi', not bare 'g'/'r'/'i'.
        band_colors = [("g", "#2ca02c", "g"), ("r", "#d62728", "r"), ("i", "#9467bd", "i")]
        for filt, grp in ztf_window.groupby("FILTER"):
            filt_lc = str(filt).lower()
            match = next((bc for bc in band_colors if bc[0] in filt_lc), None)
            color, band_label = (match[1], match[2]) if match else ("#ff7f0e", str(filt))
            ax2b.errorbar(grp["MJD"], grp["OPT_FLUX"], yerr=grp["OPT_FLUX_ERR"],
                          fmt="s", ms=6, color=color, ecolor=color,
                          markeredgecolor="black", markeredgewidth=0.4,
                          alpha=0.9, elinewidth=1.0, zorder=4,
                          label=f"ZTF {band_label}-band (observed)")
        ax2b.set_ylabel("Observed ZTF optical flux [mJy]", color="dimgray")
        ax2b.tick_params(axis="y", labelcolor="dimgray")

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
        ax2.set_title("Simulated vs. observed optical light curve (independent axes -- shape comparison only)",
                       fontsize=11)
    else:
        ax2.legend(fontsize=8, loc="upper right")
        ax2.set_title("Simulated optical light curve (no ZTF coverage in this outburst window)", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_outburst_spans():
    """Load the fred_model outburst summary and return, per source, the
    list of (t_start, t_end, outburst_idx) tuples for the X-ray band."""
    if os.path.exists(SUMMARY_CSV):
        df = pd.read_csv(SUMMARY_CSV)
    else:
        # fall back to stitching together per-source CSVs (e.g. if the
        # master summary hasn't been rebuilt since the latest run)
        rows = []
        for source_dir in glob.glob(os.path.join(PLOT_DIR, "*")):
            source_key = os.path.basename(source_dir)
            src_csv = os.path.join(source_dir, f"{source_key}_norris_tophat_outbursts.csv")
            if os.path.exists(src_csv):
                rows.append(pd.read_csv(src_csv))
        if not rows:
            print(f"[ERROR] No outburst summary found at {SUMMARY_CSV} and no "
                  f"per-source outburst CSVs found under {PLOT_DIR}.")
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

    os.makedirs(OPTICAL_DIR, exist_ok=True)

    lookup = load_classification(CLASSIFICATION_CSV)
    spans_by_source = load_outburst_spans()
    if not spans_by_source:
        return

    maxi_files = fm.find_maxi_files(SHORTLISTED_DIR)
    ztf_files = find_ztf_files(SHORTLISTED_DIR)
    ztf_lookup = build_canonical_lookup(ztf_files)

    # -----------------------------------------------------------------
    # Step 1: establish beta_BH and beta_NS from quasi-simultaneous
    # MAXI/ZTF pairs, pooled across every outburst of every source in
    # each compact-object class.
    # -----------------------------------------------------------------
    print("\n[1/2] Fitting global beta_BH / beta_NS from quasi-simultaneous pairs...")
    pooled = {"BH": ([], [], []), "NS": ([], [], [])}
    for source_key, spans in spans_by_source.items():
        co_label, _ = classify_source(source_key, lookup)
        if co_label not in ("BH", "NS"):
            continue
        xr, opt, opterr = collect_quasi_simultaneous_pairs(source_key, [(s[0], s[1]) for s in spans])
        if len(xr):
            pooled[co_label][0].extend(xr)
            pooled[co_label][1].extend(opt)
            pooled[co_label][2].extend(opterr)

    beta_results = {}
    for co_label in ("BH", "NS"):
        xr, opt, opterr = pooled[co_label]
        fit = fit_beta(xr, opt, opterr)
        if fit is None:
            print(f"  [{co_label}] insufficient quasi-simultaneous pairs "
                  f"({len(xr)} found, need >= {MIN_PAIRS_FOR_FIT}); "
                  f"falling back to baseline beta = {BETA_BASELINE}")
            beta_results[co_label] = {"beta": BETA_BASELINE, "beta_err": float("nan"), "n_pairs": len(xr)}
        else:
            print(f"  [{co_label}] beta = {fit['beta']:.3f} +/- {fit['beta_err']:.3f}  "
                  f"(n_pairs = {fit['n_pairs']})")
            beta_results[co_label] = fit
    beta_results["UNKNOWN"] = {"beta": BETA_BASELINE, "beta_err": float("nan"), "n_pairs": 0}

    beta_df = pd.DataFrame([
        {"compact_object": k, "beta": v["beta"], "beta_err": v["beta_err"], "n_pairs": v["n_pairs"]}
        for k, v in beta_results.items()
    ])
    beta_df.to_csv(BETA_CSV_PATH, index=False)
    with open(BETA_LOG_PATH, "w", encoding="utf-8") as fh:
        fh.write(f"Optical reprocessing beta fit -- {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC\n")
        fh.write(f"Model: optical_flux = 1.0 * xray_flux**beta\n")
        fh.write(f"Baseline (unfit) reference value: beta = {BETA_BASELINE} (van Paradijs & McClintock 1994)\n")
        fh.write(f"Match tolerance for quasi-simultaneous pairs: {MATCH_TOL_DAYS} day(s)\n\n")
        fh.write(beta_df.to_string(index=False))
        fh.write("\n")
    print(f"  [saved] {BETA_CSV_PATH}")
    print(f"  [saved] {BETA_LOG_PATH}")

    # -----------------------------------------------------------------
    # Step 2: forward-simulate + plot each outburst
    # -----------------------------------------------------------------
    print("\n[2/2] Forward-simulating optical light curves per outburst...")
    for source_key, spans in sorted(spans_by_source.items()):
        if source_key not in maxi_files:
            print(f"  [skip] {source_key}: no MAXI file found")
            continue

        co_label, src_type = classify_source(source_key, lookup)
        beta = beta_results.get(co_label, beta_results["UNKNOWN"])["beta"]

        xdf = pd.read_csv(maxi_files[source_key])
        xdf.columns = [c.strip().lower() for c in xdf.columns]
        if "mjd" not in xdf.columns:
            print(f"  [skip] {source_key}: MAXI file missing an mjd column")
            continue
        try:
            xdf = standardize_maxi_columns(xdf)
        except KeyError as exc:
            print(f"  [skip] {source_key}: {exc}")
            continue
        xdf = add_xray_flux_columns(xdf)

        zdf = None
        ztf_entry = ztf_lookup.get(canonical_source_key(source_key))
        if ztf_entry is not None:
            zdf = load_ztf_lightcurve(ztf_entry[1])
            if zdf is not None and not zdf.empty:
                zdf["OPT_FLUX"], zdf["OPT_FLUX_ERR"] = mag_to_flux(zdf["MAG"].values, zdf["MAGERR"].values)

        source_dir = os.path.join(OPTICAL_DIR, source_key)
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

            optical_flux, phase, peak_idx = simulate_outburst_optical(mjd, xray_flux, beta)

            # --- output CSV of the simulation ---
            sim_df = pd.DataFrame({
                "mjd": mjd,
                "x_flux_crab_rel": xray_flux,
                "x_flux_err_crab_rel": xray_err,
                "predicted_optical_flux_arb": optical_flux,
                "phase": phase,
            })
            sim_csv_path = os.path.join(source_dir, f"{source_key}_outburst{outburst_idx}_optical_simulation.csv")
            sim_df.to_csv(sim_csv_path, index=False)

            # --- Plot 1: predicted optical + X-ray, 2-panel, no real optical data ---
            pred_plot_path = os.path.join(source_dir, f"{source_key}_outburst{outburst_idx}_optical_predicted.png")
            plot_predicted_only(source_key, XRAY_BAND, outburst_idx, mjd, xray_flux, xray_err,
                                 optical_flux, beta, co_label, t_start, t_end, pred_plot_path)

            # --- Plot 2: X-ray top, simulated + actual ZTF overlaid bottom ---
            ztf_window = None
            if zdf is not None and not zdf.empty:
                ztf_window = zdf[(zdf["MJD"] >= t_start) & (zdf["MJD"] <= t_end)]
            comp_plot_path = os.path.join(source_dir, f"{source_key}_outburst{outburst_idx}_optical_comparison.png")
            plot_comparison(source_key, XRAY_BAND, outburst_idx, mjd, xray_flux, xray_err,
                             optical_flux, beta, co_label, t_start, t_end, ztf_window, comp_plot_path)

            print(f"  [done] {source_key} outburst #{outburst_idx} ({co_label}, beta={beta:.3f}) -> "
                  f"{os.path.basename(pred_plot_path)}, {os.path.basename(comp_plot_path)}")

    print(f"\nDone. Outputs written under:\n  {OPTICAL_DIR}\\<source>\\ (per-outburst plots + CSVs)\n"
          f"  {OPTICAL_DIR}\\ (global beta fit: {os.path.basename(BETA_CSV_PATH)}, {os.path.basename(BETA_LOG_PATH)})")


if __name__ == "__main__":
    main()