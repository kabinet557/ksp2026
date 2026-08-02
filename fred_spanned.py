import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Cursor
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
from scipy.signal import fftconvolve
from datetime import datetime

warnings.filterwarnings("ignore")

SHORTLISTED_DIR = r"C:\Users\abhin\Documents\ksp2026\Shortlisted_Data"
FRED_DIR        = r"C:\Users\abhin\Documents\ksp2026\FRED_Data"
PLOT_DIR        = os.path.join(FRED_DIR, "outburst_plots")

BANDS = [
    ("flux_2_20",  "2-20 keV",  "steelblue"),
    ("flux_2_4",   "2-4 keV",   "darkorange"),
    ("flux_4_10",  "4-10 keV",  "forestgreen"),
    ("flux_10_20", "10-20 keV", "crimson"),
]

# ---------------------------------------------------------------------------
# SIGMA_THRESH only draws a reference line on the selection plot as a guide.
# ---------------------------------------------------------------------------
SIGMA_THRESH = 5.0

# Extra context (days) shown around the fitted span in saved plots.
VIEW_PAD_DAYS = 20.0

MIN_SPAN_DAYS = 0.5   # ignore accidental drag-clicks shorter than this
MIN_FIT_POINTS = 10   # minimum points needed to fit 5 parameters

# ---------------------------------------------------------------------------
# RE-RUN BEHAVIOR:
# - FORCE_REFIT_SOURCES: list source keys to force re-fitting even if source
#   CSV already exists.
# - FORCE_RESELECT_SPANS: list source keys to force interactive GUI selection
#   even if stored spans exist in FRED_DIR.
# ---------------------------------------------------------------------------
FORCE_REFIT_SOURCES   = set()
FORCE_RESELECT_SPANS = set()


# ---------------------------------------------------------------------------
# SPAN STORAGE / LOADING UTILITIES
# ---------------------------------------------------------------------------
def load_stored_spans(source_key, fred_dir, plot_dir):
    """
    Looks for existing spans in CSV files in FRED_DIR or PLOT_DIR.
    Checks central span files, source-specific span files, or existing fit summaries.
    
    Recognizes column variants:
      - Start MJD: 'span_start_mjd', 't_start', 'start_mjd', 'mjd_start'
      - End MJD:   'span_end_mjd', 't_end', 'end_mjd', 'mjd_end'
    """
    candidate_files = [
        os.path.join(fred_dir, "outburst_spans.csv"),
        os.path.join(fred_dir, "spans.csv"),
        os.path.join(fred_dir, "norris_tophat_outburst_summary.csv"),
        os.path.join(fred_dir, f"{source_key}_spans.csv"),
        os.path.join(plot_dir, source_key, f"{source_key}_norris_tophat_outbursts.csv"),
    ]

    start_cols = ["span_start_mjd", "t_start", "start_mjd", "mjd_start"]
    end_cols   = ["span_end_mjd",   "t_end",   "end_mjd",   "mjd_end"]

    spans = []

    for fp in candidate_files:
        if not os.path.exists(fp):
            continue
        try:
            cdf = pd.read_csv(fp)
            cdf.columns = [c.strip().lower() for c in cdf.columns]

            # Filter by source if 'source' column exists
            if "source" in cdf.columns:
                cdf = cdf[cdf["source"].astype(str).str.strip().str.upper() == source_key.strip().upper()]

            if cdf.empty:
                continue

            s_col = next((c for c in start_cols if c in cdf.columns), None)
            e_col = next((c for c in end_cols if c in cdf.columns), None)

            if s_col and e_col:
                for _, row in cdf.iterrows():
                    try:
                        t_s = float(row[s_col])
                        t_e = float(row[e_col])
                        if np.isfinite(t_s) and np.isfinite(t_e) and t_e > t_s:
                            span_tuple = (t_s, t_e)
                            if span_tuple not in spans:
                                spans.append(span_tuple)
                    except (ValueError, TypeError):
                        continue
                if spans:
                    print(f"  [spans] Loaded {len(spans)} stored span(s) from: {fp}")
                    return spans
        except Exception:
            continue

    return []


def save_stored_spans(source_key, spans, fred_dir):
    """
    Saves manually selected spans to FRED_DIR/outburst_spans.csv for future runs.
    """
    spans_file = os.path.join(fred_dir, "outburst_spans.csv")
    new_rows = [{"source": source_key, "span_start_mjd": t_s, "span_end_mjd": t_e} for t_s, t_e in spans]
    
    if os.path.exists(spans_file):
        try:
            existing_df = pd.read_csv(spans_file)
            if "source" in existing_df.columns:
                existing_df = existing_df[existing_df["source"].astype(str).str.strip().str.upper() != source_key.strip().upper()]
            updated_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
        except Exception:
            updated_df = pd.DataFrame(new_rows)
    else:
        updated_df = pd.DataFrame(new_rows)

    updated_df.to_csv(spans_file, index=False)
    print(f"  [spans] Saved {len(spans)} span(s) to: {spans_file}")


# ---------------------------------------------------------------------------
# MODEL: Norris pulse (*) normalized tophat
# ---------------------------------------------------------------------------
def norris_model(t, A, t0, tau1, tau2):
    """The 'bare' Norris pulse, no smearing."""
    dt = t - t0
    safe = dt > 0
    result = np.zeros_like(t, dtype=float)
    result[safe] = A * np.exp(-tau1 / dt[safe] - dt[safe] / tau2)
    return result


def tophat_kernel(width, dt):
    """Normalized boxcar kernel of given width (days)."""
    n = max(1, int(round(width / dt)))
    if n % 2 == 0:
        n += 1
    return np.ones(n) / n


def make_norris_tophat_model(t_grid):
    """Closure evaluating Norris pulse convolved with tophat on uniform t_grid."""
    dt = t_grid[1] - t_grid[0]

    def model(t, A, t0, tau1, tau2, width):
        pulse = norris_model(t_grid, A, t0, tau1, tau2)
        w = max(width, dt)
        kernel = tophat_kernel(w, dt)
        smeared = fftconvolve(pulse, kernel, mode="same")
        return np.interp(t, t_grid, smeared)

    return model


def aic(n_params, n_data, chi2):
    return chi2 + 2 * n_params + (2 * n_params * (n_params + 1)) / max(n_data - n_params - 1, 1)


def _fit_single(model_fn, p0, bounds, t, f, sig):
    try:
        popt, pcov = curve_fit(
            model_fn, t, f,
            p0=p0, sigma=sig, absolute_sigma=True,
            bounds=bounds, maxfev=30000,
        )
        perr     = np.sqrt(np.diag(pcov))
        resid    = f - model_fn(t, *popt)
        chi2     = float(np.sum((resid / sig) ** 2))
        dof      = len(f) - len(popt)
        chi2_red = chi2 / dof if dof > 0 else np.inf
        aic_val  = aic(len(popt), len(f), chi2)
        return dict(popt=popt, perr=perr,
                    chi2=chi2, chi2_red=chi2_red, dof=dof, aic=aic_val)
    except Exception as exc:
        return {"error": str(exc)}


def compute_shape_params(model_fn, popt, t_start, t_end, n=5000):
    t0, tau1, tau2, width = popt[1], popt[2], popt[3], popt[4]

    t_fine = np.linspace(t_start, t_end, n)
    f = model_fn(t_fine, *popt)
    f = np.clip(f, 0, None)
    if f.max() <= 0:
        return None

    peak_idx = int(np.argmax(f))
    t_peak = t_fine[peak_idx] - t0

    cum = cumulative_trapezoid(f, t_fine, initial=0.0)
    total = cum[-1]
    if total <= 0:
        return None
    frac = cum / total
    t5  = float(np.interp(0.05, frac, t_fine))
    t95 = float(np.interp(0.95, frac, t_fine))
    t90 = t95 - t5

    return dict(t0=t0, t_rise=tau1, t_decay=tau2, width=width, t_peak=t_peak,
                t5=t5, t95=t95, t90=t90)


def robust_baseline_threshold(flux, sigma_thresh=SIGMA_THRESH):
    f = flux[np.isfinite(flux)]
    baseline = float(np.median(f))
    mad = float(np.median(np.abs(f - baseline))) * 1.4826
    if mad <= 0:
        mad = float(np.std(f)) or 1e-6
    threshold = baseline + sigma_thresh * mad
    return baseline, threshold


def fit_norris_tophat_span(mjd, flux, err, t_start, t_end):
    mask = np.isfinite(flux) & np.isfinite(err) & (err > 0) & (mjd >= t_start) & (mjd <= t_end)
    if mask.sum() < MIN_FIT_POINTS:
        return None

    t, f, sig = mjd[mask], flux[mask], err[mask]
    peak_idx = int(np.argmax(f))
    A0   = max(f[peak_idx], 1e-6)
    t0_0 = np.clip(t[peak_idx] - 5.0, t.min() - 50, t[peak_idx] - 0.5)

    sorted_t = np.sort(t)
    cadence = float(np.median(np.diff(sorted_t))) if len(sorted_t) > 1 else 1.0
    W0 = max(cadence, 0.3)

    span = max(t_end - t_start, 1.0)
    width_hi = max(span * 0.5, W0 * 5)

    dt_grid = max(span / 3000.0, 0.02)
    pad = max(width_hi * 2, 15.0)
    t_grid = np.arange(t_start - pad, t_end + pad + dt_grid, dt_grid)
    model = make_norris_tophat_model(t_grid)

    res = _fit_single(
        model,
        p0=[A0 * 1.5, t0_0, 2.0, 20.0, W0],
        bounds=([0, t.min() - 100, 0.01, 0.1, 0.01],
                [np.inf, t[peak_idx], 500., 1000., width_hi]),
        t=t, f=f, sig=sig,
    )
    if "popt" not in res:
        return res

    res["t_data"], res["f_data"], res["sig_data"] = t, f, sig
    res["model_fn"] = model
    res["shape"] = compute_shape_params(model, res["popt"], t.min(), t.max())

    model_f = model(t, *res["popt"])
    resid = f - model_f
    res["residuals"]       = resid
    res["residuals_sigma"] = resid / sig
    res["resid_mean"]      = float(np.mean(resid))
    res["resid_std"]       = float(np.std(resid))
    res["resid_max_abs"]   = float(np.max(np.abs(resid)))
    return res


# ---------------------------------------------------------------------------
# Interactive manual span selection
# ---------------------------------------------------------------------------
def interactive_select_spans(mjd, flux, err, baseline, threshold, source_key, band_label):
    spans = []
    span_patches = []

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.plot(mjd, flux, "-o", ms=2.5, lw=0.8, color="steelblue", alpha=0.85, label="Lightcurve")
    ax.axhline(baseline,  color="darkgreen", ls="--", lw=1.2, label="Quiescent Baseline")
    ax.axhline(threshold, color="crimson",   ls=":",  lw=1.2, label=f"Reference ({SIGMA_THRESH:.0f}σ)")
    ax.set_title(
        f"{source_key}  --  {band_label}\n"
        "Drag to mark an outburst span   |   'z' = undo last   |   Enter or close window = done",
        fontsize=11,
    )
    ax.set_xlabel("MJD")
    ax.set_ylabel("Flux (ph/cm2/s)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper right")

    cursor = Cursor(ax, useblit=True, color="gray", linewidth=0.6)

    def onselect(xmin, xmax):
        if xmax - xmin < MIN_SPAN_DAYS:
            return
        spans.append((xmin, xmax))
        patch = ax.axvspan(xmin, xmax, color="orange", alpha=0.25, zorder=0)
        span_patches.append(patch)
        fig.canvas.draw_idle()
        print(f"    [select] span {len(spans)}: MJD {xmin:.3f} - {xmax:.3f}  "
              f"(duration {xmax - xmin:.2f} d)")

    def onkey(event):
        if event.key == "z" and spans:
            spans.pop()
            span_patches.pop().remove()
            fig.canvas.draw_idle()
            print("    [select] undid last span")
        elif event.key == "enter":
            plt.close(fig)

    selector = SpanSelector(
        ax, onselect, "horizontal",
        useblit=True,
        props=dict(alpha=0.25, facecolor="orange"),
        interactive=True,
        drag_from_anywhere=True,
        minspan=MIN_SPAN_DAYS,
    )
    fig.canvas.mpl_connect("key_press_event", onkey)

    plt.show()
    return spans


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_fred_characterization(source_key, band_label, mjd, flux, err,
                                baseline, threshold, t_start, t_end,
                                fit_res, out_path):
    view_lo, view_hi = t_start - VIEW_PAD_DAYS, t_end + VIEW_PAD_DAYS
    view_mask = np.isfinite(mjd) & (mjd >= view_lo) & (mjd <= view_hi)
    mjd_v, flux_v, err_v = mjd[view_mask], flux[view_mask], err[view_mask]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.errorbar(mjd_v, flux_v, yerr=err_v, fmt="o-", ms=3, lw=0.8,
                color="steelblue", ecolor="lightsteelblue", alpha=0.85,
                label="Lightcurve", zorder=3)
    ax.axhline(baseline,  color="darkgreen", ls="--", lw=1.4, label="Quiescent Baseline")
    ax.axhline(threshold, color="crimson",   ls=":",  lw=1.4, label="Reference Threshold")
    ax.axvspan(t_start, t_end, color="steelblue", alpha=0.12, label="Selected Fit Window")

    if fit_res and "popt" in fit_res and fit_res.get("shape"):
        p, sh = fit_res["popt"], fit_res["shape"]
        model_fn = fit_res["model_fn"]
        t_fine  = np.linspace(t_start, t_end, 3000)
        f_model = model_fn(t_fine, *p)
        ax.plot(t_fine, f_model, color="red", lw=2.2, label="Norris ⊛ Tophat Fit", zorder=5)

        ax.axvspan(sh["t5"], sh["t95"], color="orange", alpha=0.18, zorder=1, label="T90 window")

        t_peak_abs = sh["t0"] + sh["t_peak"]
        f_peak = model_fn(np.array([t_peak_abs]), *p)[0]
        ax.scatter([t_peak_abs], [f_peak], marker="x", s=90, color="black", zorder=6, label="Peak")

        y_arrow = f_peak * 0.5 if f_peak > 0 else max(flux_v.max(), 1e-6) * 0.5
        ax.annotate("", xy=(t_peak_abs, y_arrow), xytext=(sh["t0"], y_arrow),
                     arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1))
        ax.text((sh["t0"] + t_peak_abs) / 2, y_arrow, "rise", fontsize=8,
                color="dimgray", ha="center", va="bottom")

        info = (
            "Fit Parameters (Norris ⊛ Tophat):\n"
            f"A     = {p[0]:.3f}\n"
            f"t0    = {p[1]:.2f}\n"
            f"tau1  = {p[2]:.2f}\n"
            f"tau2  = {p[3]:.2f}\n"
            f"W     = {p[4]:.2f}\n"
            f"t_peak= {sh['t_peak']:.2f}\n"
            f"T90   = {sh['t90']:.2f}\n"
            f"chi2_v= {fit_res['chi2_red']:.3f}"
        )
        ax.text(0.02, 0.97, info, transform=ax.transAxes,
                va="top", ha="left", fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="#fdf6e3", alpha=0.9))
    else:
        ax.text(0.5, 0.5, "Fit failed / insufficient data",
                transform=ax.transAxes, ha="center", va="center", color="gray")

    ax.set_xlim(view_lo, view_hi)
    if len(flux_v):
        ymin = min(0, np.nanmin(flux_v)) * 1.1 if np.nanmin(flux_v) < 0 else 0
        ymax = max(np.nanmax(flux_v), threshold) * 1.25
        ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel("MJD")
    ax.set_ylabel("Flux (ph/cm2/s)")
    ax.set_title(f"{source_key}  --  {band_label}  FRED Characterization")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.2)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(source_key, band_label, fit_res, t_start, t_end, out_path):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    if fit_res and "residuals_sigma" in fit_res:
        ax.axhline(0, color="black", lw=1)
        ax.axhspan(-1, 1, color="gray", alpha=0.15, label="±1σ")
        ax.axhspan(-2, 2, color="gray", alpha=0.08, label="±2σ")
        ax.errorbar(fit_res["t_data"], fit_res["residuals_sigma"], yerr=1.0,
                    fmt="o", ms=3.5, color="steelblue", alpha=0.8, ecolor="lightsteelblue")
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No fit / no residuals available",
                transform=ax.transAxes, ha="center", va="center", color="gray")
    ax.set_xlim(t_start - VIEW_PAD_DAYS, t_end + VIEW_PAD_DAYS)
    ax.set_xlabel("MJD")
    ax.set_ylabel("Residual (sigma)")
    ax.set_title(f"{source_key}  --  {band_label}  Fit Residuals")
    ax.grid(alpha=0.2)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_param_distributions(df, out_dir):
    params = [
        ("t_rise",  "Rise timescale  tau1  [days]"),
        ("t_decay", "Decay timescale  tau2  [days]"),
        ("width",   "Tophat width  W  [days]"),
        ("t_peak",  "Time to peak  (t_peak - t0)  [days]"),
        ("t90",     "T90  [days]"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for ax, (col, label) in zip(axes, params):
        if col not in df.columns:
            ax.set_title(f"{label}\n(column missing)")
            ax.axis("off")
            continue
        vals = df[col].dropna().values
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            ax.set_title(f"{label}\n(no data)")
            ax.axis("off")
            continue
        n_bins = int(np.clip(len(vals) // 2, 5, 20))
        ax.hist(vals, bins=n_bins, color="steelblue", edgecolor="black", alpha=0.75)
        ax.axvline(np.median(vals), color="crimson", ls="--", lw=1.5,
                   label=f"median = {np.median(vals):.2f}")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(label.split("[")[0].strip())
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    for ax in axes[len(params):]:
        ax.axis("off")
    fig.suptitle(f"Distributions of Outburst Shape Parameters (n={len(df)} fits)", fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "shape_parameter_distributions.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] shape-parameter distributions -> {out_path}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _fmt_outburst_block(band_label, idx, t_start, t_end, baseline, threshold, fit_res):
    lines = [f"  Outburst #{idx}  ({band_label})",
             f"    Window   : MJD {t_start:.4f} to {t_end:.4f}",
             f"    Baseline : {baseline:.5f}   Reference threshold : {threshold:.5f}"]
    if fit_res is None:
        lines.append("    SKIPPED (insufficient data in window)")
    elif "error" in fit_res:
        lines.append(f"    FIT FAILED: {fit_res['error']}")
    else:
        p, pe = fit_res["popt"], fit_res["perr"]
        lines += [
            "    Model  : Norris pulse (*) normalized tophat",
            f"    A      = {p[0]:.6e}  +/-  {pe[0]:.6e}",
            f"    t0     = {p[1]:.4f}   +/-  {pe[1]:.4f}   [MJD]",
            f"    tau1   = {p[2]:.4f}   +/-  {pe[2]:.4f}   [days]",
            f"    tau2   = {p[3]:.4f}   +/-  {pe[3]:.4f}   [days]",
            f"    W      = {p[4]:.4f}   +/-  {pe[4]:.4f}   [days]  (tophat width)",
            f"    chi2   = {fit_res['chi2']:.4f}   dof = {fit_res['dof']}   chi2/dof = {fit_res['chi2_red']:.4f}",
            f"    AIC    = {fit_res['aic']:.4f}",
        ]
        sh = fit_res.get("shape")
        if sh:
            lines += [
                f"    t_peak = {sh['t_peak']:.4f} days after t0",
                f"    t5,t95 = {sh['t5']:.4f}, {sh['t95']:.4f}  [MJD]",
                f"    T90    = {sh['t90']:.4f} days",
            ]
        lines += [
            f"    resid mean = {fit_res['resid_mean']:.5f}",
            f"    resid std  = {fit_res['resid_std']:.5f}",
            f"    resid max|.| = {fit_res['resid_max_abs']:.5f}",
        ]
        if "t_data" in fit_res:
            lines.append("    Residuals (MJD, resid, resid_sigma):")
            for tt, rr, rs in zip(fit_res["t_data"], fit_res["residuals"], fit_res["residuals_sigma"]):
                lines.append(f"      {tt:.4f}   {rr:.6e}   {rs:.3f}")
    lines.append("")
    return lines


def write_source_log(source_key, per_band_outbursts, log_path, mode="w"):
    sep = "=" * 72
    lines = [sep, f"SOURCE : {source_key}",
              f"TIME   : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
              "MODEL  : Norris pulse (*) tophat", sep, ""]
    for band_label, outbursts in per_band_outbursts.items():
        if not outbursts:
            lines.append(f"  {band_label}: no spans available\n")
            continue
        for idx, (t_start, t_end, baseline, threshold, fit_res) in enumerate(outbursts, start=1):
            lines += _fmt_outburst_block(band_label, idx, t_start, t_end, baseline, threshold, fit_res)
    lines += [sep, ""]
    with open(log_path, mode, encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  [log]   written -> {log_path}")


def find_maxi_files(directory):
    patterns = [
        os.path.join(directory, "*_maxi_lc.csv"),
        os.path.join(directory, "*_maxi.csv"),
        os.path.join(directory, "*_maxi_lc"),
        os.path.join(directory, "*_maxi"),
    ]
    found = {}
    for pat in patterns:
        for fp in glob.glob(pat):
            base = os.path.splitext(os.path.basename(fp))[0]
            key  = re.sub(r"(_maxi_lc|_maxi)$", "", base)
            if key not in found:
                found[key] = fp
    return found


def main():
    os.makedirs(FRED_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    master_log = os.path.join(FRED_DIR, "all_sources_norris_tophat_conv.log")

    files = find_maxi_files(SHORTLISTED_DIR)
    if not files:
        print(f"[ERROR] No MAXI CSV files found in:\n  {SHORTLISTED_DIR}")
        return
    print(f"Found {len(files)} source(s): {', '.join(sorted(files))}\n")

    summary_rows = []

    for source_key in sorted(files):
        fp = files[source_key]

        source_dir = os.path.join(PLOT_DIR, source_key)
        src_csv = os.path.join(source_dir, f"{source_key}_norris_tophat_outbursts.csv")

        # Skip if source is completely done and not in FORCE_REFIT_SOURCES
        if os.path.exists(src_csv) and source_key not in FORCE_REFIT_SOURCES:
            existing = pd.read_csv(src_csv)
            summary_rows.extend(existing.to_dict("records"))
            print(f"\n{'='*60}\nSource : {source_key}\n"
                  f"  [skip] already fit ({len(existing)} outburst rows found at {src_csv})")
            continue

        print(f"\n{'='*60}\nSource : {source_key}\nFile   : {fp}")
        try:
            df = pd.read_csv(fp)
        except Exception as exc:
            print(f"  [SKIP] Read error: {exc}")
            continue

        df.columns = [c.strip().lower() for c in df.columns]
        if "mjd" not in df.columns:
            print(f"  [SKIP] No MJD column. Got: {df.columns.tolist()}")
            continue
        mjd = df["mjd"].values

        os.makedirs(source_dir, exist_ok=True)

        # -------------------------------------------------------------------
        # 1. LOAD SPANS OR OPEN INTERACTIVE SELECTION
        # -------------------------------------------------------------------
        spans = []
        if source_key not in FORCE_RESELECT_SPANS:
            spans = load_stored_spans(source_key, FRED_DIR, PLOT_DIR)

        if not spans:
            # Fallback to interactive selection if no pre-stored spans found
            for flux_col, band_label, _ in BANDS:
                if band_label == "2-20 keV" and flux_col in df.columns:
                    err_col = flux_col.replace("flux_", "err_")
                    flux = df[flux_col].values
                    err  = df[err_col].values if err_col in df.columns else np.ones(len(mjd)) * 0.01
                    valid = np.isfinite(flux) & np.isfinite(err) & (err > 0)
                    if valid.sum() >= 12:
                        baseline, threshold = robust_baseline_threshold(flux[valid])
                        print(f"  {band_label:12s} -> opening interactive selection window "
                              f"(baseline={baseline:.4f}, reference={threshold:.4f})")
                        spans = interactive_select_spans(mjd, flux, err, baseline, threshold,
                                                      source_key, band_label)
                        print(f"  {band_label:12s} -> {len(spans)} span(s) selected interactively")
                        if spans:
                            save_stored_spans(source_key, spans, FRED_DIR)
                    break

        if not spans:
            print(f"  [SKIP] No outburst spans available for {source_key}.")
            continue

        # -------------------------------------------------------------------
        # 2. FIT NORRIS TOPHAT MODEL & SAVE RESULTS ACROSS BANDS
        # -------------------------------------------------------------------
        per_band_outbursts = {}
        source_rows = []

        for flux_col, band_label, _ in BANDS:
            if flux_col not in df.columns:
                continue
            err_col = flux_col.replace("flux_", "err_")
            flux = df[flux_col].values
            err  = df[err_col].values if err_col in df.columns else np.ones(len(mjd)) * 0.01

            valid = np.isfinite(flux) & np.isfinite(err) & (err > 0)
            if valid.sum() < 12:
                per_band_outbursts[band_label] = []
                continue

            baseline, threshold = robust_baseline_threshold(flux[valid])

            outbursts = []
            for i, (t_start, t_end) in enumerate(spans, start=1):
                fit_res = fit_norris_tophat_span(mjd, flux, err, t_start, t_end)
                outbursts.append((t_start, t_end, baseline, threshold, fit_res))

                fred_path  = os.path.join(source_dir, f"{flux_col}_outburst{i}_fred.png")
                resid_path = os.path.join(source_dir, f"{flux_col}_outburst{i}_resid.png")
                plot_fred_characterization(source_key, band_label, mjd, flux, err,
                                            baseline, threshold, t_start, t_end,
                                            fit_res, fred_path)
                plot_residuals(source_key, band_label, fit_res, t_start, t_end, resid_path)

                if fit_res and "popt" in fit_res and fit_res.get("shape"):
                    p, sh = fit_res["popt"], fit_res["shape"]
                    row = {
                        "source": source_key, "band": band_label, "outburst_idx": i,
                        "span_start_mjd": t_start, "span_end_mjd": t_end,
                        "baseline": baseline, "threshold": threshold,
                        "A": p[0], "t0": p[1], "tau1": p[2], "tau2": p[3], "width": p[4],
                        "t_rise": sh["t_rise"], "t_decay": sh["t_decay"],
                        "chi2": fit_res["chi2"], "dof": fit_res["dof"],
                        "chi2_red": fit_res["chi2_red"], "aic": fit_res["aic"],
                        "t_peak": sh["t_peak"], "t5": sh["t5"], "t95": sh["t95"], "t90": sh["t90"],
                        "resid_mean": fit_res["resid_mean"], "resid_std": fit_res["resid_std"],
                        "resid_max_abs": fit_res["resid_max_abs"],
                        "fred_plot_path": fred_path, "resid_plot_path": resid_path,
                    }
                    summary_rows.append(row)
                    source_rows.append(row)
            per_band_outbursts[band_label] = outbursts

        src_log = os.path.join(source_dir, f"{source_key}_norris_tophat_log.txt")
        write_source_log(source_key, per_band_outbursts, src_log, mode="w")

        if source_rows:
            pd.DataFrame(source_rows).to_csv(src_csv, index=False)
            print(f"  [csv]   written -> {src_csv}")

    # Rebuild master log
    with open(master_log, "w", encoding="utf-8") as out_fh:
        out_fh.write(f"Norris (*) Tophat Batch Log\nRebuilt: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC\n"
                      + "=" * 72 + "\n\n")
        for source_key in sorted(files):
            src_log = os.path.join(PLOT_DIR, source_key, f"{source_key}_norris_tophat_log.txt")
            if os.path.exists(src_log):
                with open(src_log, "r", encoding="utf-8") as in_fh:
                    out_fh.write(in_fh.read())
    print(f"\n[log] master log rebuilt -> {master_log}")

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_csv = os.path.join(FRED_DIR, "norris_tophat_outburst_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n[summary] {len(summary_df)} outburst fits -> {summary_csv}")
        plot_param_distributions(summary_df, FRED_DIR)
    else:
        print("\n[summary] No successful fits -- nothing to summarize.")

    print(f"\nDone. Per-source folders (plots, log, csv) in: {PLOT_DIR}\\<source_name>\\")


if __name__ == "__main__":
    main()