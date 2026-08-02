"""
outburst_hardness.py
---------------------
Companion to the FRED (Norris * tophat) outburst-fitting script.

That script lets you manually drag-select outburst span(s) per source and
writes them to:
    outburst_plots/<source>/<source>_norris_tophat_outbursts.csv
(one row per band per outburst, with 'span_start_mjd'/'span_end_mjd' columns
that are IDENTICAL across bands for a given outburst_idx -- the span is
picked once on the 2-20 keV lightcurve and reused for all bands).

This script:
  1. Reads those span(s) back out (from the master summary CSV if it
     exists, else from the per-source CSVs directly).
  2. Loads each source's MAXI light curve from SHORTLISTED_DIR and computes
     HR_X = flux_10-20 / flux_4-10 with error propagation (same quality
     filtering as the hardness pipeline: drop non-finite/negative flux,
     drop >100% relative error).
  3. For EACH outburst span, crops the HR time series down to exactly that
     MJD window and plots it -- no data outside the selected span is shown.
  4. Also saves a flux+HR two-panel plot for context, an HR-vs-intensity
     "mini HID" for just that outburst (useful for checking hysteresis
     behavior within a single outburst), and a CSV of the cropped HR data.

Outputs land next to the existing FRED plots, in the same per-source folder:
    outburst_plots/<source>/<source>_outburst<i>_hardness.png
    outburst_plots/<source>/<source>_outburst<i>_hardness_HID.png
    outburst_plots/<source>/<source>_outburst<i>_hardness.csv

Run this AFTER the FRED script has at least one source with a completed
fit (i.e. its *_norris_tophat_outbursts.csv exists).
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Keep these in sync with the FRED script's paths.
SHORTLISTED_DIR = r"C:\Users\abhin\Documents\ksp2026\Shortlisted_Data"
FRED_DIR        = r"C:\Users\abhin\Documents\ksp2026\FRED_Data"
PLOT_DIR        = os.path.join(FRED_DIR, "outburst_plots")

# Small breathing-room margin (days) added to the x-axis ONLY -- data points
# plotted are still cropped strictly to the selected span. Set to 0 for an
# axis that starts/ends exactly at the span edges.
AXIS_PAD_DAYS = 1.0


# ---------------------------------------------------------------------------
# Locate a source's MAXI file (mirrors find_maxi_files in the FRED script)
# ---------------------------------------------------------------------------
def find_maxi_files(directory):
    import re
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
            key = re.sub(r"(_maxi_lc|_maxi)$", "", base)
            if key not in found:
                found[key] = fp
    return found


# ---------------------------------------------------------------------------
# Load MAXI light curve + compute HR_X with error propagation
# ---------------------------------------------------------------------------
def load_maxi_hr(fp):
    """Returns a dataframe: mjd, flux_2_20, hr_x, hr_x_err (sorted by mjd),
    after dropping non-finite/negative-flux/high-relative-error rows."""
    df = pd.read_csv(fp)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"mjd", "flux_4_10", "flux_10_20"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [SKIP] {fp}: missing required column(s) {missing}. "
              f"Got: {df.columns.tolist()}")
        return None

    for col in ["err_4_10", "err_10_20", "flux_2_20"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df[["mjd", "flux_4_10", "err_4_10", "flux_10_20", "err_10_20", "flux_2_20"]].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["mjd", "flux_4_10", "flux_10_20"])

    df = df[(df["flux_4_10"] > 0) & (df["flux_10_20"] > 0)]
    for flux_col, err_col in [("flux_4_10", "err_4_10"), ("flux_10_20", "err_10_20")]:
        has_err = df[err_col].notna()
        bad_rel_err = has_err & ((df[err_col] / df[flux_col]).abs() > 1.0)
        df = df[~bad_rel_err]

    df["hr_x"] = df["flux_10_20"] / df["flux_4_10"]
    df["hr_x_err"] = df["hr_x"] * np.sqrt(
        (df["err_10_20"] / df["flux_10_20"]) ** 2 + (df["err_4_10"] / df["flux_4_10"]) ** 2
    )
    return df.sort_values("mjd").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Load outburst spans written by the FRED script
# ---------------------------------------------------------------------------
def load_outburst_spans():
    """Returns {source_key: [(outburst_idx, t_start, t_end), ...], ...}.
    Prefers the master summary CSV (rebuilt fully on every FRED run); falls
    back to scanning per-source CSVs directly if that isn't there yet."""
    master_csv = os.path.join(FRED_DIR, "norris_tophat_outburst_summary.csv")
    if os.path.exists(master_csv):
        df = pd.read_csv(master_csv)
    else:
        rows = []
        for fp in glob.glob(os.path.join(PLOT_DIR, "*", "*_norris_tophat_outbursts.csv")):
            rows.append(pd.read_csv(fp))
        if not rows:
            return {}
        df = pd.concat(rows, ignore_index=True)

    required = {"source", "outburst_idx", "span_start_mjd", "span_end_mjd"}
    if not required.issubset(df.columns):
        raise ValueError(f"Outburst CSV is missing expected columns: "
                          f"{required - set(df.columns)}")

    # Span times are identical across bands for a given (source, outburst_idx)
    # -- keep one row per outburst.
    dedup = df.drop_duplicates(subset=["source", "outburst_idx"])

    spans_by_source = {}
    for _, row in dedup.iterrows():
        spans_by_source.setdefault(row["source"], []).append(
            (int(row["outburst_idx"]), float(row["span_start_mjd"]), float(row["span_end_mjd"]))
        )
    for src in spans_by_source:
        spans_by_source[src].sort(key=lambda x: x[0])
    return spans_by_source


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_outburst_hardness(source_key, hr_df, idx, t_start, t_end, out_dir):
    mask = (hr_df["mjd"] >= t_start) & (hr_df["mjd"] <= t_end)
    sub = hr_df[mask].copy()

    if sub.empty:
        print(f"  [warn] {source_key} outburst #{idx}: no MAXI HR points fall "
              f"inside MJD {t_start:.3f}-{t_end:.3f} -- skipping plots.")
        return

    has_flux = sub["flux_2_20"].notna().any()
    n_panels = 2 if has_flux else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3.2 * n_panels), sharex=True)
    axes = [axes] if n_panels == 1 else list(axes)

    i = 0
    if has_flux:
        axes[i].plot(sub["mjd"], sub["flux_2_20"], "-o", ms=3, lw=0.8, color="steelblue")
        axes[i].set_ylabel("Flux 2-20 keV\n(ph/s/cm$^2$)")
        axes[i].set_title(f"{source_key} -- Outburst #{idx}  (MJD {t_start:.2f}-{t_end:.2f})")
        i += 1

    axes[i].errorbar(sub["mjd"], sub["hr_x"], yerr=sub["hr_x_err"],
                      fmt="o", ms=3.5, elinewidth=0.8, capsize=0, color="firebrick")
    axes[i].set_ylabel(r"HR$_X$ = F(10-20)/F(4-10)")
    axes[i].set_xlabel("MJD")
    if not has_flux:
        axes[i].set_title(f"{source_key} -- Outburst #{idx}  (MJD {t_start:.2f}-{t_end:.2f})")

    for ax in axes:
        ax.set_xlim(t_start - AXIS_PAD_DAYS, t_end + AXIS_PAD_DAYS)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{source_key}_outburst{idx}_hardness.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [plot]  -> {out_path}  ({len(sub)} HR point(s) in span)")

    # Mini HID for just this outburst, colored by time -- lets you see
    # whether the hard-to-soft transition traces a hysteresis-style loop
    # within the outburst itself.
    if has_flux and sub["flux_2_20"].notna().sum() >= 2:
        fig2, ax2 = plt.subplots(figsize=(5.5, 5))
        sc = ax2.scatter(sub["flux_2_20"], sub["hr_x"], c=sub["mjd"], cmap="viridis", s=28)
        ax2.plot(sub["flux_2_20"], sub["hr_x"], "-", lw=0.6, color="gray", alpha=0.5, zorder=0)
        ax2.set_xlabel("Flux 2-20 keV (ph/s/cm$^2$)")
        ax2.set_ylabel(r"HR$_X$ = F(10-20)/F(4-10)")
        ax2.set_title(f"{source_key} -- Outburst #{idx} HID")
        cbar = fig2.colorbar(sc, ax=ax2)
        cbar.set_label("MJD")
        fig2.tight_layout()
        hid_path = os.path.join(out_dir, f"{source_key}_outburst{idx}_hardness_HID.png")
        fig2.savefig(hid_path, dpi=150)
        plt.close(fig2)
        print(f"  [plot]  -> {hid_path}")

    csv_path = os.path.join(out_dir, f"{source_key}_outburst{idx}_hardness.csv")
    sub.to_csv(csv_path, index=False)
    print(f"  [csv]   -> {csv_path}")


def main():
    spans_by_source = load_outburst_spans()
    if not spans_by_source:
        print("No outburst spans found. Run the FRED script first so at "
              "least one source has a completed "
              "*_norris_tophat_outbursts.csv.")
        return

    maxi_files = find_maxi_files(SHORTLISTED_DIR)

    for source_key, spans in spans_by_source.items():
        print(f"\n{'='*60}\nSource : {source_key}  ({len(spans)} outburst span(s))")
        fp = maxi_files.get(source_key)
        if fp is None:
            print(f"  [SKIP] No MAXI file found for '{source_key}' in {SHORTLISTED_DIR}")
            continue

        hr_df = load_maxi_hr(fp)
        if hr_df is None or hr_df.empty:
            print(f"  [SKIP] Could not build a usable HR series from {fp}")
            continue

        out_dir = os.path.join(PLOT_DIR, source_key)
        os.makedirs(out_dir, exist_ok=True)

        for idx, t_start, t_end in spans:
            plot_outburst_hardness(source_key, hr_df, idx, t_start, t_end, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()