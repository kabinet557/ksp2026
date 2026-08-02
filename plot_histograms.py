import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse optical_sim's classification loader + source-name canonicalizer
# (query_name <-> file_stem formatting differences, e.g. "AQL X-1" vs
# "AQL_X-1") instead of duplicating that matching logic here.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optical_sim as osim  # noqa: E402

FRED_DIR    = osim.FRED_DIR   # Fit_Fred -- same summary CSV fred_model.py writes
SUMMARY_CSV = os.path.join(FRED_DIR, "norris_tophat_outburst_summary.csv")
OUT_DIR     = os.path.join(FRED_DIR, "summary_histograms")

# Per-source classification (BH/NS, LMXB/HMXB), produced by the
# classification pipeline -- same file optical_sim.py reads.
CLASSIFICATION_CSV = osim.CLASSIFICATION_CSV

# Where the outburst summary + its auto-derived classification columns are
# saved, for inspection/provenance.
CLASSIFIED_SUMMARY_CSV = os.path.join(FRED_DIR, "norris_tophat_outburst_summary_classified.csv")

# ---------------------------------------------------------------------------
# Band Selection & Column Configuration
# ---------------------------------------------------------------------------
BAND_COL    = "band"          # Expected column name in CSV for energy band
TARGET_BAND = "2-20 keV"      # Target energy band label for plot titles

COMPACT_OBJECT_COL    = "compact_object"   # expected values: "BH", "NS"
SYSTEM_TYPE_COL       = "system_type"      # expected values: "LMXB", "HMXB"
COMPACT_OBJECT_VALUES = ["BH", "NS"]
SYSTEM_TYPE_VALUES    = ["LMXB", "HMXB"]


def filter_by_band(df, band_col=BAND_COL):
    """
    Filter dataframe to only include outbursts in the 2-20 keV energy band.
    Uses regex to handle minor variations like '2-20 keV', '2-20keV', or '2.0-20.0 keV'.
    """
    if band_col not in df.columns:
        # Check alternative common column names
        alt_cols = [c for c in df.columns if c.lower() in ["energy_band", "e_band", "passband"]]
        if alt_cols:
            band_col = alt_cols[0]
        else:
            print(f"[WARN] Band column '{band_col}' not found. Plotting all bands.")
            return df

    initial_count = len(df)
    # Match patterns like 2-20, 2.0-20.0, etc.
    pattern = r"2(?:\.0)?\s*-\s*20(?:\.0)?"
    mask = df[band_col].astype(str).str.contains(pattern, case=False, regex=True, na=False)
    filtered_df = df[mask].copy()

    print(f"[band filter] Retained {len(filtered_df)}/{initial_count} outbursts for band '{TARGET_BAND}' (from column '{band_col}')")
    return filtered_df


def attach_classification(df, classification_csv=CLASSIFICATION_CSV):
    """
    Look up each row's `source` (a fred_model file_stem, e.g. 'AQL_X-1') in
    the classification catalog (keyed by `query_name`, e.g. 'AQL X-1') and
    add COMPACT_OBJECT_COL / SYSTEM_TYPE_COL columns. "Unknown"/unresolved
    sources are left as NaN (excluded from grouped histograms, same as a
    blank manually-entered cell).
    """
    lookup = osim.load_classification(classification_csv)
    if not lookup:
        print(f"[WARN] No classification entries loaded from {classification_csv} -- "
              f"'{COMPACT_OBJECT_COL}'/'{SYSTEM_TYPE_COL}' will be empty for every row.")

    def _lookup(source, field):
        entry = lookup.get(osim.canonical_source_key(source))
        if entry is None:
            return np.nan
        val = entry.get(field, "UNKNOWN")
        return np.nan if str(val).upper() == "UNKNOWN" else val

    df = df.copy()
    df[COMPACT_OBJECT_COL] = df["source"].apply(lambda s: _lookup(s, "compact_object"))
    df[SYSTEM_TYPE_COL]    = df["source"].apply(lambda s: _lookup(s, "src_type"))

    n_rows = len(df)
    n_co   = df[COMPACT_OBJECT_COL].notna().sum()
    n_type = df[SYSTEM_TYPE_COL].notna().sum()
    print(f"[classify] {n_co}/{n_rows} outburst rows matched a compact-object class, "
          f"{n_type}/{n_rows} matched a system type (source: {classification_csv})")
    return df


# Parameters to histogram, one outburst per row.
PARAMS = [
    ("t_rise",  "Rise timescale  tau1  [days]"),
    ("t_decay", "Decay timescale  tau2  [days]"),
    ("t_peak",  "Time to peak  (t_peak - t0)  [days]"),
    ("t90",     "T90  [days]"),
]

COLORS = {
    "BH": "#4477AA", "NS": "#EE6677",
    "LMXB": "#228833", "HMXB": "#CCBB44",
    "BH-LMXB": "#4477AA", "BH-HMXB": "#66CCEE",
    "NS-LMXB": "#EE6677", "NS-HMXB": "#CCBB44",
}


def _check_column(df, col, expected_values):
    """Warn about anything that would silently vanish from a grouped plot."""
    if col not in df.columns:
        print(f"[WARN] Column '{col}' not found -- attach_classification() didn't run "
              f"or the summary CSV has no 'source' column. Expected values: {expected_values}")
        return False

    present = df[col].notna()
    unrecognized = df.loc[present & ~df[col].isin(expected_values), col]
    if len(unrecognized):
        print(f"[WARN] {len(unrecognized)} row(s) have an unrecognized value in "
              f"'{col}': {sorted(unrecognized.unique())}. These will be excluded "
              f"from histograms grouped by '{col}'.")

    n_missing = (~present).sum()
    if n_missing:
        print(f"[WARN] {n_missing} row(s) have no value in '{col}' (blank) -- "
              "excluded from histograms grouped by this column.")

    return True


def plot_grouped_histograms(df, group_col, group_values, title, out_path, source_col="source"):
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(5.5 * len(PARAMS), 4.5))
    axes = np.atleast_1d(axes)

    total_outbursts = len(df)
    total_sources = df[source_col].nunique() if source_col in df.columns else 0

    for ax, (col, label) in zip(axes, PARAMS):
        if col not in df.columns:
            ax.set_title(f"{label}\n(column missing)")
            ax.axis("off")
            continue

        any_data = False
        for gv in group_values:
            # Filter rows for the current group and parameter
            sub_df = df[df[group_col] == gv]
            valid_df = sub_df[sub_df[col].notna()].copy()
            vals = valid_df[col].values
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                continue

            any_data = True
            n_outbursts = len(vals)
            n_sources = valid_df[source_col].nunique() if source_col in valid_df.columns else 0

            n_bins = int(np.clip(n_outbursts // 2, 4, 15))
            
            # Format legend label with both outburst and unique source counts
            legend_label = f"{gv} ({n_outbursts} outbursts, {n_sources} sources)"

            ax.hist(vals, bins=n_bins, alpha=0.55, edgecolor="black",
                    color=COLORS.get(gv), label=legend_label)
            ax.axvline(np.median(vals), color=COLORS.get(gv), ls="--", lw=1.5)

        if not any_data:
            ax.set_title(f"{label}\n(no data)")
            ax.axis("off")
            continue

        ax.set_title(label, fontsize=10)
        ax.set_xlabel(label.split("[")[0].strip())
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"{title}\n(Total: {total_outbursts} outbursts, {total_sources} unique sources | Band: {TARGET_BAND})",
        fontweight="bold", y=1.03
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] -> {out_path}")


def main():
    if not os.path.exists(SUMMARY_CSV):
        print(f"[ERROR] Summary CSV not found: {SUMMARY_CSV}")
        print("        Run the FRED-fitting script first, or check FRED_DIR/SUMMARY_CSV above.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(SUMMARY_CSV)
    print(f"Loaded {len(df)} outburst rows from {SUMMARY_CSV}")

    # 1) Attach classifications (BH/NS, LMXB/HMXB)
    df = attach_classification(df)

    # 2) Filter to only select 2-20 keV energy band
    df = filter_by_band(df)

    # Save classified & filtered summary CSV
    df.to_csv(CLASSIFIED_SUMMARY_CSV, index=False)
    print(f"[saved] outburst summary + classification -> {CLASSIFIED_SUMMARY_CSV}")

    for col, _ in PARAMS:
        if col not in df.columns:
            print(f"[WARN] Expected parameter column '{col}' missing from the summary CSV.")

    # 3) Grouped by compact object (BH vs NS)
    if _check_column(df, COMPACT_OBJECT_COL, COMPACT_OBJECT_VALUES):
        plot_grouped_histograms(
            df, COMPACT_OBJECT_COL, COMPACT_OBJECT_VALUES,
            "Outburst Shape Parameters by Compact Object",
            os.path.join(OUT_DIR, "by_compact_object_2-20keV.png"),
        )

    # 4) Grouped by system type (LMXB vs HMXB)
    if _check_column(df, SYSTEM_TYPE_COL, SYSTEM_TYPE_VALUES):
        plot_grouped_histograms(
            df, SYSTEM_TYPE_COL, SYSTEM_TYPE_VALUES,
            "Outburst Shape Parameters by System Type",
            os.path.join(OUT_DIR, "by_system_type_2-20keV.png"),
        )

    # 5) Combined breakdown (BH-LMXB, BH-HMXB, NS-LMXB, NS-HMXB)
    if COMPACT_OBJECT_COL in df.columns and SYSTEM_TYPE_COL in df.columns:
        both_present = df[COMPACT_OBJECT_COL].notna() & df[SYSTEM_TYPE_COL].notna()
        df = df.copy()
        df["_combined"] = np.where(
            both_present,
            df[COMPACT_OBJECT_COL].astype(str) + "-" + df[SYSTEM_TYPE_COL].astype(str),
            np.nan,
        )
        combos = [f"{a}-{b}" for a in COMPACT_OBJECT_VALUES for b in SYSTEM_TYPE_VALUES]
        plot_grouped_histograms(
            df, "_combined", combos,
            "Outburst Shape Parameters by Compact Object x System Type",
            os.path.join(OUT_DIR, "by_combined_type_2-20keV.png"),
        )

    print(f"\nDone. Histograms saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()