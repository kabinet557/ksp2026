"""
run_classification.py
----------------------
Task 1 entry point: scans Total_Data, parses filenames into standardized
source identifiers, classifies each source as BH/NS and LMXB/HMXB via
cached catalog cross-matching, and writes two review CSVs.

USAGE
-----
python run_classification.py --data-dir /path/to/Total_Data --out-dir ./classification_output

Recommended first run: use --limit 5 to sanity check on a handful of
sources before committing to a full run (SIMBAD fallback calls and the
first-time catalog downloads are the slow part).
"""

import argparse
import logging
from pathlib import Path

from filename_parser import parse_directory, write_review_csv
from classifier import classify_sources, write_results_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Path to Total_Data")
    ap.add_argument("--out-dir", default="./classification_output")
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--limit", type=int, default=None,
                     help="Only process the first N parsed sources (for testing)")
    ap.add_argument("--match-radius-arcsec", type=float, default=10.0)
    ap.add_argument("--no-simbad-fallback", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: parse filenames ---
    parsed = parse_directory(args.data_dir)
    print(f"Parsed {len(parsed)} files from {args.data_dir}")

    low_conf = [p for p in parsed if p.confidence == "low"]
    if low_conf:
        print(f"WARNING: {len(low_conf)} filenames parsed with LOW confidence "
              f"-- these will still be classified, but the query_name may be "
              f"wrong. Check filename_review.csv before trusting results.")

    write_review_csv(parsed, str(out_dir / "filename_review.csv"))

    # De-duplicate: the same physical source often shows up twice
    # (once as _maxi, once as _ztf). Classification only needs to run
    # once per unique source_key.
    unique_sources = {}
    for p in parsed:
        unique_sources.setdefault(p.source_key, p)
    unique_list = list(unique_sources.values())
    print(f"{len(unique_list)} unique sources after de-duplicating "
          f"multi-instrument filenames of the same target")

    if args.limit:
        unique_list = unique_list[: args.limit]
        print(f"--limit set: only classifying first {len(unique_list)} sources")

    # --- Step 2: classify ---
    results = classify_sources(
        unique_list,
        cache_dir=args.cache_dir,
        match_radius_arcsec=args.match_radius_arcsec,
        use_simbad_fallback=not args.no_simbad_fallback,
    )

    write_results_csv(results, str(out_dir / "classification_results.csv"))
    print(f"Wrote {len(results)} classification results to "
          f"{out_dir / 'classification_results.csv'}")

    # --- Summary ---
    from collections import Counter
    type_counts = Counter(r.xrb_type for r in results)
    compact_counts = Counter(r.compact_object for r in results)
    conf_counts = Counter(r.confidence for r in results)
    print("\n--- Summary ---")
    print("XRB type:      ", dict(type_counts))
    print("Compact object:", dict(compact_counts))
    print("Confidence:    ", dict(conf_counts))
    print(f"\nReview flagged/low-confidence rows in classification_results.csv "
          f"before using them in the population histograms.")


if __name__ == "__main__":
    main()