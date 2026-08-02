"""
classifier.py
-------------
Programmatic BH-vs-NS and HMXB-vs-LMXB classification for a list of XRB
sources, following the hierarchical strategy from the design doc, but
restructured for speed and reliability:

  KEY DESIGN CHOICE vs. the original plan
  ----------------------------------------
  The design doc's Phase 2 queries VizieR/HEASARC per-source, in a loop.
  Instead, this module downloads each reference catalog ONCE, caches it
  to disk, and then cross-matches your whole source list against it
  locally using astropy SkyCoord cone-matching. This means:
    - no per-source network round trips (fast: seconds, not minutes)
    - no SIMBAD rate-limit risk for the bulk of sources
    - fully offline / reproducible after the first run (cache_dir)
    - SIMBAD is only hit for (a) name->coordinate resolution when a
      filename gives you a name rather than coordinates, and (b) the
      final fallback for sources that don't match any curated catalog.

  Hierarchical decision tree (unchanged from the design doc):
    1. BH filter:    WATCHDOG (J/ApJS/222/15) + BlackCAT (J/A+A/587/A61)
    2. NS/LMXB filter: XRBcats Galactic LMXB (J/A+A/675/A199)
    3. HMXB filter:  HEASARC hmxbcat
    4. Fallback:     SIMBAD otype (best-effort, low confidence)

REQUIREMENTS
------------
pip install astropy astroquery --break-system-packages

NOTE ON NETWORK ACCESS
-----------------------
This module needs to reach vizier.u-strasbg.fr / cdsarc / simbad /
heasarc servers. Run it on your machine (not in a network-locked
sandbox) the first time so the catalogs get cached; after that it can
run fully offline against the cache_dir.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List

import numpy as np

logger = logging.getLogger("xrb_classifier")

CONE_MATCH_RADIUS_ARCSEC = 10.0  # tune based on catalog positional precision

# Used only to CONFIRM which declination-precision reading of an ambiguous
# catalog-style name (see _catalog_coordinate_candidates) is correct, by
# checking it lands near *some* curated-catalog entry. Deliberately looser
# than CONE_MATCH_RADIUS_ARCSEC: legacy Uhuru/Ariel-V-era positions can be
# off by several arcmin, but that's still tiny next to the ~0.5-1 degree
# gap between a correct and an incorrect digit-count interpretation, so
# this stays safe as a disambiguator without being usable as a real
# source cross-match radius.
SANITY_MATCH_RADIUS_ARCSEC = 600.0


@dataclass
class ClassificationResult:
    query_name: str
    resolved_ra: Optional[float] = None
    resolved_dec: Optional[float] = None
    xrb_type: str = "Unknown"        # 'LMXB' | 'HMXB' | 'Unknown'
    compact_object: str = "Unknown"  # 'BH' | 'NS' | 'Unknown'
    matched_catalog: str = "None"
    match_separation_arcsec: Optional[float] = None
    simbad_otype: Optional[str] = None
    confidence: str = "low"          # 'high' | 'medium' | 'low'
    notes: str = ""


class CatalogStore:
    """
    Downloads and caches the reference catalogs used for classification.
    Each is stored as an astropy Table pickled to cache_dir, so repeat
    runs don't re-hit VizieR/HEASARC.
    """

    VIZIER_CATALOGS = {
        # name -> (vizier_id, table_name_filter or None)
        "lmxb_xrbcats": "J/A+A/675/A199",
        "watchdog_bhxb": "J/ApJS/222/15",
        "blackcat_bh": "J/A+A/587/A61",
    }

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tables = {}

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.ecsv"

    def get(self, name: str):
        """Return an astropy Table for the given catalog, from cache if present."""
        if name in self._tables:
            return self._tables[name]

        cache_path = self._cache_path(name)
        if cache_path.exists():
            from astropy.table import Table
            logger.info(f"Loading {name} from cache: {cache_path}")
            t = Table.read(cache_path, format="ascii.ecsv")
            self._tables[name] = t
            return t

        t = self._download(name)
        self._tables[name] = t
        try:
            t.write(cache_path, format="ascii.ecsv", overwrite=True)
        except Exception as e:
            logger.warning(f"Could not cache {name} to disk: {e}")
        return t

    def _download(self, name: str):
        from astroquery.vizier import Vizier

        if name == "hmxb_heasarc":
            return self._download_heasarc_hmxb()

        vizier_id = self.VIZIER_CATALOGS[name]
        logger.info(f"Downloading {name} ({vizier_id}) from VizieR ...")
        v = Vizier(columns=["**"])  # fetch all columns
        v.ROW_LIMIT = -1
        result = v.get_catalogs(vizier_id)
        if len(result) == 0:
            raise RuntimeError(f"VizieR returned no tables for {vizier_id}")

        # Multi-table VizieR catalogs (WATCHDOG, BlackCAT, etc.) return
        # several sub-tables -- e.g. a main source list, a stage-history
        # table, a bibliography table. Picking "the biggest one" is wrong
        # (it can grab a bib table that just happens to have more rows).
        # Instead, require the sub-table to actually contain RA/Dec-like
        # columns, and log every sub-table seen so a bad match is easy
        # to diagnose from the log alone.
        subtable_info = [(key, result[key].colnames, len(result[key]))
                          for key in result.keys()]
        logger.debug(f"{name}: {len(result)} sub-table(s) returned: {subtable_info}")

        candidates = [
            (key, result[key]) for key in result.keys()
            if _find_col(result[key], "RAJ2000", "RA_ICRS", "RAdeg", "RA")
            and _find_col(result[key], "DEJ2000", "DE_ICRS", "DEdeg", "DEC", "Dec")
        ]
        if not candidates:
            raise RuntimeError(
                f"No sub-table of {vizier_id} contains RA/Dec columns.\n"
                f"Sub-tables seen: {subtable_info}\n"
                f"-> Open this catalog on VizieR (https://vizier.cds.unistra.fr/"
                f"viz-bin/VizieR?-source={vizier_id}) and find which table name "
                f"holds the positions, then hardcode it, e.g. "
                f"Vizier.get_catalogs('{vizier_id}/table1')."
            )

        key, table = max(candidates, key=lambda kv: len(kv[1]))
        logger.info(f"  -> selected sub-table '{key}' with {len(table)} rows "
                    f"(had RA/Dec columns)")
        return table

    def _download_heasarc_hmxb(self):
        # astroquery.heasarc's call signature has shifted across versions;
        # the TAP endpoint is the most stable way to pull a full table with
        # no spatial filter (we want the whole catalog, not a cone search,
        # since we're doing the cone-matching ourselves afterward).
        from astroquery.heasarc import Heasarc
        logger.info("Downloading HEASARC hmxbcat via TAP ...")
        heasarc = Heasarc()
        table = heasarc.query_tap("SELECT * FROM hmxbcat").to_table()
        logger.info(f"  -> got {len(table)} rows")
        logger.debug(f"hmxb_heasarc columns: {table.colnames}")
        return table


def _infer_lmxb_compact_object(tbl, row_idx, bh_ruled_out=False):
    """
    XRBcats Galactic LMXB (J/A+A/675/A199) doesn't have a blunt "BH/NS"
    column -- compact object type has to be inferred from physical
    evidence columns instead:
      - Ppulse (pulsation period) populated -> coherent pulsations -> NS
      - tMSPFlag -> millisecond pulsar flag -> NS
      - XrayType containing burst/atoll/Z-source language -> NS
        (thermonuclear bursts only happen on a solid NS surface)
      - XrayType containing 'BH' -> BH
    Returns (compact_object, note).
    """
    ns_evidence = []
    bh_evidence = []

    ppulse_col = _find_col(tbl, "Ppulse")
    if ppulse_col is not None:
        val = str(tbl[ppulse_col][row_idx]).strip().lower()
        if val not in ("", "--", "nan", "none", "masked"):
            ns_evidence.append(f"Ppulse={val}")

    msp_col = _find_col(tbl, "tMSPFlag")
    if msp_col is not None:
        val = str(tbl[msp_col][row_idx]).strip().lower()
        if val in ("1", "true", "y", "yes"):
            ns_evidence.append("tMSPFlag set")

    xtype_col = _find_col(tbl, "XrayType")
    if xtype_col is not None:
        val = str(tbl[xtype_col][row_idx]).upper()
        if any(k in val for k in ("BURST", "ATOLL", "Z SOURCE", "Z-SOURCE")):
            ns_evidence.append(f"XrayType='{val}'")
        if "BH" in val:
            bh_evidence.append(f"XrayType='{val}'")

    if ns_evidence and not bh_evidence:
        return "NS", "LMXB catalog match; NS inferred from: " + "; ".join(ns_evidence)
    if bh_evidence and not ns_evidence:
        return "BH", "LMXB catalog match; BH inferred from: " + "; ".join(bh_evidence)
    if ns_evidence and bh_evidence:
        return "Unknown", ("LMXB catalog match; conflicting evidence -- NS signals "
                            f"[{'; '.join(ns_evidence)}] vs BH signals "
                            f"[{'; '.join(bh_evidence)}] -- check manually.")
    if bh_ruled_out:
        # WATCHDOG + BlackCAT are essentially a complete census of confirmed
        # BH LMXBs (a few dozen known systems). A source that's a confirmed
        # LMXB but doesn't match either BH catalog is, in practice, almost
        # always an NS system (Z-sources/atolls with no coherent pulsations
        # -- e.g. Sco X-1 -- won't have Ppulse/tMSPFlag/burst-language
        # evidence either, but they're still NS).
        return "NS", ("LMXB catalog match; no direct pulsation/burst evidence, "
                       "but source is not in WATCHDOG/BlackCAT -- defaulting to "
                       "NS (most LMXBs without a BH-catalog match are NS) -- "
                       "verify manually if this is a rare BH LMXB not yet "
                       "in those catalogs.")
    return "Unknown", ""


def _infer_hmxb_compact_object(tbl, row_idx):
    """
    HEASARC hmxbcat's real columns (confirmed from a live query) are
    'pulse_per' and 'class', not the guessed 'SpType2'/'PulsarType'
    names used in the first pass of this code. Evidence:
      - pulse_per populated -> coherent pulsations -> NS (accreting pulsar)
      - class containing 'BH' -> BH (rare in this catalog -- Cyg X-1,
        LMC X-3 style systems -- most cataloged HMXBs are NS)
    Returns (compact_object, note).
    """
    ns_evidence = []
    bh_evidence = []

    pulse_col = _find_col(tbl, "pulse_per")
    if pulse_col is not None:
        val = str(tbl[pulse_col][row_idx]).strip().lower()
        if val not in ("", "--", "nan", "none", "masked"):
            ns_evidence.append(f"pulse_per={val}")

    class_col = _find_col(tbl, "class", "xray_type")
    if class_col is not None:
        val = str(tbl[class_col][row_idx]).upper()
        if "BH" in val:
            bh_evidence.append(f"class='{val}'")

    if ns_evidence and not bh_evidence:
        return "NS", "HMXB catalog match; NS inferred from: " + "; ".join(ns_evidence)
    if bh_evidence and not ns_evidence:
        return "BH", "HMXB catalog match; BH inferred from: " + "; ".join(bh_evidence)
    if ns_evidence and bh_evidence:
        return "Unknown", ("HMXB catalog match; conflicting evidence -- "
                            f"NS signals [{'; '.join(ns_evidence)}] vs BH signals "
                            f"[{'; '.join(bh_evidence)}] -- check manually.")
    return "Unknown", ("HMXB catalog match; no pulsation or BH flag found -- "
                        "most catalogued HMXBs are NS, but this isn't confirmed "
                        "from columns. Check literature.")


def _find_col(table, *candidates):
    """Case-insensitive column lookup with several fallback names."""
    lower_map = {c.lower(): c for c in table.colnames}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _table_skycoord(table):
    """
    Build a SkyCoord array from whatever RA/Dec-like columns a VizieR
    table happens to expose. VizieR column names vary catalog to
    catalog (RAJ2000/DEJ2000, RA_ICRS/DE_ICRS, RAdeg/DEdeg, etc), AND
    the *format* varies too: RA_ICRS/DE_ICRS are normally plain decimal
    degrees, but RAJ2000/DEJ2000 are sometimes stored as sexagesimal
    strings (e.g. "04 19 42.141") rather than degrees -- seen in
    WATCHDOG and BlackCAT. This detects the format from the data itself
    rather than trusting the column name.

    Rows with missing/blank coordinates are dropped. Returns
    (SkyCoord, valid_row_indices) where valid_row_indices[i] gives the
    row number in the *original* table that SkyCoord entry i came from
    -- use this to look back up other columns (Type, Ppulse, etc.) for
    a matched source.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    ra_col = _find_col(table, "RAJ2000", "RA_ICRS", "RAdeg", "RA")
    de_col = _find_col(table, "DEJ2000", "DE_ICRS", "DEdeg", "DEC", "Dec")
    if ra_col is None or de_col is None:
        raise ValueError(
            f"Could not find RA/Dec columns in table with columns: {table.colnames}"
        )

    ra_strs = [str(v).strip() for v in table[ra_col]]
    de_strs = [str(v).strip() for v in table[de_col]]

    blank_tokens = {"", "--", "nan", "none", "masked", "n/a"}
    valid_indices = [
        i for i, (r, d) in enumerate(zip(ra_strs, de_strs))
        if r.lower() not in blank_tokens and d.lower() not in blank_tokens
    ]
    if not valid_indices:
        raise ValueError(f"No valid RA/Dec values found in columns {ra_col}/{de_col}")

    ra_valid = [ra_strs[i] for i in valid_indices]
    de_valid = [de_strs[i] for i in valid_indices]

    # Decide format from the first valid value: sexagesimal strings
    # contain whitespace or colons ("04 19 42.141" / "04:19:42.14");
    # plain decimal degrees don't.
    looks_sexagesimal = any(c in ra_valid[0] for c in (" ", ":"))

    if looks_sexagesimal:
        coords = SkyCoord(ra=ra_valid, dec=de_valid, unit=(u.hourangle, u.deg))
    else:
        ra_deg = np.array([float(v) for v in ra_valid])
        de_deg = np.array([float(v) for v in de_valid])
        coords = SkyCoord(ra=ra_deg * u.deg, dec=de_deg * u.deg)

    return coords, np.array(valid_indices)


def _parse_ra_dec_value(ra_val, dec_val):
    """
    Parses a single RA/Dec pair that could be in either format:
      - decimal degrees (what astroquery>=0.4.8's new TAP-based SIMBAD
        client returns from query_object -- confirmed from astroquery's
        own docstring examples: 'ra'/'dec' columns are in deg)
      - sexagesimal hourangle/deg strings (what older astroquery
        versions, and some VizieR tables, use)
    Auto-detects from whether the value contains a space or colon.
    Returns (ra_deg, dec_deg) as floats.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    ra_str = str(ra_val).strip()
    dec_str = str(dec_val).strip()
    looks_sexagesimal = (" " in ra_str) or (":" in ra_str)

    if looks_sexagesimal:
        c = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
        return c.ra.degree, c.dec.degree
    else:
        return float(ra_str), float(dec_str)


def resolve_coordinates(query_name: str):
    """
    Resolve a name to (ra_deg, dec_deg) using an EXACT SIMBAD name-resolver
    lookup only. If query_name already parses as a coordinate string, this
    is skipped upstream in classify_sources().

    This is intentionally the "try the string as given" step. Fuzzy/variant
    resolution lives in _simbad_resolve_with_variants(), which calls this
    first and only reaches for wildcards if this fails -- see the note
    there for why that ordering matters.
    """
    from astroquery.simbad import Simbad
    try:
        result = Simbad.query_object(query_name)
        if result is None or len(result) == 0:
            return None, None
        ra_col = _find_col(result, "ra", "RA")
        de_col = _find_col(result, "dec", "DEC")
        if ra_col is None or de_col is None:
            logger.warning(f"SIMBAD result for {query_name} has no ra/dec column: "
                            f"{result.colnames}")
            return None, None
        return _parse_ra_dec_value(result[ra_col][0], result[de_col][0])
    except Exception as e:
        # Log the exception TYPE, not just its message -- "no match" and
        # a genuine HTTP 429/500/timeout both used to get logged
        # identically as a generic "failed", making rate-limit and
        # server errors indistinguishable from an honest miss.
        logger.warning(f"SIMBAD coordinate resolution failed for {query_name} "
                        f"[{type(e).__name__}]: {e}")
        return None, None


def _parse_iau_ra_block(ra_block: str):
    """
    Parse the RA digit-block of an IAU J-name into an hourangle string
    SkyCoord understands, plus the implied quantization (seconds of
    time) of the LAST digit written. Handles both standard truncation
    levels:
      'HHMM'[.frac]   -> whole/fractional minutes of time
      'HHMMSS'[.frac] -> whole/fractional seconds of time
    Returns (ra_str, quantum_sec_of_time) or None if the block's digit
    count doesn't match either convention.
    """
    int_part, _, frac_part = ra_block.partition(".")

    if len(int_part) == 4:
        hh, mm = int_part[:2], int_part[2:4]
        quantum = 60.0 if not frac_part else 60 * 10 ** (-len(frac_part))
        frac_sfx = f".{frac_part}" if frac_part else ""
        return f"{hh}h{mm}{frac_sfx}m", quantum
    elif len(int_part) == 6:
        hh, mm, ss = int_part[:2], int_part[2:4], int_part[4:6]
        quantum = 1.0 if not frac_part else 10 ** (-len(frac_part))
        frac_sfx = f".{frac_part}" if frac_part else ""
        return f"{hh}h{mm}m{ss}{frac_sfx}s", quantum
    return None


def _parse_iau_dec_block(sign: str, dec_block: str):
    """
    Parse the declination digit-block of an IAU J-name (sign given
    separately) into a degree string SkyCoord understands, plus the
    implied precision in arcsec. Digit-count convention (standard IAU
    name truncation -- see https://cds.unistra.fr/Dic/iau-spec.html):
      2 digits -> DD      whole degrees          (~3600" precision)
      3 digits -> DD.d    tenths of a degree      (~360"  precision)
      4 digits -> DDMM    degrees + arcmin        (~60"   precision)
      5 digits -> DDMM.m  + tenths of arcmin      (~6"    precision)
      6 digits -> DDMMSS  degrees+arcmin+arcsec   (~1"    precision)
    This is exactly the convention MAXI/XTE-style 'J2303+088' /
    'J1709-267' names use for their 3-digit declination -- the OLD
    regex here only recognized 4-digit declinations and returned no
    match at all for these, forcing every such source through a live
    SIMBAD lookup (which then also fails whenever the filename's
    truncated spelling doesn't exactly match SIMBAD's registered
    alias). This is very likely the single biggest cause of low
    classification coverage, since MAXI J-names are the dominant
    naming convention in Total_Data.
    Returns (dec_str, precision_arcsec) or None for any other length.
    """
    int_part, _, frac_part = dec_block.partition(".")
    L = len(int_part)

    if L == 2:
        return f"{sign}{int_part}d", 3600.0
    elif L == 3:
        deg, tenth = int_part[:2], int_part[2]
        return f"{sign}{deg}.{tenth}d", 360.0
    elif L == 4:
        deg, arcmin = int_part[:2], int_part[2:4]
        frac_sfx = f".{frac_part}" if frac_part else ""
        return f"{sign}{deg}d{arcmin}{frac_sfx}m", 60.0
    elif L == 5:
        deg, arcmin, tenth = int_part[:2], int_part[2:4], int_part[4]
        return f"{sign}{deg}d{arcmin}.{tenth}m", 6.0
    elif L == 6:
        deg, arcmin, arcsec = int_part[:2], int_part[2:4], int_part[4:6]
        frac_sfx = f".{frac_part}" if frac_part else ""
        return f"{sign}{deg}d{arcmin}m{arcsec}{frac_sfx}s", 1.0
    return None


def _parse_j_name_coordinate(query_name: str):
    """
    Parse a 'J'-prefixed coordinate name (e.g. 'MAXI J2303+088',
    'RX J0146.9+6121', 'XTE J1709-267') at WHATEVER truncation level it
    was actually written to, instead of only recognizing one fixed
    digit count.

    NOTE: this only matches names containing a literal 'J'. Older
    catalog-prefixed names that ALSO encode coordinates directly but
    without a 'J' -- 4U, 1A, 1E, A, GX, MXB, KS, GS, AX, SAX, EXO, OAO,
    3A (see filename_parser.MISSION_PREFIX_MAP) -- are handled
    separately by _catalog_coordinate_candidates() below.

    Returns (ra_deg, dec_deg, precision_arcsec) where precision_arcsec
    is a rough WORST-CASE positional uncertainty implied by the
    truncation (the coarser of the RA/Dec truncation, with the RA
    minute/second-of-time quantum converted to on-sky arcsec via
    cos(dec)) -- or None if no 'J<digits><sign><digits>' block is found
    at all. Callers decide what to do with a coarse (large
    precision_arcsec) result; this function only parses, it doesn't
    judge trustworthiness.
    """
    import re
    import math
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    m = re.search(r"J(\d{4,6}(?:\.\d+)?)([+-])(\d{2,6}(?:\.\d+)?)", query_name)
    if not m:
        return None
    ra_block, sign, dec_block = m.groups()

    ra_parsed = _parse_iau_ra_block(ra_block)
    dec_parsed = _parse_iau_dec_block(sign, dec_block)
    if ra_parsed is None or dec_parsed is None:
        return None
    ra_str, ra_quantum_sec = ra_parsed
    dec_str, dec_precision_arcsec = dec_parsed

    try:
        coord = SkyCoord(f"{ra_str} {dec_str}", unit=(u.hourangle, u.deg))
    except Exception:
        return None

    ra_precision_arcsec = ra_quantum_sec * 15.0 * math.cos(math.radians(coord.dec.deg))
    precision_arcsec = max(abs(ra_precision_arcsec), dec_precision_arcsec)
    return coord.ra.degree, coord.dec.degree, precision_arcsec


def _catalog_coordinate_candidates(query_name: str):
    """
    For catalog-prefixed names WITHOUT a 'J' (4U, 1A, 1E, A, GX, MXB,
    KS, GS, AX, SAX, EXO, OAO, 3A, and bare 'XTE'/'IGR' without J),
    try to pull RA/Dec directly out of the trailing 'hhmm+dd' or
    'hhmm+ddd' block, e.g. '4U 0535+262' or '4U 1700-37'.

    These older catalogs are NOT consistent about declination precision
    -- the same catalog uses both a 2-digit whole-degree convention
    ('4U 1700-37' = -37 deg) and a 3-digit tenths-of-a-degree convention
    ('4U 1957+115' = +11.5 deg) -- so this deliberately does NOT pick a
    winner. It returns every plausible interpretation as a candidate and
    leaves the decision to the caller, which sanity-checks each
    candidate against the already-cached curated catalogs (WATCHDOG /
    BlackCAT / LMXB / HMXB) rather than trusting a guessed declination
    blindly. If nothing can be confirmed that way, the caller falls
    back to SIMBAD name resolution instead of using an unconfirmed guess.

    Returns a list of (ra_deg, dec_deg, label) tuples; possibly empty.
    """
    import re
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    m = re.search(r"(\d{2})(\d{2}(?:\.\d+)?)\s*([+-])(\d{2,3}(?:\.\d+)?)\s*$", query_name.strip())
    if not m:
        return []
    hh, mm, sign, dd_raw = m.groups()

    candidates = []
    dd_int_part = dd_raw.split(".")[0]

    if len(dd_int_part) == 2:
        # Unambiguous: only the whole-degree reading is possible.
        try:
            coord = SkyCoord(f"{hh}h{mm}m {sign}{dd_raw}d", unit=(u.hourangle, u.deg))
            candidates.append((coord.ra.degree, coord.dec.degree, "whole-degree"))
        except Exception:
            pass
    elif len(dd_int_part) == 3:
        # Ambiguous: could be tenths-of-a-degree (common in the 4U
        # catalog, e.g. '115' -> 11.5 deg) or a truncated/extra digit
        # that should just be dropped. Offer both.
        try:
            tenths_deg = float(dd_raw) / 10.0
            coord = SkyCoord(f"{hh}h{mm}m {sign}{tenths_deg}d", unit=(u.hourangle, u.deg))
            candidates.append((coord.ra.degree, coord.dec.degree, "tenths-of-degree"))
        except Exception:
            pass
        try:
            coord = SkyCoord(f"{hh}h{mm}m {sign}{dd_int_part[:2]}d", unit=(u.hourangle, u.deg))
            candidates.append((coord.ra.degree, coord.dec.degree, "whole-degree-truncated"))
        except Exception:
            pass

    return candidates


def _best_catalog_sanity_match(ra_deg, dec_deg, catalog_coords, match_radius_arcsec):
    """
    Given a candidate (ra, dec), return (separation_arcsec, matched_ra,
    matched_dec, catalog_name) for the closest entry across all curated
    catalogs, or None if none of the catalogs have any coordinates
    loaded.

    Returns the MATCHED entry's own coordinates (not just the
    separation) so callers can adopt a curated catalog's precise
    position instead of trusting a name-derived guess that's only
    accurate to whatever precision the filename happened to be
    truncated to (whole degrees, tenths of a degree, etc.) -- see
    _resolve_direct().
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    best = None
    for cat_name, coords in catalog_coords.items():
        if coords is None or len(coords) == 0:
            continue
        sep = target.separation(coords)
        idx = int(np.argmin(sep))
        sep_arcsec = float(sep[idx].arcsec)
        if best is None or sep_arcsec < best[0]:
            best = (sep_arcsec, float(coords[idx].ra.deg), float(coords[idx].dec.deg), cat_name)
    return best


def _coord_hint_for_simbad(query_name: str):
    """
    Best-effort (ra_deg, dec_deg, precision_arcsec) guess for query_name,
    used ONLY to point a SIMBAD coordinate cone search roughly in the
    right place when local catalog confirmation in _resolve_direct()
    didn't succeed -- this is deliberately NOT trusted as a final answer
    on its own.

    IMPORTANT: only J-name-derived guesses are returned here, never
    _catalog_coordinate_candidates() guesses (4U/1A/1E/... style). The
    J-name case has a defensible worst-case precision bound derived
    directly from how many digits the name was truncated to (see
    _parse_j_name_coordinate). Old Uhuru-catalog-style names have NO
    such bound -- their numeric part reflects a decades-old, low-
    resolution measurement, which can be off by far more than any
    cone-search radius we'd consider safe (e.g. '4U 1735-444' truncates
    to a position over 2500" from the source's true location). Using
    that kind of guess to seed a SIMBAD cone search doesn't fail safe --
    it can confidently return the WRONG, unrelated nearby object. For
    that family of names, _resolve_direct()'s local sanity-check against
    curated catalogs (which requires independent confirmation, not just
    "nearest within a generous radius") is the only guess we trust;
    if that doesn't confirm it, we fall through to an identifier-string
    SIMBAD lookup only, never a coordinate-based one.

    Returns None if query_name doesn't parse as a J-name coordinate.
    """
    return _parse_j_name_coordinate(query_name)


def _simbad_resolve_by_coordinates(ra_deg: float, dec_deg: float, radius_arcsec: float):
    """
    Resolve via a SIMBAD coordinate cone search instead of guessing at
    identifier spellings. This is SIMBAD's OWN documented recommendation
    for this exact situation ("SIMBAD: Query from identifiers" help
    page): "if your identifier is not accepted by SIMBAD, or simply not
    found... enter the coordinates of the object; if the object exists
    in SIMBAD under a different name, you still have a chance to find
    it." It's also a single indexed query -- unlike wildcard identifier
    matching, which SIMBAD's docs warn "can be quite long (a few
    minutes), mainly because the DBMS doesn't always make use of indices
    when querying through patterns."

    Accepts the CLOSEST object within radius_arcsec (capped at
    SANITY_MATCH_RADIUS_ARCSEC by the caller). Returns
    (ra_deg, dec_deg, note) or (None, None, None).
    """
    from astroquery.simbad import Simbad
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    try:
        center = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        result = Simbad.query_region(center, radius=radius_arcsec * u.arcsec)
    except Exception as e:
        logger.warning(f"SIMBAD region query failed at RA={ra_deg:.5f} "
                        f"Dec={dec_deg:.5f} [{type(e).__name__}]: {e}")
        return None, None, None
    if result is None or len(result) == 0:
        return None, None, None

    ra_col = _find_col(result, "ra", "RA")
    de_col = _find_col(result, "dec", "DEC")
    main_id_col = _find_col(result, "main_id", "MAIN_ID")
    if ra_col is None or de_col is None:
        return None, None, None

    try:
        coords = SkyCoord(ra=np.asarray(result[ra_col], dtype=float) * u.deg,
                           dec=np.asarray(result[de_col], dtype=float) * u.deg)
        sep = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg).separation(coords)
        idx = int(np.argmin(sep))
        sep_arcsec = float(sep[idx].arcsec)
        match_ra, match_dec = float(coords[idx].ra.deg), float(coords[idx].dec.deg)
    except Exception:
        # ra/dec came back as sexagesimal strings in this SIMBAD response
        # flavor -- fall back to the first row rather than failing.
        try:
            match_ra, match_dec = _parse_ra_dec_value(result[ra_col][0], result[de_col][0])
            sep_arcsec = float("nan")
            idx = 0
        except Exception:
            return None, None, None

    name = str(result[main_id_col][idx]) if main_id_col else "unknown object"
    return match_ra, match_dec, (
        f"Resolved via SIMBAD coordinate cone search ({radius_arcsec:.1f}\" "
        f"radius around a locally-parsed guess -- SIMBAD's own recommended "
        f"fallback when an identifier isn't found directly); nearest match "
        f"'{name}'" + (f" ({sep_arcsec:.1f}\" away)." if sep_arcsec == sep_arcsec else "."))


def _generate_name_variants(query_name: str):
    """
    Given a catalog-prefixed name, generate alternate spellings that are
    plausible aliases for the SAME source but weren't in the filename
    verbatim -- e.g. '4U 0535+262' -> '4U 0535+26' (drop tenths digit)
    and '4U 0535+26' -> '4U 0535+262'/'4U 0535+260' (add a tenths digit).
    Also generates a trailing-wildcard version of each for use with
    SIMBAD's wildcard=True matching.

    Returns a list of (name_string, is_wildcard) tuples, most-specific
    first. Does not include query_name itself -- try the exact string
    first via resolve_coordinates() before falling back to this list.
    """
    import re

    m = re.search(r"^(.*?\D)(\d{2})(\d{2}(?:\.\d+)?)\s*([+-])(\d{2,3}(?:\.\d+)?)\s*$",
                  query_name.strip())
    if not m:
        return []
    prefix, hh, mm, sign, dd_raw = m.groups()
    dd_int_part = dd_raw.split(".")[0]

    stems = set()
    if len(dd_int_part) == 3:
        # Tenths-of-degree spelling and the shorter whole-degree spelling.
        stems.add(dd_int_part[:2])          # e.g. '262' -> '26'
        stems.add(dd_int_part)              # keep the original too
    elif len(dd_int_part) == 2:
        # Whole-degree spelling and a couple of plausible tenths spellings.
        stems.add(dd_int_part)
        stems.add(dd_int_part + "0")
        stems.add(dd_int_part + "5")

    variants = []
    for stem in sorted(stems):
        name = f"{prefix}{hh}{mm}{sign}{stem}"
        if name != query_name:
            variants.append((name, False))
        variants.append((f"{prefix}{hh}{mm}{sign}{dd_int_part[:2]}*", True))

    # De-duplicate while preserving order.
    seen = set()
    unique_variants = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique_variants.append(v)
    return unique_variants


def _simbad_resolve_with_variants(query_name: str, sleep_between_calls: float = 0.0,
                                    coord_hint=None):
    """
    Robust name -> coordinate resolution for catalog-prefixed names.

    Order of attempts (cheapest / most trustworthy first):
      1. Exact SIMBAD lookup on query_name as given (resolve_coordinates).
         Indexed, fast.
      2. If coord_hint is given (a local, possibly-coarse (ra, dec,
         precision_arcsec) guess -- see _coord_hint_for_simbad()): a
         SIMBAD *coordinate* cone search around it. This is SIMBAD's own
         documented recommendation for exactly this situation (see
         _simbad_resolve_by_coordinates()), and it's a single indexed
         query.
      3. Exact SIMBAD lookup on each generated alias spelling (still an
         exact match server-side, just against a different alias string
         -- e.g. trying '4U 0535+26' after '4U 0535+262' failed).
      4. SIMBAD WILDCARD lookup on a stem pattern (e.g. '4U 0535+26*'),
         ONLY accepted if it resolves to exactly one distinct object.
         This is deliberately LAST: SIMBAD's own docs warn wildcard
         identifier queries "can be quite long (a few minutes), mainly
         because the DBMS doesn't always make use of indices when
         querying through patterns" -- this was the dominant cause of
         per-source slowness before coord_hint was added, since it used
         to be tried for nearly every unresolved source.

    This function can still fire several SIMBAD requests back-to-back for
    a single source. sleep_between_calls is applied between every
    individual network call to stay under SIMBAD's rate limit.

    Returns (ra_deg, dec_deg, note) with note describing which step
    (and which alias string) actually resolved it, or (None, None, note)
    if nothing could be confirmed.
    """
    from astroquery.simbad import Simbad

    def _pause():
        if sleep_between_calls:
            time.sleep(sleep_between_calls)

    ra, dec = resolve_coordinates(query_name)
    _pause()
    if ra is not None:
        return ra, dec, f"Resolved via exact SIMBAD match on '{query_name}'."

    if coord_hint is not None:
        hint_ra, hint_dec, hint_precision_arcsec = coord_hint
        radius = min(max(hint_precision_arcsec, 15.0), SANITY_MATCH_RADIUS_ARCSEC)
        ra, dec, note = _simbad_resolve_by_coordinates(hint_ra, hint_dec, radius)
        _pause()
        if ra is not None:
            return ra, dec, note

    for variant_name, is_wildcard in _generate_name_variants(query_name):
        if not is_wildcard:
            ra, dec = resolve_coordinates(variant_name)
            _pause()
            if ra is not None:
                return ra, dec, (f"Exact SIMBAD match failed on '{query_name}'; "
                                  f"resolved via alias spelling '{variant_name}'.")
            continue

        try:
            result = Simbad.query_object(variant_name, wildcard=True)
        except Exception as e:
            logger.warning(f"SIMBAD wildcard query failed for '{variant_name}' "
                            f"[{type(e).__name__}]: {e}")
            _pause()
            continue
        _pause()
        if result is None or len(result) == 0:
            continue

        main_id_col = _find_col(result, "main_id", "MAIN_ID")
        distinct_ids = set(str(v) for v in result[main_id_col]) if main_id_col else {None}
        if len(distinct_ids) != 1:
            logger.info(f"SIMBAD wildcard '{variant_name}' matched "
                        f"{len(distinct_ids)} distinct objects -- ambiguous, skipping "
                        f"rather than guessing.")
            continue

        ra_col = _find_col(result, "ra", "RA")
        de_col = _find_col(result, "dec", "DEC")
        if ra_col is None or de_col is None:
            continue
        try:
            ra, dec = _parse_ra_dec_value(result[ra_col][0], result[de_col][0])
        except Exception:
            continue
        if ra is not None:
            return ra, dec, (f"Exact SIMBAD match failed on '{query_name}' and its "
                              f"alias spellings; resolved via unambiguous wildcard "
                              f"match on '{variant_name}'.")

    return None, None, (f"Could not resolve '{query_name}': exact SIMBAD match, "
                          f"alias spellings, and wildcard search all failed or "
                          f"were ambiguous.")


def _resolve_direct(query_name: str, catalog_coords, match_radius_arcsec: float):
    """
    Steps A+B of the resolution chain, factored out so classify_sources()
    can run them against both a source's primary query_name AND (when
    filename_parser flagged the declination sign as ambiguous, e.g. a
    bare-underscore filename with no p/m marker) its alternate-sign
    reading, WITHOUT ever hitting the network. Only a curated-catalog-
    confirmed match, or a name-derived coordinate that's already precise
    enough on its own, is trusted -- an unconfirmed coarse guess is
    never returned.

    Returns (ra_deg, dec_deg, note) or (None, None, None) if nothing
    could be parsed/confirmed directly.
    """
    # Step A: J-style coordinate names ('MAXI J2303+088', 'RX J0146.9+6121', ...).
    parsed = _parse_j_name_coordinate(query_name)
    if parsed is not None:
        ra, dec, precision_arcsec = parsed
        if precision_arcsec <= match_radius_arcsec:
            # Precise enough (e.g. arcsec-level 'HHMMSS+DDMMSS' names) to
            # trust as the final coordinate without any catalog check.
            return ra, dec, (f"Parsed directly from a 'J'-style coordinate in "
                              f"the name (~{precision_arcsec:.1f}\" implied "
                              f"precision, within the match radius).")

        # Coarse: e.g. MAXI/XTE's common 3-digit ('tenths of a degree')
        # declination truncation implies ~360" of positional uncertainty
        # on its own -- far bigger than a 10" cone-match radius. Confirm
        # it lands near a curated catalog entry at a loose sanity radius,
        # then ADOPT THAT ENTRY'S OWN COORDINATES rather than the
        # truncated guess -- the guess was only ever good enough to find
        # the right row, not to serve as the final position.
        best = _best_catalog_sanity_match(ra, dec, catalog_coords, match_radius_arcsec)
        if best is not None and best[0] <= SANITY_MATCH_RADIUS_ARCSEC:
            sep, cat_ra, cat_dec, cat_name = best
            return cat_ra, cat_dec, (
                f"Name-derived coordinate is coarse (~{precision_arcsec:.1f}\" "
                f"implied precision from filename truncation); confirmed "
                f"{sep:.1f}\" from a {cat_name} entry and adopted that entry's "
                f"coordinates instead of the truncated guess.")

    # Step B: catalog-prefixed names without 'J' (4U, 1A, 1E, A, GX, ...).
    # Several plausible declination-precision readings are generated;
    # confirm each against a curated catalog and adopt THAT catalog
    # entry's own coordinates for whichever reading matches best.
    best = None
    for cra, cdec, label in _catalog_coordinate_candidates(query_name):
        cand = _best_catalog_sanity_match(cra, cdec, catalog_coords, match_radius_arcsec)
        if cand is not None and cand[0] <= SANITY_MATCH_RADIUS_ARCSEC:
            if best is None or cand[0] < best[0]:
                best = cand + (label,)
    if best is not None:
        sep, cat_ra, cat_dec, cat_name, label = best
        return cat_ra, cat_dec, (
            f"Coordinates parsed directly from the catalog-style name "
            f"('{label}' declination reading), confirmed {sep:.1f}\" from a "
            f"{cat_name} entry and adopted that entry's coordinates.")

    return None, None, None


def classify_sources(
    parsed_sources: List,
    cache_dir: str = "./cache",
    match_radius_arcsec: float = CONE_MATCH_RADIUS_ARCSEC,
    use_simbad_fallback: bool = True,
    sleep_between_simbad_calls: float = 0.3,
    debug_nearest: bool = False,
) -> List[ClassificationResult]:
    """
    parsed_sources: list of objects with a `.query_name` attribute
                     (e.g. output of filename_parser.parse_filename),
                     OR a plain list of strings.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    store = CatalogStore(cache_dir=cache_dir)

    # Pre-load all catalogs once.
    catalogs = {}
    for name in ["watchdog_bhxb", "blackcat_bh", "lmxb_xrbcats"]:
        try:
            catalogs[name] = store.get(name)
        except Exception as e:
            logger.error(f"Failed to load catalog {name} [{type(e).__name__}]: {e}")
            catalogs[name] = None
    try:
        catalogs["hmxb_heasarc"] = store.get("hmxb_heasarc")
    except Exception as e:
        logger.error(f"Failed to load HEASARC hmxbcat [{type(e).__name__}]: {e}")
        catalogs["hmxb_heasarc"] = None

    catalog_coords = {}
    catalog_valid_idx = {}
    for name, tbl in catalogs.items():
        if tbl is not None:
            try:
                coords, valid_idx = _table_skycoord(tbl)
                catalog_coords[name] = coords
                catalog_valid_idx[name] = valid_idx
            except Exception as e:
                logger.error(f"Could not build coordinates for {name}: {e}")
                catalog_coords[name] = None
                catalog_valid_idx[name] = None
        else:
            catalog_coords[name] = None
            catalog_valid_idx[name] = None

    results = []
    for src in parsed_sources:
        query_name = getattr(src, "query_name", src)
        # filename_parser flags a source as sign_ambiguous when the
        # filename encoded declination with a bare underscore and no
        # p/m marker (e.g. "4U0053_604") -- the true sign can't be
        # recovered from the filename alone, so it hands back both
        # readings and we try both here rather than silently trusting
        # the '+' default guess.
        query_name_alt = getattr(src, "query_name_alt", None) if getattr(src, "sign_ambiguous", False) else None
        res = ClassificationResult(query_name=query_name)

        # Steps A+B: parse/confirm coordinates directly, no network call.
        # Try the primary reading first, then the alternate-sign reading
        # if the source's declination sign was ambiguous in the filename.
        ra, dec, resolution_note = _resolve_direct(query_name, catalog_coords, match_radius_arcsec)
        if ra is None and query_name_alt:
            ra, dec, resolution_note = _resolve_direct(query_name_alt, catalog_coords, match_radius_arcsec)
            if ra is not None:
                resolution_note = (f"Primary reading '{query_name}' did not confirm; "
                                    f"declination sign was ambiguous in the filename, "
                                    f"alternate reading '{query_name_alt}' confirmed "
                                    f"instead. ") + resolution_note

        # Step C: nothing parsed or confirmed directly -- fall back to
        # SIMBAD, now trying alias spellings and a guarded wildcard search
        # (not just a single exact-string lookup) before giving up. Also
        # tries the alternate-sign reading if the primary one is
        # unresolved. sleep_between_simbad_calls is now applied between
        # EVERY individual SIMBAD request inside the cascade (see
        # _simbad_resolve_with_variants), not just once per source --
        # that per-call throttle is what actually prevents SIMBAD's
        # rate limiter (HTTP 429) from tripping on sources that need
        # several lookups back-to-back.
        if ra is None:
            coord_hint = _coord_hint_for_simbad(query_name)
            ra, dec, resolution_note = _simbad_resolve_with_variants(
                query_name, sleep_between_calls=sleep_between_simbad_calls,
                coord_hint=coord_hint)
            if ra is None and query_name_alt:
                coord_hint_alt = _coord_hint_for_simbad(query_name_alt)
                ra, dec, note_alt = _simbad_resolve_with_variants(
                    query_name_alt, sleep_between_calls=sleep_between_simbad_calls,
                    coord_hint=coord_hint_alt)
                if ra is not None:
                    resolution_note = (f"Primary reading '{query_name}' unresolved via "
                                        f"SIMBAD; alternate sign reading '{query_name_alt}' "
                                        f"resolved instead. ") + note_alt

        if ra is None:
            res.notes = resolution_note or "Could not resolve coordinates (parsed nor SIMBAD)."
            results.append(res)
            continue

        res.notes = resolution_note
        res.resolved_ra, res.resolved_dec = ra, dec
        target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

        if debug_nearest:
            logger.info(f"{query_name}: resolved to RA={ra:.5f}, Dec={dec:.5f}")
            for cat_name, coords in catalog_coords.items():
                if coords is None or len(coords) == 0:
                    continue
                sep = target.separation(coords)
                nidx = np.argmin(sep)
                orig = catalog_valid_idx[cat_name][nidx] if catalog_valid_idx.get(cat_name) is not None else nidx
                logger.info(f"  nearest in {cat_name}: {sep[nidx].arcsec:.2f} arcsec "
                            f"(catalog row {orig}, catalog coord "
                            f"RA={coords[nidx].ra.deg:.5f} Dec={coords[nidx].dec.deg:.5f})")


        # --- Step 1: BH confirmation (WATCHDOG / BlackCAT) ---
        # Resolved INDEPENDENTLY of xrb_type now. A BH-catalog match no
        # longer short-circuits the type lookup, and a type-catalog match
        # no longer short-circuits the BH lookup -- both questions get
        # answered from whichever catalog actually has the evidence.
        bh_match = None  # (cat_name, orig_idx, sep_arcsec)
        for cat_name in ["watchdog_bhxb", "blackcat_bh"]:
            coords = catalog_coords.get(cat_name)
            if coords is None or len(coords) == 0:
                continue
            sep = target.separation(coords)
            idx = np.argmin(sep)
            if sep[idx].arcsec <= match_radius_arcsec:
                bh_match = (cat_name, catalog_valid_idx[cat_name][idx], float(sep[idx].arcsec))
                break

        # --- Step 2: xrb_type (LMXB vs HMXB) ---
        # Prefer whichever of the two type catalogs is the closer match;
        # if the BH catalog itself carries an explicit Type/Class/XBType
        # column, that's used as a tiebreaker note but doesn't override
        # a tighter positional match elsewhere.
        lmxb_coords = catalog_coords.get("lmxb_xrbcats")
        hmxb_coords = catalog_coords.get("hmxb_heasarc")
        lmxb_sep = (target.separation(lmxb_coords).arcsec.min()
                    if lmxb_coords is not None and len(lmxb_coords) else np.inf)
        hmxb_sep = (target.separation(hmxb_coords).arcsec.min()
                    if hmxb_coords is not None and len(hmxb_coords) else np.inf)

        type_source = None  # 'lmxb_xrbcats' | 'hmxb_heasarc' | None
        if lmxb_sep <= match_radius_arcsec or hmxb_sep <= match_radius_arcsec:
            if hmxb_sep <= lmxb_sep:
                res.xrb_type = "HMXB"
                type_source = "hmxb_heasarc"
                res.match_separation_arcsec = float(hmxb_sep)
            else:
                res.xrb_type = "LMXB"
                type_source = "lmxb_xrbcats"
                res.match_separation_arcsec = float(lmxb_sep)
            res.matched_catalog = type_source
            res.confidence = "high"
        elif bh_match is not None:
            # No independent type-catalog match, but we do have a BH
            # catalog hit -- check if it carries an explicit type column
            # before falling back to the "most BH transients are LMXBs"
            # heuristic.
            cat_name, orig_idx, sep_arcsec = bh_match
            tbl = catalogs[cat_name]
            type_col = _find_col(tbl, "Type", "Class", "XBType")
            if type_col is not None:
                val = str(tbl[type_col][orig_idx])
                res.xrb_type = "HMXB" if "HM" in val.upper() else "LMXB"
                res.notes = f"xrb_type from {cat_name}.{type_col}='{val}'."
            else:
                res.xrb_type = "LMXB"
                res.notes = ("BH catalog match with no independent LMXB/HMXB "
                              "catalog match and no Type column -- xrb_type "
                              "defaulted to LMXB (most BH transients are LMXBs) "
                              "-- verify manually.")
            res.matched_catalog = cat_name
            res.match_separation_arcsec = sep_arcsec
            res.confidence = "medium"

        # --- Step 3: compact_object (BH vs NS) ---
        # A BH-catalog match is treated as authoritative for compact_object
        # regardless of which catalog ended up deciding xrb_type above.
        if bh_match is not None:
            cat_name, orig_idx, sep_arcsec = bh_match
            res.compact_object = "BH"
            res.confidence = "high"
            if res.matched_catalog in (None, "None"):
                res.matched_catalog = cat_name
                res.match_separation_arcsec = sep_arcsec
            else:
                res.notes = (res.notes + " " if res.notes else "") + \
                    f"BH confirmed via {cat_name} ({sep_arcsec:.2f}\")."
        elif type_source == "lmxb_xrbcats":
            tbl = catalogs["lmxb_xrbcats"]
            orig_idx = catalog_valid_idx["lmxb_xrbcats"][np.argmin(target.separation(lmxb_coords))]
            # We only reach this branch when bh_match is None (see Step 3
            # ordering below), so BH has already been ruled out for this
            # source -- let the inference function use that.
            compact_object, evidence_note = _infer_lmxb_compact_object(
                tbl, orig_idx, bh_ruled_out=True)
            res.compact_object = compact_object
            if compact_object == "Unknown":
                res.notes = ((res.notes + " ") if res.notes else "") + \
                    ("LMXB catalog match; compact object type not "
                     "resolvable from columns -- check burst/pulsation "
                     "flags manually.")
            else:
                res.notes = ((res.notes + " ") if res.notes else "") + evidence_note
        elif type_source == "hmxb_heasarc":
            tbl = catalogs["hmxb_heasarc"]
            orig_idx = catalog_valid_idx["hmxb_heasarc"][np.argmin(target.separation(hmxb_coords))]
            compact_object, evidence_note = _infer_hmxb_compact_object(tbl, orig_idx)
            res.compact_object = compact_object
            if compact_object == "Unknown":
                res.confidence = "medium"
            res.notes = ((res.notes + " ") if res.notes else "") + evidence_note

        if res.matched_catalog not in (None, "None"):
            results.append(res)
            continue

        # --- Step 4: SIMBAD fallback ---
        # otype lookup by IDENTIFIER STRING alone repeats the exact same
        # query that may have already failed earlier in this loop (e.g.
        # sources that only resolved a coordinate via the coordinate
        # cone search in _simbad_resolve_by_coordinates() -- that's a
        # nearby SIMBAD object under a DIFFERENT name, so re-querying
        # query_name as an identifier here predictably fails again and
        # used to overwrite the notes with a misleading "no match in any
        # catalog, including SIMBAD" even though we already have a
        # confirmed position). Look up otype by the COORDINATE we
        # already resolved first; only fall back to the identifier-string
        # lookup if that finds nothing (e.g. a catalog-confirmed
        # coordinate with no separate SIMBAD entry at all).
        if use_simbad_fallback:
            otype = _simbad_otype_by_coordinates(res.resolved_ra, res.resolved_dec,
                                                   radius_arcsec=match_radius_arcsec)
            if sleep_between_simbad_calls:
                time.sleep(sleep_between_simbad_calls)
            if not otype:
                otype = _simbad_otype_fallback(query_name)
                if sleep_between_simbad_calls:
                    time.sleep(sleep_between_simbad_calls)
            res.simbad_otype = otype
            res.confidence = "low"
            res.matched_catalog = "SIMBAD (fallback)"
            if otype:
                otype_u = otype.upper()
                if "HXB" in otype_u:
                    res.xrb_type = "HMXB"
                elif "LXB" in otype_u:
                    res.xrb_type = "LMXB"
                elif "XB" in otype_u:
                    res.xrb_type = "Unknown"
                if "PSR" in otype_u or "PULSAR" in otype_u:
                    res.compact_object = "NS"
                elif "BH" in otype_u:
                    res.compact_object = "BH"
                res.notes = ((resolution_note + " ") if resolution_note else "") + \
                    f"Unmatched in curated catalogs; SIMBAD otype='{otype}' used as weak evidence."
            else:
                res.notes = ((resolution_note + " ") if resolution_note else "") + \
                    "No match in any catalog, including SIMBAD. Needs manual literature check."

        results.append(res)

    return results


def _simbad_otype_by_coordinates(ra_deg: float, dec_deg: float, radius_arcsec: float):
    """
    Look up the SIMBAD otype of the nearest object within radius_arcsec
    of (ra_deg, dec_deg). Used by Step 4 of classify_sources() so the
    final 'weak evidence' otype check reuses a coordinate we already
    trust, rather than repeating an identifier-string query that may
    have already failed earlier for this exact source.
    """
    from astroquery.simbad import Simbad
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if ra_deg is None or dec_deg is None:
        return None
    try:
        custom = Simbad()
        custom.add_votable_fields("otype")
        center = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        result = custom.query_region(center, radius=radius_arcsec * u.arcsec)
    except Exception as e:
        logger.warning(f"SIMBAD coordinate otype query failed at RA={ra_deg:.5f} "
                        f"Dec={dec_deg:.5f} [{type(e).__name__}]: {e}")
        return None
    if result is None or len(result) == 0:
        return None

    otype_col = _find_col(result, "OTYPE", "otype")
    ra_col = _find_col(result, "ra", "RA")
    de_col = _find_col(result, "dec", "DEC")
    if otype_col is None:
        return None
    if ra_col is None or de_col is None:
        return str(result[otype_col][0])

    try:
        coords = SkyCoord(ra=np.asarray(result[ra_col], dtype=float) * u.deg,
                           dec=np.asarray(result[de_col], dtype=float) * u.deg)
        idx = int(np.argmin(SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg).separation(coords)))
    except Exception:
        idx = 0
    return str(result[otype_col][idx])


def _simbad_otype_fallback(query_name: str):
    from astroquery.simbad import Simbad
    try:
        custom = Simbad()
        custom.add_votable_fields("otype")
        result = custom.query_object(query_name)
        if result is None or len(result) == 0:
            return None
        col = _find_col(result, "OTYPE", "otype")
        return str(result[col][0]) if col else None
    except Exception as e:
        logger.warning(f"SIMBAD otype fallback failed for {query_name} "
                        f"[{type(e).__name__}]: {e}")
        return None


def write_results_csv(results: List[ClassificationResult], out_path: str):
    import csv
    with open(out_path, "w", newline="") as fh:
        fieldnames = list(asdict(results[0]).keys()) if results else []
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))