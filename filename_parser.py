"""
filename_parser.py
-------------------
Parses filenames from the Total_Data directory into a normalized,
query-ready source identifier, following the naming conventions
described in the pipeline design doc (MAXI GSC daily lightcurves,
ZTF optical photometry).

Because the naming convention is "a bit variable", this parser is
regex/whitelist driven rather than hard-coded to a handful of examples,
so it should degrade gracefully on filenames it hasn't seen before
(it will just fall back to a best-effort guess and flag it as
low-confidence so you can eyeball a review CSV instead of trusting
it blindly).

USAGE
-----
from filename_parser import parse_filename, parse_directory

result = parse_filename("MAXI_J2303p088_maxi_lc.csv")
# ParsedSource(raw_filename=..., core_identifier='J2303p088',
#              query_name='MAXI J2303+088', band='xray', instrument='MAXI',
#              confidence='high')

df = parse_directory("/path/to/Total_Data")
"""

from __future__ import annotations

import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
import csv


# ---------------------------------------------------------------------------
# Known mission / catalog prefixes.
# Extend this dict as you encounter new prefixes in Total_Data -- that's the
# main maintenance surface for "the naming convention is a bit variable".
# Key: prefix as it appears in the filename (matched case-insensitively).
# Value: the string to reconstruct in the standardized query identifier.
# ---------------------------------------------------------------------------
MISSION_PREFIX_MAP = {
    "MAXI": "MAXI",
    "XTE": "XTE",
    "IGR": "IGR",
    "SWIFT": "Swift",
    "GRS": "GRS",
    "GX": "GX",
    "SAX": "SAX",
    "EXO": "EXO",
    "KS": "KS",
    "GS": "GS",
    "AX": "AX",
    "OAO": "OAO",
    "MXB": "MXB",
    "4U": "4U",
    "1E": "1E",
    "H": "H",
    "A": "A",     # Ariel V catalog (e.g. "A0535+262")
    "1A": "1A",   # HEAO A-1 catalog (e.g. "1A0620-00")
    "3A": "3A",
}

# ROSAT-style prefixes that map to a *different* string, not just
# reformatted -- "RJ" -> "RX J" per the design doc.
SPECIAL_PREFIX_MAP = {
    "RJ": "RX J",
    "1RXSJ": "1RXS J",
}

# Suffixes that identify the instrument/band. Longest-match-first ordering
# matters (_maxi_lc before _maxi).
SUFFIX_MAP = [
    (r"_maxi_lc$", "xray", "MAXI"),
    (r"_maxi$", "xray", "MAXI"),
    (r"_gsc_lc$", "xray", "MAXI"),
    (r"_bat_lc$", "xray", "SwiftBAT"),
    (r"_bat$", "xray", "SwiftBAT"),
    (r"_xrt_lc$", "xray", "SwiftXRT"),
    (r"_xrt$", "xray", "SwiftXRT"),
    (r"_ztf_lc$", "optical", "ZTF"),
    (r"_ztf$", "optical", "ZTF"),
    (r"_atlas_lc$", "optical", "ATLAS"),
    (r"_atlas$", "optical", "ATLAS"),
]

FILE_EXT_RE = re.compile(r"\.(csv|txt|dat|fits|lc|json)$", re.IGNORECASE)

# Tokens that are genuine acronyms (Large/Small Magellanic Cloud, etc.) and
# should stay upper-case rather than being title-cased like an ordinary
# common star name (SCO -> "Sco", but LMC must NOT become "Lmc").
# Extend this as new all-caps acronym tokens show up in Total_Data.
KNOWN_ACRONYMS = {"LMC", "SMC"}


@dataclass
class ParsedSource:
    raw_filename: str
    core_identifier: str
    query_name: str          # standardized string to feed to catalog queries
    source_key: str          # filesystem/dict-safe key (e.g. "J2303+088")
    band: str                # 'xray' | 'optical' | 'unknown'
    instrument: str          # 'MAXI' | 'ZTF' | 'SwiftBAT' | ...
    confidence: str          # 'high' | 'medium' | 'low'
    notes: str = ""
    query_name_alt: str = None   # opposite-sign reading, only set when
                                  # sign_ambiguous is True
    sign_ambiguous: bool = False


def _strip_extension(name: str) -> str:
    return FILE_EXT_RE.sub("", name)


def _detect_band_and_strip_suffix(stem: str):
    for pattern, band, instrument in SUFFIX_MAP:
        if re.search(pattern, stem, flags=re.IGNORECASE):
            core = re.sub(pattern, "", stem, flags=re.IGNORECASE)
            return core, band, instrument
    return stem, "unknown", "unknown"


def _apply_polarity_mapping(coord: str) -> str:
    """
    Replace filesystem-safe polarity substitutions with real +/- signs.
    Only applies between digits, so it won't mangle real letters
    (e.g. won't touch a 'm' that happens to sit outside a
    declination-sign position).
    """
    coord = re.sub(r"(?<=\d)p(?=\d)", "+", coord)
    coord = re.sub(r"(?<=\d)m(?=\d)", "-", coord)
    return coord


def _resolve_ambiguous_sign(coord: str):
    """
    Some filenames in Total_Data separate RA/Dec with a bare underscore
    and no p/m marker at all (e.g. "4U0053_604", "MAXI_J0028_592") --
    a DIFFERENT convention from the p/m-encoded sign handled above.
    Unlike 'p'/'m', a bare '_' carries no sign information, so the true
    declination sign genuinely cannot be recovered from the filename
    alone.

    Returns (best_guess, alternate_or_None, ambiguous_bool):
      - best_guess: coord with '_' replaced by '+' (the more common
        convention for the sources observed so far) -- used as the
        primary query_name so downstream code has *something* to try
        first.
      - alternate: the same coord with '_' replaced by '-', so callers
        can retry / cross-check against curated catalogs rather than
        silently trusting the '+' guess.
      - ambiguous_bool: True if a bare underscore sign was found.
    """
    if not re.search(r"(?<=\d)_(?=\d)", coord):
        return coord, None, False
    best_guess = re.sub(r"(?<=\d)_(?=\d)", "+", coord)
    alternate = re.sub(r"(?<=\d)_(?=\d)", "-", coord)
    return best_guess, alternate, True


def _title_case_common_name(core: str) -> str:
    """
    Title-cases an ALL_CAPS_WITH_UNDERSCORES common name (e.g.
    SCO_X-1 -> "Sco X-1"), except for tokens in KNOWN_ACRONYMS which
    must stay upper-case (e.g. LMC_X-4 -> "LMC X-4", not "Lmc X-4").
    """
    parts = core.split("_")
    out = [p.upper() if p.upper() in KNOWN_ACRONYMS else p.title() for p in parts]
    return " ".join(out)


def _split_prefix(core: str):
    """
    Try to split a leading mission/catalog prefix off the core identifier.
    Returns (prefix_or_None, remainder).
    """
    # Special two-token ROSAT-style prefixes first (RJ -> RX J)
    for pfx in sorted(SPECIAL_PREFIX_MAP, key=len, reverse=True):
        if core.upper().startswith(pfx):
            return pfx.upper(), core[len(pfx):]

    # underscore-delimited prefix: MAXI_J2303p088 -> ('MAXI', 'J2303p088')
    if "_" in core:
        head, _, tail = core.partition("_")
        if head.upper() in MISSION_PREFIX_MAP:
            return head.upper(), tail

    # no-underscore prefix directly followed by a coordinate-looking token
    # e.g. "4U1636-536" -> ('4U', '1636-536'), "1A0620-00" -> ('1A', '0620-00')
    # Some catalog prefixes (4U, 1E, 1A, 3A) start with a digit themselves,
    # so the prefix pattern has to allow an optional leading digit before
    # the letters -- a plain [A-Za-z]+ prefix regex misses these entirely
    # and silently glues the whole thing into one unspaced token that
    # SIMBAD's name resolver won't recognize (e.g. "4U0535+262" instead
    # of "4U 0535+262").
    m = re.match(r"^(\d?[A-Za-z]+)(\d.*)$", core)
    if m and m.group(1).upper() in MISSION_PREFIX_MAP:
        return m.group(1).upper(), m.group(2)

    return None, core


def parse_filename(filename: str) -> ParsedSource:
    raw = filename
    stem = _strip_extension(os.path.basename(filename))
    core, band, instrument = _detect_band_and_strip_suffix(stem)

    if band == "unknown":
        notes = "No recognized instrument suffix (_maxi/_ztf/...); band unknown."
        confidence = "low"
    else:
        notes = ""
        confidence = "high"

    prefix, remainder = _split_prefix(core)

    query_name_alt = None
    sign_ambiguous = False

    if prefix is not None:
        if prefix in SPECIAL_PREFIX_MAP:
            mission_str = SPECIAL_PREFIX_MAP[prefix]
            coord = _apply_polarity_mapping(remainder)
            coord, alt_coord, sign_ambiguous = _resolve_ambiguous_sign(coord)
            query_name = f"{mission_str}{coord}"
            core_identifier = coord
            if sign_ambiguous:
                query_name_alt = f"{mission_str}{alt_coord}"
        else:
            mission_str = MISSION_PREFIX_MAP[prefix]
            coord = _apply_polarity_mapping(remainder)
            coord, alt_coord, sign_ambiguous = _resolve_ambiguous_sign(coord)
            query_name = f"{mission_str} {coord}"
            core_identifier = coord
            if sign_ambiguous:
                query_name_alt = f"{mission_str} {alt_coord}"
    else:
        # No recognized mission prefix.
        if re.match(r"^J?\d", core):
            # Looks like a bare coordinate (e.g. "J2303+088" or "0146.9+6121")
            coord = _apply_polarity_mapping(core)
            coord, alt_coord, sign_ambiguous = _resolve_ambiguous_sign(coord)
            core_identifier = coord
            query_name = core_identifier
            if sign_ambiguous:
                query_name_alt = alt_coord
        elif "_" in core and core.upper() == core:
            # ALL_CAPS_WITH_UNDERSCORES common name -> "Title Case With Spaces"
            # e.g. SCO_X-1 -> "Sco X-1", but LMC_X-4 -> "LMC X-4" (acronym
            # tokens in KNOWN_ACRONYMS are kept upper-case, not title-cased).
            core_identifier = core
            query_name = _title_case_common_name(core)
        else:
            # Fall back: best-effort, flag for manual review.
            core_identifier = core
            query_name = core.replace("_", " ")
            confidence = "low"
            notes = (notes + " Unrecognized prefix pattern; used raw fallback."
                      ).strip()

    if sign_ambiguous:
        confidence = "low"
        notes = (notes + " Filename encodes declination sign as a bare "
                 "underscore with no p/m marker; sign could not be recovered "
                 f"from the filename. Defaulted to '+' in query_name; "
                 f"'{query_name_alt}' is the alternate ('-') reading -- "
                 "verify against catalog cross-match.").strip()

    source_key = query_name.replace(" ", "_").replace("/", "-")

    return ParsedSource(
        raw_filename=raw,
        core_identifier=core_identifier,
        query_name=query_name,
        source_key=source_key,
        band=band,
        instrument=instrument,
        confidence=confidence,
        notes=notes,
        query_name_alt=query_name_alt,
        sign_ambiguous=sign_ambiguous,
    )


def parse_directory(directory: str, extensions=(".csv", ".txt", ".dat", ".fits", ".lc")) -> List[ParsedSource]:
    directory = Path(directory)
    results = []
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix.lower() in extensions:
            results.append(parse_filename(f.name))
    return results


def write_review_csv(parsed: List[ParsedSource], out_path: str):
    """
    Dumps the parse results to a CSV so you can quickly eyeball anything
    flagged low/medium confidence before it goes into the catalog matcher.
    """
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["raw_filename", "core_identifier", "query_name",
                          "query_name_alt", "sign_ambiguous", "source_key",
                          "band", "instrument", "confidence", "notes"])
        for p in parsed:
            writer.writerow([p.raw_filename, p.core_identifier, p.query_name,
                              p.query_name_alt, p.sign_ambiguous, p.source_key,
                              p.band, p.instrument, p.confidence, p.notes])


if __name__ == "__main__":
    # Self-test against every example in the design doc's mapping table.
    test_cases = {
        "MAXI_J2303p088_maxi_lc": "MAXI J2303+088",
        "MAXI_J2303+088_ztf": "MAXI J2303+088",
        "MAXI_J2304m086_maxi_lc": "MAXI J2304-086",
        "RJ0146.9+6121_maxi": "RX J0146.9+6121",
        "RJ0146.9+6121_ztf": "RX J0146.9+6121",
        "SCO_X-1_maxi": "Sco X-1",
        "SER_X-1_maxi": "Ser X-1",
        "V0332+53_maxi": "V0332+53",
        "VELA_X-1_maxi": "Vela X-1",
        "XTE_J1709-267_maxi": "XTE J1709-267",
        # Bare-underscore declination (no p/m sign) -- newly handled.
        "4U0053_604_maxi": "4U 0053+604",
        "4U0115_634_ztf": "4U 0115+634",
        "MAXI_J0028_592_ztf": "MAXI J0028+592",
        # Acronym that must NOT be title-cased.
        "LMC_X-4_maxi": "LMC X-4",
    }
    all_ok = True
    for fname, expected in test_cases.items():
        r = parse_filename(fname)
        ok = r.query_name == expected
        all_ok &= ok
        status = "OK " if ok else "FAIL"
        extra = f" alt={r.query_name_alt}" if r.sign_ambiguous else ""
        print(f"[{status}] {fname:30s} -> {r.query_name:20s} (expected {expected}) "
              f"[{r.band}, conf={r.confidence}]{extra}")
    print("\nALL PASS" if all_ok else "\nSOME FAILED -- check regex/whitelist above")