"""
Zero report cleaning: merge KABS 1.0 (current + archive exports), KABS 2.0 (ZRF) and the
KABS zero report export (Raw_Datasets/Zero Report_KABS.xls, form v10-v15, with Id) into one
dataset with clean county / sub-county / ward names.

No single export is complete. KABS 1.0 + 2.0 form the base; the .xls export only adds
reports the base does not already have: per report date x county x ward x species it adds
its surplus over the base (rows that also match on every disease answer are treated as the
same report first), so the same report is never counted twice.

Zero reports say, per species, that a priority disease was looked for and NOT seen.
Each disease column has three distinct states that must never be collapsed:
    "No"              disease checked, not present (the zero report itself)
    "Not applicable"  disease does not apply to the species (raw "N/A")
    blank             question not answered / not on that form version
No disease value is ever converted to 0, and blanks stay blank. Raw "N/A" is
written as "Not applicable" because pandas, Excel and R read the literal text
"N/A" as missing by default, which would silently merge it with blanks.

Steps
  1. Read all three files with "N/A" kept as text.
  2. KABS 1.0: drop exact duplicate rows (identical to the millisecond submission
     timestamp, so they are export repeats). KABS 2.0 has no submission timestamp,
     so its identical rows are only flagged in possible_duplicate.
  3. Old KABS 1.0 form: combined questions CCPPCBPP / RPPPRFMD are copied into the
     individual disease columns (only where those are blank), and multi-species
     reports '["Cattle", "Sheep"]' become one row per species (same report_id).
  4. Dates, GPS (swapped lat/long fixed, points outside Kenya nulled), species,
     locality and production system standardised.
  5. County / sub-county / ward names cleaned to the Kenya shapefile names
     (clean_admin_names.py); GPS points outside their county flagged.
  6. Zero reports only cover specific diseases per species (ZERO_REPORT_SCOPE):
     Cattle RVF/RP/FMD/CBPP, Goats RVF/RP/FMD/PPR/CCPP, Sheep RVF/RP/PPR/FMD,
     Pigs FMD/ASF, Camel RVF, Chicken AI. Every other species/disease cell is set
     to "Out of scope", so a "No" there is never counted as a zero report.
     In-scope answers are untouched; in-scope N/A or blanks are listed in
     in_scope_marked_na / in_scope_blank for follow-up.

Outputs:
    Cleaned Data/zero_report_combined_clean.csv   one row per report x species (wide)
    Cleaned Data/zero_report_combined_long.csv    one row per report x species x in-scope disease;
                                                  zero_reported = True ("No"), False (marked N/A),
                                                  blank (not reported)

Run from the project root:
    python 02cleaning_scripts/zero_report_cleaning.py
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from clean_admin_names import clean_admin_names
from harmonise import epoch_ms_to_date, latest_export, read_raw, standardise_species, validate_coords
from harmonise import kabs2_date as text_to_date

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("zero_report_cleaning")

ROOT = Path(__file__).resolve().parents[1]

DISEASES = ["rvf", "rp", "fmd", "ppr", "cbpp", "ccpp", "avian_influenza", "asf"]
NOT_APPLICABLE = "Not applicable"
OUT_OF_SCOPE = "Out of scope"

# Diseases that are zero-reported for each species (KABS 2.0 zero report list).
# Any other species/disease pair is not a zero report, even if the form recorded "No".
ZERO_REPORT_SCOPE = {
    "Cattle": ["rvf", "rp", "fmd", "cbpp"],
    "Goats": ["rvf", "rp", "fmd", "ppr", "ccpp"],
    "Sheep": ["rvf", "rp", "ppr", "fmd"],
    "Pigs": ["fmd", "asf"],
    "Camel": ["rvf"],
    "Chicken": ["avian_influenza"],
}
DISEASE_NAMES = {
    "rvf": "Rift Valley Fever (RVF)", "rp": "Rinderpest (RP)", "fmd": "Foot and Mouth Disease (FMD)",
    "ppr": "Peste des Petits Ruminants (PPR)", "cbpp": "Contagious Bovine Pleuropneumonia (CBPP)",
    "ccpp": "Contagious Caprine Pleuropneumonia (CCPP)", "avian_influenza": "Avian Influenza (AI)",
    "asf": "African Swine Fever (ASF)",
}

OUTPUT_COLS = [
    "source", "report_id", "report_date", "submission_date",
    "county", "county_iso", "subcounty", "ward", "subcounty_raw", "ward_raw", "admin_match",
    "locality", "latitude", "longitude", "coords_in_county",
    "species", "multi_species_report", "species_in_scope", "production_system",
    *DISEASES,
    "in_scope_marked_na", "in_scope_blank",
    "diseases_text", "notes", "possible_duplicate",
]
LONG_COLS = [
    "source", "report_id", "report_date", "county", "county_iso", "subcounty", "ward", "locality",
    "latitude", "longitude", "coords_in_county", "species", "production_system",
    "disease_code", "disease", "answer", "zero_reported", "possible_duplicate",
]


def clean_answer(s: pd.Series) -> pd.Series:
    """Standardise a disease answer without ever filling blanks."""
    s = s.str.strip().str.lower()
    mapped = s.map({"no": "No", "yes": "Yes", "n/a": NOT_APPLICABLE, "na": NOT_APPLICABLE,
                    "not applicable": NOT_APPLICABLE})
    unexpected = s.notna() & mapped.isna()
    if unexpected.any():
        log.warning("%s: %d unexpected values left as-is: %s", s.name, unexpected.sum(),
                    s[unexpected].value_counts().head(5).to_dict())
    return mapped.fillna(s.where(unexpected))


def parse_species(value) -> list[str]:
    if pd.isna(value):
        return []
    s = str(value).strip()
    if s.startswith("["):
        try:
            return [str(x).strip() for x in json.loads(s) if str(x).strip()]
        except json.JSONDecodeError:
            s = s.strip("[]")
    return [x.strip().strip('"') for x in s.split(",") if x.strip().strip('"')]


# --------------------------------------------------------------------------- #
# KABS 1.0
# --------------------------------------------------------------------------- #
def load_kabs1(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        raw = read_raw(p)
        before = len(raw)
        raw = raw.drop_duplicates()
        log.info("KABS 1.0 %s: %d rows, dropped %d exact duplicates", p.name, before, before - len(raw))
        frames.append(raw)
    v1 = pd.concat(frames, ignore_index=True)
    overlap = v1.duplicated().sum()
    if overlap:
        log.info("KABS 1.0: dropped %d rows present in both current and archive files", overlap)
        v1 = v1.drop_duplicates()
    v1 = v1.reset_index(drop=True)

    # Old form: combined questions -> individual diseases (only where blank).
    for combined, targets in [("CCPPCBPP", ["CCPP", "CBPP"]), ("RPPPRFMD", ["RP", "PPR", "FMD"])]:
        for t in targets:
            fill = v1[t].isna() & v1[combined].notna()
            v1.loc[fill, t] = v1.loc[fill, combined]
        log.info("KABS 1.0: %s copied into %s for %d rows", combined, "/".join(targets), v1[combined].notna().sum())

    out = pd.DataFrame(index=v1.index)
    out["source"] = "KABS 1.0"
    out["report_id"] = "K1-" + pd.Series(range(1, len(v1) + 1), index=v1.index).astype(str)
    out["report_date"] = epoch_ms_to_date(v1["ReportDate"])
    out["submission_date"] = epoch_ms_to_date(v1["submissionDate"])
    for src, dst in [("County", "county"), ("SubCounty", "subcounty"), ("Ward", "ward"),
                     ("Locality", "locality"), ("ProdSystem", "production_system"),
                     ("Notes", "notes"), ("Disease", "diseases_text")]:
        out[dst] = v1[src]
    out["latitude"] = pd.to_numeric(v1["LocLat"], errors="coerce")
    out["longitude"] = pd.to_numeric(v1["LocLong"], errors="coerce")
    out["species_list"] = [
        [o if s.lower() == "other" and pd.notna(o) else s for s in parse_species(sp)] or ([o] if pd.notna(o) else [])
        for sp, o in zip(v1["Species"], v1["OtherSpecies"])
    ]
    for src, dst in [("RVF", "rvf"), ("RP", "rp"), ("FMD", "fmd"), ("PPR", "ppr"), ("CBPP", "cbpp"),
                     ("CCPP", "ccpp"), ("AI", "avian_influenza"), ("ASF", "asf")]:
        out[dst] = v1[src]
    out["possible_duplicate"] = False  # exact duplicates already removed
    return out


# --------------------------------------------------------------------------- #
# KABS 2.0
# --------------------------------------------------------------------------- #
def load_kabs2(path: Path) -> pd.DataFrame:
    v2 = read_raw(path)
    stray = [c for c in v2.columns if c.isdigit()]
    blank = v2["report_date"].isna()
    log.info("KABS 2.0 %s: %d rows; dropping %d test/blank rows and %d numbered stray columns",
             path.name, len(v2), blank.sum(), len(stray))
    v2 = v2.loc[~blank].drop(columns=stray).reset_index(drop=True)

    out = pd.DataFrame(index=v2.index)
    out["source"] = "KABS 2.0"
    out["report_id"] = "K2-" + pd.Series(range(1, len(v2) + 1), index=v2.index).astype(str)
    out["report_date"] = text_to_date(v2["report_date"])
    out["submission_date"] = pd.NaT
    suffix = {"county": r"\s+County$", "subcounty": r"\s+Sub[\s-]*County$", "ward": r"\s+Ward$"}
    for col, pat in suffix.items():
        out[col] = v2[col].str.replace(pat, "", regex=True, flags=re.I)
    for col in ["locality", "production_system", "notes"]:
        out[col] = v2[col]
    out["diseases_text"] = np.nan
    gps = v2["gps_location"].str.extract(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]")
    out["latitude"] = pd.to_numeric(gps[0], errors="coerce")
    out["longitude"] = pd.to_numeric(gps[1], errors="coerce")
    out["species_list"] = v2["species"].map(parse_species)
    for col in DISEASES:
        out[col] = v2[col]

    # No submission timestamp in KABS 2.0, so identical rows may be genuine repeat reports.
    out["possible_duplicate"] = v2.duplicated(keep=False)
    log.info("KABS 2.0: %d rows are identical to another row (flagged, not dropped)", out["possible_duplicate"].sum())
    return out


# --------------------------------------------------------------------------- #
# combine
# --------------------------------------------------------------------------- #
def load_reports_export(path: Path) -> pd.DataFrame:
    """KABS zero report export (.xls, one sheet per form version), 'N/A' kept as text."""
    sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False, na_values=[""])
    x = pd.concat(sheets.values(), ignore_index=True)
    x = x.apply(lambda c: c.str.strip().mask(lambda v: v == ""))
    log.info("KABS zero report export %s: %d rows from %d sheets", path.name, len(x), len(sheets))
    out = pd.DataFrame(index=x.index)
    out["source"] = "KABS zero report export"
    out["report_id"] = x["Id"]
    out["report_date"] = pd.to_datetime(x["Date of Report"].str[:10], format="%Y-%m-%d", errors="coerce")
    out["submission_date"] = pd.NaT
    for src, dst in [("County", "county"), ("Sub-County", "subcounty"), ("Ward", "ward"), ("Locality", "locality"),
                     ("Production System", "production_system"), ("Notes", "notes")]:
        out[dst] = x[src]
    out["diseases_text"] = np.nan
    out["latitude"] = pd.to_numeric(x["Latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(x["Longitude"], errors="coerce")
    out["species_list"] = [
        [o if s.lower() == "other" and pd.notna(o) else s for s in parse_species(sp)] or ([o] if pd.notna(o) else [])
        for sp, o in zip(x["Species examined"], x["Other Species"])
    ]
    for src, dst in [("RVF", "rvf"), ("RP", "rp"), ("FMD", "fmd"), ("PPR", "ppr"), ("CBPP", "cbpp"),
                     ("CCPP", "ccpp"), ("Avian Influenza", "avian_influenza"), ("ASF", "asf")]:
        out[dst] = x[src] if src in x else np.nan
    out["possible_duplicate"] = False
    return out


def _report_keys(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per species row: (date x county x ward x species, that + every disease answer)."""
    norm = lambda s: s.astype("string").str.lower().str.replace(r"\b(county|ward)\b", "", regex=True) \
        .str.replace(r"[^a-z]", "", regex=True).replace({"tharakanithi": "tharaka"}).fillna("")
    date = pd.to_datetime(df["report_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    group = date + "|" + norm(df["county"]) + "|" + norm(df["ward"]) + "|" + norm(df["species"]).str.rstrip("s")
    answers = df[DISEASES].apply(lambda c: c.astype("string").str.lower().fillna("")).agg("|".join, axis=1)
    return group, group + "|" + answers


def add_missing_reports(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Reports in `new` not already in `base` (see module docstring)."""
    b = base.explode("species_list").rename(columns={"species_list": "species"})
    n = new.assign(_n=range(len(new))).explode("species_list").rename(columns={"species_list": "species"})
    b_group, b_detail = _report_keys(b)
    n_group, n_detail = _report_keys(n)
    avail = b_detail.value_counts().to_dict()
    exact = []
    for d in n_detail:
        exact.append(avail.get(d, 0) > 0)
        if exact[-1]:
            avail[d] -= 1
    n["_exact"], n["_group"] = exact, n_group.to_numpy()
    counts = n_group.value_counts()
    surplus = (counts - b_group.value_counts().reindex(counts.index).fillna(0)).clip(lower=0)
    n = n.sort_values("_exact", kind="stable")
    keep_rows = n[n.groupby("_group").cumcount() < n["_group"].map(surplus)]
    keep = new.iloc[sorted(set(keep_rows["_n"]))]
    log.info("%s: %d reports; %d already in KABS 1.0 / 2.0, %d added",
             new["source"].iloc[0], len(new), len(new) - len(keep), len(keep))
    return keep


def explode_species(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["multi_species_report"] = df["species_list"].str.len() > 1
    n_reports = df["multi_species_report"].sum()
    df = df.explode("species_list").rename(columns={"species_list": "species"}).reset_index(drop=True)
    log.info("Split %d multi-species reports into one row per species (now %d rows)", n_reports, len(df))
    return df


def standardise_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["species"] = standardise_species(df["species"])
    df["locality"] = df["locality"].str.strip()
    df["production_system"] = df["production_system"].str.strip().str.capitalize().replace(
        {"N/a": NOT_APPLICABLE, "Na": NOT_APPLICABLE})
    for col in ["notes", "diseases_text"]:
        df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()
    for col in DISEASES:
        df[col] = clean_answer(df[col].rename(col))
    return df


def apply_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Mark species/disease pairs that are not zero-reportable; never touch in-scope answers."""
    df = df.copy()
    df["species_in_scope"] = df["species"].isin(ZERO_REPORT_SCOPE)
    in_scope = pd.DataFrame({c: df["species"].map(lambda s: c in ZERO_REPORT_SCOPE.get(s, [])) for c in DISEASES})

    moved = {c: df.loc[~in_scope[c], c].fillna("<blank>").value_counts().to_dict() for c in DISEASES}
    log.info("Out-of-scope species/disease answers set to '%s' (previous values):\n%s", OUT_OF_SCOPE,
             pd.DataFrame(moved).fillna(0).astype(int).to_string())
    for c in DISEASES:
        df.loc[~in_scope[c], c] = OUT_OF_SCOPE

    # In-scope questions answered N/A or left blank are not zero reports: list them for follow-up.
    def which(mask: pd.DataFrame) -> pd.Series:
        return mask.apply(lambda r: "; ".join(c for c in DISEASES if r[c]) or np.nan, axis=1)

    df["in_scope_marked_na"] = which(in_scope & df[DISEASES].eq(NOT_APPLICABLE))
    df["in_scope_blank"] = which(in_scope & df[DISEASES].isna())
    log.info("Rows for species with no zero-report diseases: %d %s", (~df["species_in_scope"]).sum(),
             df.loc[~df["species_in_scope"], "species"].value_counts(dropna=False).head(6).to_dict())
    log.info("Rows with an in-scope disease marked N/A: %d; with an in-scope disease left blank: %d",
             df["in_scope_marked_na"].notna().sum(), df["in_scope_blank"].notna().sum())
    return df


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per report x species x in-scope disease."""
    long = df.melt(id_vars=[c for c in LONG_COLS if c in df.columns], value_vars=DISEASES,
                   var_name="disease_code", value_name="answer")
    long = long[long["answer"].ne(OUT_OF_SCOPE)].copy()
    long["disease"] = long["disease_code"].map(DISEASE_NAMES)
    # True = zero reported ("No"), False = marked not applicable, blank = not reported.
    long["zero_reported"] = long["answer"].map({"No": True, NOT_APPLICABLE: False}).astype("boolean")
    return long[LONG_COLS].sort_values(["report_date", "report_id", "disease_code"], kind="stable")


def run_pipeline(v1_paths: list[Path], v2_path: Path, export_path: Path | None, out: Path) -> pd.DataFrame:
    df = pd.concat([load_kabs1(v1_paths), load_kabs2(v2_path)], ignore_index=True)
    if export_path is not None and export_path.exists():
        df = pd.concat([df, add_missing_reports(df, load_reports_export(export_path))], ignore_index=True)

    # Expected answer counts after the species split: each raw answer repeated once per species.
    weight = df["species_list"].str.len().clip(lower=1)
    state = lambda s: s.str.strip().str.lower().map({"no": "No", "n/a": NOT_APPLICABLE}).fillna(s).fillna("<blank>")
    expected = {c: weight.groupby(state(df[c])).sum() for c in DISEASES}

    df = validate_coords(df)
    df = explode_species(df)
    df = standardise_text(df)
    df = clean_admin_names(df)

    # Guard: No / Not applicable / blank counts must be exactly preserved (no NA -> 0, no NA -> No).
    for c in DISEASES:
        got = df[c].fillna("<blank>").value_counts().sort_index()
        if not got.equals(expected[c].sort_index().rename(got.name)):
            raise ValueError(f"{c}: answer counts changed during cleaning\nexpected {expected[c].to_dict()}\ngot {got.to_dict()}")
    log.info("Check passed: No / Not applicable / blank counts preserved for all %d diseases", len(DISEASES))

    df = apply_scope(df)
    log.info("Disease answers after scoping (rows):\n%s",
             df[DISEASES].apply(lambda s: s.value_counts(dropna=False)).fillna(0).astype(int).to_string())

    future = df["report_date"] > pd.Timestamp.today()
    old = df["report_date"] < "2015-01-01"
    log.warning("Report dates in the future: %d, before 2015: %d (left as-is)", future.sum(), old.sum())

    df = df[OUTPUT_COLS].sort_values(["report_date", "source", "report_id"], kind="stable").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, date_format="%Y-%m-%d")
    log.info("Rows by source:\n%s", df["source"].value_counts().to_string())
    log.info("Wrote %d rows x %d columns -> %s", *df.shape, out)

    long = to_long(df)
    long_out = out.with_name(out.stem.replace("_clean", "") + "_long.csv")
    long.to_csv(long_out, index=False, date_format="%Y-%m-%d")
    log.info("Zero reports by species x disease (in scope only):\n%s",
             pd.crosstab([long["species"]], long["disease_code"], values=long["zero_reported"].fillna(False),
                         aggfunc="sum").fillna(0).astype(int).to_string())
    log.info("Wrote long format (%d rows: one per report x species x in-scope disease) -> %s", len(long), long_out)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, nargs="+", default=[
        latest_export(ROOT / "data/kabs1.0_raw", "zero_report"),
        latest_export(ROOT / "data/kabs1.0_raw", "zero_report_archive"),
    ])
    parser.add_argument("--v2", type=Path, default=latest_export(ROOT / "data/kabs2.0_raw", "zrf"))
    parser.add_argument("--export", type=Path, default=ROOT / "Raw_Datasets/Zero Report_KABS.xls")
    parser.add_argument("--out", type=Path, default=ROOT / "Cleaned Data/zero_report_combined_clean.csv")
    args = parser.parse_args()
    run_pipeline(args.v1, args.v2, args.export, args.out)
