"""
KABS ND1 (disease event) harmonisation.
Maps the KABS 1.0 export (CamelCase, epoch-ms dates) and the KABS 2.0 export
(snake_case, text dates) onto one schema and stacks them into a single CSV.

Run from the project root:
    python 02cleaning_scripts/00cleaning.py
    python 02cleaning_scripts/00cleaning.py --v1 data/kabs1.0_raw/nd1_28102026.csv \
        --v2 data/kabs2.0_raw/nd1_29102026.csv --out "Cleaned Data/nd1_final.csv"
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kabs_nd1_harmonise")

ROOT = Path(__file__).resolve().parents[1]

# Rough Kenya bounding box, used to catch swapped / junk coordinates.
LAT_RANGE = (-5.0, 5.5)
LON_RANGE = (33.5, 42.5)

# Syndrome block names: KABS 1.0 prefix -> unified (KABS 2.0) name.
SYNDROMES = {
    "Abortion": "abortion",
    "SuddenDeath": "sudden_death",
    "Hemorrhagic": "haemorrhagic",
    "Neurologic": "neurologic",
    "Bites": "animal_bites",
    "Respiratory": "respiratory_signs",
    "Vesicular": "oral_foot_lesions",
    "Cutaneous": "cutaneous_skin_lesions",
    "GIT": "gastrointestinal_tract_syndromes",
    "OtherSyn": "other_syndromes",
}

COUNT_COLS = [
    "number_at_risk", "number_sick", "number_dead", "number_slaughtered",
    "number_destroyed", "number_vaccinated", "humans_affected_number",
    "death_number", *[f"{s}_number" for s in SYNDROMES.values()],
]

OUTPUT_COLS = [
    "source", "report_date", "outbreak_event_date", "submission_date",
    "county", "subcounty", "ward", "locality", "latitude", "longitude", "onsite",
    "species", "production_system",
    *[c for s in SYNDROMES.values() for c in (s, f"{s}_number")],
    "other_syndromes_list", "bite_type", "gastrointestinal_bloody",
    "death", "death_number",
    "disease_condition", "diagnosis_nature", "laboratory_test_taken",
    "number_at_risk", "number_sick", "number_dead", "humans_affected_number",
    "number_slaughtered", "number_destroyed", "number_vaccinated",
    "vaccination_type", "vaccine_batch_no", "control_methods",
    "comments", "notes",
]

DIAGNOSIS_MAP = {"Laboratory Test": "Laboratory"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_list(value) -> list[str]:
    """'["FMD", "CBPP"]' -> ['FMD', 'CBPP']; plain text -> [text]."""
    if pd.isna(value) or not str(value).strip():
        return []
    s = str(value).strip()
    try:
        parsed = json.loads(s)
        items = parsed if isinstance(parsed, list) else [parsed]
    except (json.JSONDecodeError, TypeError):
        items = s.strip("[]").split(",") if s.startswith("[") else [s]
    return [str(x).strip().strip('"') for x in items if str(x).strip()]


def resolve_other(items: list[str], other_text) -> list[str]:
    """Replace the 'Other' option with the free-text 'other' answer, if any."""
    other = [] if pd.isna(other_text) or not str(other_text).strip() else [str(other_text).strip()]
    if not items:
        return other
    out = []
    for item in items:
        out.extend(other if item == "Other" and other else [item])
    return list(dict.fromkeys(out))  # de-dup, keep order


def join_list(items: list[str]) -> str | float:
    return "; ".join(items) if items else np.nan


def clean_text(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return s.mask(s.str.upper().isin(["", "N/A", "NA", "NONE", "NAN"]))


def load_county_lookup() -> dict[str, str]:
    path = ROOT / "data" / "canonical_counties.csv"
    if not path.exists():
        return {}
    ref = pd.read_csv(path, dtype=str)
    return dict(zip(ref["key"], ref["canonical"]))


def standardise_admin(df: pd.DataFrame, counties: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    suffixes = {"county": r"\s+County$", "subcounty": r"\s+Sub[\s-]*County$", "ward": r"\s+Ward$"}
    for col, suffix in suffixes.items():
        s = clean_text(df[col]).str.replace(suffix, "", regex=True, flags=re.I)
        df[col] = s.str.title().str.replace(r"'([A-Z])", lambda m: "'" + m[1].lower(), regex=True)
    key = df["county"].str.lower().str.replace(r"[^a-z]", "", regex=True)
    df["county"] = key.map(counties).fillna(df["county"])
    return df


def validate_coords(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lat, lon = df["latitude"], df["longitude"]
    in_box = lambda a, b: a.between(*LAT_RANGE) & b.between(*LON_RANGE)
    swapped = ~in_box(lat, lon) & in_box(lon, lat)
    df.loc[swapped, ["latitude", "longitude"]] = df.loc[swapped, ["longitude", "latitude"]].to_numpy()
    bad = df["latitude"].notna() & ~in_box(df["latitude"], df["longitude"])
    df.loc[bad, ["latitude", "longitude"]] = np.nan
    log.info("Coordinates: swapped %d lat/long pairs, nulled %d outside Kenya", swapped.sum(), bad.sum())
    return df


def to_count(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").round().astype("Int64")


# --------------------------------------------------------------------------- #
# KABS 1.0
# --------------------------------------------------------------------------- #
def epoch_ms_to_date(s: pd.Series) -> pd.Series:
    ms = pd.to_numeric(s, errors="coerce")
    return pd.to_datetime(ms, unit="ms", utc=True).dt.tz_convert("Africa/Nairobi").dt.date


def collapse_old_form(v1: pd.DataFrame) -> pd.DataFrame:
    """
    Early KABS 1.0 submissions (no generic Risk/NoSick/Disease) store outcome
    data per syndrome block (e.g. RespiratoryNoSick, GITDisease). Collapse
    those into the generic columns. Counts take the max across blocks because
    the same herd is repeated in each block, so summing would double count.
    """
    v1 = v1.copy()
    old = v1["Risk"].isna() & v1["NoSick"].isna() & v1["Disease"].isna()
    blocks = list(SYNDROMES)

    for generic, suffix in [("Risk", "Risk"), ("NoSick", "NoSick"), ("NoDead", "NoDead"), ("NoHumans", "NoHumans")]:
        cols = [f"{b}{suffix}" for b in blocks if f"{b}{suffix}" in v1]
        v1.loc[old, generic] = v1.loc[old, cols].apply(pd.to_numeric, errors="coerce").max(axis=1)

    def first(row, suffix):
        vals = [row.get(f"{b}{suffix}") for b in blocks]
        return next((v for v in vals if pd.notna(v) and str(v).strip()), np.nan)

    def union_disease(row):
        items = []
        for b in blocks:
            items += resolve_other(parse_list(row.get(f"{b}Disease")), row.get(f"{b}DiseaseOther"))
        return json.dumps(list(dict.fromkeys(items))) if items else np.nan

    sub = v1.loc[old]
    v1.loc[old, "Disease"] = sub.apply(union_disease, axis=1)
    v1.loc[old, "Diagnosis"] = sub.apply(first, axis=1, suffix="Diagnosis")
    v1.loc[old, "LabTest"] = sub.apply(first, axis=1, suffix="LabTest")

    # Per-syndrome affected counts: old rows only have <Block>NoSick.
    for b in blocks:
        affected = "DiarrhoeaNoAffected" if b == "GIT" else f"{b}NoAffected"
        if f"{b}NoSick" in v1:
            v1[affected] = v1[affected].fillna(v1[f"{b}NoSick"].where(old))

    log.info("KABS 1.0: collapsed %d old-form rows into generic fields", old.sum())
    return v1


def harmonise_v1(path: Path) -> pd.DataFrame:
    log.info("Loading KABS 1.0: %s", path)
    v1 = pd.read_csv(path, dtype=str)
    log.info("KABS 1.0: %d rows, %d columns", *v1.shape)
    v1 = collapse_old_form(v1)

    out = pd.DataFrame(index=v1.index)
    out["source"] = "KABS 1.0"
    out["report_date"] = epoch_ms_to_date(v1["ReportDate"])
    out["outbreak_event_date"] = epoch_ms_to_date(v1["OutbreakStart"])
    out["submission_date"] = epoch_ms_to_date(v1["submissionDate"])
    for src, dst in [("County", "county"), ("SubCounty", "subcounty"), ("Ward", "ward"),
                     ("Locality", "locality"), ("ProdSystem", "production_system")]:
        out[dst] = v1[src]
    out["latitude"] = pd.to_numeric(v1["LocLat"], errors="coerce")
    out["longitude"] = pd.to_numeric(v1["LocLong"], errors="coerce")
    out["onsite"] = np.nan
    out["species"] = [join_list(resolve_other([s] if pd.notna(s) else [], o))
                      for s, o in zip(v1["Species"], v1["OtherSpecies"])]

    for old, new in SYNDROMES.items():
        flag = v1["Diarrhoea"].fillna(v1["GIT"]) if old == "GIT" else v1[old]
        out[new] = flag
        out[f"{new}_number"] = v1["DiarrhoeaNoAffected" if old == "GIT" else f"{old}NoAffected"]
    out["other_syndromes_list"] = v1["OtherSynText"]
    out["bite_type"] = np.nan
    out["gastrointestinal_bloody"] = v1["DiarrhoeaBleed"].fillna(v1["GITBleed"])
    out["death"] = v1["Death"]
    out["death_number"] = v1["DeathNoAffected"]

    out["disease_condition"] = [join_list(resolve_other(parse_list(d), o))
                                for d, o in zip(v1["Disease"], v1["DiseaseOther"])]
    out["diagnosis_nature"] = v1["Diagnosis"]
    out["laboratory_test_taken"] = v1["LabTest"]
    for src, dst in [("Risk", "number_at_risk"), ("NoSick", "number_sick"), ("NoDead", "number_dead"),
                     ("NoHumans", "humans_affected_number"), ("NoSlaughter", "number_slaughtered"),
                     ("NoDestroy", "number_destroyed"), ("NoVaccinate", "number_vaccinated")]:
        out[dst] = v1[src]
    out["vaccination_type"] = np.nan
    out["vaccine_batch_no"] = np.nan
    out["control_methods"] = [join_list(resolve_other(parse_list(c), o))
                              for c, o in zip(v1["ControlMeasure"], v1["ControlMeasureOther"])]
    out["comments"] = v1["Notes"]
    out["notes"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# KABS 2.0
# --------------------------------------------------------------------------- #
def text_to_date(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.replace(r"\s+(EAT|East Africa Time)$", "", regex=True).str.strip()
    return pd.to_datetime(s, format="%a, %b %d, %Y", errors="coerce").dt.date


def split_gps(s: pd.Series) -> pd.DataFrame:
    parts = s.astype("string").str.extract(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]")
    return parts.apply(pd.to_numeric, errors="coerce").set_axis(["latitude", "longitude"], axis=1)


def harmonise_v2(path: Path) -> pd.DataFrame:
    log.info("Loading KABS 2.0: %s", path)
    v2 = pd.read_csv(path, dtype=str)
    log.info("KABS 2.0: %d rows, %d columns", *v2.shape)

    # Rows with no report_date are 2024 test submissions whose answers landed
    # in numbered stray columns ('2', '5', ... '54') with junk values. Drop both.
    stray = [c for c in v2.columns if c.isdigit()]
    blank = v2["report_date"].isna()
    log.info("KABS 2.0: dropping %d test/blank rows and %d numbered stray columns", blank.sum(), len(stray))
    v2 = v2.loc[~blank].drop(columns=stray).reset_index(drop=True)

    out = pd.DataFrame(index=v2.index)
    out["source"] = "KABS 2.0"
    out["report_date"] = text_to_date(v2["report_date"])
    out["outbreak_event_date"] = text_to_date(v2["outbreak_event_date"])
    out["submission_date"] = np.nan
    for col in ["county", "subcounty", "ward", "locality", "production_system", "onsite"]:
        out[col] = v2[col]
    out[["latitude", "longitude"]] = split_gps(v2["gps_location"])
    out["species"] = [join_list(resolve_other([s] if pd.notna(s) else [], o))
                      for s, o in zip(v2["species"], v2["other_species"])]

    for new in SYNDROMES.values():
        out[new] = v2[new]
        out[f"{new}_number"] = v2[f"{new}_number"]
    # Some submissions used the older 'gastrointestinal_tract_infections' question instead.
    gi = "gastrointestinal_tract_syndromes"
    out[gi] = v2[gi].fillna(v2["gastrointestinal_tract_infections"])
    out[f"{gi}_number"] = v2[f"{gi}_number"].fillna(v2["gastrointestinal_tract_infections_number"])
    out["other_syndromes_list"] = v2["other_syndromes_list"]
    out["bite_type"] = v2["bite_type"]
    out["gastrointestinal_bloody"] = np.nan
    out["death"] = np.nan
    out["death_number"] = np.nan

    out["disease_condition"] = [join_list(resolve_other(parse_list(d), o))
                                for d, o in zip(v2["disease_condition"], v2["other_disease_condition"])]
    for col in ["diagnosis_nature", "laboratory_test_taken", "number_at_risk", "number_sick",
                "number_dead", "humans_affected_number", "number_slaughtered", "number_destroyed",
                "number_vaccinated", "vaccination_type", "vaccine_batch_no", "comments", "notes"]:
        out[col] = v2[col]
    out["control_methods"] = [join_list(resolve_other(parse_list(c), o))
                              for c, o in zip(v2["control_methods"], v2["other_control_methods"])]
    return out


# --------------------------------------------------------------------------- #
# combine
# --------------------------------------------------------------------------- #
def finalise(df: pd.DataFrame) -> pd.DataFrame:
    df = standardise_admin(df, load_county_lookup())
    df = validate_coords(df)

    for col in ["locality", "species", "production_system", "diagnosis_nature",
                "other_syndromes_list", "bite_type", "laboratory_test_taken",
                "vaccination_type", "vaccine_batch_no", "comments", "notes"]:
        df[col] = clean_text(df[col])
    df["diagnosis_nature"] = df["diagnosis_nature"].replace(DIAGNOSIS_MAP)
    df["production_system"] = df["production_system"].str.capitalize()

    yes_no = ["onsite", "death", "gastrointestinal_bloody", *SYNDROMES.values()]
    for col in yes_no:
        df[col] = clean_text(df[col]).str.capitalize()

    for col in COUNT_COLS:
        df[col] = to_count(df[col])
        neg = df[col] < 0
        if neg.any():
            log.warning("%s: nulling %d negative values", col, neg.sum())
            df.loc[neg, col] = pd.NA

    for col in ["report_date", "outbreak_event_date", "submission_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    late_start = df["outbreak_event_date"] > df["report_date"]
    if late_start.any():
        log.warning("%d rows have outbreak_event_date after report_date (left as-is)", late_start.sum())

    return df[OUTPUT_COLS].sort_values(["report_date", "source"], kind="stable").reset_index(drop=True)


def run_pipeline(v1_path: Path, v2_path: Path, out_path: Path) -> pd.DataFrame:
    df = pd.concat([harmonise_v1(v1_path), harmonise_v2(v2_path)], ignore_index=True)
    df = finalise(df)

    nulls = df.isna().mean().mul(100).round(1)
    log.info("Rows by source:\n%s", df["source"].value_counts().to_string())
    log.info("Columns >50%% empty:\n%s", nulls[nulls > 50].to_string())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, date_format="%Y-%m-%d")
    log.info("Wrote %d rows x %d columns -> %s", *df.shape, out_path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, default=ROOT / "data/kabs1.0_raw/nd1_28102026.csv")
    parser.add_argument("--v2", type=Path, default=ROOT / "data/kabs2.0_raw/nd1_29102026.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "Cleaned Data/nd1_final.csv")
    args = parser.parse_args()
    run_pipeline(args.v1, args.v2, args.out)
