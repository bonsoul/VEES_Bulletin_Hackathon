"""
KABS ND1 (disease event) harmonisation: one complete, de-duplicated event dataset.

Sources, mapped onto one schema (source / record_id say where each row came from):
  KABS 1.0                data/kabs1.0_raw/nd1_*.csv          2007-2026, CamelCase, epoch-ms dates
  KABS 2.0                data/kabs2.0_raw/nd1_*.csv          2025-2026, snake_case, text dates
  KABS reports export     Raw_Datasets/combined_reports.xlsx  2020-Aug 2026, ND1 form v15-v27, with Id
  KABS live export        data/live_kabs_export.csv           2019-2026, with id, no syndromes
  KABS events export      data/kabs_events_admin_corrected.csv one row per sign, with id
  KABS 2020-2026 export   data/kabs_2020_2026.xlsx            no id, no syndromes

No single export is complete, so the exports are merged without double counting:
  1. KABS 1.0 exact duplicate rows are dropped (identical to the millisecond submission
     timestamp = export repeats). KABS 1.0 + KABS 2.0 form the base.
  2. Each further export only adds events the base does not already have. Records with
     an id already taken from an earlier export are skipped. Otherwise, per report date x
     county x ward x species, the export adds only its surplus over what is already there
     (rows that also match on at-risk / sick / dead are treated as the same event first).
     The same event recorded with different detail (e.g. 'Other' vs 'Milk Fever') is
     therefore not counted twice.
  Fields an export does not have (e.g. syndromes in the live export) stay blank, never 'No'.

Run from the project root:
    python 02cleaning_scripts/00cleaning.py
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from harmonise import latest_export

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
    "source", "record_id", "report_date", "outbreak_event_date", "submission_date",
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
    before = len(v1)
    v1 = v1.drop_duplicates().reset_index(drop=True)
    log.info("KABS 1.0: dropped %d exact duplicate rows (export repeats)", before - len(v1))
    v1 = collapse_old_form(v1)

    out = pd.DataFrame(index=v1.index)
    out["source"] = "KABS 1.0"
    out["record_id"] = "K1-" + v1["submissionDate"]
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
    out["record_id"] = np.nan
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
# other exports
# --------------------------------------------------------------------------- #
# combined_reports: syndrome -> its 'Number Affected' column
COMBINED_SYNDROMES = {
    "abortion": ("Abortion", "Number Affected"),
    "sudden_death": ("Sudden Death", "Number Affected.1"),
    "death": ("Death", "Number Affected.2"),
    "haemorrhagic": ("Hemorrhagic Signs", "Number Affected.3"),
    "neurologic": ("Neurologic signs", "Number Affected.4"),
    "animal_bites": ("Animal Bites", "Number Affected.5"),
    "respiratory_signs": ("Respiratory Signs", "Number Affected.6"),
    "oral_foot_lesions": ("Oral/Foot Lesions", "Number Affected.7"),
    "cutaneous_skin_lesions": ("Cutaneous/Skin Lesions", "Number Affected.8"),
    "gastrointestinal_tract_syndromes": ("Gastrointestinal tract syndromes", "Number Affected.9"),
    "other_syndromes": ("Other Syndromes", "Number Affected.10"),
}
# events export: sign code -> syndrome
EVENT_SIGNS = {
    "abortion": "abortion", "sudden_death": "sudden_death", "hemorrhagic": "haemorrhagic",
    "neurologic": "neurologic", "animal_bites": "animal_bites", "respiratory": "respiratory_signs",
    "oral_foot": "oral_foot_lesions", "cutaneous": "cutaneous_skin_lesions",
    "gi": "gastrointestinal_tract_syndromes", "other_syndrome": "other_syndromes",
}


def empty_frame(index) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=index, columns=OUTPUT_COLS, dtype=object)


def iso_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", format="%Y-%m-%d").dt.date


def us_or_iso_date(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, format="%m/%d/%y", errors="coerce")
    return d.fillna(pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")).dt.date


def comma_list(value, other=None) -> str | float:
    items = [] if pd.isna(value) else [x.strip() for x in str(value).strip("[]").split(",") if x.strip()]
    return join_list(resolve_other(items, other))


def harmonise_combined(path: Path) -> pd.DataFrame:
    c = pd.read_excel(path, dtype=str)
    log.info("KABS reports export: %d rows (%s)", len(c), path.name)
    out = empty_frame(c.index)
    out["source"], out["record_id"] = "KABS reports export", c["Id"]
    out["report_date"] = iso_date(c["Date of Report"])
    out["outbreak_event_date"] = iso_date(c["Date of Start of Outbreak/Event"])
    for src, dst in [("County", "county"), ("Sub-County", "subcounty"), ("Ward", "ward"), ("Locality", "locality"),
                     ("Production System", "production_system"), ("Nature of Diagnosis", "diagnosis_nature"),
                     ("Test Used", "laboratory_test_taken"), ("Number at Risk", "number_at_risk"),
                     ("Number Dead", "number_dead"), ("Number of Humans Affected (If zoonosis)", "humans_affected_number"),
                     ("Number Slaughtered", "number_slaughtered"), ("Number Destroyed", "number_destroyed"),
                     ("Number Vaccinated", "number_vaccinated")]:
        out[dst] = c[src]
    out["latitude"] = pd.to_numeric(c["Latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(c["Longitude"], errors="coerce")
    out["species"] = [join_list(resolve_other([s] if pd.notna(s) else [], o))
                      for s, o in zip(c["Species Affected"], c["Other Species"])]
    out["number_sick"] = c["Number Sick / Bitten"].fillna(c["Number Sick"])
    for dst, (flag, num) in COMBINED_SYNDROMES.items():
        out[dst], out[f"{dst}_number"] = c[flag], c[num]
    out["disease_condition"] = [join_list(resolve_other([d] if pd.notna(d) else [], o))
                                for d, o in zip(c["Disease/Condition"], c["Other Disease"])]
    out["control_methods"] = [comma_list(m, o) for m, o in zip(c["Disease Control Method"], c["Other Control Method"])]
    return out


def harmonise_live(path: Path) -> pd.DataFrame:
    l = pd.read_csv(path, dtype=str)
    log.info("KABS live export: %d rows (%s)", len(l), path.name)
    out = empty_frame(l.index)
    out["source"], out["record_id"] = "KABS live export", l["id"]
    out["report_date"] = iso_date(l["Date of Report"])
    for src, dst in [("County", "county"), ("Sub-County", "subcounty"), ("Ward", "ward"),
                     ("Reporting Site", "locality"), ("production_system", "production_system"),
                     ("Species Affected", "species"), ("Disease/Condition", "disease_condition"),
                     ("diagnosis_basis", "diagnosis_nature"), ("Number at Risk", "number_at_risk"),
                     ("Number Sick", "number_sick"), ("Number Dead", "number_dead"),
                     ("Number Vaccinated", "number_vaccinated")]:
        out[dst] = l[src]
    out["latitude"] = pd.to_numeric(l["Latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(l["Longitude"], errors="coerce")
    out["control_methods"] = l["control_method"].map(comma_list)
    return out


def harmonise_events(path: Path) -> pd.DataFrame:
    e = pd.read_csv(path, dtype=str)
    log.info("KABS events export: %d rows / %d events (%s)", len(e), e["id"].nunique(), path.name)
    first = e.drop_duplicates("id").set_index("id")
    signs = e.dropna(subset=["sign"]).drop_duplicates(["id", "sign"])
    out = empty_frame(first.index)
    out["source"], out["record_id"] = "KABS events export", first.index
    out["report_date"] = us_or_iso_date(first["date"])
    out["outbreak_event_date"] = us_or_iso_date(first["date_start"])
    for src, dst in [("county", "county"), ("subcounty", "subcounty"), ("ward", "ward"), ("locality", "locality"),
                     ("production_system", "production_system"), ("diagnosis_basis", "diagnosis_nature"),
                     ("test_used", "laboratory_test_taken"), ("at_risk", "number_at_risk"), ("sick", "number_sick"),
                     ("dead", "number_dead"), ("humans_affected", "humans_affected_number"),
                     ("n_slaughtered", "number_slaughtered"), ("n_destroyed", "number_destroyed"),
                     ("n_vaccinated", "number_vaccinated"), ("field_notes", "comments")]:
        out[dst] = first[src]
    out["latitude"] = pd.to_numeric(first["lat_clean"], errors="coerce")
    out["longitude"] = pd.to_numeric(first["lon_clean"], errors="coerce")
    out["species"] = [join_list(resolve_other([s] if pd.notna(s) else [], o))
                      for s, o in zip(first["species"], first["other_species"])]
    out["disease_condition"] = [join_list(resolve_other([d] if pd.notna(d) else [], o if o != "0" else np.nan))
                                for d, o in zip(first["disease"], first["other_disease"])]
    out["control_methods"] = [comma_list(m, o) for m, o in zip(first["control_method"], first["other_control"])]
    out["other_syndromes_list"] = first["other_syndrome_detail"].replace({"0": np.nan})
    # Only the signs present are listed; signs not listed stay blank (not asked vs 'No' is unknown).
    for code, dst in EVENT_SIGNS.items():
        hit = signs[signs["sign"] == code].set_index("id")["n_affected"]
        out.loc[hit.index, dst] = "Yes"
        out.loc[hit.index, f"{dst}_number"] = hit
    return out.reset_index(drop=True)


def harmonise_xlsx(path: Path) -> pd.DataFrame:
    x = pd.read_excel(path, dtype=str)
    log.info("KABS 2020-2026 export: %d rows (%s)", len(x), path.name)
    out = empty_frame(x.index)
    out["source"] = "KABS 2020-2026 export"
    out["report_date"] = iso_date(x["Report_Date"].str[:10])
    out["outbreak_event_date"] = iso_date(x["Start_Date"].str[:10])
    for src, dst in [("county", "county"), ("sub_county", "subcounty"), ("Ward", "ward"), ("Locality", "locality"),
                     ("Species_Affected", "species"), ("Production_System", "production_system"),
                     ("Disease_Condition", "disease_condition"), ("Number_at_Risk", "number_at_risk"),
                     ("Number_Sick", "number_sick"), ("Number_Dead", "number_dead"),
                     ("Humans_Affected", "humans_affected_number"), ("Number_Slaughtered", "number_slaughtered"),
                     ("Number_Destroyed", "number_destroyed"), ("Number_Vaccinated", "number_vaccinated")]:
        out[dst] = x[src]
    out["latitude"] = pd.to_numeric(x["Latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(x["Longitude"], errors="coerce")
    return out


# --------------------------------------------------------------------------- #
# merge without double counting
# --------------------------------------------------------------------------- #
def _norm(s: pd.Series, drop_words: str = "") -> pd.Series:
    s = s.astype("string").str.lower()
    if drop_words:
        s = s.str.replace(drop_words, "", regex=True)
    return s.str.replace(r"[^a-z]", "", regex=True).replace({"tharakanithi": "tharaka"}).fillna("")


def event_keys(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(group key: date x county x ward x species, detail key: group + at risk / sick / dead)."""
    date = pd.to_datetime(df["report_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    species = _norm(df["species"]).str.rstrip("s")
    group = date + "|" + _norm(df["county"], r"county") + "|" + _norm(df["ward"], r"\bward\b") + "|" + species
    num = lambda c: pd.to_numeric(df[c], errors="coerce").round().astype("Int64").astype("string").fillna("")
    detail = group + "|" + num("number_at_risk") + "|" + num("number_sick") + "|" + num("number_dead")
    return group, detail


def add_missing(master: pd.DataFrame, new: pd.DataFrame, seen_ids: set) -> pd.DataFrame:
    """Rows of `new` that are not already in `master` (see module docstring)."""
    name = new["source"].iloc[0]
    known = new["record_id"].isin(seen_ids)
    new = new[~known].copy()

    m_group, m_detail = event_keys(master)
    n_group, n_detail = event_keys(new)
    # Exact detail matches are the same event: consume them first.
    avail = m_detail.value_counts().to_dict()
    exact = []
    for d in n_detail:
        exact.append(avail.get(d, 0) > 0)
        if exact[-1]:
            avail[d] -= 1
    new["_exact"], new["_group"] = exact, n_group.to_numpy()
    surplus = (n_group.value_counts() - m_group.value_counts().reindex(n_group.value_counts().index).fillna(0)).clip(lower=0)
    new = new.sort_values("_exact", kind="stable")  # unmatched rows first within each group
    new["_rank"] = new.groupby("_group").cumcount()
    keep = new["_rank"] < new["_group"].map(surplus)
    log.info("%s: %d rows; %d skipped (id already used), %d already in the data, %d added",
             name, len(new) + known.sum(), known.sum(), (~keep).sum(), keep.sum())
    return new[keep].drop(columns=["_exact", "_group", "_rank"])


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


def run_pipeline(v1_path: Path, v2_path: Path, extra: list[tuple], out_path: Path) -> pd.DataFrame:
    df = pd.concat([harmonise_v1(v1_path), harmonise_v2(v2_path)], ignore_index=True)
    seen_ids: set = set()
    for loader, path in extra:
        if not path.exists():
            log.warning("Skipping missing export %s", path)
            continue
        new = loader(path)
        added = add_missing(df, new, seen_ids)
        seen_ids |= set(new["record_id"].dropna())
        df = pd.concat([df, added], ignore_index=True)
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
    parser.add_argument("--v1", type=Path, default=latest_export(ROOT / "data/kabs1.0_raw", "nd1"))
    parser.add_argument("--v2", type=Path, default=latest_export(ROOT / "data/kabs2.0_raw", "nd1"))
    parser.add_argument("--combined", type=Path, default=ROOT / "Raw_Datasets/combined_reports.xlsx")
    parser.add_argument("--live", type=Path, default=ROOT / "data/live_kabs_export.csv")
    parser.add_argument("--events", type=Path, default=ROOT / "data/kabs_events_admin_corrected.csv")
    parser.add_argument("--xlsx", type=Path, default=ROOT / "data/kabs_2020_2026.xlsx")
    parser.add_argument("--out", type=Path, default=ROOT / "Cleaned Data/nd1_final.csv")
    args = parser.parse_args()
    extra = [(harmonise_combined, args.combined), (harmonise_live, args.live),
             (harmonise_events, args.events), (harmonise_xlsx, args.xlsx)]
    run_pipeline(args.v1, args.v2, extra, args.out)
