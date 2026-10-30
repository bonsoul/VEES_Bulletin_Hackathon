"""
Wildlife syndrome reports (KABS 1.0) cleaning.

The form has no county / sub-county / ward, only GPS and a protected-area name, so
county / sub-county / ward (Kenya shapefile names) are assigned from the GPS point
(admin_from_gps). Reports without valid GPS have no admin names.

  - dates (epoch ms) parsed; GPS checked (projected / junk coordinates nulled)
  - old form stores disease / diagnosis / counts per syndrome block; these are merged
    into one set (disease = union of all blocks, harmonised with the ND1 dictionary;
    counts = the generic field, else the largest block value, never summed)
  - syndrome flags Yes/No kept; <syndrome>_number from NoAffected, else NoSick
  - wildlife species tidied ('Other' -> SpeciesOther, Chimpanzees -> Chimpanzee, ...)
  - test submissions ('This is a test', keyboard-mash area names) flagged in
    possible_test_record, not removed
  - counts are numbers; blanks stay blank (never 0). WildNoRisk is a range ('11-50') and stays text

Run from the project root:
    python 02cleaning_scripts/clean_wildlife.py
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from clean_admin_names import admin_from_gps
from harmonise import (UNKNOWN, epoch_ms_to_date, harmonise_disease_cell, latest_export, parse_list_cell, read_raw,
                       validate_coords)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clean_wildlife")

ROOT = Path(__file__).resolve().parents[1]

# Form syndrome -> output name
SYNDROMES = {
    "Death": "death", "Growths": "growths", "Blindness": "blindness", "Cutaneous": "cutaneous",
    "Diarrhoea": "diarrhoea", "Vesicular": "vesicular", "Aggression": "aggression", "Emaciation": "emaciation",
    "Respiratory": "respiratory", "CNS": "nervous_signs", "Reproduction": "reproduction",
    "TraumaticInjury": "traumatic_injury", "OtherSyn": "other_syndromes",
}
SPECIES_FIXES = {
    "chimpanzees": "Chimpanzee", "chimpizee": "Chimpanzee", "chimpanzee": "Chimpanzee", "tommies": "Thomson gazelles",
    "impalla": "Impala", "zebra": "Equidae (Zebra)", "dikdik": "Dik-dik", "chimpanze": "Chimpanzee",
    "chimps": "Chimpanzee", "chips": "Chimpanzee", "baboon": "Baboons", "aardvak": "Aardvark",
    "blackand white colobus monkey": "Black and white colobus monkey",
}


def first_valid(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    cols = [c for c in cols if c in df]
    return df[cols].bfill(axis=1).iloc[:, 0] if cols else pd.Series(np.nan, index=df.index)


def max_of(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    cols = [c for c in cols if c in df]
    return df[cols].apply(pd.to_numeric, errors="coerce").max(axis=1) if cols else pd.Series(np.nan, index=df.index)


def tidy_species(species: pd.Series, other: pd.Series) -> pd.Series:
    s = species.where(species.str.lower() != "other", other)
    key = s.str.lower().str.strip()
    return key.map(SPECIES_FIXES).fillna(s.str.strip().str.replace(r"\s+", " ", regex=True))


def looks_like_test(notes: pd.Series, area: pd.Series) -> pd.Series:
    note_test = notes.str.contains(r"(?<!lab )\btests?\b|\btrial\b", case=False, na=False) & \
        ~notes.str.contains(r"lab tests?|samples? (?:were )?collected", case=False, na=False)
    a = area.fillna("").str.strip()
    mash = a.str.fullmatch(r"[A-Za-z]{2,5}") & ~a.str.contains(r"[aeiouAEIOU]") & ~a.str.isupper()
    return note_test | mash | a.str.lower().eq("test")


def run(src: Path, out: Path) -> pd.DataFrame:
    w = read_raw(src)
    log.info("Loaded %d rows from %s", len(w), src)
    blocks = [s for s in SYNDROMES if s != "TraumaticInjury"]

    out_df = pd.DataFrame(index=w.index)
    out_df["report_id"] = "W1-" + w["submissionDate"]
    for src_col, dst in [("ReportDate", "report_date"), ("OutbreakStart", "outbreak_start_date"),
                         ("OutbreakReported", "outbreak_reported_date"), ("submissionDate", "submission_date")]:
        out_df[dst] = epoch_ms_to_date(w[src_col])
    out_df["latitude"], out_df["longitude"] = w["LocLat"], w["LocLong"]
    out_df = validate_coords(out_df)
    out_df = out_df.join(admin_from_gps(out_df))
    out_df["region"] = w["Region"]
    out_df["area_name"] = w["AreaName"].str.strip()
    out_df["area_type"] = w["AreaType"].where(w["AreaType"] != "Other", w["AreaTypeOther"]).str.replace(
        "Protected Areas:", "Protected Area:", regex=False)
    out_df["season"] = w["Season"]
    out_df["species"] = tidy_species(w["Species"], w["SpeciesOther"])
    out_df["species_status"] = w["SpeciesStatus"].replace({"I don't know": UNKNOWN})
    out_df["animal_category"] = w["AnimalCategory"]
    out_df["number_at_risk_range"] = w["WildNoRisk"]

    for form, name in SYNDROMES.items():
        out_df[name] = w[form]
        num = first_valid(w, [f"{form}NoAffected", f"{form}NoSick"]) if form != "TraumaticInjury" else w["InjuryNoAffected"]
        out_df[f"{name}_number"] = pd.to_numeric(num, errors="coerce").astype("Int64")
    out_df["other_syndromes_text"] = w["OtherSynText"]
    out_df["bleeding_with_death"] = w["DeathBleed"]
    out_df["bloody_diarrhoea"] = w["DiarrhoeaBleed"]
    out_df["injury_type"] = w["Injury"]
    out_df["human_inflicted_injury"] = w["HumanInflicted"]

    # Disease: union of the generic field and every syndrome block.
    pairs = [("Disease", "DiseaseOther")] + [(f"{b}Disease", f"{b}DiseaseOther") for b in blocks]
    pairs = [(d, o) for d, o in pairs if d in w]

    def diseases(row) -> str | float:
        names = []
        for d, o in pairs:
            h = harmonise_disease_cell(row[d], row.get(o))
            if pd.notna(h):
                names += [n for n in h.split("; ") if n not in names]
        if len(names) > 1 and UNKNOWN in names:
            names.remove(UNKNOWN)
        return "; ".join(names) if names else np.nan

    out_df["disease"] = w.apply(diseases, axis=1)
    raw_dz = w[[c for p in pairs for c in p if c in w]].apply(
        lambda r: "; ".join(dict.fromkeys(x for v in r.dropna() for x in parse_list_cell(v))) or np.nan, axis=1)
    out_df["disease_raw"] = raw_dz
    out_df["diagnosis"] = first_valid(w, ["Diagnosis"] + [f"{b}Diagnosis" for b in blocks]).replace({"N/A": np.nan})
    out_df["lab_test"] = first_valid(w, ["LabTest"] + [f"{b}LabTest" for b in blocks])

    # Counts: generic field, else the largest syndrome-block value (same animals repeated per block).
    out_df["number_sick"] = pd.to_numeric(w["NoSick"], errors="coerce").fillna(
        max_of(w, [f"{b}NoSick" for b in blocks])).astype("Int64")
    out_df["number_dead"] = pd.to_numeric(w["NoDead"], errors="coerce").fillna(
        pd.to_numeric(w["DeathNoDead"], errors="coerce")).astype("Int64")
    out_df["humans_affected"] = pd.to_numeric(w["NoHumans"], errors="coerce").fillna(
        max_of(w, [f"{b}NoHumans" for b in blocks])).astype("Int64")
    out_df["number_destroyed"] = pd.to_numeric(w["NoDestroy"], errors="coerce").astype("Int64")
    out_df["number_vaccinated"] = pd.to_numeric(w["NoVaccinate"], errors="coerce").astype("Int64")

    controls = [list(dict.fromkeys((o.strip() if c.lower() == "other" and pd.notna(o) else c)
                                   for c in parse_list_cell(cm))) for cm, o in zip(w["ControlMeasure"], w["ControlMeasureOther"])]
    out_df["control_measures"] = ["; ".join(c) if c else np.nan for c in controls]
    out_df["post_mortem"] = w["PostMortem"]
    out_df["post_mortem_findings"] = w["PostMortemFindings"]
    out_df["reporter_cadre"] = w["Cadre"]
    out_df["notes"] = w["Notes"].str.replace(r"\s+", " ", regex=True).str.strip()
    out_df["possible_test_record"] = looks_like_test(w["Notes"], w["AreaName"])
    log.info("Possible test submissions flagged: %d", out_df["possible_test_record"].sum())

    out_df = out_df.sort_values("report_date", kind="stable")
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False, date_format="%Y-%m-%d")
    log.info("Species: %s", out_df["species"].value_counts().head(12).to_dict())
    log.info("Diseases: %s", out_df["disease"].str.split("; ").explode().value_counts().head(12).to_dict())
    log.info("Wrote %d rows x %d columns -> %s", *out_df.shape, out)
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=latest_export(ROOT / "data/kabs1.0_raw", "wildlife_syndrome"))
    parser.add_argument("--out", type=Path, default=ROOT / "Cleaned Data/wildlife_clean.csv")
    args = parser.parse_args()
    run(args.src, args.out)
