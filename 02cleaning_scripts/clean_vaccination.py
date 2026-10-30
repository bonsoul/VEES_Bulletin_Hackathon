"""
Vaccination form (KABS 2.0 VF) cleaning.

  - county / sub-county / ward = Kenya shapefile names (clean_admin_names.py)
  - vaccination date parsed; GPS split and checked
  - species and disease names as in ND1 ('Other' replaced by other_disease);
    combination vaccines keep every disease, e.g. 'Canine Distemper; Leptospirosis; Parvovirus'
  - counts are numbers; blanks stay blank (never 0)
  - identical rows are flagged in possible_duplicate (the form has no submission
    timestamp, so they may be genuine repeat entries)

Run from the project root:
    python 02cleaning_scripts/clean_vaccination.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from clean_admin_names import clean_admin_names
from harmonise import (harmonise_disease_cell, kabs2_date, latest_export, parse_gps, read_raw, standardise_species,
                       strip_admin_suffixes, validate_coords)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clean_vaccination")

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_COLS = [
    "source", "vaccination_date", "county", "county_iso", "subcounty", "ward", "subcounty_raw", "ward_raw",
    "admin_match", "vaccination_site", "latitude", "longitude", "coords_in_county",
    "species", "disease", "disease_raw", "vaccination_type", "total_number_vaccinated",
    "number_of_beneficiaries", "batch_number", "possible_duplicate",
]


def run(src: Path, out: Path) -> pd.DataFrame:
    raw = read_raw(src)
    log.info("Loaded %d rows from %s", len(raw), src)
    df = strip_admin_suffixes(raw)
    df["source"] = "KABS 2.0"
    df["possible_duplicate"] = raw.duplicated(keep=False)
    log.info("%d rows are identical to another row (flagged, not dropped)", df["possible_duplicate"].sum())

    df["vaccination_date"] = kabs2_date(df["vaccination_date"])
    df = df.join(parse_gps(df.pop("gps_location")))
    df = validate_coords(df)
    df["species"] = standardise_species(df["species"])
    df["disease_raw"] = df["disease"]
    df["disease"] = [harmonise_disease_cell(d, o) for d, o in zip(df["disease"], df["other_disease"])]
    for col in ["total_number_vaccinated", "number_of_beneficiaries"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["vaccination_site"] = df["vaccination_site"].str.strip()

    df = clean_admin_names(df)
    df = df[OUTPUT_COLS].sort_values("vaccination_date", kind="stable")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, date_format="%Y-%m-%d")
    log.info("Diseases: %s", df["disease"].value_counts().head(12).to_dict())
    log.info("Wrote %d rows -> %s", len(df), out)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=latest_export(ROOT / "data/kabs2.0_raw", "vf"))
    parser.add_argument("--out", type=Path, default=ROOT / "Cleaned Data/vaccination_clean.csv")
    args = parser.parse_args()
    run(args.src, args.out)
