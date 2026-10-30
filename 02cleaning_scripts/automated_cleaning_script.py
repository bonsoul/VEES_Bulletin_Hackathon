"""
ND1 admin + disease harmonisation.
Reads Cleaned Data/nd1_final.csv and:
  1. Cleans county / sub-county / ward names to the Kenya shapefile names and
     flags GPS points outside their county (see clean_admin_names.py).
  2. Harmonises disease names (one spelling/category per disease) and the
     free-text syndrome list.

Run from the project root:
    python 02cleaning_scripts/automated_cleaning_script.py

Outputs:
    Cleaned Data/nd1_final_clean.csv            cleaned dataset
    Cleaned Data/admin_lookup.csv               county -> sub-county -> ward (shapefile names)
    Cleaned Data/disease_name_mapping.csv       raw disease text -> harmonised name (for review)
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from clean_admin_names import clean_admin_names, write_admin_lookup
from harmonise import PLACEHOLDERS, SIGNS, UNKNOWN, harmonise_disease_item, match_all, norm, standardise_species

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nd1_harmonise")

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# disease and syndrome names (rules live in harmonise.py)
# --------------------------------------------------------------------------- #
def harmonise_diseases(df: pd.DataFrame, mapping_out: Path) -> pd.DataFrame:
    df = df.copy()
    df["disease_condition_raw"] = df["disease_condition"]
    items = df["disease_condition"].dropna().str.split("; ").explode()
    lookup = {raw: harmonise_disease_item(raw) for raw in items.unique()}

    def combine(cell):
        if pd.isna(cell):
            return UNKNOWN
        names = list(dict.fromkeys(n for raw in cell.split("; ") for n in lookup[raw][0]))
        if len(names) > 1 and UNKNOWN in names:
            names.remove(UNKNOWN)
        return "; ".join(names)

    df["disease_condition"] = df["disease_condition"].map(combine)

    counts = items.value_counts()
    mapping = pd.DataFrame(
        [(raw, "; ".join(lookup[raw][0]), lookup[raw][1], counts[raw]) for raw in counts.index],
        columns=["raw_value", "harmonised", "method", "rows"],
    )
    mapping.to_csv(mapping_out, index=False)

    n_before, n_after = items.nunique(), df["disease_condition"].str.split("; ").explode().nunique()
    unmatched = mapping[mapping["method"] == "unmatched"]
    log.info("Disease names: %d distinct raw values -> %d harmonised", n_before, n_after)
    log.info("Unmatched disease values: %d distinct (%d rows); review in %s",
             len(unmatched), unmatched["rows"].sum(), mapping_out.name)
    return df


def harmonise_syndrome_list(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col = "other_syndromes_list"
    df[col + "_raw"] = df[col]

    def clean(cell):
        if pd.isna(cell):
            return np.nan
        names = []
        for part in re.split(r"[,;/\n]|\band\b|\bwith\b|\d+[.)]", str(cell).lower()):
            t = norm(part)
            if len(re.sub(r"[^a-z]", "", t)) < 3 or t in PLACEHOLDERS:
                continue
            found = match_all(t, SIGNS) or [t.capitalize()]
            names += [n for n in found if n not in names]
        return "; ".join(names) if names else np.nan

    uniq = {v: clean(v) for v in df[col].dropna().unique()}
    df[col] = df[col].map(uniq)
    signs = df[col].dropna().str.split("; ").explode()
    known = {n for n, _ in SIGNS}
    log.info("Syndrome list: %d distinct raw values -> %d signs (%d standard, %d free-text leftovers)",
             len(uniq), signs.nunique(), signs[signs.isin(known)].nunique(), signs[~signs.isin(known)].nunique())
    return df


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(src: Path, out: Path) -> pd.DataFrame:
    log.info("Loading %s", src)
    df = pd.read_csv(src, low_memory=False)
    log.info("%d rows, %d columns", *df.shape)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = clean_admin_names(df)
    df["species"] = standardise_species(df["species"])
    write_admin_lookup(out.parent / "admin_lookup.csv")

    df = harmonise_diseases(df, out.parent / "disease_name_mapping.csv")
    df = harmonise_syndrome_list(df)

    # Keep the original column order, with new columns next to the ones they describe.
    cols = list(pd.read_csv(src, nrows=0).columns)
    for new, after in [("county_iso", "county"), ("subcounty_raw", "ward"), ("ward_raw", "subcounty_raw"),
                       ("admin_match", "ward_raw"), ("coords_in_county", "longitude"),
                       ("disease_condition_raw", "disease_condition"),
                       ("other_syndromes_list_raw", "other_syndromes_list")]:
        cols.insert(cols.index(after) + 1, new)
    df = df[cols]

    df.to_csv(out, index=False)
    log.info("Wrote %d rows x %d columns -> %s", *df.shape, out)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=ROOT / "Cleaned Data/nd1_final.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "Cleaned Data/nd1_final_clean.csv")
    args = parser.parse_args()
    run_pipeline(args.src, args.out)
