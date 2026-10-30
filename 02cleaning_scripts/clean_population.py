"""
Reference / denominator tables: human population by county, livestock population
by county and by sub-county.

Aligns them with the cleaned surveillance data:
  - county / sub-county names = Kenya shapefile names (clean_admin_names.py), plus county_iso
  - species names as in ND1 / zero reports; the breed or production type in brackets
    ('Cattle(Beef)', 'Chicken (exoticBroilers)') moves to species_category
  - values are numbers; missing values stay missing (never filled with 0)

County totals: the sub-county table also contains one row per county/species named
after the county (e.g. sub-county 'Baringo' in Baringo) holding the county total.
Those rows are removed from the sub-county output, and the county livestock table is
rebuilt from them: the original Aggregated_Animal_Species.csv summed the totals with
the sub-county rows, so its values are double the true county figures.

Rows named like the county that are bigger than all other sub-counties combined but
are not an exact total (e.g. 'Garissa Township', 'Isiolo') are kept and flagged in
possible_county_total for checking against the source.

Sub-counties that are old district / veterinary names with no single shapefile
sub-county (admin_match = 'unmatched') keep their name and are listed in the log;
their numbers cannot be split onto shapefile sub-counties.

Run from the project root:
    python 02cleaning_scripts/clean_population.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from clean_admin_names import clean_admin_names, county_key
from harmonise import read_raw, split_species_category

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clean_population")

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "Cleaned Data"


def to_number(s: pd.Series) -> pd.Series:
    n = pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")
    bad = s.notna() & n.isna()
    if bad.any():
        log.warning("%d non-numeric values set to missing: %s", bad.sum(), s[bad].unique()[:5].tolist())
    return n


def human_population() -> pd.DataFrame:
    df = read_raw(DATA / "county_population.csv")
    df = clean_admin_names(df, subcounty_col=None, ward_col=None, lat_col=None, lon_col=None)
    df["population"] = to_number(df.pop("pop")).astype("Int64")
    return df[["county_iso", "county", "population"]]


def livestock_by_subcounty() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (sub-county rows without county totals, county totals)."""
    df = read_raw(DATA / "aggregated_PopulationRK.csv").rename(columns={"sub_county": "subcounty"})
    df = clean_admin_names(df, ward_col=None, lat_col=None, lon_col=None)
    df = df.join(split_species_category(df["Species"]))
    df["population"] = to_number(df["value"])

    # A row is a county total if it is named like the county (or 'Nairobi city') and
    # equals the sum of the other sub-counties for that county / species / category.
    named_like_county = df["subcounty_raw"].map(county_key).isin({"nairobicity"}) | (
        df["subcounty_raw"].map(county_key) == df["county"].map(county_key))
    group = [df["county"], df["species"], df["species_category"].fillna("")]
    others = df["population"].groupby(group).transform("sum") - df["population"]
    is_total = named_like_county & np.isclose(df["population"], others)
    log.info("Removed %d county-total rows from the sub-county table", is_total.sum())
    # Rows named like the county (or a county 'Township') that are larger than all other
    # sub-counties combined but are not an exact total: may be partial totals. Flagged, kept.
    named_like_total = named_like_county | df["subcounty_raw"].str.contains("township", case=False, na=False)
    df["possible_county_total"] = ~is_total & named_like_total & (df["population"] > others) & (others > 0)
    if df["possible_county_total"].any():
        log.warning("Possible partial county totals (kept, flagged in possible_county_total):\n%s",
                    df[df["possible_county_total"]].groupby(["county", "subcounty_raw"]).size().to_string())
    totals, df = df[is_total], df[~is_total]

    unmatched = df[df["admin_match"] != "subcounty_only"]
    if len(unmatched):
        log.warning("Sub-counties not in the shapefile (kept as entered; %d rows):\n%s", len(unmatched),
                    unmatched.groupby(["county", "subcounty_raw"]).size().to_string())
    cols = ["county_iso", "county", "subcounty", "subcounty_raw", "admin_match",
            "species", "species_category", "population", "possible_county_total"]
    return df[cols], totals


def livestock_by_county(subcounty: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """County totals from the sub-county source (total row where present, else sum of sub-counties)."""
    keys = ["county_iso", "county", "species", "species_category"]
    summed = subcounty.groupby(keys, dropna=False)["population"].sum(min_count=1)
    county = summed.to_frame().join(totals.set_index(keys)["population"].rename("total_row"))
    county["population"] = county["total_row"].fillna(county["population"])
    county = county.drop(columns="total_row").reset_index()

    # Compare with the original county table, which double counts.
    orig = read_raw(DATA / "Aggregated_Animal_Species.csv")
    orig = clean_admin_names(orig, subcounty_col=None, ward_col=None, lat_col=None, lon_col=None)
    orig = orig.join(split_species_category(orig["Species"]))
    ratio = (orig.assign(v=to_number(orig["value"])).groupby(["county", "species"])["v"].sum()
             / county.groupby(["county", "species"])["population"].sum()).round(2)
    log.info("Original Aggregated_Animal_Species / corrected county totals (county x species):\n%s",
             ratio.value_counts().head(5).to_string())
    return county


def main() -> None:
    OUT.mkdir(exist_ok=True)
    subcounty, totals = livestock_by_subcounty()
    for name, df in [("population_human_county_clean", human_population()),
                     ("population_livestock_county_clean", livestock_by_county(subcounty, totals)),
                     ("population_livestock_subcounty_clean", subcounty)]:
        df.to_csv(OUT / f"{name}.csv", index=False)
        log.info("Wrote %d rows -> %s", len(df), OUT / f"{name}.csv")


if __name__ == "__main__":
    main()
