"""
Meat inspection (food safety) cleaning: merges
    KABS 1.0  data/food_safety_clean.rds            (reports / conditions / organs tables, read via Rscript)
    KABS 2.0  data/kabs2.0_raw/meat-inspection_*.csv (wide form)
    KABS 2.0  data/kiserian_meat_inspection_raw.json (Kiserian / Keekonyokie abattoirs; older field names)
into three tables linked by report_id:
    Cleaned Data/meat_inspection_reports_clean.csv     one row per inspection report
    Cleaned Data/meat_inspection_conditions_clean.csv  one row per report x condition asked
    Cleaned Data/meat_inspection_organs_clean.csv      one row per report x condemned organ

  - county / sub-county / ward = Kenya shapefile names (clean_admin_names.py); GPS checked
  - animal source counties = shapefile county names
  - species as in ND1; abattoir names grouped within a county ('Choice Meat' = 'Choice Meats')
  - KABS 2.0 'Female' = non-pregnant female (pregnant females are counted separately)
  - the KABS 2.0 generic partially_condemned / fully_condemned columns belong to multiple abscesses
  - counts are numbers; blanks stay blank (never 0); a condition row is only written where the
    question was answered, so 'not asked' is never turned into 'No'
  - KABS 2.0 identical rows flagged in possible_duplicate (no submission timestamp)

Needs R (Rscript) to read the .rds file.

Run from the project root:
    python 02cleaning_scripts/clean_meat_inspection.py
"""
from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from clean_admin_names import ADMIN_LOOKUP, clean_admin_names, county_key
from harmonise import kabs2_date, latest_export, parse_gps, parse_list_cell, read_raw, standardise_species, strip_admin_suffixes, \
    validate_coords

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clean_meat_inspection")
# Concatenating KABS 1.0 / 2.0 tables where one side has an all-blank column is intended here.
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with empty or all-NA")

ROOT = Path(__file__).resolve().parents[1]

# Older Kiserian JSON field names -> KABS 2.0 CSV names
JSON_RENAME = {
    "abbattior": "abbattoir", "source_of_animals": "animals_source",
    "number_approved_for_slaughter": "slaughtered_number_approved",
    "slaughter_number_approved": "slaughtered_number_approved",
    "number_of_whole_carcasses_condemned": "carcasses_condemned",
    "cases_of_partial_condemnation": "partial_condemnation_cases", "over_scalding": "overscalding",
    "other_conditions_seen": "other_conditions", "were_organs_condemned": "was_organ_condemned",
    "number_of_condemned_liver": "number_liver_condemned", "reason_for_liver_condemnation": "liver_condemndation_reason",
    "liver_condemnation_reason": "liver_condemndation_reason", "number_of_condemned_lungs": "number_lungs_condemned",
    "reason_for_lung_condemnation": "lung_condemnation_reason",
    "number_of_condemned_stomach_and_intestines": "number_condemned_stomach_intestines",
    "reason_for_stomach_and_intestine_condemnation": "stomach_intestines_condemnation_reason",
    "reason_for_partial_condemnation": "partial_condemnation_reason",
    "number_of_condemned_kidneys": "number_condemned_kidneys", "reason_for_kidney_condemnation": "kidney_condemnation_reason",
    "reason_for_carcass_condemnation": "carcass_condemnation_reason", "number_with_jaundice": "jaundice_no",
    "number_of_condemned_hearts": "number_condemned_hearts", "reason_for_heart_condemnation": "heart_condemnation_reason",
    "number_of_emaciation_cases": "emaciation_no",
    "number_of_emacation_cases_partially_condemned": "emaciation_partially_condemned",
    "emacation_partially_condemned": "emaciation_partially_condemned",
    "number_of_emacation_cases_fully_condemned": "emaciation_fully_condemned",
    "emacation_fully_condemned": "emaciation_fully_condemned", "inadequate_bleeding": "inadequeate_bleeding",
}

# Condition name -> KABS 2.0 columns (present, number affected, partially condemned, fully condemned)
K2_CONDITIONS = {
    "Multiple abscesses": ("multiple_abscesses", "multiple_abscesses_no", "partially_condemned", "fully_condemned"),
    "Emaciation": ("emaciation", "emaciation_no", "emaciation_partially_condemned", "emaciation_fully_condemned"),
    "Septicaemia": ("septicaemia", "septicaemia_no", "septicaemia_partially_condemned", "septicaemia_fully_condemned"),
    "Septic pleurisy": ("septic_pleurisy", "septic_pleurisy_no", "septic_pleurisy_partially_condemned",
                        "septic_pleurisy_fully_condemned"),
    "Septic peritonitis": ("septic_peritonitis", "septic_peritonitis_no", "septic_peritonitis_partially_condemned",
                           "septic_peritonitis_fully_condemned"),
    "Jaundice": ("jaundice", "jaundice_no", "jaundice_partially_condemned", "jaundice_fully_condemned"),
    "C. bovis": ("c_bovis", "c_bovis_no", "c_bovis_partially_condemned", "c_bovis_fully_condemned"),
    # poultry / pig conditions: yes/no only
    "Arthritis": ("arthritis", None, None, None), "Ascites": ("ascites", None, None, None),
    "Bruising": ("bruising", None, None, None), "Contamination": ("contamination", None, None, None),
    "Folliculitis": ("folliculitis", None, None, None),
    "Inadequate bleeding": ("inadequeate_bleeding", None, None, None),
    "Myotic disease": ("myotic_disease", None, None, None), "Overscalding": ("overscalding", None, None, None),
    "Peritonitis": ("peritonitis", None, None, None), "Pigmentation": ("pigmentation", None, None, None),
}
K1_CONDITION_NAMES = {"Multiple Abscesses": "Multiple abscesses", "Septic Pleurisy": "Septic pleurisy",
                      "Septic Peritonitis": "Septic peritonitis", "C. Bovis": "C. bovis"}

# Organ -> KABS 2.0 (count column, reason column, unit)
K2_ORGANS = {
    "Liver": ("number_liver_condemned", "liver_condemndation_reason", "count"),
    "Lungs": ("number_lungs_condemned", "lung_condemnation_reason", "count"),
    "Kidney": ("number_condemned_kidneys", "kidney_condemnation_reason", "count"),
    "Heart": ("number_condemned_hearts", "heart_condemnation_reason", "count"),
    "Head and tongue": ("number_condemned_heads", "head_tongue_condemnation_reason", "count"),
    "Stomach and intestines": ("number_condemned_stomach_intestines", "stomach_intestines_condemnation_reason", "count"),
    "Breast muscle": ("weight_condemned_breast_muscle", "breast_muscle_condemnation_reason", "weight"),
    "Gizzards": ("weight_condemned_gizzards", "gizzards_condemnation_reason", "weight"),
    "Skin / crop": (None, "skin_crop_condemnation_reason", "count"),
}
K1_ORGAN_NAMES = {"Head & Tongue": "Head and tongue", "Stomach & Intestines": "Stomach and intestines"}

REPORT_COLS = [
    "report_id", "source", "reporting_date", "abattoir", "abattoir_raw",
    "county", "county_iso", "subcounty", "ward", "subcounty_raw", "ward_raw", "admin_match",
    "latitude", "longitude", "coords_in_county", "species", "breed", "animals_source",
    "number_male", "number_female_non_pregnant", "number_female_pregnant", "number_broilers",
    "number_dead", "number_rejected", "number_isolated", "number_approved_for_slaughter",
    "carcasses_condemned", "carcass_condemnation_reason", "partial_condemnation", "partial_condemnation_reason",
    "other_conditions_seen", "other_conditions", "other_condition_animals", "other_condition_condemned",
    "organs_condemned", "organs_condemned_list", "possible_duplicate",
]
LINK_COLS = ["report_id", "source", "reporting_date", "abattoir", "county", "county_iso", "subcounty", "ward", "species"]


def to_int(s) -> pd.Series:
    """Whole-number counts as Int64; if a column has fractions, keep it as decimals and log them."""
    n = pd.to_numeric(s, errors="coerce")
    frac = n.notna() & (n % 1 != 0)
    if frac.any():
        log.warning("%s: %d non-whole counts kept as decimals: %s", getattr(s, "name", "count"), frac.sum(),
                    n[frac].unique()[:5].tolist())
        return n
    return n.astype("Int64")


def join_list(cell, other=None) -> str | float:
    items = [i.strip() for i in parse_list_cell(cell) if i.strip()]
    if other is not None and pd.notna(other) and str(other).strip():
        items = [str(other).strip() if i.lower() in ("other", "others") else i for i in items] or [str(other).strip()]
    items = [i for i in dict.fromkeys(items) if i not in ("0",)]
    return "; ".join(items) if items else np.nan


def yes_no(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower().map({"yes": "Yes", "no": "No"})


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def load_rds(path: Path) -> dict[str, pd.DataFrame]:
    if not shutil.which("Rscript"):
        raise RuntimeError("Rscript not found: install R to read the KABS 1.0 .rds file")
    with tempfile.TemporaryDirectory() as tmp:
        code = (f'x <- readRDS("{path}"); for (n in names(x)) '
                f'write.csv(x[[n]], file.path("{tmp}", paste0(n, ".csv")), row.names = FALSE, na = "")')
        subprocess.run(["Rscript", "-e", code], check=True, capture_output=True)
        return {p.stem: read_raw(p) for p in Path(tmp).glob("*.csv")}


def load_kabs2(csv_path: Path, json_path: Path) -> pd.DataFrame:
    csv = read_raw(csv_path)
    csv["possible_duplicate"] = csv.duplicated(keep=False)
    records = json.loads(json_path.read_text())
    js = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()}
                       for r in records]).astype("string")
    js = js.apply(lambda c: c.str.strip().mask(lambda x: x == ""))
    for old, new in JSON_RENAME.items():  # older field names: fill the current-name column
        if old in js:
            js[new] = js[new].fillna(js[old]) if new in js else js[old]
            js = js.drop(columns=old)
    js["possible_duplicate"] = js.duplicated(keep=False)
    log.info("KABS 2.0: %d CSV rows (%d identical to another), %d Kiserian JSON rows (%d identical)",
             len(csv), csv["possible_duplicate"].sum(), len(js), js["possible_duplicate"].sum())

    both = pd.concat([csv.assign(_file="csv"), js.assign(_file="json")], ignore_index=True)
    shared = [c for c in csv.columns if c in js.columns and c != "possible_duplicate"]
    overlap = both.duplicated(subset=shared, keep="first") & both["_file"].eq("json") & \
        both[shared].apply(tuple, axis=1).isin(set(csv[shared].apply(tuple, axis=1)))
    if overlap.any():
        log.info("Dropped %d Kiserian JSON rows already in the KABS 2.0 CSV", overlap.sum())
    both = both[~overlap].reset_index(drop=True)
    both["report_id"] = "MI2-" + pd.Series(range(1, len(both) + 1)).astype(str)
    return both


# --------------------------------------------------------------------------- #
# KABS 1.0 tables
# --------------------------------------------------------------------------- #
def kabs1_tables(t: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r = t["reports"]
    rep = pd.DataFrame({
        "report_id": r["id"], "source": "KABS 1.0",
        "reporting_date": pd.to_datetime(r["reporting_date"], errors="coerce"),
        "abattoir_raw": r["abattoir_name"], "county": r["county"], "subcounty": r["sub_county"], "ward": r["ward"],
        "latitude": r["latitude"], "longitude": r["longitude"], "species": r["species"], "breed": np.nan,
        "animals_source": r["source_of_the_animals"].map(join_list),
        "number_male": to_int(r["male"]), "number_female_non_pregnant": to_int(r["non_pregnant_female"]),
        "number_female_pregnant": to_int(r["pregnant_female"]), "number_broilers": pd.NA,
        "number_dead": to_int(r["number_dead"]), "number_rejected": to_int(r["number_rejected"]),
        "number_isolated": to_int(r["number_isolated"]),
        "number_approved_for_slaughter": to_int(r["number_approved_for_slaughter"]),
        "carcasses_condemned": to_int(r["number_of_whole_carcasses_condemned"]),
        "carcass_condemnation_reason": [join_list(a, b) for a, b in
                                        zip(r["reason_for_carcass_condemnation"], r["specify_reason_carcass_condemnation"])],
        "partial_condemnation": yes_no(r["were_there_any_cases_of_partial_condemnation"]),
        "partial_condemnation_reason": [join_list(a, b) for a, b in
                                        zip(r["reasons_for_partial_condemnation"], r["specify_reason_partial_condemnation"])],
        "other_conditions_seen": yes_no(r["were_there_other_conditions_seen"]),
        "other_conditions": r["what_are_the_other_observed_conditions"],
        "other_condition_animals": to_int(r["how_many_animal_had_the_other_condition_s"]),
        "other_condition_condemned": to_int(r["how_many_led_to_condemnation_s"]),
        "organs_condemned": yes_no(r["was_any_organs_condemned"]),
        "organs_condemned_list": r["which_organs_were_condemned"].map(join_list),
        "possible_duplicate": False,
    })
    c = t["conditions"]
    cond = pd.DataFrame({
        "report_id": c["id"], "condition": c["condition"].replace(K1_CONDITION_NAMES), "present": yes_no(c["present"]),
        "number_affected": to_int(c["n_affected"]), "partially_condemned": to_int(c["partially_condemned"]),
        "fully_condemned": to_int(c["fully_condemned"]),
    })
    o = t["organs"]
    org = pd.DataFrame({
        "report_id": o["id"], "organ": o["organ"].replace(K1_ORGAN_NAMES), "number_condemned": to_int(o["n_condemned"]),
        "weight_condemned": np.nan, "reason": o["reason"].map(join_list),
    })
    return rep, cond, org


# --------------------------------------------------------------------------- #
# KABS 2.0 tables
# --------------------------------------------------------------------------- #
def kabs2_tables(m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    m = strip_admin_suffixes(m)
    get = lambda c: m[c] if c in m else pd.Series(np.nan, index=m.index)
    gps = parse_gps(get("gps_location"))
    breed = get("breed").map(join_list)
    rep = pd.DataFrame({
        "report_id": m["report_id"], "source": "KABS 2.0", "reporting_date": kabs2_date(get("reporting_date")),
        "abattoir_raw": get("abbattoir"), "county": m["county"], "subcounty": m["subcounty"], "ward": m["ward"],
        "latitude": gps["latitude"], "longitude": gps["longitude"], "species": get("species"), "breed": breed,
        "animals_source": [join_list(a, b) for a, b in zip(get("animals_source"), get("other_animal_source"))],
        "number_male": to_int(get("no_of_males")), "number_female_non_pregnant": to_int(get("no_of_females")),
        "number_female_pregnant": to_int(get("no_of_pregnant_females")),
        "number_broilers": to_int(get("broilers_number").fillna(get("boilers_number"))),
        "number_dead": to_int(get("number_dead")), "number_rejected": to_int(get("number_rejected")),
        "number_isolated": to_int(get("number_isolated")),
        "number_approved_for_slaughter": to_int(get("slaughtered_number_approved")),
        "carcasses_condemned": to_int(get("carcasses_condemned")),
        "carcass_condemnation_reason": [join_list(a, b) for a, b in
                                        zip(get("carcass_condemnation_reason"), get("other_condemnation_reason"))],
        "partial_condemnation": yes_no(get("partial_condemnation_cases")),
        "partial_condemnation_reason": [join_list(a, b) for a, b in
                                        zip(get("partial_condemnation_reason"), get("other_reason_for_partial_condemnation"))],
        "other_conditions_seen": yes_no(get("other_conditions")),
        "other_conditions": get("observed_conditions").replace({"0": np.nan}),
        "other_condition_animals": to_int(get("animals_with_other_condition")),
        "other_condition_condemned": to_int(get("condemnation_led_no")),
        "organs_condemned": yes_no(get("was_organ_condemned")),
        "organs_condemned_list": [join_list(a, b) for a, b in zip(get("organs_condemned"), get("other_condemned_organ"))],
        "possible_duplicate": m["possible_duplicate"].fillna(False).astype(bool),
    })

    conds = []
    for name, (present, n, part, full) in K2_CONDITIONS.items():
        if present not in m:
            continue
        asked = m[present].notna()
        conds.append(pd.DataFrame({
            "report_id": m.loc[asked, "report_id"], "condition": name, "present": yes_no(m.loc[asked, present]),
            "number_affected": to_int(get(n)[asked]) if n else pd.NA,
            "partially_condemned": to_int(get(part)[asked]) if part else pd.NA,
            "fully_condemned": to_int(get(full)[asked]) if full else pd.NA,
        }))
    orgs = []
    for organ, (num, reason, unit) in K2_ORGANS.items():
        qty, why = (get(num) if num else pd.Series(np.nan, index=m.index)), get(reason)
        has = qty.notna() | why.notna()
        orgs.append(pd.DataFrame({
            "report_id": m.loc[has, "report_id"], "organ": organ,
            "number_condemned": to_int(qty[has]) if unit == "count" else pd.NA,
            "weight_condemned": pd.to_numeric(qty[has], errors="coerce") if unit == "weight" else np.nan,
            "reason": why[has].map(join_list),
        }))
    return rep, pd.concat(conds, ignore_index=True), pd.concat(orgs, ignore_index=True)


# --------------------------------------------------------------------------- #
# shared clean-up
# --------------------------------------------------------------------------- #
def group_abattoirs(rep: pd.DataFrame) -> pd.Series:
    """Same abattoir spelled differently within a county -> its most common spelling."""
    raw = rep["abattoir_raw"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    key = raw.str.lower().str.replace(r"\b(slaughter ?house|slaughter|abattoir|abbattoir|abbattior|slab|sh|ward|ltd|limited|s)\b", " ", regex=True) \
        .str.replace(r"[^a-z0-9]", "", regex=True)
    display = {}
    for county, grp in rep.assign(key=key, raw=raw).dropna(subset=["key"]).groupby("county"):
        reps: list[str] = []
        for k in grp["key"].value_counts().index:  # most frequent first
            match = next((r for r in reps if difflib.SequenceMatcher(None, k, r).ratio() >= 0.88), None)
            if match is None:
                reps.append(k)
                match = k
            display[(county, k)] = match
        names = grp.assign(rep=grp["key"].map(lambda k: display[(county, k)]))
        best = names.groupby("rep")["raw"].agg(lambda s: s.value_counts().index[0].title())
        for k in grp["key"].unique():
            display[(county, k)] = best[display[(county, k)]]
    out = pd.Series([display.get((c, k)) if pd.notna(k) else pd.NA for c, k in zip(rep["county"], key)], index=rep.index)
    log.info("Abattoir names: %d raw spellings -> %d abattoirs", raw.nunique(), out.nunique())
    return out


def clean_source_counties(s: pd.Series) -> pd.Series:
    names = pd.read_csv(ADMIN_LOOKUP).drop_duplicates("county")["county"]
    by_key = {county_key(n): n for n in names}
    def fix(cell):
        if pd.isna(cell):
            return np.nan
        parts = [by_key.get(county_key(re.sub(r"\s+County$", "", p.strip(), flags=re.I)), p.strip())
                 for p in str(cell).split(";")]
        return "; ".join(dict.fromkeys(p for p in parts if p))
    return s.map(fix)


def run(rds: Path, k2_csv: Path, k2_json: Path, out_dir: Path) -> None:
    rep1, cond1, org1 = kabs1_tables(load_rds(rds))
    log.info("KABS 1.0: %d reports", len(rep1))
    rep2, cond2, org2 = kabs2_tables(load_kabs2(k2_csv, k2_json))
    log.info("KABS 2.0: %d reports", len(rep2))

    rep = pd.concat([rep1, rep2], ignore_index=True)
    rep = validate_coords(rep)
    rep["species"] = standardise_species(rep["species"])
    rep = clean_admin_names(rep)
    rep["abattoir"] = group_abattoirs(rep)
    rep["animals_source"] = clean_source_counties(rep["animals_source"])
    rep = rep[REPORT_COLS].sort_values(["reporting_date", "source"], kind="stable")

    links = rep[LINK_COLS]
    cond = links.merge(pd.concat([cond1, cond2], ignore_index=True), on="report_id", how="inner")
    org = links.merge(pd.concat([org1, org2], ignore_index=True), on="report_id", how="inner")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in [("reports", rep), ("conditions", cond), ("organs", org)]:
        path = out_dir / f"meat_inspection_{name}_clean.csv"
        df.to_csv(path, index=False, date_format="%Y-%m-%d")
        log.info("Wrote %d rows -> %s", len(df), path)
    log.info("Reports by source:\n%s", rep["source"].value_counts().to_string())
    log.info("Conditions present (Yes) by source:\n%s",
             pd.crosstab(cond["condition"], cond["source"], values=cond["present"].eq("Yes"), aggfunc="sum")
             .fillna(0).astype(int).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rds", type=Path, default=ROOT / "data/food_safety_clean.rds")
    parser.add_argument("--k2-csv", type=Path, default=latest_export(ROOT / "data/kabs2.0_raw", "meat-inspection"))
    parser.add_argument("--k2-json", type=Path, default=ROOT / "data/kiserian_meat_inspection_raw.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "Cleaned Data")
    args = parser.parse_args()
    run(args.rds, args.k2_csv, args.k2_json, args.out_dir)
