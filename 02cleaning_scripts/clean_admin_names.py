"""
Clean county, sub-county and ward names.

Renames county / sub-county / ward to exactly the names in the Kenya boundary
files (data/kenya_counties.geojson, kenya_subcounties.geojson, kenya_wards.geojson;
geoBoundaries ADM1/2/3), so any dataset joins to the maps on name and each county
only lists its own sub-counties and wards. The lookup comes from
data/reference/kenya_shapefile_admin_lookup.csv (built by build_admin_lookup.py).

How a row is resolved, in order:
  1. ward alias (known misspellings, see WARD_ALIASES)
  2. exact ward name within the county (ignoring case, spacing, punctuation, word order)
  3. fuzzy ward name within the county (>= 0.85, never swapping East/West etc.)
  4. ward + sub-county found together in exactly one other county -> county corrected
  5. ward missing or not in the shapefile -> only the sub-county is corrected
     (exact / fuzzy, or SUBCOUNTY_ALIASES for renamed districts, e.g. Mbeere North -> Siakago)
  6. no sub-county or ward in the data -> county only (admin_match = 'county_only')
The sub-county always comes from the matched ward, so old veterinary sub-county
names (e.g. Merti, Kuresoi) are replaced by the shapefile sub-county.

Adds: county_iso, <subcounty>_raw, <ward>_raw, admin_match and, if the data has
GPS columns, coords_in_county (point inside the reported county polygon).

Run from the project root on any CSV:
    python 02cleaning_scripts/clean_admin_names.py --src "Cleaned Data/nd1_final.csv" --out out.csv
    python 02cleaning_scripts/clean_admin_names.py --src data/x.csv --out y.csv \
        --county-col County --subcounty-col "Sub-County" --ward-col Ward

Or from Python:
    from clean_admin_names import clean_admin_names
    df = clean_admin_names(df, county_col="County", subcounty_col="Sub-County", ward_col="Ward")
"""
from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clean_admin_names")

ROOT = Path(__file__).resolve().parents[1]
ADMIN_LOOKUP = ROOT / "data/reference/kenya_shapefile_admin_lookup.csv"
COUNTY_GEOJSON = ROOT / "data/kenya_counties.geojson"

FUZZY_CUTOFF = 0.85
DIRECTIONS = {"north", "south", "east", "west", "central", "upper", "lower"}

# Data county keys that differ from the shapefile's (shapefile calls Tharaka-Nithi 'Tharaka').
COUNTY_KEY_ALIASES = {"tharakanithi": "tharaka"}

# Ward spellings too far from the shapefile name for fuzzy matching.
# (county key, ward key) -> shapefile ward name
WARD_ALIASES = {
    ("kwale", "waangombeni"): "WAA",                  # Ng'ombeni is a location in Waa ward
    ("mandera", "lebahiya"): "LIBEHIA",
    ("busia", "nambale"): "NAMBALE TOWNSHIP",
    ("marsabit", "logologo"): "LOG LOGO",
    ("homabay", "rumakaksingri"): "RUMA KAKSINGRI EAST",
    ("isiolo", "cherub"): "CHERAB",
    ("nyandarua", "kanjuirirange"): "KANJUIRI RIDGE",
    ("taitataveta", "bomani"): "BOMENI",
    # No Wumingu/Kishushe polygon in the ward shapefile; 47 of 58 geotagged reports fall in Werugha.
    ("taitataveta", "wumingukishushe"): "Werugha",
    ("nairobi", "matopenispringvalley"): "Matopeni",
}

# Old district / veterinary sub-county names that are a straight rename of one shapefile
# sub-county. Only used when the ward cannot be matched. (county key, sub-county key) -> shapefile name
SUBCOUNTY_ALIASES = {
    ("embu", "mbeerenorth"): "Siakago",
    ("embu", "mbeeresouth"): "Gachoka",
    ("garissa", "garissatownship"): "Dujis",
    ("garissa", "daadab"): "Dadaab",
    ("meru", "centralimenti"): "Cental Imenti",       # spelled this way in the shapefile
    ("kitui", "mwingicentral"): "Mwingi East",
    ("tharaka", "chukaigambangombe"): "Nithi",
    ("nyeri", "mukurweini"): "Mukurweni",
    ("homabay", "homabaytown"): "Homa Bay",
    ("homabay", "suba"): "Suba South",
    ("homabay", "mbita"): "Suba North",
    ("kirinyaga", "kirinyagaeast"): "Gichugu",
    ("kirinyaga", "kirinyagawest"): "Ndia",
    ("kirinyaga", "mweaeast"): "Mwea",
    ("kirinyaga", "mweawest"): "Mwea",
    ("bungoma", "cheptais"): "Mt. Elgon",
}


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def tokens(s) -> list[str]:
    if pd.isna(s):
        return []
    toks = re.findall(r"[a-z0-9]+", str(s).lower().replace("&", " and "))
    return [t for t in toks if t not in {"ward", "county", "sub", "subcounty"}]


def key(s) -> str:
    return "".join(tokens(s))


def sorted_key(s) -> str:
    return "".join(sorted(tokens(s)))


def nice(s):
    """'SANG'ALO WEST' -> "Sang'alo West", 'MAAI-MAHIU' -> 'Maai-Mahiu'."""
    if pd.isna(s):
        return s
    s = re.sub(r"\s+", " ", str(s)).strip()
    return re.sub(r"[A-Za-z0-9]+(?:'[A-Za-z]+)*", lambda m: m[0].capitalize(), s)


def same_direction(a: str, b: str) -> bool:
    """Block fuzzy matches like 'West Sang'alo' -> 'East Sang'alo'."""
    return set(tokens(a)) & DIRECTIONS == set(tokens(b)) & DIRECTIONS


def county_key(s) -> str:
    k = key(s)
    return COUNTY_KEY_ALIASES.get(k, k)


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
def load_reference(path: Path = ADMIN_LOOKUP) -> pd.DataFrame:
    ref = pd.read_csv(path).drop_duplicates(["county", "subcounty", "ward"])
    ref["ck"] = ref["county"].map(county_key)
    # The shapefile disambiguates repeated names, e.g. 'Township (Thika)'; match on the bare name too.
    ref["wbk"] = ref["ward"].str.replace(r"\s*\(.*?\)", "", regex=True).map(key)
    ref["sk"], ref["wk"], ref["wsk"] = ref["subcounty"].map(key), ref["ward"].map(key), ref["ward"].map(sorted_key)
    return ref


def fuzzy_pick(name: str, options: pd.DataFrame, col: str) -> pd.DataFrame:
    """Best fuzzy match on exact and token-sorted keys, guarded by direction words."""
    best, best_score = None, FUZZY_CUTOFF
    for k_fn, k_col in [(key, col), (sorted_key, col + "_sorted")]:
        target = k_fn(name)
        for idx, cand in options[k_col].items():
            score = difflib.SequenceMatcher(None, target, cand).ratio()
            if score >= best_score and same_direction(name, options.at[idx, "label"]):
                best, best_score = idx, score
    return options.loc[[best]] if best is not None else options.iloc[0:0]


def resolve_admin(county, subcounty, ward, ref: pd.DataFrame) -> dict:
    ck, sk, wk = county_key(county), key(subcounty), key(ward)
    in_county = ref[ref["ck"] == ck]
    shape_county = in_county.iloc[0]["county"] if not in_county.empty else nice(county)
    out = {"county": shape_county, "subcounty": nice(subcounty), "ward": nice(ward),
           "admin_match": "unmatched" if not in_county.empty else "county_unmatched"}

    def pick(cands: pd.DataFrame, method: str) -> dict | None:
        if cands.empty:
            return None
        # Duplicate ward names within a county (e.g. Biashara in Nakuru): use the stated sub-county.
        if len(cands) > 1:
            narrowed = cands[cands["sk"] == sk]
            if narrowed.empty:  # e.g. stated 'Nakuru Town' -> 'Nakuru Town East' rather than 'Naivasha'
                score = cands["sk"].map(lambda c: difflib.SequenceMatcher(None, sk, c).ratio())
                narrowed = cands[(score == score.max()) & (score >= 0.6)]
            if len(narrowed) != 1:
                return {**out, "ward": cands.iloc[0]["ward"], "admin_match": "ward_ambiguous"}
            cands = narrowed
        row = cands.iloc[0]
        return {"county": row["county"], "subcounty": row["subcounty"], "ward": row["ward"], "admin_match": method}

    if wk:
        # --- ward within the stated county
        if (ck, wk) in WARD_ALIASES:
            hit = pick(in_county[in_county["wk"] == key(WARD_ALIASES[(ck, wk)])], "ward_alias")
            if hit:
                return hit
        hit = pick(in_county[(in_county["wk"] == wk) | (in_county["wsk"] == sorted_key(ward)) | (in_county["wbk"] == wk)],
                   "ward_exact")
        if hit:
            return hit
        opts = in_county.assign(label=in_county["ward"], w=in_county["wk"], w_sorted=in_county["wsk"])
        hit = pick(fuzzy_pick(ward, opts, "w"), "ward_fuzzy")
        if hit:
            return hit

        # --- ward + sub-county both point to a single other county: county was mis-entered
        elsewhere = ref[(ref["wk"] == wk) & (ref["sk"] == sk) & (ref["ck"] != ck)]
        if sk and len(elsewhere) == 1:
            return pick(elsewhere, "county_corrected")

    # --- ward missing or not in the shapefile: at least fix the sub-county
    if not sk and not wk and not in_county.empty:
        return {**out, "admin_match": "county_only"}
    if sk and not in_county.empty:
        subs = in_county.drop_duplicates("sk")
        if (ck, sk) in SUBCOUNTY_ALIASES:
            match = subs[subs["subcounty"] == SUBCOUNTY_ALIASES[(ck, sk)]]
        else:
            match = subs[(subs["sk"] == sk) | (subs["subcounty"].map(sorted_key) == sorted_key(subcounty))]
        if match.empty:
            opts = subs.assign(label=subs["subcounty"], s=subs["sk"], s_sorted=subs["subcounty"].map(sorted_key))
            match = fuzzy_pick(subcounty, opts, "s")
        if not match.empty:
            out["subcounty"], out["admin_match"] = match.iloc[0]["subcounty"], "subcounty_only"
    return out


# --------------------------------------------------------------------------- #
# GPS check
# --------------------------------------------------------------------------- #
def _in_ring(x: np.ndarray, y: np.ndarray, ring) -> np.ndarray:
    r = np.asarray(ring, dtype=float)
    xi, yi = r[:, 0], r[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    inside = np.zeros(len(x), dtype=bool)
    for a, b, c, d in zip(xi, yi, xj, yj):
        if b == d:
            continue
        crosses = ((b > y) != (d > y)) & (x < (c - a) * (y - b) / (d - b) + a)
        inside ^= crosses
    return inside


def flag_coords_in_county(df: pd.DataFrame, county_col: str = "county", lat_col: str = "latitude",
                          lon_col: str = "longitude", geojson: Path = COUNTY_GEOJSON) -> pd.DataFrame:
    df = df.copy()
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    df["coords_in_county"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    for f in json.loads(geojson.read_text())["features"]:
        rows = df.index[(df[county_col] == f["properties"]["shapeName"]) & lat.notna() & lon.notna()]
        if rows.empty:
            continue
        x, y = lon[rows].to_numpy(), lat[rows].to_numpy()
        geom = f["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        inside = np.zeros(len(rows), dtype=bool)
        for poly in polys:
            in_poly = _in_ring(x, y, poly[0])
            for hole in poly[1:]:
                in_poly &= ~_in_ring(x, y, hole)
            inside |= in_poly
        df.loc[rows, "coords_in_county"] = inside
    outside = int((df["coords_in_county"] == False).sum())  # noqa: E712
    log.info("GPS points outside their reported county: %d (flagged in coords_in_county)", outside)
    return df


def admin_from_gps(df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude",
                   wards_geojson: Path = ROOT / "data/kenya_wards.geojson") -> pd.DataFrame:
    """County / sub-county / ward (shapefile names) for each GPS point, for data that has no admin names."""
    lookup = pd.read_csv(ADMIN_LOOKUP).set_index("ward_shape_id")
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    has = lat.notna() & lon.notna()
    x, y = lon[has].to_numpy(), lat[has].to_numpy()
    shape_id = np.full(len(x), None, dtype=object)
    for f in json.loads(wards_geojson.read_text())["features"]:
        geom = f["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            ring = np.asarray(poly[0], dtype=float)
            cand = (shape_id == None) & (x >= ring[:, 0].min()) & (x <= ring[:, 0].max()) \
                & (y >= ring[:, 1].min()) & (y <= ring[:, 1].max())  # noqa: E711
            if not cand.any():
                continue
            inside = _in_ring(x[cand], y[cand], ring)
            for hole in poly[1:]:
                inside &= ~_in_ring(x[cand], y[cand], hole)
            idx = np.flatnonzero(cand)[inside]
            shape_id[idx] = f["properties"]["shapeID"]
    found = pd.Series(shape_id, index=lat[has].index).reindex(df.index)
    out = lookup.reindex(found)[["county_iso", "county", "subcounty", "ward"]].set_axis(df.index)
    log.info("Admin names from GPS: %d of %d points placed in a ward", found.notna().sum(), len(df))
    return out


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def clean_admin_names(df: pd.DataFrame, county_col: str = "county", subcounty_col: str | None = "subcounty",
                      ward_col: str | None = "ward", lat_col: str | None = "latitude",
                      lon_col: str | None = "longitude", ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a copy of df with admin names set to the shapefile names (columns keep their names)."""
    ref = load_reference() if ref is None else ref
    df = df.copy()
    for col in [c for c in (subcounty_col, ward_col) if c]:
        df[f"{col}_raw"] = df[col]

    county = df[county_col]
    subcounty = df[subcounty_col] if subcounty_col else pd.Series(np.nan, index=df.index)
    ward = df[ward_col] if ward_col else pd.Series(np.nan, index=df.index)
    triples = pd.DataFrame({"c": county, "s": subcounty, "w": ward}).astype("object")
    triples = triples.where(triples.notna(), None)
    uniq = triples.drop_duplicates()
    resolved = {tuple(t): resolve_admin(*t, ref) for t in uniq.itertuples(index=False)}
    fixed = pd.DataFrame([resolved[tuple(t)] for t in triples.itertuples(index=False)], index=df.index)

    df[county_col] = fixed["county"]
    if subcounty_col:
        df[subcounty_col] = fixed["subcounty"]
    if ward_col:
        df[ward_col] = fixed["ward"]
    df["admin_match"] = fixed["admin_match"]
    df["county_iso"] = df[county_col].map(ref.drop_duplicates("county").set_index("county")["county_iso"])

    log.info("Admin match (rows):\n%s", df["admin_match"].value_counts().to_string())
    bad_county = df.loc[df["admin_match"] == "county_unmatched", county_col].dropna().unique()
    if len(bad_county):
        log.warning("Counties not in the shapefile: %s", sorted(map(str, bad_county)))
    left = df[df["admin_match"].isin(["unmatched", "subcounty_only"]) & ward.notna()]
    if ward_col and len(left):
        log.warning("Wards not in the shapefile (kept as entered):\n%s",
                    left.groupby([county_col, subcounty_col or county_col, ward_col]).size()
                    .sort_values(ascending=False).head(20).to_string())

    if lat_col and lon_col and lat_col in df and lon_col in df:
        df = flag_coords_in_county(df, county_col, lat_col, lon_col)
    return df


def write_admin_lookup(out: Path, ref: pd.DataFrame | None = None) -> None:
    ref = load_reference() if ref is None else ref
    cols = ["county_iso", "county", "subcounty", "ward"]
    ref[cols].sort_values(cols).to_csv(out, index=False)
    log.info("Wrote admin lookup (%d wards) -> %s", len(ref), out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--county-col", default="county")
    parser.add_argument("--subcounty-col", default="subcounty", help="pass '' if the data has no sub-county column")
    parser.add_argument("--ward-col", default="ward", help="pass '' if the data has no ward column")
    parser.add_argument("--lat-col", default="latitude")
    parser.add_argument("--lon-col", default="longitude")
    args = parser.parse_args()

    data = pd.read_csv(args.src, low_memory=False)
    log.info("Loaded %d rows from %s", len(data), args.src)
    cleaned = clean_admin_names(data, args.county_col, args.subcounty_col or None, args.ward_col or None,
                                args.lat_col, args.lon_col)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.out, index=False)
    log.info("Wrote %d rows -> %s", len(cleaned), args.out)
