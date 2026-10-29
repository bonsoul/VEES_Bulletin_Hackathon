"""
Build the county -> sub-county -> ward lookup from the Kenya boundary files
(geoBoundaries KEN ADM1/ADM2/ADM3, 2020), so the cleaned data uses exactly the
names that appear in the shapefiles.

The ward and sub-county layers carry no parent names, so each ward is placed in
its sub-county and county using a point guaranteed to lie inside the ward.

Run from the project root (only needed when the boundary files change):
    python 02cleaning_scripts/build_admin_lookup.py
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_admin_lookup")

ROOT = Path(__file__).resolve().parents[1]
COUNTIES = ROOT / "data/kenya_counties.geojson"
SUBCOUNTIES = ROOT / "data/kenya_subcounties.geojson"
WARDS = ROOT / "data/kenya_wards.geojson"
OUT = ROOT / "data/reference/kenya_shapefile_admin_lookup.csv"

# The Buret polygon's interior falls inside the Bomet county polygon (ADM1/ADM2
# boundaries disagree), but Bureti constituency is in Kericho County.
SUBCOUNTY_COUNTY_OVERRIDES = {"Buret": "Kericho"}

# Official 2012 ward list, used only to settle wards whose polygon spills into another county.
OFFICIAL_WARDS = ROOT / "data/reference/kenya_county_constituency_ward_2012.csv"
OFFICIAL_COUNTY_ALIASES = {"tharakanithi": "tharaka"}  # shapefile calls the county 'Tharaka'


def norm_key(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def polygons(geom: dict) -> list:
    return [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]


def in_ring(x: np.ndarray, y: np.ndarray, ring) -> np.ndarray:
    r = np.asarray(ring, dtype=float)
    xi, yi = r[:, 0], r[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    inside = np.zeros(len(x), dtype=bool)
    for a, b, c, d in zip(xi, yi, xj, yj):
        if b != d:
            inside ^= ((b > y) != (d > y)) & (x < (c - a) * (y - b) / (d - b) + a)
    return inside


def in_geom(x: np.ndarray, y: np.ndarray, geom: dict) -> np.ndarray:
    inside = np.zeros(len(x), dtype=bool)
    for poly in polygons(geom):
        p = in_ring(x, y, poly[0])
        for hole in poly[1:]:
            p &= ~in_ring(x, y, hole)
        inside |= p
    return inside


def interior_point(geom: dict) -> tuple[float, float]:
    """Centroid of the largest part if it is inside, else the midpoint of a scanline through it."""
    def area(ring):
        r = np.asarray(ring, dtype=float)
        return 0.5 * abs(np.dot(r[:, 0], np.roll(r[:, 1], 1)) - np.dot(r[:, 1], np.roll(r[:, 0], 1)))

    poly = max(polygons(geom), key=lambda p: area(p[0]))
    r = np.asarray(poly[0], dtype=float)
    x, y = r[:, 0], r[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2
    if a:
        cx = ((x + np.roll(x, -1)) * cross).sum() / (6 * a)
        cy = ((y + np.roll(y, -1)) * cross).sum() / (6 * a)
        if in_geom(np.array([cx]), np.array([cy]), geom)[0]:
            return cx, cy
    # Scanline at mid-height: crossings come in inside/outside pairs.
    sy = (y.min() + y.max()) / 2 + 1e-9
    xs = sorted(
        x0 + (sy - y0) * (x1 - x0) / (y1 - y0)
        for x0, y0, x1, y1 in zip(x, y, np.roll(x, -1), np.roll(y, -1))
        if (y0 > sy) != (y1 > sy)
    )
    widest = max(range(0, len(xs) - 1, 2), key=lambda i: xs[i + 1] - xs[i])
    return (xs[widest] + xs[widest + 1]) / 2, sy


def locate(points: pd.DataFrame, layer: list[dict]) -> pd.Series:
    out = pd.Series(pd.NA, index=points.index, dtype="object")
    x, y = points["x"].to_numpy(), points["y"].to_numpy()
    for f in layer:
        hit = in_geom(x, y, f["geometry"]) & out.isna().to_numpy()
        out[hit] = f["properties"]["shapeName"]
    return out


def main() -> None:
    counties = json.loads(COUNTIES.read_text())["features"]
    subcounties = json.loads(SUBCOUNTIES.read_text())["features"]
    wards = json.loads(WARDS.read_text())["features"]

    pts = pd.DataFrame(
        [(f["properties"]["shapeName"], f["properties"]["shapeID"], *interior_point(f["geometry"])) for f in wards],
        columns=["ward", "ward_shape_id", "x", "y"],
    )
    pts["subcounty"] = locate(pts, subcounties)
    pts["county"] = locate(pts, counties)

    # Coastal / island wards whose interior point misses every sub-county polygon:
    # use the sub-county holding most of the ward's boundary vertices.
    by_id = {f["properties"]["shapeID"]: f for f in wards}
    for i in pts.index[pts["subcounty"].isna()]:
        verts = np.vstack([np.asarray(p[0]) for p in polygons(by_id[pts.at[i, "ward_shape_id"]]["geometry"])])
        votes = {f["properties"]["shapeName"]: in_geom(verts[:, 0], verts[:, 1], f["geometry"]).sum() for f in subcounties}
        pts.at[i, "subcounty"] = max(votes, key=votes.get)

    # Sub-county -> county from the sub-county's own interior point (more robust than per ward).
    sub_pts = pd.DataFrame([(f["properties"]["shapeName"], *interior_point(f["geometry"])) for f in subcounties],
                           columns=["subcounty", "x", "y"])
    sub_pts["county"] = locate(sub_pts, counties)
    sub_county = sub_pts.set_index("subcounty")["county"]
    sub_county.update(pd.Series(SUBCOUNTY_COUNTY_OVERRIDES))
    pts["county"] = pts["subcounty"].map(sub_county).fillna(pts["county"])

    # Ward polygons that spill across a county line (e.g. Nanyuki lands in Meru, Athi River
    # in Kajiado): if the official 2012 list has that ward name in exactly one county, move
    # it there and pick that county's sub-county covering most of the ward's vertices.
    official = pd.read_csv(OFFICIAL_WARDS, encoding="utf-8-sig")
    official["ck"] = official["COUNTY NAME"].map(norm_key).replace(OFFICIAL_COUNTY_ALIASES)
    official["wk"] = official["WARD NAME"].map(norm_key)
    official_counties = official.groupby("wk")["ck"].unique()
    county_by_key = {norm_key(f["properties"]["shapeName"]): f["properties"]["shapeName"] for f in counties}
    sub_by_county = sub_county.groupby(sub_county).groups
    shape_name_counts = pts["ward"].map(norm_key).value_counts()
    for i in pts.index:
        wk, ck = norm_key(pts.at[i, "ward"]), norm_key(pts.at[i, "county"])
        homes = official_counties.get(wk, [])
        # Skip names the shapefile itself repeats (e.g. a second Githiga in Laikipia).
        if len(homes) != 1 or homes[0] == ck or shape_name_counts[wk] > 1:
            continue
        home = county_by_key[homes[0]]
        verts = np.vstack([np.asarray(p[0]) for p in polygons(by_id[pts.at[i, "ward_shape_id"]]["geometry"])])
        votes = {f["properties"]["shapeName"]: in_geom(verts[:, 0], verts[:, 1], f["geometry"]).sum()
                 for f in subcounties if f["properties"]["shapeName"] in sub_by_county[home]}
        log.info("Moved ward %s: %s/%s -> %s/%s", pts.at[i, "ward"], pts.at[i, "county"], pts.at[i, "subcounty"],
                 home, max(votes, key=votes.get))
        pts.at[i, "county"], pts.at[i, "subcounty"] = home, max(votes, key=votes.get)

    iso = {f["properties"]["shapeName"]: f["properties"]["shapeISO"] for f in counties}
    pts["county_iso"] = pts["county"].map(iso)

    missing = pts[pts[["county", "subcounty"]].isna().any(axis=1)]
    if len(missing):
        log.warning("Wards not placed in a sub-county/county:\n%s", missing[["ward", "subcounty", "county"]].to_string())

    lookup = pts[["county_iso", "county", "subcounty", "ward", "ward_shape_id"]].sort_values(["county", "subcounty", "ward"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(OUT, index=False)
    log.info("%d counties, %d sub-counties, %d wards -> %s",
             lookup["county"].nunique(), lookup["subcounty"].nunique(), len(lookup), OUT)


if __name__ == "__main__":
    main()
