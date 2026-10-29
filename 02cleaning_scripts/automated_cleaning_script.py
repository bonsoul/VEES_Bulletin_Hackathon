"""
ND1 admin + disease harmonisation.
Reads Cleaned Data/nd1_final.csv and:
  1. Renames county / sub-county / ward to exactly the names in the Kenya
     boundary files (data/kenya_counties.geojson, kenya_subcounties.geojson,
     kenya_wards.geojson; geoBoundaries ADM1/2/3), so the data joins to the
     maps on name. Each county only lists its own sub-counties and wards.
     The lookup comes from data/reference/kenya_shapefile_admin_lookup.csv
     (built by build_admin_lookup.py).
  2. Flags GPS points that fall outside their county polygon.
  3. Harmonises disease names (one spelling/category per disease) and the
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
import difflib
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nd1_harmonise")

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
}


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
def tokens(s) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", str(s).lower().replace("&", " and "))
    return [t for t in toks if t != "ward"]


def key(s) -> str:
    return "".join(tokens(s))


def sorted_key(s) -> str:
    return "".join(sorted(tokens(s)))


def nice(s: str) -> str:
    """'SANG'ALO WEST' -> "Sang'alo West", 'MAAI-MAHIU' -> 'Maai-Mahiu'."""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return re.sub(r"[A-Za-z0-9]+(?:'[A-Za-z]+)*", lambda m: m[0].capitalize(), s)


def same_direction(a: str, b: str) -> bool:
    """Block fuzzy matches like 'West Sang'alo' -> 'East Sang'alo'."""
    return set(tokens(a)) & DIRECTIONS == set(tokens(b)) & DIRECTIONS


# --------------------------------------------------------------------------- #
# 1. admin names
# --------------------------------------------------------------------------- #
def county_key(s) -> str:
    k = key(s)
    return COUNTY_KEY_ALIASES.get(k, k)


def load_reference() -> pd.DataFrame:
    ref = pd.read_csv(ADMIN_LOOKUP).drop_duplicates(["county", "subcounty", "ward"])
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


def resolve_admin(county: str, subcounty: str, ward: str, ref: pd.DataFrame) -> dict:
    ck, sk, wk = county_key(county), key(subcounty), key(ward)
    in_county = ref[ref["ck"] == ck]
    official_county = in_county.iloc[0]["county"] if not in_county.empty else county
    out = {"county": official_county, "subcounty": nice(subcounty) if pd.notna(subcounty) else subcounty,
           "ward": nice(ward) if pd.notna(ward) else ward, "admin_match": "unmatched"}

    def pick(cands: pd.DataFrame, method: str) -> dict | None:
        if cands.empty:
            return None
        # Duplicate ward names within a county (e.g. Township in Kiambu): use the stated sub-county.
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
    if len(elsewhere) == 1:
        return pick(elsewhere, "county_corrected")

    # --- ward not in the official list: at least fix the sub-county
    subs = in_county.drop_duplicates("sk")
    match = subs[(subs["sk"] == sk) | (subs["subcounty"].map(sorted_key) == sorted_key(subcounty))]
    if match.empty:
        opts = subs.assign(label=subs["subcounty"], s=subs["sk"], s_sorted=subs["subcounty"].map(sorted_key))
        match = fuzzy_pick(subcounty, opts, "s")
    if not match.empty:
        out["subcounty"], out["admin_match"] = match.iloc[0]["subcounty"], "subcounty_only"
    return out


def harmonise_admin(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["subcounty_raw"], df["ward_raw"] = df["subcounty"], df["ward"]
    triples = df[["county", "subcounty", "ward"]].drop_duplicates()
    resolved = pd.DataFrame(
        [resolve_admin(c, s, w, ref) for c, s, w in triples.itertuples(index=False)],
        index=pd.MultiIndex.from_frame(triples),
    )
    fixed = resolved.reindex(pd.MultiIndex.from_frame(df[["county", "subcounty", "ward"]]))
    for col in ["county", "subcounty", "ward", "admin_match"]:
        df[col] = fixed[col].to_numpy()

    df["county_iso"] = df["county"].map(ref.drop_duplicates("county").set_index("county")["county_iso"])

    log.info("Admin match (rows):\n%s", df["admin_match"].value_counts().to_string())
    changed = (df["subcounty"] != df["subcounty_raw"].map(nice)) | (df["ward"] != df["ward_raw"].map(nice))
    log.info("Rows with sub-county or ward corrected: %d", changed.sum())
    left = df[df["admin_match"].isin(["unmatched", "subcounty_only"])]
    if len(left):
        log.warning("Wards not in the shapefile (kept as entered):\n%s",
                    left.groupby(["county", "subcounty", "ward"]).size().sort_values(ascending=False).head(20).to_string())
    return df


def write_admin_lookup(ref: pd.DataFrame, out: Path) -> None:
    cols = ["county_iso", "county", "subcounty", "ward"]
    ref[cols].sort_values(cols).to_csv(out, index=False)
    log.info("Wrote admin lookup (%d wards) -> %s", len(ref), out)


# --------------------------------------------------------------------------- #
# 2. county boundaries
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


def flag_coords_in_county(df: pd.DataFrame, gj: dict) -> pd.DataFrame:
    df = df.copy()
    df["coords_in_county"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    for f in gj["features"]:
        rows = df.index[(df["county"] == f["properties"]["shapeName"]) & df["latitude"].notna()]
        if rows.empty:
            continue
        x, y = df.loc[rows, "longitude"].to_numpy(), df.loc[rows, "latitude"].to_numpy()
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


# --------------------------------------------------------------------------- #
# 3. disease and syndrome names
# --------------------------------------------------------------------------- #
# Order matters: specific names first. Each match is blanked out before the next
# pattern runs, so 'sheep pox' is not also counted as generic 'pox'.
DISEASES = [
    ("Contagious Caprine Pleuropneumonia (CCPP)", r"(contagious )?cap\w* pl\w*pneumonia|\bccpp\b"),
    ("Contagious Bovine Pleuropneumonia (CBPP)", r"(contagious )?bovine pl\w*pneumonia|\bcbpp\b"),
    ("East Coast Fever (ECF)", r"\be ?c ?f\b|\becf\w*|east coast"),
    ("Foot and Mouth Disease (FMD)", r"\bfmd\b|foot (and )?mouth"),
    ("Lumpy Skin Disease (LSD)", r"\blsd\b|l[ua]mpy skin"),
    ("Peste des Petits Ruminants (PPR)", r"\bppr\b|peste"),
    ("Rift Valley Fever (RVF)", r"\brvf\b|rift valley"),
    ("Newcastle Disease (ND)", r"\bn ?c ?d\b|\bnd\b|new ?castle"),
    ("African Swine Fever (ASF)", r"\basf\b|african swine"),
    ("Avian Influenza", r"avian influenza|\bhpai\b"),
    ("MERS-CoV", r"\bmers\b"),
    ("Nairobi Sheep Disease (NSD)", r"\bnsd\b|nairobi sheep"),
    ("Acute Camel Death Syndrome (ACDS)", r"\bacds\b|acute camel"),
    ("Infectious Bursal Disease (Gumboro)", r"gumboro|bursa\w? disease"),
    ("Infectious Coryza", r"cor+[iy]z+a"),
    ("Infectious Bronchitis", r"infectious bronchitis"),
    ("Chronic Respiratory Disease (CRD)", r"\bcrd\b|chronic resp"),
    ("Fowl Typhoid", r"fowl ?typh"),
    ("Fowl Pox", r"fowl ?pox"),
    ("Marek's Disease (Fowl Paralysis)", r"fowl paralysis|marek"),
    ("Avian Leukosis", r"leu[ck]osis"),
    ("Pullorum Disease", r"pullor"),
    ("Camel Pox", r"camel ?pox"),
    ("Sheep and Goat Pox", r"(sheep|goat)s?( and goats?)? ?pox"),
    ("Cowpox / Pseudocowpox", r"(pseudo )?cow ?pox|pox mam\w*"),
    ("Swine Pox", r"swine ?pox"),
    ("Orf (Contagious Ecthyma)", r"\borf[es]?\b|contagious ecthyma|pustular dermatitis|pastular dermatitis"),
    ("Anthrax", r"anthrax"),
    ("Black Quarter", r"black ?qu?art?er|black ?quat\w*|\bb ?q\b|black ?leg"),
    ("Haemorrhagic Septicaemia", r"h(a)?emorr?hagic [sc]ept\w*|\bh ?s\b"),
    ("Pasteurellosis", r"paste?u?r?\w*llosis|pasteulosis"),
    ("Rabies", r"rab+ies"),
    ("Brucellosis", r"brucell"),
    ("Tuberculosis", r"tubercul|\bbtb\b|\btb\b"),
    ("Johne's Disease", r"johne"),
    ("Q Fever", r"\bq ?fever"),
    ("Leptospirosis", r"leptospir"),
    ("Bluetongue", r"blue ?tongue"),
    ("Bovine Ephemeral Fever", r"\be\w{0,3}h[ei]\w{0,3}ral\b|(three|3) ?days? ?sick|\bbef\b|b e fever"),
    ("Malignant Catarrhal Fever", r"catarrh"),
    ("Heartwater", r"heart ?water"),
    ("Babesiosis (Redwater)", r"babes\w*|red ?water|red urine"),
    ("Anaplasmosis", r"anapla\w*"),
    ("Theileriosis", r"theiler"),
    ("Ehrlichiosis", r"e?h?r?l?h?ichiosis|ehrlich|erhlich"),
    ("Tick Paralysis", r"tick paralysis"),
    ("Tick-borne Disease", r"tick ?(borne )?(fever|disease)|\btbd\b"),
    ("Trypanosomiasis", r"\btr[iy]p[aos]\w*|\btryps?\b|nagana"),
    ("Ringworm (Dermatophytosis)", r"ring ?worms?|dermato?ph[yi]t\w*|dermatomycos\w*|\btinea"),
    ("Dermatophilosis (Streptothricosis)", r"strep\w*(thr|tr)\w*(cosis|chosis)|dermatoph[iy]l+\w*"),
    ("Fasciolosis (Liver Fluke)", r"fas?c?[iy]?ol\w*|fabiola|liver ?flukes?|distomatosis"),
    ("Hydatidosis (Echinococcosis)", r"\bh[yi]a?d[aei]t?\w*d\w*|echinoc\w*"),
    ("Coenurosis", r"coe?ne?ur\w*|cenor\w*"),
    ("Cysticercosis", r"cysticerc\w*"),
    ("Coccidiosis", r"cocc?id\w*"),
    ("Myiasis", r"\bmyi?asis|maggot\w*|bot ?fly|mango\w* worms?|miaces"),
    ("Helminthiasis", r"\bh[ae]*l?m[ie]n?th\w*|\belm[ie]n?th\w*|\bhelm[ei]\w*|haemonch\w*|hemonch\w*|stil[ei]\w*|stelesia|ster[ie]sia|"
                      r"\bworm\w*|tapeworm|ascari\w*|\bgit worms\b"),
    ("Mange", r"\bm[ae]nges?\b|\bmites?\b|scabies|sarcopt\w*|demodec\w*"),
    ("Sheep Scab", r"sheep scab"),
    ("Oestrosis (Nasal Bot)", r"oestrus ov\w*|nasal ?bot"),
    ("Flea Infestation", r"\bfleas?\b"),
    ("Lice Infestation", r"\blice\b|pedicul\w*"),
    ("Tick Infestation", r"\bticks?\b"),
    ("Ectoparasite Infestation", r"ecto ?parasit\w*|external parasit\w*"),
    ("Parasitic Infestation (unspecified)", r"parasit\w*"),
    ("Mastitis", r"\bm[ai]s?t\w*it[iy]s?|mastity|ma?m+[ei]l+itis"),
    ("Metritis", r"(endo ?)?metr[iy]?t?is\b|metrisis|pyomet\w*|endometriosis|(uterine|postpartum|post pertum) (uterine )?infection"),
    ("Retained Placenta", r"retaine?d? (after ?birth|placenta|pacenta)|retainplacenta"),
    ("Dystocia", r"d[iy]s?[ty]o[ck]h?ia|difficult calving|\bbree?a?ch\w*"),
    ("Prolapse", r"prolap\w*"),
    ("Abortion", r"abortions?|miscarriage|premature birth"),
    ("Stillbirth", r"still ?birth"),
    ("Infertility / Reproductive Disorder", r"infertil\w*|an?o?estrus|repeat breeder|silent heat|follicular|luteal|hormonal"),
    ("Milk Fever (Hypocalcaemia)", r"milk ?fever|hypo ?ca\w*|calcium deficiency"),
    ("Downer Cow Syndrome", r"downer"),
    ("Hypomagnesaemia (Grass Tetany)", r"hypomagn\w*|grass tetany"),
    ("Ketosis", r"ketosis"),
    ("Bloat", r"bloat\w*"),
    ("Ruminal Acidosis", r"acidosis|grain overload"),
    ("Traumatic Reticuloperitonitis (Hardware Disease)", r"hard ?ware|traumatic reticul\w*|\btrp\b|foreign (body|objects?)"),
    ("Enterotoxaemia", r"entero ?tox\w*"),
    ("Colibacillosis", r"col+[iy] ?bacil+\w*|\be ?coli\b"),
    ("Salmonellosis", r"salm\w*n\w*l\w*"),
    ("Diarrhoea / Scours", r"di?a?r+h?o?e?a|dirhoea|diarrohea|scour\w*|dys?ent\w*|calf score|kids? scores?"),
    ("Enteritis", r"ente?rit[ie]s|gastro ?enter\w*|git (infection|disorder)"),
    ("Pneumonia", r"pneumon\w*"),
    ("Kennel Cough", r"kennel cough"),
    ("Camel Respiratory Disease", r"camel (cough|flu)\w*"),
    ("Atrophic Rhinitis", r"atrophic rhinitis"),
    ("Strangles", r"strangles?"),
    ("Equine Influenza", r"equine influenza"),
    ("Respiratory Infection (unspecified)", r"respirat\w*|\buri\b|shipping fever|rhinitis|sinusitis"),
    ("Canine Distemper", r"distemper"),
    ("Parvovirus", r"parvo ?\w*"),
    ("Tetanus", r"tet[ae]nus"),
    ("Botulism", r"botulism"),
    ("Listeriosis", r"listeri\w*"),
    ("Swine Erysipelas", r"erysipel\w*"),
    ("Actinobacillosis (Wooden Tongue)", r"actinobac\w*|wooden tongue"),
    ("Actinomycosis (Lumpy Jaw)", r"actinomyc\w*|lumpy jaw"),
    ("Trichomoniasis", r"tri?c?h?omon\w*"),
    ("Toxoplasmosis", r"toxoplasm\w*"),
    ("Navel Ill", r"nav[ae]l"),
    ("Foot Rot", r"\b(foot|fot|for|foit) ?rote?\b|foul in the foot"),
    ("Laminitis", r"laminitis"),
    ("Lameness", r"\blame\w*|limping"),
    ("Pink Eye (Infectious Keratoconjunctivitis)", r"pin[kc] ?eye|k?c?erato ?conj\w*|\bibk\b|contagious bovine opth\w*"),
    ("Eye Infection", r"conj\w*ctivitis|eye (infection|problem|inflam)\w*|occulitis|keratitis"),
    ("Transmissible Venereal Tumour", r"(transmissible )?vener\w*( tum\w*)?( in dogs)?"),
    ("Tumour", r"carcinoma|cancer|cancar|tumou?r|sarcoid|neoplas\w*"),
    ("Papillomatosis (Warts)", r"\bwarts?\b|\bwatts\b|papill?om\w*"),
    ("Caseous Lymphadenitis", r"ca[sc]eous( lymph\w*)?"),
    ("Abscess", r"\bab+s?e?c+e?s+\w*|\babs+es+\w*|\balbcess\w*|\babscy"),
    ("Predation / Wildlife Attack", r"(predat\w*|attack\w*|h?a?yena|hynes|leopard|\blion\b|killed by)( [a-z]+)?( bites?)?"),
    ("Snake Bite", r"snakes? ?bites?"),
    ("Bee Sting", r"\bbees? (sting|stink|bite)s?"),
    ("Animal Bites", r"\b(dog|animals?|bite) ?(bites?|injury)|\bbite from|\bbites?\b"),
    ("Trauma / Injury", r"wound\w*|injur\w*|trauma\w*|fr[au]cture\w*|laceration|\bcut\b|sprain|dislocat\w*|\bburns?\b|"
                        r"beat(en|ing)|accident|thorn|broken horn|electrocut\w*"),
    ("Poisoning", r"poiso\w*|toxic|aflatox\w*|afflatox\w*|toxicosis"),
    ("Anaemia", r"an(a)?emi\w*"),
    ("Nutritional Deficiency", r"malnutr\w*|mulnutr\w*|malnourish\w*|nutrition\w* imbalance|starvation|emaciat\w*|poor body|defic+ien\w*|avitamin\w*|drought"),
    ("Hernia", r"h[ae]rnia"),
    ("Arthritis", r"a?rt?hritis|athritis"),
    ("Colic", r"\bcolic\b"),
    ("Choke", r"\bchoke\b"),
    ("Impaction / Constipation", r"impaction|constipation|blockage|intussus\w*"),
    ("Septicaemia", r"septic(a)?emia|sepsis"),
    ("Fungal Infection", r"fung\w*|mycotic"),
    ("Dermatitis / Skin Infection", r"dermatitis|eczema|alopecia|skin (infection|disease)"),
    ("Photosensitisation", r"photo ?sensiti\w*"),
    ("Allergy", r"allerg\w*|hypersens\w*|anaphyla\w*"),
    ("Urinary Tract Infection", r"urinary|urino\w*|kidney stones?|urolith\w*"),
    ("Small Hive Beetle", r"small hive beetle"),
    ("Wax Moth", r"wax ?moth"),
    ("Varroosis", r"varroa?\w*"),
    ("Foulbrood", r"foul ?brood"),
    ("Stomatitis", r"stomatitis"),
    ("Sweating Sickness", r"sweating sickness"),
    ("Contagious Agalactia", r"agalactia"),
    ("Habronemiasis", r"ha?e?bron\w*"),
    ("Wry Neck", r"wry ?neck"),
    ("Heat Stress", r"stress"),
    ("Routine Procedure (non-disease)", r"pregnancy diagnosis|\bpd\b|synchroni[sz]ation|herd health|castration|dehorn\w*|docking|prophyla\w*|iron (injection|supplement)\w*|"
                                        r"dry cow therapy|vaccinat\w*|deworm\w*"),
    ("Pox (unspecified)", r"\bpox\b"),
]
DISEASES = [(name, re.compile(p)) for name, p in DISEASES]

PLACEHOLDERS = {
    "", "0", "00", "o", "no", "non", "none", "nil", "na", "n a", "unknown", "not known", "not know",
    "i don t know", "i dont know", "other", "others", "unspecified", "not confirmed", "non specific",
    "nonspecific", "non specific disease", "non specific diagnosis", "ghhnn", "yes", "no idea", "test", "sheep",
}
UNKNOWN = "Unknown"

SIGNS = [
    ("Bloody diarrhoea", r"blood\w* (tin\w* )?diarr\w*|dysent\w*"),
    ("Diarrhoea", r"di?a?r+h?o?e?a|dirhoea|scour\w*|loose (stool|f[ae]e?ces)"),
    ("Constipation", r"constip\w*|hard (stool|dung|f[ae]e?ces)"),
    ("Fever", r"fever|pyrexia|high temp\w*|\btemp\w*"),
    ("Anorexia / inappetence", r"anorex\w*|in ?app?e?t+[ae]n?ce|(loss|lack|reduced) of appetite|reduced appetite|off feed|not feeding"),
    ("Emaciation / weight loss", r"em+[ae]r?c?i?at\w*|emerc\w*|wast(ing|ed)|(loss of|poor) (body )?(weight|condition)|weight loss|unthrift\w*"),
    ("Lacrimation", r"lac?h?r[iy]m\w*|lachrin\w*|tearing|watery eyes"),
    ("Ocular discharge", r"(eye|ocular) discharges?"),
    ("Corneal opacity", r"corneal opac\w*"),
    ("Eye infection", r"eye infections?|conj\w*ctivitis|pin[kc] ?eye"),
    ("Blindness", r"blind\w*"),
    ("Photophobia", r"photophob\w*"),
    ("Coughing", r"cou?g?h\w*|couph\w*"),
    ("Sneezing", r"sneez\w*"),
    ("Nasal discharge", r"n[ao]sal discharges?"),
    ("Dyspnoea", r"dys?p?o?n\w*|labou?red breath\w*|difficult(y)? (in )?breath\w*|respiratory distress"),
    ("Enlarged lymph nodes", r"(enlarged|swollen) (\w+ )?(lymph|lyph|lymp|glands?)\w*( ?nodes?)?|lymph\w*aden\w*|lymphatrophy|lymph ?nodes?"),
    ("Udder inflammation / abnormal milk", r"udder|clot+\w* (in )?milk|clots in milk|blood\w* (stain\w* )?(in )?milk|mastitis|mammary"),
    ("Reduced milk yield", r"(reduced|drop in) milk\w*|milk (drop|yield)|reduced production"),
    ("Lameness", r"lam(e|ness|ness)\b|lamness|limp\w*|limbing|laminitis"),
    ("Bloat", r"bloat\w*"),
    ("Rough hair coat", r"(rough|star\w*|stared) (hair ?)?co[au]?r?t|rough hair\w*"),
    ("Ruffled feathers", r"ruffled feathers?"),
    ("Drooping wings", r"dro+p\w* wings?"),
    ("Salivation", r"saliv\w*|drool\w*"),
    ("Recumbency", r"recumb\w*|unable to stand|downer"),
    ("Paralysis", r"paraly\w*"),
    ("Anaemia / pale mucous membranes", r"an(a)?emi\w*|pale mucu?o?u?s"),
    ("Lethargy / weakness", r"dull\w*|depress\w*|l[ei]th[ae]rg\w*|malaise|weak\w*"),
    ("Jaundice", r"jaund\w*|icter\w*"),
    ("Bottle jaw / oedema", r"bottle jaw|o?edema|swelling|facial swelling"),
    ("Red urine / haematuria", r"(red|bloody|blood( in)?) urine|h(a)?emoglobinur\w*|h(a)?ematur\w*"),
    ("Abortion", r"abortion\w*"),
    ("Retained placenta", r"retaine?d? (after ?birth|placenta)"),
    ("Dystocia", r"d[iy]sto[ck]h?ia"),
    ("Prolapse", r"prolap\w*"),
    ("Infertility", r"infertil\w*|reproductive\w*"),
    ("Nervous signs", r"nervous|circling|convuls\w*|seizure\w*|incoordinat\w*|teeth grinding"),
    ("Aggression", r"aggress\w*"),
    ("Hydrophobia", r"hydroph\w*"),
    ("Bellowing", r"bellow\w*"),
    ("Vomiting", r"vomit\w*"),
    ("Dehydration", r"dehydrat\w*"),
    ("Hair loss", r"alopecia|(loss of|hair) (hair|loss)"),
    ("Itching", r"itch\w*|prurit\w*"),
    ("Petechiae / bleeding", r"petech\w*|bleed\w*|h(a)?emorrhag\w*"),
    ("Stomatitis / mouth lesions", r"stomatitis|mouth (lesion|ulcer)\w*"),
    ("Skin lesions / wounds", r"wound\w*|skin lesion\w*|nodule\w*"),
    ("Abscess", r"\bab+s?c?es+\w*"),
    ("Musculoskeletal disorder", r"musculoskeletal\w*"),
    ("Sudden death", r"sudden death|\bdeath\b"),
]
SIGNS = [(name, re.compile(p)) for name, p in SIGNS]


def norm(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9()/ ]", " ", str(s).lower().replace("&", " and "))).strip()


def match_all(text: str, patterns) -> list[str]:
    found = []
    for name, pat in patterns:
        text, n = pat.subn(lambda m: " " * len(m[0]), text)
        if n and name not in found:
            found.append(name)
    return found


def harmonise_disease_item(raw: str) -> tuple[list[str], str]:
    t = norm(raw)
    if t in PLACEHOLDERS or t.replace(" ", "").isdigit():
        return [UNKNOWN], "placeholder"
    found = match_all(t, DISEASES)
    if found:
        return found, "dictionary"
    return [nice(re.sub(r"[\s.,;:?]+$", "", str(raw).strip()))], "unmatched"


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

    ref = load_reference()
    df = harmonise_admin(df, ref)
    write_admin_lookup(ref, out.parent / "admin_lookup.csv")

    missing = set(df["county"]) - set(ref["county"])
    if missing:
        raise ValueError(f"Counties not in the shapefile: {missing}")
    df = flag_coords_in_county(df, json.loads(COUNTY_GEOJSON.read_text()))

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
