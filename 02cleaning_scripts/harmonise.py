"""
Shared harmonisation rules used by every VEES cleaning script, so all cleaned
datasets use the same names and checks:

  - disease names      DISEASES dictionary, harmonise_disease_item / harmonise_disease_cell
  - clinical signs     SIGNS dictionary, harmonise_signs_cell
  - species names      standardise_species
  - dates              epoch_ms_to_date (KABS 1.0), kabs2_date (KABS 2.0 'Wed, Oct 28, 2026 EAT')
  - GPS                validate_coords (swapped lat/long fixed, points outside Kenya nulled)
  - raw files          read_raw (keeps the text 'N/A' distinct from an empty cell)

County / sub-county / ward names are handled by clean_admin_names.py.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from clean_admin_names import nice

log = logging.getLogger("harmonise")

# --------------------------------------------------------------------------- #
# disease and syndrome names
# --------------------------------------------------------------------------- #
# Order matters: specific names first. Each match is blanked out before the next
# pattern runs, so 'sheep pox' is not also counted as generic 'pox'.
DISEASES = [
    ("Contagious Caprine Pleuropneumonia (CCPP)", r"(contagious )?ca\w{0,3}ine pl\w*pneumonia|(contagious )?cap\w* pl\w*pneumonia|\bccpp\b"),
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
    ("Bloody diarrhoea", r"blood\w* (tin\w* )?diarr\w*|dysent\w*|blood\w* (f[ae]e?ces|stool)"),
    ("Discoloured faeces", r"discolou?red (f[ae]e?ces|stool)"),
    ("Diarrhoea", r"di?a?r+h?o?e?a|dirhoea|scour\w*|loose (stool|f[ae]e?ces)|kuhara"),
    ("Constipation", r"constip\w*|hard (stool|dung|f[ae]e?ces)"),
    ("Fever", r"fever|pyrexia|high temp\w*|\btemp\w*"),
    ("Anorexia / inappetence", r"anorex\w*|in ?app?e?t+[ae]n?ce|(loss|lack|reduced) of appetite|reduced appetite|off feed|not feeding"),
    ("Emaciation / weight loss", r"em+[ae]r?c?i?at\w*|emerc\w*|wast(ing|ed)|(loss of|poor) (body )?(weight|condition)|weight loss|unthrift\w*"),
    ("Lacrimation", r"lac?h?r[iy]m\w*|lachrin\w*|tearing|watery eyes|machozi"),
    ("Ocular discharge", r"(eye|ocular) discharges?|discharges? from (the )?eyes?|^the eyes$"),
    ("Corneal opacity", r"corneal opac\w*"),
    ("Eye infection", r"eye infections?|conj\w*ctivitis|pin[kc] ?eye"),
    ("Blindness", r"blind\w*"),
    ("Photophobia", r"photophob\w*"),
    ("Coughing", r"cou?g?h\w*|couph\w*"),
    ("Sneezing", r"sneez\w*"),
    ("Nasal discharge", r"n[ao]sal discharges?|makamas\w*"),
    ("Dyspnoea", r"dys?p?o?n\w*|labou?red breath\w*|difficult(y)? (in )?breath\w*|respiratory distress"),
    ("Enlarged lymph nodes", r"(enlarged|swollen) (\w+ )?(lymph|lyph|lymp|glands?)\w*( ?nodes?)?|lymph\w*aden\w*|lymphatrophy|lymph ?nodes?"),
    ("Udder inflammation / abnormal milk", r"udder|clot+\w* (in )?milk|clots in milk|blood\w* (stain\w* )?(in )?milk|mastitis|mammary"),
    ("Reduced milk yield", r"(reduced|drop in) milk\w*|milk (drop|yield)|reduced production"),
    ("Lameness", r"lam(e|ness|ness)\b|lam+e?n?e?s+\b|l\w?meness\w*|limp\w*|limbing|laminitis"),
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
    ("Bottle jaw / oedema", r"bottle jaw|o?edema|(?<!skin )swelling|facial swelling"),
    ("Red urine / haematuria", r"(red|bloody|blood( in)?) urine|h(a)?emoglobinur\w*|h(a)?ematur\w*"),
    ("Abortion", r"abortion\w*"),
    ("Retained placenta", r"retaine?d? (after ?birth|placenta)"),
    ("Dystocia", r"d[iy]sto[ck]h?ia"),
    ("Prolapse", r"prolap\w*"),
    ("Infertility", r"infertil\w*|reproductive\w*"),
    ("Nervous signs", r"nervous|circling|walking in circle|convuls\w*|seizure\w*|incoordinat\w*|teeth grinding|stagger\w*|kombo|head pressing"),
    ("Aggression", r"aggre+s\w*|\bbiting\b|kuuma"),
    ("Hydrophobia", r"hydroph\w*"),
    ("Bellowing", r"bellow\w*"),
    ("Vomiting", r"vomit\w*|kutapika"),
    ("Dehydration", r"dehydrat\w*"),
    ("Hair loss", r"alopecia|(loss of|hair) (hair|loss)"),
    ("Itching", r"itch\w*|prurit\w*"),
    ("Petechiae / bleeding", r"petech\w*|bleed\w*|h(a)?emorrhag\w*"),
    ("Stomatitis / mouth lesions", r"stomatitis|mouth (lesion|ulcer)\w*"),
    ("Skin lesions / wounds", r"wound\w*|would\b|skin lesion\w*|nodule\w*|skin swelling\w*|ngozi"),
    ("Wattle / comb swelling or discolouration", r"wattles?|\bcombs?\b"),
    ("Soft-shelled eggs", r"soft shell\w* eggs?"),
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


def parse_list_cell(cell) -> list[str]:
    """'["Mange", "Other"]' -> ['Mange', 'Other']; 'a; b' -> ['a', 'b']; plain text -> [text]."""
    if pd.isna(cell) or not str(cell).strip():
        return []
    s = str(cell).strip()
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            return [str(x).strip() for x in (parsed if isinstance(parsed, list) else [parsed]) if str(x).strip()]
        except json.JSONDecodeError:
            s = s.strip("[]")
            return [x.strip().strip('"') for x in s.split(",") if x.strip().strip('"')]
    return [x.strip() for x in s.split(";") if x.strip()]


def harmonise_disease_cell(cell, other=None) -> str | float:
    """Harmonise one cell (plain text or JSON list); 'Other' is replaced by the free-text 'other' answer."""
    items = parse_list_cell(cell)
    if other is not None and pd.notna(other) and str(other).strip():
        items = [str(other).strip() if i.lower() == "other" else i for i in items] or [str(other).strip()]
    names = list(dict.fromkeys(n for i in items for n in harmonise_disease_item(i)[0]))
    if len(names) > 1 and UNKNOWN in names:
        names.remove(UNKNOWN)
    return "; ".join(names) if names else np.nan


def harmonise_signs_cell(cell) -> str | float:
    """Free-text clinical signs -> '; '-joined standard sign names (unknown text kept, capitalised)."""
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


# --------------------------------------------------------------------------- #
# species
# --------------------------------------------------------------------------- #
# Same names as ND1 / zero reports.
SPECIES_SYNONYMS = {
    "camels": "Camel", "camel": "Camel", "donkeys": "Donkey", "donkey": "Donkey", "dog": "Dogs", "dogs": "Dogs",
    "cat": "Cats", "cats": "Cats", "pig": "Pigs", "pigs": "Pigs", "swine": "Pigs", "porcine": "Pigs",
    "goat": "Goats", "goats": "Goats", "caprine": "Goats", "sheep": "Sheep", "ovine": "Sheep",
    "cattle": "Cattle", "cow": "Cattle", "cows": "Cattle", "bovine": "Cattle",
    "chicken": "Chicken", "chickens": "Chicken", "avian domestic": "Chicken",
    "rabbit": "Rabbits", "rabbits": "Rabbits", "duck": "Ducks", "ducks": "Ducks",
    "horse": "Horse", "horses": "Horse", "equine": "Horse", "bees": "Bees", "bee": "Bees",
    "calf": "Cattle", "calves": "Cattle", "canine": "Dogs", "feline": "Cats", "kitten": "Cats", "puppy": "Dogs",
}


def _one_species(v):
    if pd.isna(v):
        return pd.NA
    k = re.sub(r"[^a-z]+", " ", str(v).lower()).strip()
    if not k:  # numbers / punctuation only, e.g. '0', '6677'
        return pd.NA
    return SPECIES_SYNONYMS.get(k, str(v).strip().capitalize())


def standardise_species(s: pd.Series) -> pd.Series:
    """Species names as in ND1 / zero reports; '; '-separated lists are standardised item by item."""
    def one_cell(v):
        if pd.isna(v):
            return pd.NA
        parts = [_one_species(x) for x in str(v).split(";")]
        parts = list(dict.fromkeys(x for x in parts if pd.notna(x)))
        return "; ".join(parts) if parts else pd.NA
    uniq = {v: one_cell(v) for v in s.dropna().unique()}
    return s.map(uniq).astype("string")


def split_species_category(s: pd.Series) -> pd.DataFrame:
    """'Cattle(Beef)' / 'Cattle (exoticDairy)' -> species 'Cattle', category 'Beef' / 'Exotic dairy'."""
    parts = s.astype("string").str.extract(r"^\s*([^(]+?)\s*(?:\((.*)\))?\s*$")
    cat = parts[1].str.replace(r"(?<=[a-z])(?=[A-Z])", " ", regex=True).str.strip().str.capitalize()
    return pd.DataFrame({"species": standardise_species(parts[0]), "species_category": cat}, index=s.index)


# --------------------------------------------------------------------------- #
# dates, GPS, raw files
# --------------------------------------------------------------------------- #
LAT_RANGE = (-5.0, 5.5)   # rough Kenya bounding box
LON_RANGE = (33.5, 42.5)


def epoch_ms_to_date(s: pd.Series) -> pd.Series:
    ms = pd.to_numeric(s, errors="coerce")
    return pd.to_datetime(ms, unit="ms", utc=True).dt.tz_convert("Africa/Nairobi").dt.tz_localize(None).dt.normalize()


def kabs2_date(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.replace(r"\s+(EAT|East Africa Time)$", "", regex=True).str.strip()
    return pd.to_datetime(s, format="%a, %b %d, %Y", errors="coerce")


def parse_gps(s: pd.Series) -> pd.DataFrame:
    """KABS 2.0 '[lat, long]' text or [lat, long] lists -> latitude, longitude."""
    txt = s.map(lambda v: f"[{v[0]}, {v[1]}]" if isinstance(v, (list, tuple)) and len(v) == 2 else v).astype("string")
    parts = txt.str.extract(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]")
    return pd.DataFrame({"latitude": pd.to_numeric(parts[0], errors="coerce"),
                         "longitude": pd.to_numeric(parts[1], errors="coerce")}, index=s.index)


def validate_coords(df: pd.DataFrame, lat: str = "latitude", lon: str = "longitude") -> pd.DataFrame:
    df = df.copy()
    df[lat], df[lon] = pd.to_numeric(df[lat], errors="coerce"), pd.to_numeric(df[lon], errors="coerce")
    in_box = lambda a, b: a.between(*LAT_RANGE) & b.between(*LON_RANGE)
    swapped = ~in_box(df[lat], df[lon]) & in_box(df[lon], df[lat])
    df.loc[swapped, [lat, lon]] = df.loc[swapped, [lon, lat]].to_numpy()
    bad = (df[lat].notna() | df[lon].notna()) & ~in_box(df[lat], df[lon])
    df.loc[bad, [lat, lon]] = np.nan
    log.info("Coordinates: swapped %d lat/long pairs, nulled %d outside Kenya (incl. 0,0)", swapped.sum(), bad.sum())
    return df


def latest_export(folder: Path, prefix: str) -> Path:
    """Newest '<prefix>_DDMMYYYY.csv' in folder (by the date in the name), so a fresh pull is picked up."""
    dated = {}
    for p in folder.glob(f"{prefix}_*.csv"):
        m = re.fullmatch(re.escape(prefix) + r"_(\d{8})", p.stem)
        if m:
            dated[p] = pd.to_datetime(m.group(1), format="%d%m%Y", errors="coerce")
    dated = {p: d for p, d in dated.items() if pd.notna(d)}
    if not dated:
        raise FileNotFoundError(f"No {prefix}_DDMMYYYY.csv in {folder}")
    return max(dated, key=dated.get)


def read_raw(path: Path) -> pd.DataFrame:
    """Read everything as text; only truly empty cells are missing ('N/A' stays 'N/A')."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], low_memory=False)
    return df.apply(lambda c: c.str.strip().mask(lambda x: x == ""))


def strip_admin_suffixes(df: pd.DataFrame, county="county", subcounty="subcounty", ward="ward") -> pd.DataFrame:
    """KABS 2.0 'Nandi County' / 'Chesumei Sub County' / 'Kapsabet Ward' -> bare names."""
    df = df.copy()
    for col, pat in [(county, r"\s+County$"), (subcounty, r"\s+Sub[\s-]*County$"), (ward, r"\s+Ward$")]:
        if col and col in df:
            df[col] = df[col].astype("string").str.strip().str.replace(pat, "", regex=True, flags=re.I)
    return df
