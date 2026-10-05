"""Text normalisation and data loading.

Original columns are always kept; normalised versions are added as new columns.
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

# Legal-form words carry little identity information -> removed from the "core" name.
LEGAL_SUFFIXES = frozenset({
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "pvt", "private",
    "corp", "corporation", "co", "company", "pc", "plc", "pllc", "sarl", "sasu",
    "sas", "sci", "eurl", "sa", "gmbh", "opc", "ag", "the", "and", "of",
    # common outputs of transliterating "प्राइवेट लिमिटेड" / "एलएलपी" etc.
    "praivet", "piraivet", "elelp", "limitet",
})

ADDRESS_ABBREVIATIONS = {
    "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "rd": "road",
    "dr": "drive", "blvd": "boulevard", "bd": "boulevard", "ln": "lane", "ct": "court",
    "cir": "circle", "hwy": "highway", "pkwy": "parkway", "pl": "place", "sq": "square",
    "ter": "terrace", "trl": "trail", "expy": "expressway", "n": "north", "s": "south",
    "e": "east", "w": "west", "ne": "northeast", "nw": "northwest", "se": "southeast",
    "sw": "southwest", "opp": "opposite", "nr": "near", "flr": "floor", "fl": "floor",
    "mt": "mount", "ft": "fort",
}
# Words that only describe the kind of sub-address and vary between sources.
ADDRESS_NOISE = frozenset({
    "no", "nr", "unit", "apartment", "apt", "ste", "suite", "bldg", "building",
    "flat", "door", "plot", "hno", "city", "of", "village", "town", "region", "the",
})

US_STATE_CODES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california",
    "CO": "colorado", "CT": "connecticut", "DE": "delaware", "FL": "florida", "GA": "georgia",
    "HI": "hawaii", "ID": "idaho", "IL": "illinois", "IN": "indiana", "IA": "iowa",
    "KS": "kansas", "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland",
    "MA": "massachusetts", "MI": "michigan", "MN": "minnesota", "MS": "mississippi",
    "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
    "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york",
    "NC": "north carolina", "ND": "north dakota", "OH": "ohio", "OK": "oklahoma",
    "OR": "oregon", "PA": "pennsylvania", "RI": "rhode island", "SC": "south carolina",
    "SD": "south dakota", "TN": "tennessee", "TX": "texas", "UT": "utah", "VT": "vermont",
    "VA": "virginia", "WA": "washington", "WV": "west virginia", "WI": "wisconsin",
    "WY": "wyoming", "DC": "district of columbia",
}

COUNTRY_ALIASES = {
    "usa": "us", "u s": "us", "u s a": "us", "united states": "us",
    "united states of america": "us", "america": "us",
    "u k": "uk", "united kingdom": "uk", "great britain": "uk", "gb": "uk",
}

# ---------------------------------------------------------------------------
# Indic-script transliteration (rule based, no external library or lookup).
# All major Indic Unicode blocks share the ISCII layout, so any codepoint is
# mapped to the Devanagari block by offset and then romanised.
# ---------------------------------------------------------------------------
_INDIC_BLOCK_STARTS = [0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00]
_INDIC_RE = re.compile("[\u0900-\u0D7F]+")

_VOWELS = {"अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ऌ": "li",
           "ऍ": "e", "ऎ": "e", "ए": "e", "ऐ": "ai", "ऑ": "o", "ऒ": "o", "ओ": "o", "औ": "au"}
_CONSONANTS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh",
               "ज": "j", "झ": "jh", "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh",
               "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "ऩ": "n",
               "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r",
               "ऱ": "r", "ल": "l", "ळ": "l", "ऴ": "l", "व": "v", "श": "sh", "ष": "sh",
               "स": "s", "ह": "h"}
_MATRAS = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "ॅ": "e", "ॆ": "e",
           "े": "e", "ै": "ai", "ॉ": "o", "ॊ": "o", "ो": "o", "ौ": "au"}
_SIGNS = {"ं": "n", "ँ": "n", "ः": "h"}
_HALANT = "्"
_SPECIAL_LATIN = str.maketrans({"ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "đ": "d", "ł": "l"})


def _to_devanagari(ch):
    code = ord(ch)
    for start in _INDIC_BLOCK_STARTS:
        if start <= code < start + 0x80:
            return chr(code - start + 0x0900)
    return ch


def _romanise_run(run):
    """Romanise one contiguous run of Indic characters (word-final schwa dropped)."""
    out, pending_schwa = [], False
    for ch in (_to_devanagari(c) for c in run):
        if ch in _CONSONANTS:
            if pending_schwa:
                out.append("a")
            out.append(_CONSONANTS[ch])
            pending_schwa = True
        elif ch in _MATRAS:
            out.append(_MATRAS[ch])
            pending_schwa = False
        elif ch == _HALANT:
            pending_schwa = False
        elif ch in _VOWELS:
            if pending_schwa:
                out.append("a")
            out.append(_VOWELS[ch])
            pending_schwa = False
        elif ch in _SIGNS:
            if pending_schwa:
                out.append("a")
            out.append(_SIGNS[ch])
            pending_schwa = False
    return "".join(out)


def transliterate_indic(text):
    """Replace Indic-script runs with a romanised approximation."""
    if not _INDIC_RE.search(text):
        return text
    text = _INDIC_RE.sub(lambda m: _romanise_run(m.group()), text)
    return text


def strip_accents(text):
    text = unicodedata.normalize("NFKD", text.translate(_SPECIAL_LATIN))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def _base_normalise(text):
    """NFKC -> transliterate -> strip accents. Case is preserved here."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = transliterate_indic(text)
    return strip_accents(text)


_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co|fr|io|biz|info)"
                     r"(?:\.[a-z]{2})?\b")
_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def clean_text(text):
    """Generic cleaning: lowercase, accents, punctuation and whitespace."""
    text = _base_normalise(text).lower().replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def clean_name(text):
    """Full cleaned business name (web-domains reduced to their name part)."""
    text = _base_normalise(text).lower().replace("&", " and ")
    text = _URL_RE.sub(r"\1", text)
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def core_name_tokens(clean):
    """Name tokens without legal suffixes (falls back to all tokens if nothing is left)."""
    tokens = clean.split()
    core = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return core or tokens


def clean_country(text):
    value = clean_text(text)
    return COUNTRY_ALIASES.get(value, value)


def clean_address(text, country_clean=""):
    """Normalised address: abbreviations expanded, noise words removed, US state
    codes expanded (only for country == 'us', and only when a comma-separated
    segment is exactly the code, to avoid confusing them with ordinary words)."""
    raw = _base_normalise(text)
    segments = []
    for segment in raw.split(","):
        stripped = segment.strip()
        if country_clean == "us" and stripped in US_STATE_CODES:
            stripped = US_STATE_CODES[stripped]
        segments.append(stripped)
    text = " ".join(segments).lower().replace("&", " and ")
    text = _ORDINAL_RE.sub(r"\1", text)
    text = _PUNCT_RE.sub(" ", text)
    tokens = []
    for token in text.split():
        token = ADDRESS_ABBREVIATIONS.get(token, token)
        if token not in ADDRESS_NOISE:
            tokens.append(token)
    return " ".join(tokens)


def preprocess_dataframe(df):
    """Return a copy of ``df`` with normalised columns added (originals kept)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")
    df = df.copy()
    for column in ("business_name", "business_address", "country"):
        df[column] = df[column].fillna("").astype(str)

    df["country_clean"] = df["country"].map(clean_country)
    df["business_name_clean"] = df["business_name"].map(clean_name)
    core_tokens = df["business_name_clean"].map(core_name_tokens)
    df["name_core"] = core_tokens.map(" ".join)
    df["name_core_sorted"] = core_tokens.map(lambda t: " ".join(sorted(t)))
    df["name_squashed"] = core_tokens.map("".join)
    df["business_address_clean"] = [
        clean_address(a, c) for a, c in zip(df["business_address"], df["country_clean"])
    ]
    df["address_sorted"] = df["business_address_clean"].map(lambda a: " ".join(sorted(a.split())))
    return df


def load_tsv(path, limit=None):
    """Read a challenge TSV file (always tab-separated), optionally only the first rows."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df.head(limit).copy() if limit else df


def combine_sources(source2, source3):
    """Stack Source 2 and Source 3 into one table with a ``source`` column."""
    source2 = source2.assign(source="S2")
    source3 = source3.assign(source="S3")
    return pd.concat([source2, source3], ignore_index=True)
