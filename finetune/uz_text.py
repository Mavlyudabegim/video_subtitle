"""Uzbek text normalization, shared by data prep, labeling, training and evaluation.

Conventions (one script for everything, so the model learns ONE spelling):
  * Output is always Uzbek Latin, lowercase, no punctuation.
  * Cyrillic (Uzbek Cyrillic or Russian words) is transliterated to Uzbek Latin:
      "просто" -> "prosto", "вообще" -> "voobshe", "ўзбек" -> "o'zbek"
  * All apostrophe look-alikes (ʻ ʼ ‘ ’ ` ´) become ASCII "'"  (o', g', tutuq belgisi).
  * Dialect words are kept as spoken (no normalization to literary Uzbek).
"""
import re
import unicodedata

_APOSTROPHES = "\u02bb\u02bc\u2018\u2019\u0060\u00b4\u02b9\u2032'"
_APOS_RE = re.compile(f"[{re.escape(_APOSTROPHES)}]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")

# Uzbek Cyrillic + Russian letters -> Uzbek Latin (official 1995 alphabet conventions)
_CYR2LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ё": "yo", "ж": "j", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x", "ц": "ts", "ч": "ch",
    "ш": "sh", "щ": "sh", "ъ": "'", "ы": "i", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "ў": "o'", "қ": "q", "ғ": "g'", "ҳ": "h",
}
_VOWELS_CYR = set("аеёиоуыэюяў")


def cyr_to_lat(text: str) -> str:
    """Transliterate Cyrillic letters to Uzbek Latin (expects lowercase input)."""
    out = []
    for i, ch in enumerate(text):
        if ch == "е":
            prev = text[i - 1] if i > 0 else " "
            # "e" -> "ye" at word start or after a vowel / ъ / ь  (ер -> yer, поезд -> poyezd)
            out.append("ye" if (not prev.isalpha() or prev in _VOWELS_CYR or prev in "ъь") else "e")
        else:
            out.append(_CYR2LAT.get(ch, ch))
    return "".join(out)


def normalize_uz(text: str) -> str:
    """Lowercase, Cyrillic -> Latin, unify apostrophes, drop punctuation, keep in-word ' and -."""
    text = unicodedata.normalize("NFC", text).lower()
    text = cyr_to_lat(text)
    text = _APOS_RE.sub("'", text)
    text = re.sub(r"[^\w\s'\-]", " ", text)          # remove punctuation
    text = re.sub(r"(?<!\w)['\-]|['\-](?!\w)", " ", text)  # ' or - not inside a word
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def has_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text))