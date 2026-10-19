import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "finetune"))
from uz_text import has_cyrillic, normalize_uz  # noqa: E402


def test_apostrophe_variants_are_unified():
    variants = ["Oʻzbekiston", "O‘zbekiston", "O’zbekiston", "O`zbekiston", "O'zbekiston"]
    assert {normalize_uz(v) for v in variants} == {"o'zbekiston"}


def test_punctuation_removed_but_in_word_marks_kept():
    assert normalize_uz("Salom, do‘stlar! Ma’no — bu “ilm-fan”.") == "salom do'stlar ma'no bu ilm-fan"


def test_cyrillic_detection():
    assert has_cyrillic("Салом дунё")
    assert not has_cyrillic("salom dunyo")

def test_cyrillic_is_transliterated_to_uzbek_latin():
    assert normalize_uz("Просто, вообще!") == "prosto voobshe"
    assert normalize_uz("Ўзбекистон ғалаба қилди") == "o'zbekiston g'alaba qildi"
    assert normalize_uz("Ер, поезд, объект") == "yer poyezd ob'yekt"
    assert normalize_uz("koroche bu ishni qildim, понял?") == "koroche bu ishni qildim ponyal"