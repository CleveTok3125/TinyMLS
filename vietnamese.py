import re
import unicodedata
from typing import List

VOWELS = "aeiouyàáãạảăắằẵặẳâấầẫậẩèéẽẹẻêếềễệểìíĩịỉòóõọỏôốồỗộổơớờỡợởùúũụủưứừữựửỳýỹỵỷ"
TONED_VOWELS = "àáãạảắằẵặẳấầẫậẩèéẽẹẻếềễệểìíĩịỉòóõọỏốồỗộổớờỡợởùúũụủứừữựửỳýỹỵỷ"
VIETNAMESE_CHARS = "a-zA-Zàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"

_INITIALS = r"(ch|gh|gi|kh|ngh|ng|nh|ph|qu|th|tr|b|c|d|đ|g|h|k|l|m|n|p|r|s|t|v|x)"
_FINALS = r"(ch|ng|nh|c|m|n|p|t)"
_SYLLABLE_PATTERN = re.compile(rf"^{_INITIALS}?([{VOWELS}]+){_FINALS}?$")
_WORD_RE = re.compile(f"[{VIETNAMESE_CHARS}]+")

_TONE_MAP = {
    "a": "àáảãạ",
    "ă": "ằắẳẵặ",
    "â": "ầấẩẫậ",
    "e": "èéẻẽẹ",
    "ê": "ềếểễệ",
    "i": "ìíỉĩị",
    "o": "òóỏõọ",
    "ô": "ồốổỗộ",
    "ơ": "ờớởỡợ",
    "u": "ùúủũụ",
    "ư": "ừứửữự",
    "y": "ỳýỷỹỵ",
}


def _toned_variations(base_char: str) -> str:
    return _TONE_MAP.get(base_char, "")


def is_valid_word(word: str) -> bool:
    word = unicodedata.normalize("NFC", word)
    return word.isalpha() and 1 <= len(word) <= 15


def is_valid_vietnamese_syllable(word: str) -> bool:
    word = unicodedata.normalize("NFC", word)
    if not (1 <= len(word) <= 7):
        return False
    tone_count = sum(1 for char in word if char in TONED_VOWELS)
    if tone_count > 1:
        return False
    match = _SYLLABLE_PATTERN.match(word)
    if not match:
        return False
    initial = match.group(1) or ""
    vowel_part = match.group(2)
    if len(vowel_part) > 3:
        return False
    front_vowel_base = "eêiy"
    first_v_char = vowel_part[0]
    is_front_vowel = any(
        first_v_char in _toned_variations(base) for base in front_vowel_base
    )
    if initial in {"gh", "ngh", "k"} and not is_front_vowel:
        return False
    if initial in {"g", "ng", "c"} and is_front_vowel:
        return False
    return True


_SENTENCE_SEP_RE = re.compile(r'[.,!?;:()\[\]{}""\'\n\r\t\-]+')


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SEP_RE.split(text) if s.strip()]


def tokenize_words(text: str) -> List[str]:
    return [m.group().lower() for m in _WORD_RE.finditer(text)]


def extract_sentences_with_words(text: str) -> List[List[str]]:
    result = []
    for sent in split_sentences(text):
        current_seq: List[str] = []
        for w in sent.split():
            w = w.lower()
            if is_valid_word(w):
                current_seq.append(w)
            else:
                if current_seq:
                    result.append(current_seq)
                    current_seq = []
        if current_seq:
            result.append(current_seq)
    return result


def extract_ngrams(words: List[str], min_n: int = 1, max_n: int = 3) -> List[str]:
    result = []
    for n in range(min_n, max_n + 1):
        for i in range(len(words) - n + 1):
            result.append(" ".join(words[i : i + n]))
    return result
