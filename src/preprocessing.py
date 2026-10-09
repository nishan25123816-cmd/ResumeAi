"""
preprocessing.py
----------------
NLP text preprocessing pipeline for resume classification and information extraction.

Key design choices:
- Two modes: 'classification' (aggressive cleaning for ML) and 'extraction' (light cleaning to preserve named entities)
- HTML removal uses regex to avoid BeautifulSoup dependency at runtime
- Reproducible via stateless functions
"""

import re
import unicodedata
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ── NLTK downloads (silent if already present) ───────────────────────────────
def _ensure_nltk():
    for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

_ensure_nltk()

_LEMMATIZER = WordNetLemmatizer()
_STOP_WORDS = set(stopwords.words("english"))

# Words that are meaningful in resumes and should NOT be removed as stopwords
_RESUME_KEEP_WORDS = {
    "not", "no", "nor", "both", "either", "neither",
    "c", "r",  # programming languages often single-letter
}
_EFFECTIVE_STOP_WORDS = _STOP_WORDS - _RESUME_KEEP_WORDS


# ── HTML/XML removal ─────────────────────────────────────────────────────────
_HTML_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)
_HTML_ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;|&\w+;")


def remove_html(text: str) -> str:
    """Strip HTML/XML tags and decode common HTML entities."""
    text = _HTML_TAG_RE.sub(" ", text)
    text = _HTML_ENTITY_RE.sub(" ", text)
    return text


# ── Unicode normalisation ─────────────────────────────────────────────────────
def normalize_unicode(text: str) -> str:
    """Normalise unicode characters to their closest ASCII equivalents."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


# ── Whitespace cleaning ───────────────────────────────────────────────────────
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


# ── URL / e-mail removal (classification mode) ───────────────────────────────
_URL_RE = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
_EMAIL_RE = re.compile(r"[\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,}")


def remove_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


def remove_emails(text: str) -> str:
    return _EMAIL_RE.sub(" EMAIL ", text)


# ── Number normalisation ──────────────────────────────────────────────────────
_PHONE_RE = re.compile(
    r"(\+?\d[\d\s\-\(\)]{7,}\d)"
)
_NUMBER_RE = re.compile(r"\b\d+\b")


def normalize_numbers(text: str) -> str:
    """Replace standalone numbers with PHONE or NUMBER token (classification mode)."""
    text = _PHONE_RE.sub(" PHONE ", text)
    text = _NUMBER_RE.sub(" NUMBER ", text)
    return text


# ── Punctuation ───────────────────────────────────────────────────────────────
_PUNCT_RE = re.compile(r"[^a-zA-Z0-9\s]")


def remove_punctuation(text: str) -> str:
    return _PUNCT_RE.sub(" ", text)


# ── Tokenisation & lemmatisation ─────────────────────────────────────────────
def tokenize(text: str) -> list:
    return word_tokenize(text.lower())


def lemmatize_tokens(tokens: list) -> list:
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in _EFFECTIVE_STOP_WORDS and len(t) > 1]


# ── Main pipelines ────────────────────────────────────────────────────────────

def clean_for_classification(text: str) -> str:
    """
    Full cleaning pipeline for TF-IDF / ML classification.
    Aggressively normalises text to improve feature quality.
    Does NOT need to preserve exact entities.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = remove_html(text)
    text = normalize_unicode(text)
    text = remove_urls(text)
    # Keep e-mail token for classification signal, then remove
    text = _EMAIL_RE.sub(" emailaddress ", text)
    text = _PHONE_RE.sub(" phonenumber ", text)
    text = remove_punctuation(text)
    text = text.lower()
    text = _NUMBER_RE.sub(" ", text)
    text = normalize_whitespace(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


def clean_for_extraction(text: str) -> str:
    """
    Light cleaning pipeline for information extraction.
    Preserves e-mails, phone numbers, names, skill tokens, etc.
    Only removes HTML markup and normalises whitespace.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = remove_html(text)
    # Normalise unicode but keep readable characters
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    # Collapse excessive whitespace but keep newlines (section separators)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_text_stats(text: str) -> dict:
    """Return basic statistics for a resume text string."""
    words = text.split()
    return {
        "char_count": len(text),
        "word_count": len(words),
        "line_count": text.count("\n"),
    }
