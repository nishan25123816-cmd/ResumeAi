"""
feature_extraction.py
----------------------
TF-IDF feature extraction for resume classification.

Design choices:
- Subword n-grams (1,2) capture multi-word technical terms (e.g., "machine learning")
- Sublinear TF scaling helps with the wide range of term frequencies in resumes
- Max features capped at 15,000 to avoid overfitting on sparse categories
- min_df=2 removes hapax legomena that add noise
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os


DEFAULT_VECTORIZER_PARAMS = dict(
    ngram_range=(1, 2),      # unigrams + bigrams
    max_features=15_000,
    sublinear_tf=True,       # replace tf with 1 + log(tf)
    min_df=2,                # ignore terms that appear in fewer than 2 docs
    max_df=0.95,             # ignore terms that appear in >95% of docs (near-constant)
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"[a-zA-Z][a-zA-Z0-9\+\#\.]{0,49}",  # include C++, C#, .NET
)


def build_vectorizer(**kwargs) -> TfidfVectorizer:
    """
    Create a new TF-IDF vectorizer with sensible resume-specific defaults.
    Keyword args override defaults.
    """
    params = {**DEFAULT_VECTORIZER_PARAMS, **kwargs}
    return TfidfVectorizer(**params)


def fit_vectorizer(
    train_texts: list,
    vectorizer: TfidfVectorizer = None,
    **kwargs,
) -> TfidfVectorizer:
    """
    Fit (and return) a TF-IDF vectorizer on training texts.

    Parameters
    ----------
    train_texts : list of str
        Preprocessed resume texts for training.
    vectorizer : TfidfVectorizer, optional
        An existing vectorizer to fit. Creates a new one if None.
    **kwargs
        Passed to build_vectorizer() if creating a new one.

    Returns
    -------
    Fitted TfidfVectorizer.
    """
    if vectorizer is None:
        vectorizer = build_vectorizer(**kwargs)
    vectorizer.fit(train_texts)
    return vectorizer


def transform(texts: list, vectorizer: TfidfVectorizer):
    """Transform texts using a fitted vectorizer. Returns sparse matrix."""
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer: TfidfVectorizer, path: str) -> None:
    """Persist a fitted vectorizer to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path: str) -> TfidfVectorizer:
    """Load a persisted vectorizer from disk."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Vectorizer not found at: {path}")
    return joblib.load(path)
