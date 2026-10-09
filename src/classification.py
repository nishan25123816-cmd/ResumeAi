"""
classification.py
-----------------
Model training, selection, and prediction for resume classification.

Models implemented:
  1. Logistic Regression
  2. Multinomial Naive Bayes
  3. Linear SVM (LinearSVC)
  4. Random Forest

All models use TF-IDF features produced by feature_extraction.py.
Hyperparameters are tuned for text classification tasks.
"""

import os
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder


# ── Model definitions ─────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "Logistic Regression": LogisticRegression(
        C=5.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    ),
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Linear SVM": CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, random_state=42),
        cv=3,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    ),
}


def get_label_encoder(labels: list) -> LabelEncoder:
    """Fit and return a LabelEncoder on the provided labels."""
    le = LabelEncoder()
    le.fit(labels)
    return le


def train_model(name: str, X_train, y_train):
    """
    Train a single model by name.

    Parameters
    ----------
    name : str
        One of the keys in MODEL_CONFIGS.
    X_train : sparse matrix
        TF-IDF feature matrix.
    y_train : array-like
        Encoded integer labels.

    Returns
    -------
    Trained estimator.
    """
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_CONFIGS)}")

    model = MODEL_CONFIGS[name]
    # Handle NB: needs non-negative TF-IDF (sublinear_tf=True can produce negatives? No — log(1+0)=0, always ≥0)
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train, y_train) -> dict:
    """
    Train all models. Returns a dict {name: trained_model}.
    """
    trained = {}
    for name in MODEL_CONFIGS:
        print(f"  Training {name}...")
        trained[name] = train_model(name, X_train, y_train)
    return trained


def predict(model, X) -> np.ndarray:
    """Return integer label predictions."""
    return model.predict(X)


def predict_proba(model, X) -> np.ndarray:
    """Return class probability estimates (requires predict_proba support)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    raise AttributeError(f"Model {type(model).__name__} does not support predict_proba")


def predict_single(model, vectorizer, label_encoder, text: str) -> dict:
    """
    Predict the category of a single preprocessed text string.

    Parameters
    ----------
    model       : fitted sklearn estimator
    vectorizer  : fitted TfidfVectorizer
    label_encoder : fitted LabelEncoder
    text        : preprocessed resume text

    Returns
    -------
    dict with keys: predicted_label, confidence, all_probabilities
    """
    if not text or not text.strip():
        return {
            "predicted_label": "Unknown",
            "category": "Unknown",
            "confidence": 0.0,
            "all_probabilities": {},
            "top_3": [("Unknown", 0.0)],
            "error": "Empty text after preprocessing.",
        }

    X = vectorizer.transform([text])
    pred_int = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_int])[0]

    result = {
        "predicted_label": pred_label,
        "category": pred_label,
        "confidence": None,
        "all_probabilities": {},
        "top_3": [(pred_label, 1.0)],
    }

    try:
        proba = model.predict_proba(X)[0]
        classes = label_encoder.classes_
        result["confidence"] = float(np.max(proba))
        prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}
        result["all_probabilities"] = prob_dict
        result["top_3"] = sorted(prob_dict.items(), key=lambda x: -x[1])[:3]
    except (AttributeError, Exception):
        # Some models (LinearSVC without calibration) don't support predict_proba
        pass

    return result


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model, model_dir: str, model_name: str) -> str:
    """Save a model to disk. Returns the saved file path."""
    os.makedirs(model_dir, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(model_dir, f"{safe_name}.pkl")
    joblib.dump(model, path)
    return path


def load_model(path: str):
    """Load a persisted model from disk."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    return joblib.load(path)


def save_label_encoder(le: LabelEncoder, path: str) -> None:
    """Persist label encoder."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(le, path)


def load_label_encoder(path: str) -> LabelEncoder:
    """Load a persisted label encoder."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Label encoder not found at: {path}")
    return joblib.load(path)


def save_model_metadata(metadata: dict, path: str) -> None:
    """Save model metadata as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_model_metadata(path: str) -> dict:
    """Load model metadata from JSON."""
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)
