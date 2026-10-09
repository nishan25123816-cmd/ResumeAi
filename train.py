"""
train.py
--------
End-to-end training script for the Resume Classification system.

Usage:
    python train.py

This script:
  1. Loads the Resume.csv dataset
  2. Performs EDA and saves charts
  3. Preprocesses text
  4. Splits data (70% train / 15% val / 15% test) using stratified split
  5. Fits TF-IDF vectorizer
  6. Trains all 4 classification models
  7. Evaluates on validation set — selects best model
  8. Evaluates best model on test set
  9. Saves model, vectorizer, label encoder, and metadata
 10. Saves results (confusion matrix, comparison charts, classification report)

IMPORTANT:
- Random seed 42 is used throughout for reproducibility
- Test set is not touched until final evaluation
- No results are fabricated; all numbers come from actual training
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split

from src.utils import set_seed, data_path, models_path, results_path
from src.preprocessing import clean_for_classification
from src.feature_extraction import fit_vectorizer, transform, save_vectorizer
from src.classification import (
    train_all_models, predict, save_model, save_label_encoder,
    save_model_metadata, get_label_encoder,
)
from src.evaluation import (
    evaluate_model, plot_confusion_matrix, plot_model_comparison,
    plot_multi_metric_comparison, save_classification_report,
    save_comparison_csv,
)

set_seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════
CSV_PATH     = data_path("Resume.csv")
MODELS_DIR   = models_path()
VEC_PATH     = models_path("tfidf_vectorizer.pkl")
LE_PATH      = models_path("label_encoder.pkl")
META_PATH    = models_path("model_metadata.json")
RESULTS_DIR  = results_path()
FIGURES_DIR  = results_path("figures")
REPORT_PATH  = results_path("classification_report.txt")
COMPARISON_PATH = results_path("model_comparison.csv")
CM_PATH      = results_path("figures", "confusion_matrix.png")

os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 60)
print("  RESUME NLP PROJECT — TRAINING PIPELINE")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/10] Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"  Loaded {len(df)} rows × {len(df.columns)} columns")
print(f"  Columns: {list(df.columns)}")

# ── Basic inspection ──────────────────────────────────────────────────────────
print(f"\n  Missing values:\n{df.isnull().sum()}")
print(f"\n  Duplicate rows: {df.duplicated().sum()}")

# Drop duplicates
df.drop_duplicates(inplace=True)
print(f"  After deduplication: {len(df)} rows")

# Drop rows with missing Resume_str or Category
df.dropna(subset=["Resume_str", "Category"], inplace=True)
df["Resume_str"] = df["Resume_str"].astype(str)
df["Category"] = df["Category"].astype(str).str.strip()

print(f"  Final dataset size: {len(df)} rows")
print(f"  Number of categories: {df['Category'].nunique()}")
print(f"\n  Category distribution:\n{df['Category'].value_counts().to_string()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS — Charts
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/10] Generating EDA charts...")

# 2a. Category distribution
fig, ax = plt.subplots(figsize=(14, 7))
cat_counts = df["Category"].value_counts()
colors = plt.cm.tab20(np.linspace(0, 1, len(cat_counts)))
bars = ax.barh(cat_counts.index, cat_counts.values, color=colors)
ax.set_xlabel("Number of Resumes", fontsize=12)
ax.set_title("Resume Category Distribution", fontsize=14, fontweight="bold")
for bar, v in zip(bars, cat_counts.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            str(v), va="center", fontsize=9)
ax.invert_yaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "category_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ Saved category_distribution.png")

# 2b. Resume length distribution
df["word_count"] = df["Resume_str"].apply(lambda x: len(x.split()))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["word_count"], bins=50, color="#2E86AB", edgecolor="white", alpha=0.85)
axes[0].set_xlabel("Word Count")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Resume Length Distribution (Words)", fontweight="bold")
axes[0].axvline(df["word_count"].mean(), color="red", linestyle="--", label=f"Mean: {df['word_count'].mean():.0f}")
axes[0].legend()
axes[1].boxplot([df[df["Category"] == c]["word_count"].values
                  for c in cat_counts.index],
                 labels=cat_counts.index, vert=False, patch_artist=True)
axes[1].set_xlabel("Word Count")
axes[1].set_title("Word Count by Category", fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "resume_length_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ Saved resume_length_distribution.png")

# 2c. Top words (after basic cleaning, not full preprocessing)
from collections import Counter
import re

def quick_tokenize(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return [w.lower() for w in text.split() if len(w) > 3]

all_words = []
for text in df["Resume_str"].sample(min(500, len(df)), random_state=42):
    all_words.extend(quick_tokenize(text))

from nltk.corpus import stopwords
import nltk
nltk.download("stopwords", quiet=True)
stop = set(stopwords.words("english"))
word_freq = Counter(w for w in all_words if w not in stop)
top_words = word_freq.most_common(30)

fig, ax = plt.subplots(figsize=(14, 6))
words, freqs = zip(*top_words)
ax.bar(words, freqs, color=plt.cm.viridis(np.linspace(0.2, 0.9, len(words))))
ax.set_xlabel("Word", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Top 30 Most Frequent Words in Resumes", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "top_words.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  ✓ Saved top_words.png")

# Print stats
print(f"\n  ── Resume Length Statistics ──")
print(f"  Average words: {df['word_count'].mean():.0f}")
print(f"  Median  words: {df['word_count'].median():.0f}")
print(f"  Min     words: {df['word_count'].min()}")
print(f"  Max     words: {df['word_count'].max()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/10] Preprocessing text for classification...")

df["clean_text"] = df["Resume_str"].apply(clean_for_classification)

# Filter out empty texts after cleaning
empty_mask = df["clean_text"].str.strip() == ""
print(f"  Resumes with empty text after cleaning: {empty_mask.sum()}")
df = df[~empty_mask].reset_index(drop=True)
print(f"  Usable resumes: {len(df)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. LABEL ENCODING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/10] Encoding labels...")
label_encoder = get_label_encoder(sorted(df["Category"].unique()))
df["label"] = label_encoder.transform(df["Category"])
print(f"  Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. STRATIFIED TRAIN / VAL / TEST SPLIT (70 / 15 / 15)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/10] Splitting data (70% train / 15% val / 15% test, stratified)...")
X = df["clean_text"].values
y = df["label"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"  Train : {len(X_train)} samples")
print(f"  Val   : {len(X_val)} samples")
print(f"  Test  : {len(X_test)} samples")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. TF-IDF VECTORISATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6/10] Fitting TF-IDF vectorizer on training data only...")
vectorizer = fit_vectorizer(X_train)
X_train_tfidf = vectorizer.transform(X_train)
X_val_tfidf   = vectorizer.transform(X_val)
X_test_tfidf  = vectorizer.transform(X_test)
print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"  Feature matrix shape (train): {X_train_tfidf.shape}")

save_vectorizer(vectorizer, VEC_PATH)
print(f"  ✓ Vectorizer saved → {VEC_PATH}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. TRAIN ALL MODELS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[7/10] Training models...")
trained_models = train_all_models(X_train_tfidf, y_train)

# ═══════════════════════════════════════════════════════════════════════════════
# 8. VALIDATION EVALUATION — SELECT BEST MODEL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[8/10] Evaluating models on VALIDATION set...")
val_results = {}
for name, model in trained_models.items():
    metrics, _ = evaluate_model(model, X_val_tfidf, y_val, label_encoder, split_name="Validation")
    val_results[name] = metrics
    print(f"  {name}: Accuracy={metrics['accuracy']:.4f}  Weighted-F1={metrics['weighted_f1']:.4f}  Macro-F1={metrics['macro_f1']:.4f}")

best_name = max(val_results, key=lambda n: val_results[n]["weighted_f1"])
best_model = trained_models[best_name]
print(f"\n  ★ Best model on validation: {best_name} (Weighted-F1={val_results[best_name]['weighted_f1']:.4f})")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. FINAL TEST EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[9/10] Evaluating BEST model on TEST set (first and only time)...")
test_metrics, y_test_pred = evaluate_model(
    best_model, X_test_tfidf, y_test, label_encoder, split_name="Test"
)
print(f"  Test Accuracy   : {test_metrics['accuracy']:.4f}")
print(f"  Test Weighted-F1: {test_metrics['weighted_f1']:.4f}")
print(f"  Test Macro-F1   : {test_metrics['macro_f1']:.4f}")
print(f"\n  Classification Report:\n{test_metrics['classification_report']}")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. SAVE EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[10/10] Saving models and results...")

# Save all models
for name, model in trained_models.items():
    save_model(model, MODELS_DIR, name)

# Save label encoder
save_label_encoder(label_encoder, LE_PATH)

# Save model metadata
metadata = {
    "best_model_name": best_name,
    "best_model_file": os.path.join(MODELS_DIR, f"{best_name.lower().replace(' ', '_')}.pkl"),
    "tfidf_vectorizer_file": VEC_PATH,
    "label_encoder_file": LE_PATH,
    "n_classes": int(len(label_encoder.classes_)),
    "classes": list(label_encoder.classes_),
    "n_train": int(len(X_train)),
    "n_val": int(len(X_val)),
    "n_test": int(len(X_test)),
    "validation_results": {
        name: {k: v for k, v in m.items() if k != "classification_report"}
        for name, m in val_results.items()
    },
    "test_results": {k: v for k, v in test_metrics.items() if k != "classification_report"},
    "random_seed": 42,
}
save_model_metadata(metadata, META_PATH)
print(f"  ✓ Metadata saved → {META_PATH}")

# Save classification report
save_classification_report(test_metrics["classification_report"], REPORT_PATH)
print(f"  ✓ Classification report saved → {REPORT_PATH}")

# Save model comparison CSV
val_results_no_report = {
    name: {k: v for k, v in m.items() if k != "classification_report"}
    for name, m in val_results.items()
}
save_comparison_csv(val_results_no_report, COMPARISON_PATH)
print(f"  ✓ Model comparison CSV saved → {COMPARISON_PATH}")

# Save confusion matrix
classes_list = list(label_encoder.classes_)
plot_confusion_matrix(
    y_test, y_test_pred,
    classes=classes_list,
    title=f"Confusion Matrix — {best_name} (Test Set)",
    save_path=CM_PATH,
)
print(f"  ✓ Confusion matrix saved → {CM_PATH}")

# Save model comparison charts
comp_fig_path = os.path.join(FIGURES_DIR, "model_comparison.png")
plot_model_comparison(
    val_results_no_report,
    metric="weighted_f1",
    title="Model Comparison — Weighted F1 (Validation Set)",
    save_path=comp_fig_path,
)
print(f"  ✓ Model comparison chart saved → {comp_fig_path}")

multi_comp_path = os.path.join(FIGURES_DIR, "model_comparison_multi.png")
plot_multi_metric_comparison(
    val_results_no_report,
    save_path=multi_comp_path,
)
print(f"  ✓ Multi-metric comparison chart saved → {multi_comp_path}")

print("\n" + "=" * 60)
print(f"  TRAINING COMPLETE!")
print(f"  Best model: {best_name}")
print(f"  Test Accuracy:    {test_metrics['accuracy']:.4f}")
print(f"  Test Weighted-F1: {test_metrics['weighted_f1']:.4f}")
print("=" * 60)
