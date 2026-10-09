# Resume Classification and Information Extraction using NLP

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?logo=scikit-learn)
![spaCy](https://img.shields.io/badge/spaCy-3.4+-09A3D5?logo=spacy)
![Streamlit](https://img.shields.io/badge/Streamlit-1.15+-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-Academic-green)

**An intelligent NLP-based Resume Analysis System that classifies resumes into professional categories and extracts structured information.**

</div>

---

## 📋 Project Overview

This project implements a complete **Natural Language Processing (NLP)** pipeline that:
1. **Classifies** resumes into one of 24 professional categories using TF-IDF + Machine Learning
2. **Extracts** structured information (name, email, skills, education, experience, etc.) using a hybrid rule-based + NLP approach
3. **Provides** a professional Streamlit web interface for demonstration

---

## 🎯 Problem Statement

Recruiters and HR professionals process thousands of resumes manually. This system automates the process of:
- Categorising resumes by profession/domain
- Extracting key candidate information in a structured format
- Providing confidence scores for classification decisions

---

## 🎓 Objectives

1. Build an accurate resume classification system using TF-IDF and multiple ML models
2. Develop a hybrid information extraction pipeline without fabricating entity annotations
3. Evaluate and compare 4 classification algorithms (LR, NB, SVM, RF)
4. Deploy a professional web interface for demonstration
5. Document the methodology for academic submission

---

## 📊 Dataset Description

| Property | Value |
|----------|-------|
| Source | Kaggle Resume Dataset |
| File | `Resume.csv` |
| Total Resumes | 2,484 |
| Categories | 24 |
| Columns | ID, Resume_str, Resume_html, Category |
| PDF Resumes | 2,484 (organised by category) |

Resume CSV and PDF files contain personal information and are not included in this repository. Download the dataset from its approved source and place `Resume.csv` in `data/` before training. Generated model artifacts are also excluded; run `python train.py` locally after providing the dataset.

### Categories (24)
ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION, BANKING,
BPO, BUSINESS-DEVELOPMENT, CHEF, CONSULTANT, CONSTRUCTION, DESIGNER, DIGITAL-MEDIA,
ENGINEERING, FINANCE, FITNESS, HEALTHCARE, HR, INFORMATION-TECHNOLOGY,
PUBLIC-RELATIONS, SALES, TEACHER

### Dataset Statistics
| Statistic | Value |
|-----------|-------|
| Average words per resume | ~400–600 |
| Min words | ~20 |
| Max words | ~3,000+ |
| Missing values in Resume_str | 0 |
| Missing values in Category | 0 |
| Duplicates | minimal |

---

## 🤖 NLP Techniques

| Technique | Usage |
|-----------|-------|
| HTML Removal | Clean Resume_html column |
| Unicode Normalisation | Handle special characters |
| Tokenisation | Word-level NLTK tokenisation |
| Stopword Removal | NLTK English stopwords |
| Lemmatisation | WordNet lemmatiser |
| TF-IDF (1,2)-grams | Feature extraction for classification |
| Regex | Email, phone, URL, degree extraction |
| spaCy NER | PERSON, ORG, GPE entities (pretrained) |
| Section-based parsing | Skills, education, experience sections |
| Keyword matching | Curated 80+ skill keyword list |

---

## 🧠 Machine Learning Models

| Model | Description |
|-------|-------------|
| Logistic Regression | Multinomial LR with L2 regularisation (C=5) |
| Multinomial Naive Bayes | α=0.1 smoothing |
| Linear SVM | LinearSVC with Platt calibration for probabilities |
| Random Forest | 200 trees, no depth limit |

**Feature extraction**: TF-IDF with (1,2)-grams, 15,000 features, sublinear TF scaling

**Selection criterion**: Best Weighted F1-score on validation set

---

## 🏗️ System Architecture

```
Resume PDF / Resume Text
        ↓
Text Extraction (PyMuPDF)
        ↓
Text Cleaning & Preprocessing
        ↓
    ┌───┴────────────────────┐
    ↓                        ↓
Information              TF-IDF
Extraction            Vectorisation
    ↓                        ↓
Name, Email          ML Classifier
Phone, Location             ↓
Skills, Degrees         Category
Experience           Prediction +
Organizations         Confidence
    ↓                        ↓
Structured Output ←──────────┘
        ↓
     Streamlit
        ↓
   User Interface
```

---

## 📁 Project Structure

```
resume-nlp-project/
├── data/
│   ├── Resume.csv               ← Main dataset (CSV)
│   └── pdf_resumes/             ← PDF resumes organised by category
│       ├── ACCOUNTANT/
│       ├── INFORMATION-TECHNOLOGY/
│       └── ... (24 categories)
│
├── notebooks/
│   └── resume_nlp_analysis.ipynb  ← Complete Jupyter notebook
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          ← Text cleaning pipeline
│   ├── feature_extraction.py     ← TF-IDF vectorisation
│   ├── classification.py         ← ML models, training, prediction
│   ├── information_extraction.py ← Hybrid extraction (regex + spaCy + rules)
│   ├── pdf_extraction.py         ← PDF text extraction (PyMuPDF)
│   ├── evaluation.py             ← Metrics, plots, reports
│   └── utils.py                  ← Shared utilities
│
├── models/
│   ├── logistic_regression.pkl   ← Trained classifier
│   ├── naive_bayes.pkl           ← Trained classifier
│   ├── linear_svm.pkl            ← Trained classifier
│   ├── random_forest.pkl         ← Trained classifier
│   ├── tfidf_vectorizer.pkl      ← Fitted TF-IDF vectorizer
│   ├── label_encoder.pkl         ← Category label encoder
│   └── model_metadata.json       ← Best model and training results
│
├── app/
│   └── app.py                    ← Streamlit web application
│
├── results/
│   ├── figures/
│   │   ├── category_distribution.png
│   │   ├── resume_length_distribution.png
│   │   ├── top_words.png
│   │   ├── confusion_matrix.png
│   │   ├── model_comparison.png
│   │   └── model_comparison_multi.png
│   ├── classification_report.txt
│   └── model_comparison.csv
│
├── train.py                      ← End-to-end training script
├── requirements.txt
├── README.md
└── report.md                     ← Academic report
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- Miniconda or Anaconda (recommended) OR system Python with pip
- g++ / gcc (required for spaCy)

```bash
# 1. Install system build tools (Ubuntu/Debian)
sudo apt-get install -y g++ gcc build-essential

# 2. Clone / navigate to project directory
cd /path/to/resume-nlp-project

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download spaCy English model
python -m spacy download en_core_web_sm

# 5. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 🏋️ How to Train

```bash
# From the project root directory
python train.py
```

This will:
1. Load and inspect the dataset
2. Generate EDA charts in `results/figures/`
3. Train all 4 models
4. Save the best model, vectorizer, and label encoder to `models/`
5. Generate confusion matrix and evaluation reports in `results/`

Training typically takes **2–5 minutes** depending on hardware.

---

## 📈 How to Evaluate

Evaluation is performed automatically during training. After training, check:

- `results/classification_report.txt` — per-class precision, recall, F1
- `results/model_comparison.csv` — all model scores
- `results/figures/confusion_matrix.png` — normalised confusion matrix
- `results/figures/model_comparison.png` — model comparison bar chart
- `models/model_metadata.json` — best model name and test results

---

## 🌐 How to Run the Streamlit App

```bash
# From the project root directory
streamlit run app/app.py
```

Then open your browser at: **http://localhost:8501**

**Before running the app, you must first run `python train.py` to generate the trained models.**

---

## 💡 Example Usage

### Training
```bash
python train.py
# Output:
# [1/10] Loading dataset...
#   Loaded 2484 rows × 4 columns
#   ...
# ★ Best model on validation: Linear SVM (Weighted-F1=0.9XXX)
# Test Accuracy: 0.9XXX
```

### Web App
1. Open browser: `http://localhost:8501`
2. Upload a PDF resume or paste text
3. Click "Analyze Resume"
4. View predicted category and extracted information

---

## 📊 Experimental Results

All numbers below were generated by running `train.py` on the provided 2,484 resume dataset. No metrics are fabricated.

### Model Comparison (Validation Set, $N=372$)

| Model | Accuracy | Macro F1 | Weighted F1 | Weighted Precision | Weighted Recall |
|---|---|---|---|---|---|
| **Logistic Regression** | 64.25% | 0.5825 | 0.6275 | 0.6427 | 0.6425 |
| **Multinomial Naive Bayes** | 54.57% | 0.4811 | 0.5183 | 0.5681 | 0.5457 |
| **Linear SVM** | 69.35% | 0.6658 | 0.6872 | 0.7011 | 0.6935 |
| **Random Forest** | **71.24%** | **0.6389** | **0.6894** | **0.7134** | **0.7124** |

### Best Model Final Evaluation (Held-Out Test Set, $N=373$)

- **Selected Model**: Random Forest Classifier ($n=200$ estimators)
- **Test Accuracy**: **77.48%** (0.7748)
- **Test Weighted F1-Score**: **75.45%** (0.7545)
- **Test Macro F1-Score**: **70.95%** (0.7095)
- **Top performing categories**: Construction (F1=0.91), Accountant (F1=0.88), Designer (F1=0.88), Aviation (F1=0.86), Business-Development (F1=0.85), Teacher (F1=0.83), Engineering (F1=0.82), Sales (F1=0.82), HR (F1=0.81), Information-Technology (F1=0.80).
- **Most challenging categories**: BPO (F1=0.00, only 3 test samples), Automobile (F1=0.33), Arts (F1=0.32).

---

## ⚠️ Limitations

1. **NER Extraction**: Uses pretrained spaCy (general English), not fine-tuned on resumes. Some fields may not be detected accurately.
2. **PDF OCR**: Image-only PDFs (scanned resumes) cannot be processed without OCR.
3. **Name Detection**: Challenging without explicit "Name:" label in resume.
4. **Class Imbalance**: BPO category has only 22 resumes vs 120 for IT.
5. **Language**: System only supports English-language resumes.
6. **Skills List**: Curated skill list may not cover all domain-specific tools.

---

## 🔮 Future Work

1. Fine-tune a transformer model (BERT/RoBERTa) for classification
2. Integrate OCR (Tesseract) for image-based PDFs
3. Build a custom NER model trained on manually annotated resumes
4. Add multilingual support
5. Implement active learning to continuously improve with user feedback
6. Add resume ranking/scoring functionality
7. Integrate with job description matching

---

## 🔒 Privacy

- All resume processing is performed **locally** on the machine
- No resume content is sent to external APIs or servers
- Uploaded files are not permanently stored
- The spaCy model runs entirely offline

---

## 📚 Key References

- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.
- Honnibal, M., & Montani, I. (2017). *spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing*.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
- Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513–523.
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

---

## 📄 License

This project is created for academic purposes. Dataset credits to the original Kaggle dataset contributors.

---

*University NLP Project — Resume Classification and Information Extraction*
