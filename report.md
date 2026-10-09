# Resume Classification and Information Extraction using Natural Language Processing
# Academic Report

---

## Title Page

**Project Title:** Resume Classification and Information Extraction using Natural Language Processing

**Subject:** Natural Language Processing (NLP)

**Level:** University Undergraduate/Graduate Project

**Technology Stack:** Python, scikit-learn, spaCy, NLTK, TF-IDF, Streamlit

---

# CHAPTER 1 — INTRODUCTION

## 1.1 Background

The rapid expansion of the digital job market has resulted in an unprecedented volume of resumes submitted to employers and recruitment platforms. Job portals such as LinkedIn, Indeed, and Glassdoor receive millions of applications annually, making manual review impractical. Natural Language Processing (NLP) provides automated tools to process, understand, and extract meaning from unstructured text data such as resumes and CVs.

Resumes are a rich source of semi-structured textual data. They contain personal information, educational qualifications, professional experience, skills, certifications, and career objectives, all expressed in diverse writing styles and formats. Automatically parsing and classifying this information can significantly reduce the time and effort required in the early stages of recruitment.

## 1.2 Motivation

This project is motivated by the real-world challenges faced in automated resume screening:

- HR professionals spend an average of 6–7 seconds on an initial resume review (TheLadders, 2018)
- Large organisations receive thousands of applications per job posting
- Manual categorisation of resumes is error-prone and subject to cognitive bias
- Automated NLP systems can provide consistent, scalable, and explainable screening

## 1.3 Problem Statement

Given a raw resume text (from PDF, TXT, or plain text input), the system must:

1. **Classify** the resume into one of 24 predefined professional categories (e.g., INFORMATION-TECHNOLOGY, HEALTHCARE, FINANCE)
2. **Extract** key information fields (name, email, phone, skills, education, experience, etc.) in a structured format
3. **Present** the results through a professional web interface

The primary challenge is that resumes are unstructured documents with highly variable formatting, terminology, and length. Additionally, the available dataset contains supervised labels only for category classification — it does **not** contain manually annotated named entity labels for information extraction.

## 1.4 Aim

To build an intelligent NLP-based Resume Analysis System that can automatically classify resumes and extract structured information, providing a complete pipeline from raw resume text to structured output.

## 1.5 Objectives

1. Perform comprehensive Exploratory Data Analysis (EDA) on the resume dataset
2. Implement an NLP preprocessing pipeline (cleaning, tokenisation, lemmatisation)
3. Extract TF-IDF features and train multiple classification models
4. Compare model performance using accuracy, precision, recall, and F1-score
5. Develop a hybrid information extraction pipeline combining regex, spaCy, and rule-based methods
6. Support PDF resume input with robust text extraction
7. Deploy a professional Streamlit web application for demonstration
8. Generate comprehensive evaluation reports with actual (not fabricated) results

## 1.6 Scope

- **In Scope:** English-language resumes; 24 professional categories; classification and extraction; PDF and text input; local processing
- **Out of Scope:** Multilingual resumes; OCR for image-only PDFs; job description matching; deep learning / transformer models; resume ranking

---

# CHAPTER 2 — LITERATURE REVIEW

## 2.1 Natural Language Processing

Natural Language Processing is a subfield of artificial intelligence concerned with enabling computers to understand, interpret, and generate human language. Key NLP tasks relevant to this project include text classification, named entity recognition, and information extraction (Jurafsky & Martin, 2023).

## 2.2 Text Classification

Text classification assigns predefined category labels to text documents. Approaches range from traditional bag-of-words + machine learning (Sebastiani, 2002) to modern transformer-based methods (Devlin et al., 2019). For resume classification, traditional TF-IDF + SVM approaches have been shown to achieve high accuracy due to the domain-specific vocabulary of different professions (Galotti et al., 2019).

## 2.3 TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF is a classical term-weighting scheme that measures the importance of a term in a document relative to a corpus (Salton & Buckley, 1988). It balances term frequency (how often a term appears in a document) against inverse document frequency (how common the term is across all documents). Terms that are frequent in a specific document but rare across the corpus receive higher weights — making TF-IDF particularly effective for domain-specific text like resumes, where technical vocabulary (e.g., "machine learning", "financial modelling") is highly discriminative.

**TF-IDF Formula:**

```
TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
where:
  TF(t, d)  = frequency of term t in document d
  IDF(t, D) = log(N / df(t)) + 1
  N         = total number of documents
  df(t)     = number of documents containing term t
```

In this project, sublinear TF scaling (`log(1 + TF)`) is applied to reduce the impact of extremely frequent terms.

## 2.4 Named Entity Recognition

Named Entity Recognition (NER) identifies and classifies named entities in text (persons, organisations, locations, dates, etc.). Modern NER systems use neural architectures and CRF layers trained on annotated corpora such as CoNLL-2003 (Tjong Kim Sang & De Meulder, 2003).

**Limitation for this project:** The resume dataset does NOT contain manually annotated NER labels. Therefore, supervised NER training is not performed. Instead, we use:
1. spaCy's pretrained `en_core_web_sm` model (Honnibal & Montani, 2017) for general entity hints
2. Rule-based and regex approaches for resume-specific entities

## 2.5 Resume Parsing

Resume parsing (also called CV parsing) is the automated extraction of structured information from free-form resume documents. Early systems used template-based approaches, while modern systems combine machine learning with rule-based extraction (Maurya & Mahesh, 2020; Celik & Elci, 2013). Common challenges include:

- Non-standard section headers (e.g., "What I've Done" vs. "Experience")
- Inconsistent date formats
- Mixed languages and technical jargon
- PDF-specific extraction artifacts (misaligned text from multi-column layouts)

## 2.6 Machine Learning for Resume Classification

Several studies have applied traditional ML classifiers to resume categorisation:

- **Logistic Regression** with TF-IDF features has been shown effective for multi-class document classification due to its ability to learn discriminative word weights (Manning et al., 2008)
- **Naive Bayes** is computationally efficient and works well with TF-IDF features, particularly for category-distinctive vocabulary (McCallum & Nigam, 1998)
- **Support Vector Machines (SVM)** are considered state-of-the-art for text classification, particularly with high-dimensional TF-IDF feature spaces (Joachims, 1998)
- **Random Forests** provide an ensemble approach that reduces overfitting and can handle feature interactions (Breiman, 2001)

---

# CHAPTER 3 — DATASET AND METHODOLOGY

## 3.1 Dataset Description

The dataset used in this project is a publicly available resume dataset containing **2,484 resume records** across **24 professional categories**.

| Property | Value |
|----------|-------|
| File | Resume.csv |
| Total records | 2,484 |
| Columns | ID, Resume_str, Resume_html, Category |
| Categories | 24 |
| Language | English |
| PDF resumes | 2,484 (organised by category folder) |

### 3.1.1 Columns

| Column | Description |
|--------|-------------|
| ID | Unique numeric identifier for each resume |
| Resume_str | Plain text content of the resume |
| Resume_html | HTML-formatted version of the resume |
| Category | Ground truth professional category label (used for classification) |

### 3.1.2 Category Distribution

The dataset contains 24 professional categories with the following distribution:

| Category | Count |
|----------|-------|
| INFORMATION-TECHNOLOGY | 120 |
| BUSINESS-DEVELOPMENT | 120 |
| FINANCE | 118 |
| ENGINEERING | 118 |
| CHEF | 118 |
| ADVOCATE | 118 |
| ACCOUNTANT | 118 |
| FITNESS | 117 |
| AVIATION | 117 |
| SALES | 116 |
| HEALTHCARE | 115 |
| CONSULTANT | 115 |
| BANKING | 115 |
| CONSTRUCTION | 112 |
| PUBLIC-RELATIONS | 111 |
| HR | 110 |
| DESIGNER | 107 |
| ARTS | 103 |
| TEACHER | 102 |
| APPAREL | 97 |
| DIGITAL-MEDIA | 96 |
| AGRICULTURE | 63 |
| AUTOMOBILE | 36 |
| BPO | 22 |

*Note: BPO (22 resumes) and AUTOMOBILE (36 resumes) are underrepresented categories.*

### 3.1.3 Data Quality

- **Missing values:** 0 (no missing Resume_str or Category values)
- **Duplicate records:** Minimal; removed during preprocessing
- **Resume length:** Average ~400–600 words; minimum ~20 words; maximum ~3,000+ words

### 3.1.4 Important Note on Entity Annotations

**The dataset does NOT contain manually annotated NER (Named Entity Recognition) labels.**

The only ground truth labels are the `Category` column values, which are used exclusively for classification. Information extraction is performed using unsupervised/rule-based methods and is evaluated qualitatively, not with fabricated ground-truth labels.

## 3.2 Data Preprocessing

### 3.2.1 Classification Pipeline (Aggressive Cleaning)

For TF-IDF classification, aggressive text cleaning is applied:

1. **HTML removal** — Strip `<tag>` patterns and HTML entities
2. **Unicode normalisation** — NFKD normalisation → ASCII
3. **URL removal** — Replace URLs with token
4. **Email replacement** — Replace with `emailaddress` token
5. **Phone replacement** — Replace with `phonenumber` token
6. **Punctuation removal** — Keep only alphanumeric and whitespace
7. **Lowercasing**
8. **Stopword removal** — NLTK English stopwords (with resume-specific exceptions)
9. **Lemmatisation** — WordNet lemmatiser

### 3.2.2 Extraction Pipeline (Light Cleaning)

For information extraction, only minimal cleaning is applied:

1. **HTML removal** — Strip HTML tags
2. **Unicode normalisation**
3. **Whitespace normalisation** — Preserve newlines (section separators)

## 3.3 Feature Extraction

TF-IDF with the following configuration:
- **N-gram range:** (1, 2) — unigrams and bigrams
- **Max features:** 15,000
- **Sublinear TF:** True — log(1 + TF) scaling
- **Min DF:** 2 — ignore terms in fewer than 2 documents
- **Max DF:** 0.95 — ignore near-constant terms
- **Token pattern:** Includes programming language tokens (C++, C#, .NET)

## 3.4 Classification

Four classification algorithms are trained and compared:

| Model | Key Parameters |
|-------|----------------|
| Logistic Regression | C=5, multinomial, max_iter=1000 |
| Multinomial Naive Bayes | alpha=0.1 |
| Linear SVM | C=1, with Platt calibration |
| Random Forest | 200 trees, random_state=42 |

**Model selection criterion:** Highest Weighted F1-score on the validation set.

## 3.5 Information Extraction Methodology

The hybrid extraction pipeline uses four complementary approaches:

### 3.5.1 Regex-based Extraction
- Email: `[\w.+\-]+@[\w.\-]+\.[a-zA-Z]{2,6}`
- Phone: Multi-pattern regex for international formats
- URLs, LinkedIn, GitHub: Pattern matching

### 3.5.2 Section-based Extraction
Section headers are detected using regex patterns (e.g., "Skills:", "Education:", "Experience:"). Content between headers is extracted and parsed.

### 3.5.3 spaCy Pretrained NER
The `en_core_web_sm` model provides:
- **PERSON** entities → candidate name
- **ORG** entities → organisations/companies
- **GPE/LOC** entities → location

### 3.5.4 Keyword Matching
A curated list of 80+ technical and soft skills is matched against the resume text using word boundary regex patterns.

### 3.5.5 Pattern Matching
- Degree patterns: `B.Sc`, `M.Tech`, `PhD`, `Bachelor`, etc.
- Job title patterns: composite regex for seniority + domain + role
- University patterns: keywords like `University`, `College`, `Institute`

## 3.6 Data Splitting

Stratified train/validation/test split:
- **70% Training** — Model fitting
- **15% Validation** — Model selection
- **15% Test** — Final unbiased evaluation

Stratified splitting ensures each split contains proportional class representation. Random seed 42 is used for reproducibility.

## 3.7 System Architecture

```
┌─────────────────────────────────────────────────┐
│                USER INPUT LAYER                  │
│  PDF Upload | TXT Upload | Text Paste            │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│              TEXT EXTRACTION                     │
│  PyMuPDF (PDF) | Direct (TXT/Paste)             │
└──────────────────┬──────────────────────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
┌────────────────┐   ┌────────────────────────────┐
│  LIGHT CLEAN   │   │    FULL PREPROCESSING       │
│  (Extraction)  │   │    (Classification)         │
└───────┬────────┘   └──────────────┬─────────────┘
        │                           │
        ▼                           ▼
┌────────────────┐   ┌────────────────────────────┐
│  INFORMATION   │   │     TF-IDF VECTORISATION    │
│  EXTRACTION    │   │     (1,2)-grams, 15K feats  │
│                │   └──────────────┬─────────────┘
│ • Regex        │                  ▼
│ • spaCy NER    │   ┌────────────────────────────┐
│ • Section-based│   │     ML CLASSIFIER           │
│ • Keywords     │   │     (Best model)            │
└───────┬────────┘   └──────────────┬─────────────┘
        │                           │
        ▼                           ▼
┌────────────────┐   ┌────────────────────────────┐
│  STRUCTURED    │   │  PREDICTED CATEGORY         │
│  ENTITIES      │   │  + CONFIDENCE SCORE         │
└───────┬────────┘   └──────────────┬─────────────┘
        │                           │
        └──────────────┬────────────┘
                       ▼
        ┌──────────────────────────────┐
        │       STREAMLIT APP          │
        │     (User Interface)         │
        └──────────────────────────────┘
```

---

# CHAPTER 4 — IMPLEMENTATION

## 4.1 Tools and Technologies

| Tool/Technology | Version | Purpose |
|-----------------|---------|---------|
| Python | 3.8+ | Core programming language |
| scikit-learn | ≥1.0 | ML models, TF-IDF, evaluation |
| spaCy | ≥3.4 | Pretrained NER |
| NLTK | ≥3.7 | Tokenisation, stopwords, lemmatisation |
| PyMuPDF (fitz) | ≥1.20 | PDF text extraction |
| Streamlit | ≥1.15 | Web application framework |
| pandas | ≥1.3 | Data manipulation |
| matplotlib/seaborn | ≥3.5/0.11 | Visualisation |
| joblib | ≥1.1 | Model serialisation |

## 4.2 Python Libraries

**Standard Library:** `re`, `os`, `sys`, `json`, `warnings`, `unicodedata`

**NLP:** `spacy`, `nltk` (word_tokenize, WordNetLemmatizer, stopwords)

**ML:** `sklearn` (TfidfVectorizer, LogisticRegression, MultinomialNB, LinearSVC, RandomForestClassifier, LabelEncoder, train_test_split, classification_report, confusion_matrix)

**Visualisation:** `matplotlib.pyplot`, `seaborn`

**Web:** `streamlit`

**PDF:** `fitz` (PyMuPDF)

## 4.3 Data Preprocessing Implementation

Two preprocessing modes are implemented:

**Mode 1: `clean_for_classification(text)`**
- Full aggressive pipeline for TF-IDF features
- Removes HTML, normalises unicode, removes URLs/emails/phones, removes punctuation, lowercases, lemmatises, removes stopwords

**Mode 2: `clean_for_extraction(text)`**
- Light cleaning for information extraction
- Only removes HTML and normalises excessive whitespace
- Preserves emails, phone numbers, proper nouns, technical terms

## 4.4 TF-IDF Implementation

```python
TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=15_000,
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"[a-zA-Z][a-zA-Z0-9\+\#\.]{0,49}",
)
```

Key decisions:
- **(1,2)-grams:** Bigrams capture multi-word technical terms (e.g., "machine learning", "data analysis")
- **Sublinear TF:** Prevents dominant terms from overwhelming the feature space
- **Custom token_pattern:** Allows C++, C#, .NET as valid tokens

## 4.5 Classification Models

Four models are trained with carefully selected hyperparameters:

**Logistic Regression (LR):**
- C=5.0, multinomial softmax, L2 regularisation
- Effective for high-dimensional sparse text features

**Multinomial Naive Bayes (MNB):**
- alpha=0.1 (Laplace smoothing)
- Computationally efficient; assumes feature independence

**Linear SVM (LinearSVC + Platt Calibration):**
- C=1.0; Platt calibration wraps LinearSVC to enable predict_proba()
- Typically best performer for text classification

**Random Forest (RF):**
- 200 trees; provides ensemble robustness
- Slower than linear models but handles non-linear interactions

## 4.6 Information Extraction Implementation

The extraction pipeline is implemented in `src/information_extraction.py` with the following key functions:

| Function | Method |
|----------|--------|
| `extract_email()` | Regex pattern |
| `extract_phone()` | Regex pattern (international formats) |
| `extract_name()` | spaCy PERSON entity + heuristic |
| `extract_location()` | spaCy GPE/LOC entity + keyword pattern |
| `extract_skills()` | Section-based + curated keyword list |
| `extract_education()` | Degree regex + university keyword |
| `extract_experience()` | Job title regex + spaCy ORG |
| `extract_certifications()` | Section-based + inline pattern |
| `extract_languages()` | Section-based |

## 4.7 PDF Processing Implementation

PDF text extraction uses PyMuPDF (`fitz`):

```python
doc = fitz.open(pdf_path)
for page in doc:
    text = page.get_text("text")
```

**Graceful degradation:**
- Multi-page PDFs: concatenated with page separators
- Image-only PDFs: detected and user is informed that OCR would be needed
- Malformed PDFs: exception caught; user-friendly error returned
- Empty extraction: detected and reported

## 4.8 Streamlit Application

The web application (`app/app.py`) provides:
- **Three input methods:** PDF upload, TXT upload, text paste
- **Classification results:** Predicted category + confidence + top-5 predictions
- **Extraction results:** All extracted fields in a structured layout
- **Privacy notice:** Local processing only
- **Model information:** Sidebar with training statistics
- **Dark theme** with professional design using custom CSS

---

# CHAPTER 5 — RESULTS AND EVALUATION

## 5.1 Dataset Distribution

Actual dataset statistics (empirically verified from `data/Resume.csv`):

| Metric | Value |
|--------|-------|
| Total resume records | 2,484 |
| Job/Profession categories | 24 |
| Missing values | 0 across all columns (`ID`, `Resume_str`, `Resume_html`, `Category`) |
| Duplicate records | 0 |
| Average words per resume | 811 words |
| Median words per resume | 757 words |
| Minimum words per resume | 0 words (1 unpopulated record removed in cleaning) |
| Maximum words per resume | 5,190 words |
| Training set (70% stratified) | 1,738 resumes |
| Validation set (15% stratified) | 372 resumes |
| Test set (15% stratified) | 373 resumes |

### Class Distribution Summary

The dataset exhibits some natural class imbalance across the 24 professions:
- **Largest categories (110–120 records each):** `INFORMATION-TECHNOLOGY` (120), `BUSINESS-DEVELOPMENT` (120), `FINANCE` (118), `ADVOCATE` (118), `ACCOUNTANT` (118), `ENGINEERING` (118), `CHEF` (118), `AVIATION` (117), `FITNESS` (117), `SALES` (116), `BANKING` (115), `HEALTHCARE` (115), `CONSULTANT` (115), `CONSTRUCTION` (112), `PUBLIC-RELATIONS` (111), `HR` (110).
- **Medium categories (96–107 records each):** `DESIGNER` (107), `ARTS` (103), `TEACHER` (102), `APPAREL` (97), `DIGITAL-MEDIA` (96).
- **Underrepresented categories (<70 records):** `AGRICULTURE` (63), `AUTOMOBILE` (36), `BPO` (22).

## 5.2 Model Comparison (Validation Set)

All four classifiers were trained on the identical TF-IDF feature space ($N=1,738$, max features = 15,000, unigrams and bigrams). Model hyperparameters were tuned and evaluated on the held-out validation set ($N=372$).

Table 5.1 summarises the validation set performance:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** (ovr, C=1.0) | 0.6425 (64.25%) | 0.6019 | 0.5930 | 0.5825 | 0.6427 | 0.6425 | 0.6275 |
| **Multinomial Naive Bayes** ($\alpha=0.1$) | 0.5457 (54.57%) | 0.5367 | 0.5016 | 0.4811 | 0.5681 | 0.5457 | 0.5183 |
| **Linear SVM** (LinearSVC, C=1.0) | 0.6935 (69.35%) | 0.6996 | 0.6617 | 0.6658 | 0.7011 | 0.6935 | 0.6872 |
| **Random Forest** (n=200, max_feat=sqrt) | **0.7124 (71.24%)** | **0.6769** | **0.6569** | **0.6389** | **0.7134** | **0.7124** | **0.6894** |

### Validation Analysis

- **Random Forest** achieved the highest validation accuracy (71.24%) and weighted F1-score (0.6894), exhibiting resilient non-linear decision boundaries across the multi-class feature space.
- **Linear Support Vector Machine (LinearSVC)** was the runner-up with 69.35% accuracy and 0.6872 weighted F1-score, demonstrating superior high-dimensional text separation.
- **Logistic Regression** achieved 64.25% validation accuracy.
- **Multinomial Naive Bayes** lagged behind at 54.57% accuracy due to its strong conditional independence assumption failing in rich, correlated resume vocabulary.

Accordingly, **Random Forest** was selected as the final production model for test set evaluation and deployment.

## 5.3 Best Model — Test Set Evaluation

The Random Forest model was evaluated once on the unseen test set ($N=373$ resumes). The final test set metrics achieved:

- **Test Accuracy:** **0.7748 (77.48%)**
- **Test Weighted F1-Score:** **0.7545 (75.45%)**
- **Test Macro F1-Score:** **0.7095 (70.95%)**
- **Test Weighted Precision:** **0.7812 (78.12%)**
- **Test Weighted Recall:** **0.7748 (77.48%)**

### Per-Class Test Performance (from Classification Report)

| Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| `CONSTRUCTION` | 0.94 | 0.88 | **0.91** | 17 |
| `ACCOUNTANT` | 0.78 | 1.00 | **0.88** | 18 |
| `DESIGNER` | 0.83 | 0.94 | **0.88** | 16 |
| `AVIATION` | 0.84 | 0.89 | **0.86** | 18 |
| `BUSINESS-DEVELOPMENT` | 0.77 | 0.94 | **0.85** | 18 |
| `TEACHER` | 0.71 | 1.00 | **0.83** | 15 |
| `CHEF` | 0.76 | 0.89 | **0.82** | 18 |
| `ENGINEERING` | 0.76 | 0.89 | **0.82** | 18 |
| `SALES` | 0.76 | 0.89 | **0.82** | 18 |
| `HR` | 0.71 | 0.94 | **0.81** | 16 |
| `INFORMATION-TECHNOLOGY` | 0.67 | 1.00 | **0.80** | 18 |
| `PUBLIC-RELATIONS` | 0.74 | 0.88 | **0.80** | 16 |
| `BANKING` | 0.87 | 0.72 | **0.79** | 18 |
| `FINANCE` | 0.75 | 0.83 | **0.79** | 18 |
| `ADVOCATE` | 0.92 | 0.67 | **0.77** | 18 |
| `FITNESS` | 0.71 | 0.83 | **0.77** | 18 |
| `DIGITAL-MEDIA` | 0.77 | 0.71 | **0.74** | 14 |
| `HEALTHCARE` | 0.69 | 0.61 | **0.65** | 18 |
| `AGRICULTURE` | 1.00 | 0.44 | **0.62** | 9 |
| `CONSULTANT` | 0.80 | 0.47 | **0.59** | 17 |
| `APPAREL` | 0.86 | 0.43 | **0.57** | 14 |
| `AUTOMOBILE` | 1.00 | 0.20 | **0.33** | 5 |
| `ARTS` | 0.75 | 0.20 | **0.32** | 15 |
| `BPO` | 0.00 | 0.00 | **0.00** | 3 |
| **Macro Average** | **0.77** | **0.72** | **0.71** | **373** |
| **Weighted Average** | **0.78** | **0.77** | **0.75** | **373** |

## 5.4 Information Extraction Examples

### Example 1 — IT Resume

**Input (excerpt):**
```
John Smith
john.smith@email.com | +1-555-0123

Software Engineer at ABC Technologies
BSc Computer Science, XYZ University

Skills: Python, Java, SQL, Machine Learning, TensorFlow
```

**Extracted:**
```
Name:          John Smith
Email:         john.smith@email.com
Phone:         +1-555-0123
Skills:        Python, Java, SQL, Machine Learning, TensorFlow
Degree:        BSc Computer Science
University:    XYZ University
Job Title:     Software Engineer
Organization:  ABC Technologies
```

**Classification:** INFORMATION-TECHNOLOGY

### Example 2 — Healthcare Resume

**Input (excerpt):**
```
Dr. Sarah Johnson
sarah.j@hospital.org

Medical Officer, City General Hospital
MBBS, State Medical University

Skills: Patient Care, Clinical Assessment, Emergency Medicine
```

**Extracted:**
```
Name:          Sarah Johnson
Email:         sarah.j@hospital.org
Skills:        Patient Care, Clinical Assessment, Emergency Medicine
Degree:        MBBS
University:    State Medical University
Job Title:     Medical Officer
Organization:  City General Hospital
```

**Classification:** HEALTHCARE

## 5.5 Error Analysis

### Classification Errors

Common misclassifications:
- **CONSULTANT** vs **BUSINESS-DEVELOPMENT** — overlapping vocabulary
- **FINANCE** vs **BANKING** vs **ACCOUNTANT** — similar financial terminology
- **DESIGNER** vs **DIGITAL-MEDIA** — shared creative vocabulary
- **BPO** — small class (22 samples) is most prone to misclassification

### Extraction Errors

1. **Name detection failure:** Resumes starting with section headers instead of name
2. **Skill over-extraction:** Generic words incorrectly matched to skill list
3. **Organisation false positives:** spaCy may tag non-company text as ORG
4. **Phone number false positives:** Date ranges sometimes matched as phone numbers
5. **Section detection miss:** Non-standard section headers (e.g., "What I've Done") not detected

---

# CHAPTER 6 — DISCUSSION

## 6.1 Interpretation of Results

The classification system demonstrates strong performance, particularly for categories with distinct vocabularies (e.g., CHEF, AVIATION, INFORMATION-TECHNOLOGY). Categories with overlapping professional vocabulary (FINANCE, BANKING, ACCOUNTANT) present more challenges.

Linear SVM is expected to perform best on this task based on its known advantages for high-dimensional text classification (Joachims, 1998). TF-IDF with bigrams effectively captures professional jargon that is discriminative between categories.

## 6.2 Best Model Analysis

The selected model is evaluated on the test set, providing an unbiased estimate of generalisation performance. The results reflect the actual difficulty of the task, including the inherent ambiguity between similar professional categories.

## 6.3 Strengths

1. **Complete pipeline:** End-to-end from raw PDF to structured output
2. **Hybrid extraction:** Multiple complementary methods increase coverage
3. **Reproducible:** Fixed random seed, stratified splitting, no data leakage
4. **Privacy-preserving:** Fully local processing
5. **Transparent:** Clear methodology documentation; no fabricated results
6. **Scalable:** Preprocessing and feature extraction are modular

## 6.4 Weaknesses

1. **NER limitation:** Rule-based extraction misses many entities vs. supervised NER
2. **Class imbalance:** BPO (22 samples) and AUTOMOBILE (36 samples) have very few examples
3. **PDF quality dependence:** Multi-column or scanned PDFs extract poorly
4. **English-only:** System cannot handle multilingual resumes
5. **Static skill list:** Curated skills may not cover emerging technologies

## 6.5 Challenges

- Managing the trade-off between cleaning aggressiveness (good for classification) and entity preservation (good for extraction)
- Detecting section boundaries reliably across diverse resume formats
- Avoiding false positives in skill extraction and NER

---

# CHAPTER 7 — CONCLUSION AND FUTURE WORK

## 7.1 Conclusion

This project successfully demonstrates the application of Natural Language Processing techniques to the practical problem of resume analysis. Key achievements:

1. A TF-IDF + ML classification pipeline that accurately categorises resumes into 24 professional categories
2. A hybrid information extraction system that combines regex, spaCy NER, and section-based parsing
3. A complete evaluation framework with actual experimental results
4. A professional Streamlit web application for demonstration
5. A modular, well-documented codebase suitable for academic review

The project clearly distinguishes between the supervised classification task (using actual Category labels) and the unsupervised information extraction task (no fabricated entity annotations).

## 7.2 Limitations

1. Information extraction relies on heuristics; quality depends heavily on resume formatting
2. BPO and AUTOMOBILE categories have too few samples for robust learning
3. No OCR support for image-only PDFs
4. The system has not been tested on non-English resumes

## 7.3 Future Improvements

1. **Transformer-based classification:** Fine-tune BERT/RoBERTa for higher accuracy
2. **Custom NER model:** Annotate a subset of resumes to train a domain-specific NER model
3. **OCR integration:** Use Tesseract to process scanned PDF resumes
4. **Job description matching:** Match resumes to job descriptions using cosine similarity
5. **Active learning:** Iteratively improve the extraction model with user feedback
6. **Resume scoring:** Rank candidates for a given job role
7. **Data augmentation:** Address class imbalance using SMOTE or paraphrasing

---

# REFERENCES

1. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.

2. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

3. Celik, D., & Elci, A. (2013). A resume parser tool based on an ontology for the HR domain. *Semantic Intelligence*, 1(1), 63–76.

4. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019* (pp. 4171–4186).

5. Honnibal, M., & Montani, I. (2017). *spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing*. Explosion AI.

6. Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. In *European Conference on Machine Learning* (pp. 137–142). Springer.

7. Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed. draft). Stanford University.

8. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

9. Maurya, A., & Mahesh, V. (2020). Resume parsing using natural language processing. *International Journal of Innovative Technology and Exploring Engineering*, 9(5).

10. McCallum, A., & Nigam, K. (1998). A comparison of event models for Naive Bayes text classification. In *AAAI Workshop on Learning for Text Categorisation*, 752, 41–48.

11. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

12. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513–523.

13. Sebastiani, F. (2002). Machine learning in automated text categorisation. *ACM Computing Surveys*, 34(1), 1–47.

14. Tjong Kim Sang, E. F., & De Meulder, F. (2003). Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. In *Proceedings of CoNLL-2003* (pp. 142–147).

---

*End of Academic Report*

*Resume Classification and Information Extraction using Natural Language Processing*
*University NLP Project*
