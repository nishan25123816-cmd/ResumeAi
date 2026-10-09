"""
app.py
------
Minimal Streamlit Application for Resume Classification
and Information Extraction using Natural Language Processing.
"""

import base64

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np

from src.preprocessing import clean_for_classification, clean_for_extraction
from src.information_extraction import extract_all
from src.pdf_extraction import (
    extract_text_from_file,
    extract_text_from_bytes,
    extract_text_from_image_bytes,
    extract_text_from_docx_bytes,
)
from src.classification import predict_single, load_model, load_label_encoder, load_model_metadata
from src.feature_extraction import load_vectorizer
from src.quality_check import looks_like_low_quality_ocr
from src.recommendations import build_resume_recommendations
from src.utils import models_path

st.set_page_config(
    page_title="ResumeAI · Resume NLP Analysis",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0a0a0a;
    color: #d4d4d4;
  }

  #MainMenu, footer, header { visibility: hidden; }

  .block-container {
    padding-top: 3.5rem !important;
    padding-bottom: 4rem !important;
    max-width: 760px !important;
  }

  /* Header */
  .app-header {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin-bottom: 2.8rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #1f1f1f;
  }
  .app-logo-img {
    width: 48px;
    height: 48px;
    border-radius: 10px;
    object-fit: cover;
    flex-shrink: 0;
  }
  .app-header-text .app-logo {
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #525252;
    margin-bottom: 0.3rem;
  }
  .app-title {
    font-size: 1.95rem;
    font-weight: 600;
    color: #fafafa;
    letter-spacing: -0.03em;
    margin: 0 0 0.4rem 0;
    line-height: 1.25;
  }
  .app-subtitle {
    font-size: 1rem;
    color: #525252;
    margin: 0;
    line-height: 1.6;
  }

  /* Section label */
  .section-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #404040;
    margin-bottom: 0.6rem;
  }

  /* Result block */
  .result-block {
    background: #111111;
    border: 1px solid #1f1f1f;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
  }
  .result-category {
    font-size: 1.7rem;
    font-weight: 600;
    color: #fafafa;
    letter-spacing: -0.02em;
    margin: 0 0 0.15rem 0;
  }
  .result-meta {
    font-size: 0.88rem;
    color: #404040;
  }
  .result-confidence {
    font-size: 0.9rem;
    font-weight: 600;
    color: #a3a3a3;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Info grid */
  .info-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-bottom: 1rem;
  }
  .info-cell {
    background: #111111;
    border: 1px solid #1f1f1f;
    border-radius: 8px;
    padding: 0.9rem 1rem;
  }
  .info-cell-label {
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #404040;
    margin-bottom: 0.25rem;
  }
  .info-cell-value {
    font-size: 0.95rem;
    color: #d4d4d4;
    font-weight: 500;
    word-break: break-all;
  }

  /* Skills */
  .skill-tag {
    display: inline-block;
    background: #161616;
    border: 1px solid #262626;
    color: #a3a3a3;
    padding: 0.3rem 0.75rem;
    border-radius: 5px;
    font-size: 0.86rem;
    font-weight: 500;
    margin: 0.2rem 0.15rem;
  }

  /* Cards & Timeline */
  .item-card {
    background: #121212;
    border: 1px solid #222222;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
    transition: all 0.15s ease;
  }
  .item-card:hover {
    border-color: #333333;
    background: #141414;
  }
  .item-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.6rem;
    flex-wrap: wrap;
  }
  .item-title {
    font-size: 0.98rem;
    font-weight: 600;
    color: #f0f0f0;
    line-height: 1.35;
  }
  .item-sub {
    font-size: 0.88rem;
    color: #a3a3a3;
    margin-top: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .badge-date {
    font-size: 0.76rem;
    color: #f43f5e;
    background: rgba(244, 63, 94, 0.08);
    border: 1px solid rgba(244, 63, 94, 0.25);
    border-radius: 4px;
    padding: 0.15rem 0.55rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    white-space: nowrap;
  }
  .badge-edu {
    font-size: 0.74rem;
    color: #60a5fa;
    background: rgba(96, 165, 250, 0.08);
    border: 1px solid rgba(96, 165, 250, 0.2);
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    font-weight: 500;
    white-space: nowrap;
  }
  .badge-school {
    font-size: 0.74rem;
    color: #a78bfa;
    background: rgba(167, 139, 250, 0.08);
    border: 1px solid rgba(167, 139, 250, 0.2);
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    font-weight: 500;
    white-space: nowrap;
  }

  /* Alternatives */
  .alt-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid #1a1a1a;
  }
  .alt-row:last-child { border-bottom: none; }
  .alt-name {
    font-size: 0.9rem;
    color: #a3a3a3;
    font-weight: 500;
    flex: 1;
  }
  .alt-pct {
    font-size: 0.84rem;
    font-family: 'JetBrains Mono', monospace;
    color: #525252;
    min-width: 42px;
    text-align: right;
  }
  .alt-bar-wrap {
    width: 80px;
    height: 3px;
    background: #1f1f1f;
    border-radius: 2px;
    overflow: hidden;
    margin: 0 0.75rem;
  }
  .alt-bar-fill {
    height: 100%;
    background: #3d3d3d;
    border-radius: 2px;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #1f1f1f;
    padding: 0;
    border-radius: 0;
  }
  .stTabs [data-baseweb="tab"] {
    height: 40px;
    font-size: 0.9rem;
    font-weight: 500;
    color: #525252;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0 1rem;
    border-radius: 0;
    background: transparent !important;
  }
  .stTabs [aria-selected="true"] {
    color: #fafafa !important;
    border-bottom: 2px solid #fafafa !important;
    background: transparent !important;
  }

  /* Uploader */
  div[data-testid="stFileUploader"] {
    background: #111111;
    border: 1px dashed #262626;
    border-radius: 8px;
    padding: 0.5rem;
  }
  div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
    background: #d1d5db !important;
    border: 1px solid #b8bec8 !important;
    color: #687385 !important;
  }
  div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
  div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #687385 !important;
  }
  div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] {
    background: #e5e7eb !important;
    border: 1px solid #c4c9d1 !important;
    color: #687385 !important;
  }
  div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"] {
    color: #687385 !important;
  }

  /* Button */
  div.stButton > button {
    background: #fafafa !important;
    color: #0a0a0a !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1.8rem !important;
    border-radius: 7px !important;
    border: none !important;
    box-shadow: none !important;
    transition: opacity 0.15s ease !important;
    letter-spacing: -0.01em !important;
  }
  div.stButton > button:hover {
    opacity: 0.85 !important;
    transform: none !important;
    box-shadow: none !important;
  }

  /* Textarea */
  .stTextArea textarea {
    background: #111111 !important;
    border: 1px solid #1f1f1f !important;
    border-radius: 8px !important;
    color: #d4d4d4 !important;
    font-size: 0.95rem !important;
  }
  .stTextArea textarea:focus {
    border-color: #3d3d3d !important;
    box-shadow: none !important;
  }

  /* Selectbox */
  div[data-baseweb="select"] > div {
    background: #111111 !important;
    border: 1px solid #1f1f1f !important;
    border-radius: 8px !important;
    color: #d4d4d4 !important;
    font-size: 0.95rem !important;
  }

  /* Footer */
  .app-footer {
    text-align: center;
    color: #2a2a2a;
    font-size: 0.75rem;
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid #141414;
  }

  .mini-divider {
    height: 1px;
    background: #1f1f1f;
    margin: 1.5rem 0;
  }
</style>
""", unsafe_allow_html=True)

# ── Load Models ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_nlp_system():
    meta_path       = models_path("model_metadata.json")
    vec_path        = models_path("tfidf_vectorizer.pkl")
    le_path         = models_path("label_encoder.pkl")

    if not all(os.path.isfile(p) for p in [meta_path, vec_path, le_path]):
        return None, None, None, None
    try:
        meta       = load_model_metadata(meta_path)
        best_model_name = meta.get("best_model_name")
        if not best_model_name:
            return None, None, None, None
        best_model_path = models_path(f"{best_model_name.lower().replace(' ', '_')}.pkl")
        if not os.path.isfile(best_model_path):
            return None, None, None, None
        vectorizer = load_vectorizer(vec_path)
        label_enc  = load_label_encoder(le_path)
        model      = load_model(best_model_path)
        return model, vectorizer, label_enc, meta
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

model, vectorizer, label_enc, meta = load_nlp_system()

# ── Sample Presets ─────────────────────────────────────────────────────────────
SAMPLE_PRESETS = {
    "Select a sample...": "",
    "💻 Software Engineer (ML / Cloud)": """Alex Morgan
alex.morgan@cloudtech.io | +1 (555) 234-5678 | San Francisco, CA
linkedin.com/in/alex-morgan | github.com/alexmorgan

PROFESSIONAL SUMMARY
Senior Machine Learning Engineer with 7+ years architecting distributed cloud systems, deploying deep learning pipelines, and building scalable REST microservices.

EXPERIENCE
Staff ML Engineer | CloudScale Systems (2021 - Present)
- Architected NLP inference engine on AWS EKS serving 15M daily requests.
- Optimized PyTorch transformer latency by 42% with ONNX Runtime.

Senior Software Engineer | DataStream Corp (2018 - 2021)
- Built streaming data pipelines using Kafka, Spark, and PostgreSQL.
- Implemented CI/CD pipelines using Docker, GitHub Actions, and Terraform.

EDUCATION
M.S. Computer Science | Stanford University (2018)
B.S. Software Engineering | UC Berkeley (2016)

SKILLS
Python, Go, C++, SQL, PyTorch, TensorFlow, Scikit-Learn, AWS, Docker, Kubernetes
""",
    "📊 Finance & Banking (Portfolio Analyst)": """Victoria Sterling
v.sterling@capitalwealth.com | +1 (555) 890-1234 | New York, NY

PROFILE
Chartered Financial Analyst (CFA) with 8+ years in portfolio management, risk analysis, and financial modeling across global capital markets.

EXPERIENCE
Senior Portfolio Manager | Meridian Global Capital (2020 - Present)
- Managed $450M multi-asset equity fund achieving 14.2% annualized alpha.
- Conducted DCF valuations and investment committee briefings.

Financial Analyst | JPMorgan Chase (2016 - 2020)
- Built automated financial analysis models in Excel and VBA.
- Conducted credit evaluations and treasury forecasting.

EDUCATION
MBA in Finance | Columbia Business School (2016)
B.S. Economics & Accounting | NYU Stern (2014)

SKILLS
Financial Modeling, DCF Valuation, Risk Management, CFA, Equity Research, Bloomberg
""",
    "🍳 Culinary Arts (Executive Chef)": """Marcus Vance
chef.vance@culinaryarts.com | +1 (555) 432-8765 | Chicago, IL

PROFILE
Executive Chef with 12+ years leading Michelin-starred restaurant kitchens and luxury resort dining operations.

EXPERIENCE
Executive Chef | L'Etoile Fine Dining (2019 - Present)
- Directed kitchen operations and menu creation for 120-seat restaurant.
- Reduced food costs by 7% while maintaining Forbes 5-Star standards.

Head Chef | Grand Hotel & Resort (2014 - 2019)
- Managed banquet catering for 600+ guests and vendor procurement.

EDUCATION
A.A.S. Culinary Arts | The Culinary Institute of America (2012)

SKILLS
Menu Planning, Food Safety, HACCP, Fine Dining, Inventory Control, Recipe Development
"""
}

# ── Header ─────────────────────────────────────────────────────────────────────
_logo_path = os.path.join(APP_DIR, "logo.jpg")
_logo_b64  = ""
if os.path.isfile(_logo_path):
    with open(_logo_path, "rb") as _f:
        _logo_b64 = base64.b64encode(_f.read()).decode()

_logo_html = f"<img src='data:image/jpeg;base64,{_logo_b64}' class='app-logo-img' />" if _logo_b64 else ""

st.markdown(f"""
<div class='app-header'>
  {_logo_html}
  <div class='app-header-text'>
    <div class='app-logo'>ResumeAI</div>
    <h1 class='app-title'>Resume Classification &amp; NLP Analysis</h1>
    <p class='app-subtitle'>Upload or paste a resume to classify its job category and extract structured information.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Input Tabs ─────────────────────────────────────────────────────────────────
tab_upload, tab_paste, tab_examples = st.tabs(["Upload", "Paste Text", "Examples"])

resume_text = ""
source_tag  = ""

with tab_upload:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop a PDF, Word (.docx), TXT, or Image file",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png", "webp", "tiff", "bmp"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        fname     = uploaded_file.name.lower()
        file_type = uploaded_file.type or ""
        is_image  = file_type.startswith("image/") or fname.endswith((".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"))
        is_pdf    = (file_type == "application/pdf") or fname.endswith(".pdf")
        is_docx   = fname.endswith(".docx") or "word" in file_type

        raw_bytes = uploaded_file.read()

        if is_image:
            from PIL import Image
            import io
            try:
                img_preview = Image.open(io.BytesIO(raw_bytes))
                st.image(img_preview, caption=uploaded_file.name, use_container_width=True)
            except Exception:
                pass
            with st.spinner("Scanning and reading text with OCR..."):
                text, status = extract_text_from_image_bytes(raw_bytes)
                tag_label = "Image (OCR)"
        elif is_pdf:
            with st.spinner("Scanning and extracting text from PDF (with OCR fallback)..."):
                text, status = extract_text_from_bytes(raw_bytes)
                tag_label = "PDF Document"
        elif is_docx:
            with st.spinner("Extracting text and tables from Word document..."):
                text, status = extract_text_from_docx_bytes(raw_bytes)
                tag_label = "Word Document (.docx)"
        else:
            with st.spinner("Reading text file..."):
                text, status = extract_text_from_file(raw_bytes, uploaded_file.name)
                tag_label = "Text File"

        if status.startswith("error"):
            st.error(f"Extraction error: {status}")
        elif status.startswith("empty") or not text.strip():
            st.warning("No readable text found in this file. Please make sure the file contains clear text or unblurred scanned content.")
        else:
            resume_text = text
            source_tag  = f"{tag_label} · {uploaded_file.name}"
            word_count = len(text.split())
            st.success(f"✓ Scanned **{word_count} words** from `{uploaded_file.name}`")
            with st.expander("👁️ View Extracted / Scanned Text", expanded=False):
                st.text_area("Scanned Content", value=resume_text, height=180, disabled=True, label_visibility="collapsed")

with tab_paste:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    pasted_text = st.text_area(
        "Paste resume",
        height=220,
        placeholder="Paste resume text here...",
        label_visibility="collapsed",
    )
    if pasted_text.strip() and not resume_text:
        resume_text = pasted_text
        source_tag  = "Pasted text"

with tab_examples:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    selected = st.selectbox("", options=list(SAMPLE_PRESETS.keys()), label_visibility="collapsed")
    if selected and SAMPLE_PRESETS[selected]:
        resume_text = SAMPLE_PRESETS[selected]
        source_tag  = selected

# ── Analyze Button ─────────────────────────────────────────────────────────────
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
analyze_clicked = st.button("Analyze Resume →")

# ── Analysis Results ───────────────────────────────────────────────────────────
if analyze_clicked:
    if not resume_text or not resume_text.strip():
        st.warning("Please provide a resume first.")
    elif looks_like_low_quality_ocr(resume_text):
        st.warning(
            "This resume text looks like low-quality OCR from a blurry or compressed image. "
            "Please upload a clearer PDF or a high-resolution scan and try again."
        )
        st.stop()
    elif len(resume_text.split()) < 15:
        st.warning("Text too short for reliable analysis.")
    elif model is None:
        st.error("Model not found. Run `python train.py` first.")
    else:
        with st.spinner("Analyzing..."):
            clean_text = clean_for_classification(resume_text)
            prediction = predict_single(model, vectorizer, label_enc, clean_text)
            light_text = clean_for_extraction(resume_text)
            extracted  = extract_all(light_text)

        category       = prediction.get("predicted_label", "Unknown")
        confidence     = prediction.get("confidence", 0.0)
        confidence_pct = (confidence * 100) if confidence else 0.0

        st.markdown("<div class='mini-divider'></div>", unsafe_allow_html=True)

        # Prediction result
        st.markdown(
            f"<div class='result-block'>"
            f"<div class='result-meta'>Predicted category</div>"
            f"<div class='result-category'>{category}</div>"
            f"<div style='margin-top:0.5rem'>"
            f"<span class='result-confidence'>{confidence_pct:.1f}% confidence</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Top alternatives
        all_probs = prediction.get("all_probabilities", {})
        if all_probs:
            top_alts = sorted(all_probs.items(), key=lambda x: -x[1])[1:5]
            alt_rows = ""
            for name, prob in top_alts:
                pct     = prob * 100
                bar_pct = int(prob * 100)
                alt_rows += (
                    f"<div class='alt-row'>"
                    f"<span class='alt-name'>{name}</span>"
                    f"<div class='alt-bar-wrap'>"
                    f"<div class='alt-bar-fill' style='width:{bar_pct}%'></div>"
                    f"</div>"
                    f"<span class='alt-pct'>{pct:.1f}%</span>"
                    f"</div>"
                )
            st.markdown(
                f"<div class='result-block' style='padding:1rem 1.4rem;'>"
                f"<div class='section-label' style='margin-bottom:0.5rem;'>Other likely categories</div>"
                f"{alt_rows}"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Result tabs
        t1, t2, t3, t4, t5 = st.tabs(["Profile", "Skills", "Education & Experience", "Recommendations", "Export"])

        with t1:
            name_val  = extracted.get("name")  or "—"
            email_val = extracted.get("email") or "—"
            phone_val = extracted.get("phone") or "—"
            loc_val   = extracted.get("location") or "—"

            st.markdown(
                f"<div class='info-row'>"
                f"<div class='info-cell'><div class='info-cell-label'>Name</div><div class='info-cell-value'>{name_val}</div></div>"
                f"<div class='info-cell'><div class='info-cell-label'>Email</div><div class='info-cell-value'>{email_val}</div></div>"
                f"<div class='info-cell'><div class='info-cell-label'>Phone</div><div class='info-cell-value'>{phone_val}</div></div>"
                f"<div class='info-cell'><div class='info-cell-label'>Location</div><div class='info-cell-value'>{loc_val}</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            links = extracted.get("links", {})
            if isinstance(links, dict):
                if links.get("linkedin"):
                    st.markdown(f"[↗ LinkedIn]({links['linkedin']})")
                if links.get("github"):
                    st.markdown(f"[↗ GitHub]({links['github']})")

        with t2:
            skills = extracted.get("skills", [])
            if skills:
                badges = "".join([f"<span class='skill-tag'>{s}</span>" for s in skills])
                st.markdown(f"<div style='margin-top:0.5rem;line-height:2;'>{badges}</div>", unsafe_allow_html=True)
            else:
                st.caption("No specific skills detected.")

        with t3:
            col_ed, col_exp = st.columns(2)
            with col_ed:
                st.markdown("<div class='section-label'>Education</div>", unsafe_allow_html=True)
                degrees = extracted.get("degrees", [])
                unis    = extracted.get("universities", [])
                if degrees or unis:
                    for d in degrees:
                        badge_label = "Secondary" if "ncea" in d.lower() or "school" in d.lower() else "Degree"
                        badge_cls   = "badge-school" if "ncea" in d.lower() or "school" in d.lower() else "badge-edu"

                        # Separate qualification title and institution if separated by ' at ' or ' — '
                        if " at " in d:
                            d_parts = d.split(" at ", 1)
                            d_title, d_inst = d_parts[0].strip(), d_parts[1].strip()
                            st.markdown(
                                f"<div class='item-card'>"
                                f"<div class='item-header'>"
                                f"<span class='item-title'>{d_title}</span>"
                                f"<span class='{badge_cls}'>{badge_label}</span>"
                                f"</div>"
                                f"<div class='item-sub'>🏛 {d_inst}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        elif " — " in d:
                            d_parts = d.split(" — ", 1)
                            d_title, d_inst = d_parts[0].strip(), d_parts[1].strip()
                            st.markdown(
                                f"<div class='item-card'>"
                                f"<div class='item-header'>"
                                f"<span class='item-title'>{d_title}</span>"
                                f"<span class='{badge_cls}'>{badge_label}</span>"
                                f"</div>"
                                f"<div class='item-sub'>🏛 {d_inst}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"<div class='item-card'>"
                                f"<div class='item-header'>"
                                f"<span class='item-title'>{d}</span>"
                                f"<span class='{badge_cls}'>{badge_label}</span>"
                                f"</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    for u in unis:
                        st.markdown(
                            f"<div class='item-card'>"
                            f"<div class='item-header'>"
                            f"<span class='item-title'>{u}</span>"
                            f"<span class='badge-edu'>Institution</span>"
                            f"</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No degrees detected.")

            with col_exp:
                st.markdown("<div class='section-label'>Experience</div>", unsafe_allow_html=True)
                exp_entries = extracted.get("experience_entries", [])
                titles = extracted.get("job_titles", [])
                orgs   = extracted.get("organizations", [])

                if exp_entries:
                    for entry in exp_entries[:8]:
                        title = entry.get("title") or "Role"
                        company = entry.get("company") or ""
                        period = entry.get("period") or ""

                        date_badge = f"<span class='badge-date'>🗓 {period}</span>" if period else ""
                        comp_html = f"<div class='item-sub'>🏢 {company}</div>" if company else ""

                        st.markdown(
                            f"<div class='item-card'>"
                            f"<div class='item-header'>"
                            f"<span class='item-title'>{title}</span>"
                            f"{date_badge}"
                            f"</div>"
                            f"{comp_html}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                elif titles or orgs:
                    for t in titles[:5]:
                        st.markdown(
                            f"<div class='item-card'>"
                            f"<div class='item-header'><span class='item-title'>{t}</span><span class='badge-edu'>Role</span></div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    for o in orgs[:5]:
                        st.markdown(
                            f"<div class='item-card'>"
                            f"<div class='item-header'><span class='item-title'>{o}</span><span class='badge-edu'>Organization</span></div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No experience detected.")

        with t4:
            recommendations = build_resume_recommendations(extracted, category)
            score = recommendations.get("score", 0)
            job_match = recommendations.get("job_match_score", score)
            rec_list = recommendations.get("recommendations", [])
            missing_sections = recommendations.get("missing_sections", [])
            strengths = recommendations.get("strengths", [])
            keyword_gap = recommendations.get("keyword_gap", [])
            keywords = recommendations.get("keywords", [])
            exact_writing = recommendations.get("exact_writing_suggestions", [])

            st.markdown(
                f"<div class='result-block'>"
                f"<div class='result-meta'>ATS / CV quality score</div>"
                f"<div class='result-category'>{score}/100</div>"
                f"<div class='result-meta' style='margin-top:0.5rem;'>Job match score: {job_match}/100</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("<div class='section-label'>Quick wins</div>", unsafe_allow_html=True)
            for item in rec_list:
                st.markdown(f"- {item}")

            if missing_sections:
                st.markdown("<div class='section-label'>Missing sections</div>", unsafe_allow_html=True)
                st.write(", ".join(missing_sections))

            if strengths:
                st.markdown("<div class='section-label'>Strengths</div>", unsafe_allow_html=True)
                st.write("• " + "\n• ".join(strengths))

            if keywords:
                st.markdown("<div class='section-label'>Best keywords for {}</div>".format(category.upper() if category else "this role"), unsafe_allow_html=True)
                chips = " ".join([f"<span class='skill-tag'>{k}</span>" for k in keywords[:8]])
                st.markdown(f"<div style='margin-top:0.5rem;line-height:2;'>{chips}</div>", unsafe_allow_html=True)

            if keyword_gap:
                st.markdown("<div class='section-label'>Skill gap analyzer</div>", unsafe_allow_html=True)
                st.write(", ".join(keyword_gap))

            if exact_writing:
                st.markdown("<div class='section-label'>Improvement checklist</div>", unsafe_allow_html=True)
                for item in exact_writing:
                    st.markdown(f"- {item}")

            if score < 70:
                st.info("Tip: Add measurable achievements, stronger skills, and clearer job-role keywords to improve recruiter fit.")

        with t5:
            export_payload = {
                "predicted_category": category,
                "confidence_score":   round(confidence, 4),
                "candidate_profile":  extracted,
                "input_source":       source_tag,
                "recommendations":    build_resume_recommendations(extracted, category)
            }
            json_str = json.dumps(export_payload, indent=2)
            st.download_button(
                label="↓ Download JSON",
                data=json_str,
                file_name="resume_analysis.json",
                mime="application/json"
            )
            st.code(json_str, language="json")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='app-footer'>
  Resume NLP Analysis · University Capstone Project · Local Processing
</div>
""", unsafe_allow_html=True)
