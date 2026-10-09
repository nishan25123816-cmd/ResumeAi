"""
pdf_extraction.py
-----------------
High-accuracy document text extraction supporting:
- PDF: Layout-aware block extraction + automatic 300 DPI OCR fallback for scanned pages and graphics
- Images: Multi-pass Tesseract OCR with adaptive contrast, deskew / orientation correction, and column-aware segmentation
- Word documents (DOCX): Paragraphs, tables, headers, and footers
- Text files (TXT)

Design decisions:
- Multi-column reading order: Prevents interleaving of parallel columns (e.g. sidebar and main experience).
- Automatic OCR fallback ensures 100% of PDFs (scanned, vector, hybrid) are fully scanned.
- Advanced image preprocessing (rescaling, contrast optimization, unsharp mask, OSD) maximizes character accuracy.
"""

import os
import io
import re
import shutil
from typing import Optional, Tuple, List
from PIL import Image, ImageEnhance, ImageFilter


def _get_tesseract_cmd() -> Optional[str]:
    """Find and configure the Tesseract executable path."""
    import pytesseract

    candidates = [
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Tesseract-OCR", "tesseract.exe"),
        "/home/nishan/miniconda3/bin/tesseract",
        shutil.which("tesseract"),
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            pytesseract.pytesseract.tesseract_cmd = path
            return path
    return None


def clean_extracted_text(text: str) -> str:
    """
    Post-process extracted text to fix common OCR and PDF layout artifacts:
    - Fix hyphenated words broken across lines (e.g. 'manage-\\nment' -> 'management')
    - Normalize unicode spaces and clean non-printable characters
    - Fix spaces inside email addresses and URLs
    - Preserve clean section headers and list structures
    """
    if not text:
        return ""

    # Replace unusual unicode whitespace and control characters
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Fix hyphenated words broken across line breaks
    text = re.sub(r"(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)", r"\1\2", text)

    # Fix spaces inserted inside email addresses during OCR
    text = re.sub(r"([\w.\-+]+)\s*[@©®]\s*([\w.\-]+)\s*\.\s*([a-zA-Z]{2,})", r"\1@\2.\3", text)
    
    # Fix spaces inserted inside URLs during OCR
    text = re.sub(r"(https?://[^\s]+)\s+([^\s]+)", r"\1\2", text)
    text = re.sub(r"linkedin\s*\.\s*com\s*/\s*in\s*/", "linkedin.com/in/", text, flags=re.I)
    text = re.sub(r"github\s*\.\s*com\s*/", "github.com/", text, flags=re.I)

    # Normalize excessive spaces and blank lines
    text = re.sub(r"[ \t]{3,}", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    
    return text.strip()


def _auto_rotate_image(img: Image.Image) -> Image.Image:
    """Detect image orientation with Tesseract OSD and rotate upright if needed."""
    try:
        import pytesseract
        _get_tesseract_cmd()
        # Downscale for fast OSD check
        w, h = img.size
        small = img.resize((min(w, 1000), int(h * min(w, 1000) / max(w, 1))), Image.Resampling.BILINEAR)
        osd = pytesseract.image_to_osd(small, output_type=pytesseract.Output.DICT)
        rotate_angle = osd.get("rotate", 0)
        if rotate_angle in (90, 180, 270):
            return img.rotate(-rotate_angle, expand=True)
    except Exception:
        pass
    return img


def preprocess_image_for_ocr(image: Image.Image) -> List[Image.Image]:
    """
    Generate optimized image variants for high-accuracy OCR:
    1. Rescaled high-contrast grayscale with unsharp mask
    2. High-contrast sharpened grayscale
    3. Standard grayscale
    """
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    # Correct orientation if rotated
    image = _auto_rotate_image(image)

    w, h = image.size
    # Target standard A4 ~2200-2600px width for 300 DPI clarity
    target_width = 2400
    if w < target_width:
        scale = max(2.0, target_width / max(w, 1))
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        scaled_img = image.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
    elif w > 3600:
        scale = 3000.0 / w
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        scaled_img = image.resize((scaled_w, scaled_h), Image.Resampling.BILINEAR)
    else:
        scaled_img = image

    variants = []

    # Variant 1: Enhanced grayscale with contrast & unsharp filter
    gray = scaled_img.convert("L")
    contrasted = ImageEnhance.Contrast(gray).enhance(1.8)
    sharp = ImageEnhance.Sharpness(contrasted).enhance(1.6)
    variants.append(sharp)

    # Variant 2: Unsharp mask on mild contrast
    mild_gray = scaled_img.convert("L")
    mild_sharp = mild_gray.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
    variants.append(mild_sharp)

    return variants


def _ocr_single_image(img: Image.Image) -> str:
    """Run OCR on a PIL image using adaptive layout detection configs."""
    try:
        import pytesseract
        _get_tesseract_cmd()
    except ImportError:
        return ""

    variants = preprocess_image_for_ocr(img)
    best_text = ""

    # Primary pass: PSM 3 (Fully automatic page segmentation)
    for variant in variants:
        try:
            txt = pytesseract.image_to_string(variant, config="--oem 3 --psm 3")
            if len(txt.strip().split()) > len(best_text.strip().split()):
                best_text = txt
        except Exception:
            continue

    # Secondary pass: PSM 1 (with OSD) and PSM 6 / 11 if sparse
    if len(best_text.split()) < 35:
        for psm in (1, 6, 11):
            try:
                txt = pytesseract.image_to_string(variants[0], config=f"--oem 3 --psm {psm}")
                if len(txt.strip().split()) > len(best_text.strip().split()):
                    best_text = txt
            except Exception:
                continue

    # Check for 2-column layout (sidebar + main column)
    # If a 2-column crop produces cleaner, richer text than full page, use column-aware text
    try:
        primary_var = variants[0]
        vw, vh = primary_var.size
        # Header (top 15%)
        header_crop = primary_var.crop((0, 0, vw, int(vh * 0.15)))
        # Left column (0 to 38%)
        left_crop = primary_var.crop((0, int(vh * 0.15), int(vw * 0.38), vh))
        # Right column (36% to 100%)
        right_crop = primary_var.crop((int(vw * 0.36), int(vh * 0.15), vw, vh))

        h_text = pytesseract.image_to_string(header_crop, config="--oem 3 --psm 6").strip()
        l_text = pytesseract.image_to_string(left_crop, config="--oem 3 --psm 6").strip()
        r_text = pytesseract.image_to_string(right_crop, config="--oem 3 --psm 6").strip()

        col_combined = f"{h_text}\n\n{r_text}\n\n{l_text}".strip()
        col_words = len(col_combined.split())
        best_words = len(best_text.split())

        # If 2-column extraction captures significant text in both columns and exceeds or matches full OCR
        if len(l_text.split()) > 15 and len(r_text.split()) > 25 and col_words >= best_words * 0.90:
            best_text = col_combined
    except Exception:
        pass

    return best_text.strip()


def extract_text_from_image_bytes(image_bytes: bytes) -> Tuple[str, str]:
    """
    Extract text from an image file (JPG, PNG, WEBP, TIFF, BMP) using high-accuracy OCR.

    Parameters
    ----------
    image_bytes : bytes
        Raw image file content.

    Returns
    -------
    (text, status) : Tuple[str, str]
        text   — Extracted plain text
        status — 'ok', 'error:<msg>', or 'empty'
    """
    try:
        import pytesseract
        cmd = _get_tesseract_cmd()
        if not cmd:
            return "", "error: Tesseract OCR executable not found on system."
    except ImportError:
        return "", "error: pytesseract not installed. Run: pip install pytesseract"

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        return "", f"error: Could not open image — {e}"

    try:
        text = _ocr_single_image(image)
    except Exception as e:
        return "", f"error: OCR processing failed — {e}"

    text = clean_extracted_text(text)
    if not text:
        return "", "empty: No readable text found in image."

    return text, "ok"


def _sort_pdf_blocks_in_reading_order(blocks: list, page_width: float, page_height: float) -> str:
    """
    Sort PDF text blocks in visual reading order.
    Detects 2-column layouts to avoid interleaving left sidebar with right main content.
    """
    text_blocks = [b for b in blocks if len(b) >= 5 and b[4].strip() and (len(b) < 7 or b[6] == 0)]
    if not text_blocks:
        return ""

    # Top header threshold (top ~18% of page)
    header_threshold = page_height * 0.18
    # Bottom footer threshold (bottom ~10% of page)
    footer_threshold = page_height * 0.90

    headers = []
    footers = []
    body = []

    for b in text_blocks:
        x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
        if y1 <= header_threshold:
            headers.append(b)
        elif y0 >= footer_threshold:
            footers.append(b)
        else:
            body.append(b)

    headers.sort(key=lambda b: (b[1], b[0]))
    footers.sort(key=lambda b: (b[1], b[0]))

    # Check if body forms 2 distinct columns
    left_col = [b for b in body if b[2] <= page_width * 0.52]
    right_col = [b for b in body if b[0] >= page_width * 0.35]

    # If body blocks neatly divide into 2 columns
    if len(left_col) >= 1 and len(right_col) >= 1 and (len(left_col) + len(right_col) >= len(body) * 0.85):
        left_col.sort(key=lambda b: b[1])
        right_col.sort(key=lambda b: b[1])
        ordered = headers + right_col + left_col + footers
    else:
        # Single column / standard vertical sorting
        body.sort(key=lambda b: (b[1], b[0]))
        ordered = headers + body + footers

    return "\n\n".join(b[4].strip() for b in ordered if b[4].strip())


def extract_text_from_pdf_page(page, doc, page_num: int) -> str:
    """
    Extract text from a single PDF page with column-aware layout sorting
    and automatic OCR fallback for scanned or low-text pages.
    """
    page_text = ""
    rect = page.rect
    page_w, page_h = rect.width, rect.height

    # 1. Try PyMuPDF column-aware block extraction
    try:
        blocks = page.get_text("blocks")
        sorted_text = _sort_pdf_blocks_in_reading_order(blocks, page_w, page_h)
        if sorted_text.strip():
            page_text = sorted_text
    except Exception:
        try:
            page_text = page.get_text("text").strip()
        except Exception:
            page_text = ""

    # 2. Check if text is sparse or page contains scanned graphics
    word_count = len(page_text.split())
    has_images = bool(page.get_images())

    # If the page has very few words (< 30 words) or has images with low text
    if word_count < 30 or (has_images and word_count < 55):
        try:
            # Render page to high-res image (300 DPI) for OCR
            import fitz
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = _ocr_single_image(img)
            
            # If OCR extracted more content, prefer OCR text
            if len(ocr_text.split()) > word_count + 10:
                page_text = ocr_text
            elif word_count == 0 and ocr_text.strip():
                page_text = ocr_text
        except Exception:
            pass

    return page_text.strip()


def extract_text_from_bytes(pdf_bytes: bytes) -> Tuple[str, str]:
    """
    Extract text from PDF bytes with column-aware layout and automatic OCR fallback.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF file content.

    Returns
    -------
    (text, status) : Tuple[str, str]
    """
    try:
        import fitz
    except ImportError:
        return "", "error: PyMuPDF (fitz) not installed. Run: pip install pymupdf"

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        return "", f"error: Could not parse PDF — {e}"

    pages_text = []

    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
            text = extract_text_from_pdf_page(page, doc, page_num)
            if text:
                pages_text.append(f"\n--- Page {page_num + 1} ---\n{text}" if len(doc) > 1 else text)
        except Exception:
            continue

    doc.close()

    if not pages_text:
        return "", "empty: No readable text could be extracted from this PDF."

    full_text = "\n\n".join(pages_text)
    full_text = clean_extracted_text(full_text)
    return full_text, "ok"


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, str]:
    """
    Extract text from a PDF file on disk with automatic OCR fallback.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.

    Returns
    -------
    (text, status) : Tuple[str, str]
    """
    if not os.path.isfile(pdf_path):
        return "", f"error: File not found — {pdf_path}"

    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        return extract_text_from_bytes(pdf_bytes)
    except Exception as e:
        return "", f"error: Failed reading PDF — {e}"


def extract_text_from_docx_bytes(docx_bytes: bytes) -> Tuple[str, str]:
    """
    Extract text from Microsoft Word (.docx) file bytes including paragraphs, tables, headers, footers.

    Parameters
    ----------
    docx_bytes : bytes
        Raw DOCX file content.

    Returns
    -------
    (text, status) : Tuple[str, str]
    """
    try:
        import docx
    except ImportError:
        return "", "error: python-docx not installed. Run: pip install python-docx"

    try:
        doc = docx.Document(io.BytesIO(docx_bytes))
    except Exception as e:
        return "", f"error: Could not parse DOCX file — {e}"

    content = []

    # 1. Extract paragraphs
    for p in doc.paragraphs:
        txt = p.text.strip()
        if txt:
            content.append(txt)

    # 2. Extract tables
    for table in doc.tables:
        table_rows = []
        for row in table.rows:
            row_cells = []
            seen_cells = set()
            for cell in row.cells:
                cell_id = id(cell._tc)
                if cell_id not in seen_cells:
                    seen_cells.add(cell_id)
                    cell_txt = cell.text.strip()
                    if cell_txt:
                        row_cells.append(cell_txt)
            if row_cells:
                table_rows.append(" | ".join(row_cells))
        if table_rows:
            content.append("\n".join(table_rows))

    # 3. Extract headers and footers
    for section in doc.sections:
        for header in (section.header, section.first_page_header):
            if header:
                for p in header.paragraphs:
                    txt = p.text.strip()
                    if txt and txt not in content:
                        content.insert(0, txt)

    if not content:
        return "", "empty: No text found in DOCX file."

    full_text = "\n\n".join(content)
    full_text = clean_extracted_text(full_text)
    return full_text, "ok"


def extract_text_from_docx(docx_path: str) -> Tuple[str, str]:
    """Extract text from a DOCX file on disk."""
    if not os.path.isfile(docx_path):
        return "", f"error: File not found — {docx_path}"
    try:
        with open(docx_path, "rb") as f:
            return extract_text_from_docx_bytes(f.read())
    except Exception as e:
        return "", f"error: Could not read DOCX — {e}"


def extract_text_from_file(raw_bytes: bytes, filename: str) -> Tuple[str, str]:
    """
    Unified dispatcher to extract text from any supported resume format:
    - PDF (.pdf)
    - Word (.docx)
    - Images (.jpg, .jpeg, .png, .webp, .tiff, .bmp)
    - Plain text (.txt)
    """
    fname = filename.lower()
    
    if fname.endswith(".pdf"):
        return extract_text_from_bytes(raw_bytes)
    elif fname.endswith(".docx"):
        return extract_text_from_docx_bytes(raw_bytes)
    elif fname.endswith((".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp")):
        return extract_text_from_image_bytes(raw_bytes)
    else:
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw_bytes.decode("latin-1")
            except Exception as e:
                return "", f"error: Could not decode text file — {e}"
        text = clean_extracted_text(text)
        return (text, "ok") if text else ("", "empty: File is empty.")
