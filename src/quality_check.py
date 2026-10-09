import re


def looks_like_low_quality_ocr(text):
    """Heuristic check to reject garbled OCR output before classification."""
    if text is None:
        return True

    cleaned = " ".join(str(text).split())
    if len(cleaned) < 80:
        return True

    tokens = re.findall(r"\S+", cleaned)
    if not tokens or len(tokens) < 12:
        return True

    meaningful_words = re.findall(r"[A-Za-z]{3,}", cleaned)
    if not meaningful_words:
        return True

    meaningful_ratio = len(meaningful_words) / len(tokens)
    if meaningful_ratio < 0.55:
        return True

    mixed_tokens = [
        token for token in tokens
        if len(token) >= 6 and any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token)
    ]
    noisy_ratio = len(mixed_tokens) / len(tokens)

    lower = cleaned.lower()
    resume_markers = (
        "experience",
        "education",
        "skills",
        "profile",
        "summary",
        "employment",
        "projects",
        "contact",
        "objective",
        "developer",
        "engineer",
        "manager",
    )
    has_marker = any(marker in lower for marker in resume_markers)
    has_contact = bool(
        re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", cleaned)
        or re.search(r"\+?\d[\d\s().-]{7,}\d", cleaned)
    )

    if not has_marker and not has_contact and noisy_ratio >= 0.08:
        return True

    # Very short or malformed OCR output should still be rejected even if it contains a few real words.
    if not has_marker and not has_contact and meaningful_ratio < 0.65 and len(meaningful_words) < 40:
        return True

    return False
