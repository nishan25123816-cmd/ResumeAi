"""
utils.py
--------
Shared utility functions used across the project.
"""

import os
import re
import json
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy, random, and Python."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_json(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def truncate_text(text: str, max_chars: int = 300) -> str:
    """Truncate text for display purposes."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def format_extraction_result(extracted: dict) -> str:
    """
    Format the extraction result dictionary as a readable text block.
    """
    lines = []

    def add(label, value):
        if value:
            if isinstance(value, list):
                if value:
                    lines.append(f"{label}:")
                    for item in value:
                        lines.append(f"  • {item}")
            elif isinstance(value, dict):
                # Skip complex nested dicts (like links)
                pass
            else:
                lines.append(f"{label}: {value}")

    add("Name", extracted.get("name"))
    add("Email", extracted.get("email"))
    add("Phone", extracted.get("phone"))
    add("Location", extracted.get("location"))

    links = extracted.get("links", {})
    if isinstance(links, dict):
        if links.get("linkedin"):
            lines.append(f"LinkedIn: {links['linkedin']}")
        if links.get("github"):
            lines.append(f"GitHub: {links['github']}")

    add("Skills", extracted.get("skills"))
    add("Degree(s)", extracted.get("degrees"))
    add("University/College", extracted.get("universities"))
    add("Job Title(s)", extracted.get("job_titles"))
    add("Organization(s)", extracted.get("organizations"))
    add("Certifications", extracted.get("certifications"))
    add("Languages", extracted.get("languages"))

    return "\n".join(lines)


def project_root() -> str:
    """Return the project root directory (parent of src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_path(*parts) -> str:
    return os.path.join(project_root(), "data", *parts)


def models_path(*parts) -> str:
    return os.path.join(project_root(), "models", *parts)


def results_path(*parts) -> str:
    return os.path.join(project_root(), "results", *parts)
