"""
information_extraction.py
--------------------------
Hybrid resume information extraction system.

Methodology:
  1. Regex-based extraction  — email, phone, URLs
  2. Section-based extraction — identifies resume sections (Skills, Education, etc.)
                                and extracts structured content
  3. spaCy pretrained NER    — extracts PERSON, ORG, GPE, DATE entities
  4. Keyword / pattern matching — skills from curated lists, degree patterns

NOTE: The dataset does NOT contain manually annotated NER labels.
      This module performs UNSUPERVISED / RULE-BASED extraction only.
      We use spaCy's pretrained en_core_web_sm model for entity hints,
      not a model fine-tuned on resumes.
"""

import re
import spacy
from typing import Optional

# ── spaCy model (lazy load) ──────────────────────────────────────────────────
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(
                ["python", "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
            )
            _NLP = spacy.load("en_core_web_sm")
    return _NLP


# ── Regex patterns ────────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(
    r"[\w.+\-]+@[\w.\-]+\.[a-zA-Z]{2,6}", re.IGNORECASE
)
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s\-\(\)]?)?\(?\d{2,4}\)?[\s\-\.]?\d{2,4}[\s\-\.]?\d{2,6}(?:\s?(?:#|x\.?|ext\.?|extension)\s?\d{1,5})?"
)
_URL_RE = re.compile(
    r"https?://[^\s,)>\"']+|www\.[^\s,)>\"']+"
)
_LINKEDIN_RE = re.compile(
    r"linkedin\.com/in/[\w\-]+", re.IGNORECASE
)
_GITHUB_RE = re.compile(
    r"github\.com/[\w\-]+", re.IGNORECASE
)

# ── Degree patterns ───────────────────────────────────────────────────────────
_DEGREE_PATTERNS = [
    # Full words first (most reliable)
    r"\b(?:Bachelor(?:\'s)?)\s*(?:\([^\)]*\))?\s*(?:of|in|with)?\s*[^\n]*",
    r"\b(?:BSc\s*(?:\(Hons\))?|B\.Sc\.?)\s*(?:in|of|with)?\s*[^\n]*",
    r"\b(?:B\.Tech|BTech|B\.E\.?|BCA|BIT)\b[^\n]*",
    r"\b(?:Master(?:\'s)?|Masters)\s*(?:of|in|with)\s*[^\n]+",
    r"\b(?:MSc|M\.Sc\.?|M\.Tech|MTech|MCA|MBA)\b[^\n]*",
    r"\b(?:Ph\.?D\.?|Doctor(?:ate)?)\b[^\n]*",
    r"\b(?:Associate(?:\'s)?|Diploma|Certificate)\s+(?:in|of)\s+[^\n]+",
    r"\b(?:High School|Secondary|SSC|HSC|10\+2|Intermediate)\b[^\n]*",
    r"\b(?:NCEA\s+Level\s+\d+)\b[^\n]*",
    r"\bGraduated\s+with\s+NCEA[^\n]*",
]
_DEGREE_RE = re.compile("|".join(_DEGREE_PATTERNS), re.IGNORECASE)

# ── University patterns ───────────────────────────────────────────────────────
_UNIV_RE = re.compile(
    r"\b(?:(?:in\s+partnership\s+with\s+)?([A-Z][a-zA-Z\s\&\.\-]{2,40}\s+(?:University|College|Institute(?:\s+of\s+Technology)?|School|Academy|Polytechnic|Campus)(?:\s*\([^\)]+\))?(?:[\w\s,]*))|"
    r"(?:MIT|Stanford(?:\s+University)?|Harvard(?:\s+University)?|Oxford(?:\s+University)?|Cambridge(?:\s+University)?|Caltech|NYU|UCLA|CMU|UC\s+Berkeley|IIT\s+[A-Z][a-z]+|Columbia\s+University|Yale(?:\s+University)?|Princeton(?:\s+University)?))\b",
    re.IGNORECASE,
)

# ── Section headers ───────────────────────────────────────────────────────────
_SECTION_HEADERS = {
    "skills": re.compile(
        r"^\s*.*(?:skills?|technical\s+skills?|key\s+skills?|"
        r"core\s+competencies?|competencies?|expertise|proficiencies?|"
        r"areas\s+of\s+expertise|tools?\s*(?:and|&)?\s*technologies?|rececai\s+sks|reccar\s+seus)\b.*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "education": re.compile(
        r"^\s*.*(?:education(?:al)?\s*(?:background|qualification|detail)?|"
        r"academic(?:\s+background)?|qualifications?|degrees?|epucanon|edueation)\b.*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "experience": re.compile(
        r"^\s*.*(?:(?:work|professional|employment|job|career)\s*"
        r"(?:experience|history|background)|^\s*experience\s*[:\-]?\s*$)\b.*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "certifications": re.compile(
        r"^\s*.*(?:certifications?|certificates?|licensures?|credentials?|"
        r"professional\s+certifications?|certifications?\s*&\s*courses?)\b.*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "languages": re.compile(
        r"^\s*.*(?:languages?|linguistic\s+(?:skills?|abilities?)|"
        r"language\s+(?:skills?|proficiency)|vancuaces)\b.*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "soft_skills": re.compile(
        r"^\s*.*(?:soft\s+skills?|interests?|hobbies?|career\s+objective|activities)\b.*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "projects": re.compile(
        r"^\s*.*(?:projects?|key\s+projects?|notable\s+projects?)\b.*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "summary": re.compile(
        r"^\s*.*(?:summary|professional\s+summary|objective|career\s+objective|"
        r"profile|about\s+me|personal\s+statement)\b.*$",
        re.IGNORECASE | re.MULTILINE,
    ),
}

# ── Skills keyword list ───────────────────────────────────────────────────────
TECH_SKILLS = {
    # Programming languages
    "python", "java", "javascript", "js", "typescript", "ts", "c++", "c#", "c",
    "r", "ruby", "php", "swift", "kotlin", "go", "golang", "rust", "scala",
    "perl", "matlab", "bash", "shell", "powershell", "vba", "cobol", "fortran",
    # Web
    "html", "css", "react", "angular", "vue", "node.js", "nodejs", "express",
    "django", "flask", "fastapi", "spring", "laravel", "asp.net", "jquery",
    "bootstrap", "tailwind", "next.js", "nuxt.js", "wordpress", "woocommerce",
    # Data / ML
    "machine learning", "deep learning", "neural network", "nlp",
    "natural language processing", "computer vision", "tensorflow", "pytorch",
    "keras", "scikit-learn", "sklearn", "pandas", "numpy", "scipy",
    "matplotlib", "seaborn", "plotly", "tableau", "power bi", "data analysis",
    "data science", "data visualization", "statistical analysis", "spss", "sas",
    # Databases
    "sql", "mysql", "postgresql", "sqlite", "oracle", "mongodb", "nosql",
    "redis", "elasticsearch", "cassandra", "dynamodb",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git",
    "github", "gitlab", "bitbucket", "ci/cd", "terraform", "ansible", "linux",
    # Data Engineering
    "hadoop", "spark", "kafka", "airflow", "dbt", "snowflake", "redshift",
    "bigquery", "hive",
    # Productivity / Office
    "excel", "word", "powerpoint", "microsoft office", "google sheets",
    "sap", "salesforce", "jira", "confluence", "slack", "trello", "asana",
    # Networking / Security
    "networking", "cisco", "firewall", "cybersecurity", "ethical hacking",
    "penetration testing", "linux administration",
    # Design / Creative
    "photoshop", "illustrator", "adobe xd", "figma", "sketch", "indesign",
    "after effects", "premiere pro", "lightroom", "canva", "coreldraw",
    "ui design", "ux design", "ui/ux", "graphic design", "web design",
    "branding", "typography", "logo design", "motion graphics", "video editing",
    "photography", "3d modeling", "autocad", "solidworks", "blender",
    # Finance / Accounting
    "accounting", "bookkeeping", "financial modeling", "financial analysis",
    "budgeting", "forecasting", "tax", "auditing", "quickbooks", "tally",
    "dcf", "valuation", "bloomberg", "cfa", "cpa", "ifrs", "gaap",
    "accounts payable", "accounts receivable", "payroll", "ms excel",
    # Marketing / Sales
    "digital marketing", "seo", "sem", "social media marketing", "content marketing",
    "email marketing", "google analytics", "facebook ads", "instagram",
    "copywriting", "market research", "crm", "hubspot", "mailchimp",
    "brand management", "public relations", "media planning",
    # Healthcare / Medical
    "patient care", "clinical", "nursing", "first aid", "cpr", "ehr", "emr",
    "medical coding", "phlebotomy", "pharmacology",
    # Culinary / Chef
    "menu planning", "food safety", "haccp", "food preparation", "cooking",
    "baking", "pastry", "culinary arts", "kitchen management", "catering",
    "inventory management", "recipe development", "wine pairing",
    # HR / People
    "recruitment", "talent acquisition", "onboarding", "performance management",
    "employee relations", "training", "hr management", "payroll management",
    "hris", "workday", "bamboohr",
    # Engineering
    "autocad", "civil engineering", "structural analysis", "project planning",
    "quality control", "six sigma", "lean manufacturing", "cad", "cam",
    # Legal / Advocate
    "legal research", "contract drafting", "litigation", "compliance",
    "corporate law", "intellectual property", "arbitration",
    # Teaching / Education
    "curriculum development", "lesson planning", "e-learning", "lms",
    "classroom management", "student assessment",
    # General Business
    "business development", "strategic planning", "operations management",
    "supply chain", "procurement", "vendor management", "risk management",
    "agile", "scrum", "rest api", "graphql", "microservices", "oop",
}

SOFT_SKILLS = {
    "leadership", "communication", "teamwork", "problem solving",
    "critical thinking", "time management", "project management",
    "analytical", "presentation", "negotiation", "collaboration",
    "creativity", "adaptability", "customer service", "attention to detail",
    "multitasking", "decision making", "interpersonal skills",
    "written communication", "verbal communication", "research",
    "self-motivated", "fast learner", "detail-oriented",
}

ALL_SKILLS = TECH_SKILLS | SOFT_SKILLS

# ── Job title patterns ────────────────────────────────────────────────────────
_JOB_TITLE_PATTERNS = [
    r"\b(?:Senior|Junior|Lead|Principal|Chief|Head|Director|VP|Vice\s+President|"
    r"Manager|Supervisor|Coordinator|Executive|Associate|Assistant|Freelance|"
    r"Staff|Entry.Level)?\s*"
    r"(?:Software|Data|Machine\s+Learning|AI|ML|AI/ML|Business|"
    r"Financial|Marketing|Sales|HR|Human\s+Resources|"
    r"Operations|Product|Project|Program|Account|"
    r"Systems?|Network|Cloud|DevOps|Full.?Stack|Front.?End|Back.?End|"
    r"Graphic|UI|UX|UI/UX|Web|Visual|Creative|Digital|Brand|Industrial|Game|Interior|"
    r"QA|Quality|Research|Analytics?|Database|Security|IT|Content|Copy|Event|Social\s+Media)?\s*"
    r"(?:Engineer|Developer|Analyst|Scientist|Architect|Manager|"
    r"Consultant|Specialist|Administrator|Officer|Advisor|"
    r"Designer|Director|Coordinator|Coordination|Executive|Lead|Intern|"
    r"Technician|Strategist|Producer|Writer|Editor|Representative|Creator|"
    r"Assistant|Supervisor|Instructor|Teacher|Accountant|Auditor|Doctor|Nurse|Chef|Cook|Design|Promotion)\b",
]
_JOB_TITLE_RE = re.compile("|".join(_JOB_TITLE_PATTERNS), re.IGNORECASE)

# ── Location / GPE hints ──────────────────────────────────────────────────────
_LOCATION_RE = re.compile(
    r"\b(?:Address|Location|City|State|Country)?\s*[:\-]?\s*"
    r"([A-Z][a-z]+([\s,]+[A-Z][a-z]+)*)\b"
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_section_text(text: str, section_key: str) -> str:
    """
    Extract the text block following a detected section header.
    Returns content until the next section header or end of text.
    """
    pattern = _SECTION_HEADERS[section_key]
    all_header_patterns = "|".join(
        p.pattern for p in _SECTION_HEADERS.values()
    )
    all_headers_re = re.compile(all_header_patterns, re.IGNORECASE | re.MULTILINE)

    lines = text.split("\n")
    section_start = None
    for i, line in enumerate(lines):
        if pattern.search(line):
            section_start = i + 1
            break

    if section_start is None:
        return ""

    section_lines = []
    for line in lines[section_start:]:
        if all_headers_re.search(line) and line.strip():
            break
        section_lines.append(line)

    return "\n".join(section_lines).strip()


def _clean_extracted_text(text: str) -> str:
    """Remove bullets, excessive whitespace from extracted section text."""
    text = re.sub(r"^[\s•\-\*\>◆▪►\u2022\u25CF\u25AA]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_skills_text(text: str) -> list:
    """Split a skills block into individual skill tokens without breaking hyphenated skills (e.g. Scikit-Learn)."""
    # Replace bullet points and list markers with commas
    cleaned = re.sub(r"[\s•\*\u2022\u25CF\u25AA◆▪►]+", ",", text)
    cleaned = re.sub(r"\s+[\-–—]\s+", ",", cleaned)
    parts = re.split(r"[,|;\n]", cleaned)
    skills = []
    for part in parts:
        part = part.strip().strip(":").strip()
        if 1 < len(part) < 60:
            skills.append(part)
    return skills


# ═══════════════════════════════════════════════════════════════════════════════
#  Core extraction functions
# ═══════════════════════════════════════════════════════════════════════════════

def extract_email(text: str) -> Optional[str]:
    # 1. Clean spacing around @ and dots
    cleaned_text = re.sub(r"(\w+)\s*[@©®]\s*([\w\-]+(?:\s*\.\s*[a-zA-Z]{2,6})+)", r"\1@\2", text)
    cleaned_text = re.sub(r"\s*\.\s*", ".", cleaned_text)
    
    match = _EMAIL_RE.search(cleaned_text)
    if match:
        email = match.group(0).lower()
        email = re.sub(r"@g(?:mai1|mall|maii|natcon|ralicaon|malicon|ralcon)\.(?:com|con)", "@gmail.com", email)
        email = re.sub(r"@g(?:mai1|mall|maii|natcon|ralicaon|malicon|ralcon)$", "@gmail.com", email)
        return email
        
    m_noisy = re.search(r"([a-zA-Z0-9_\.\+\-]{3,35})\s*[@©®S\d\s]{1,5}\s*([a-zA-Z0-9_\.\-]{0,25}(?:gralicon|gralcon|gnaticom|gmalicon|gmail|yahoo|hotmail|outlook|com)(?:\.[a-zA-Z]{2,6})?)", text, re.I)
    if m_noisy:
        user = m_noisy.group(1).strip()
        domain = m_noisy.group(2).strip()
        if any(g in domain.lower() for g in ["gmail", "yahoo", "hotmail", "outlook", "gralicon", "gralcon", "gmalicon", "gnatcon", "gnaticom"]):
            return f"{user}@gmail.com".lower()
        if "." in domain:
            return f"{user}@{domain}".lower()
    return None


def extract_phone(text: str) -> Optional[str]:
    # Match international format +977 or standard numbers
    m_int = re.search(r"(?:\+?977[\s\-]?)?(?:98\d{8}|97\d{8}|01\d{7})\b", text)
    if m_int:
        num = m_int.group(0).strip()
        if num.startswith("98") or num.startswith("97"):
            return "+977 " + num
        if not num.startswith("+"):
            return "+" + num
        return num

    matches = _PHONE_RE.findall(text)
    if matches:
        for m in matches:
            digits = re.sub(r"\D", "", m)
            if len(digits) >= 7:
                cleaned = re.sub(r"^[^\d\+]+", "", m.strip())
                return cleaned
    return None


def extract_urls(text: str) -> list:
    urls = _URL_RE.findall(text)
    linkedin = _LINKEDIN_RE.search(text)
    github = _GITHUB_RE.search(text)
    result = {"urls": urls}
    if linkedin:
        result["linkedin"] = "https://" + linkedin.group(0)
    if github:
        result["github"] = "https://" + github.group(0)
    return result


def extract_name(text: str) -> Optional[str]:
    """
    Heuristic name extraction:
    1. Check top lines for clean all-caps or title-case candidate name.
    2. Fall back to spaCy PERSON entity in first 8 lines.
    """
    lines = [re.sub(r"^[^a-zA-Z]+", "", l).strip() for l in text.split("\n") if l.strip()][:15]

    _NOT_NAME = [
        "resume", "curriculum", "vitae", "profile", "address",
        "objective", "summary", "experience", "education", "skills",
        "http", "www", "@", "phone", "email", "mobile", "tel", "contact",
        "developer", "engineer", "designer", "creator", "kathmandu", "nepal",
        "maitidevi", "street", "road", "avenue", "city", "state", "linkedin", "github",
        "python", "javascript", "react", "html", "css", "node", "java"
    ]

    # Priority 1: First few lines for prominent name
    for line in lines[:5]:
        if any(kw in line.lower() for kw in _NOT_NAME):
            continue
        if re.search(r"[@\|/\\0-9]", line):
            continue
        tokens = line.split()
        if 2 <= len(tokens) <= 4 and all(t[0].isupper() for t in tokens if t and t[0].isalpha()):
            # Ensure tokens are not single skills
            if not any(t.lower() in ALL_SKILLS for t in tokens):
                return line.strip()

    # Priority 2: spaCy PERSON entity
    nlp = _get_nlp()
    header_text = "\n".join(lines[:8])
    doc = nlp(header_text[:800])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            name = re.sub(r"^[^a-zA-Z]+", "", name).strip()
            if (
                2 <= len(name.split()) <= 4
                and not re.search(r"[\d@\|]", name)
                and not any(kw in name.lower() for kw in _NOT_NAME)
                and not any(w.lower() in ALL_SKILLS for w in name.split())
            ):
                return name

    # Priority 3: First valid line
    for line in lines[:8]:
        if re.search(r"[@\|/\\0-9]", line):
            continue
        if any(kw in line.lower() for kw in _NOT_NAME):
            continue
        tokens = line.split()
        if 2 <= len(tokens) <= 4 and all(t[0].isupper() for t in tokens if t and t[0].isalpha()) and len(line) >= 4:
            if not any(t.lower() in ALL_SKILLS for t in tokens):
                return line.strip()

    return None


def extract_location(text: str) -> Optional[str]:
    """Extract location using address patterns, known cities, and validated GPE entities."""
    # 1. Pattern: "Address: ...", "Location: ..."
    addr_re = re.compile(
        r"(?:Address|Location|City|Residence|Based\s+in)\s*[:\-]\s*([^\n]{4,60})",
        re.IGNORECASE,
    )
    match = addr_re.search(text)
    if match:
        return match.group(1).strip()

    # 2. Known Nepal / International city patterns on clean address lines (not inside education/university)
    for line in text.split("\n"):
        line_clean = line.strip()
        if not line_clean:
            continue
        # Discard lines that mention degrees, universities, college, school, student
        if any(w in line_clean.lower() for w in ["college", "university", "school", "degree", "bachelor", "master", "bsc", "b.sc", "student", "faculty"]):
            continue
        m_city = re.search(
            r"\b(?:([A-Z][a-zA-Z]{2,25}(?:[\s\-][A-Z][a-zA-Z]{2,25})?\s*,\s*)?(?:Kathmandu|Katenandta|Karthrrendu|Kathrresdu|Katesandu|Lalitpur|Bhaktapur|Pokhara|Biratnagar|Chitwan|Nepal|New York|London|San Francisco|Sydney|Toronto|Tokyo|Berlin|Paris|Delhi|Bangalore|Mumbai|Dubai|Singapore)(?:,\s*[A-Z][a-zA-Z]+)?)\b",
            line_clean,
            re.I
        )
        if m_city:
            loc_str = m_city.group(0).strip()
            loc_str = re.sub(r"^[^\w]+", "", loc_str)
            # Fix common OCR misspellings of Kathmandu
            loc_str = re.sub(r"\b(?:Katenandta|Karthrrendu|Kathrresdu|Katesandu)\b", "Kathmandu", loc_str, flags=re.I)
            loc_str = re.sub(r"\b(?:Marites|Maries|Martidevi)\b", "Maitidevi", loc_str, flags=re.I)
            if len(loc_str) >= 4:
                return loc_str

    # 3. spaCy GPE/LOC
    _INVALID_LOC_WORDS = {
        "artificial", "intelligence", "computer", "science", "developer", "designer",
        "summary", "experience", "education", "skills", "projects", "objective",
        "profile", "contact", "student", "technical", "creative", "media", "digital",
        "college", "university", "institute", "bcu", "sunway"
    }
    nlp = _get_nlp()
    doc = nlp(text[:800])
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            loc = ent.text.strip()
            loc_clean = re.sub(r"^[^\w]+", "", loc).strip()
            if (
                len(loc_clean) >= 3
                and loc_clean.lower() not in _INVALID_LOC_WORDS
                and not any(w in loc_clean.lower() for w in _INVALID_LOC_WORDS)
            ):
                return loc_clean

    return None


_SKILL_CANONICAL = {
    "js": "JavaScript",
    "javascript": "JavaScript",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "reactjs": "React",
    "react.js": "React",
    "react": "React",
    "sklearn": "Scikit-Learn",
    "scikit-learn": "Scikit-Learn",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "golang": "Go",
    "c++": "C++",
    "c#": "C#",
    "aws": "AWS",
    "gcp": "GCP",
    "sql": "SQL",
    "nosql": "NoSQL",
    "ci/cd": "CI/CD",
    "nlp": "NLP",
    "ai": "Artificial Intelligence",
    "ml": "Machine Learning",
    "ui/ux": "UI/UX",
    "ui": "UI Design",
    "ux": "UX Design",
    "git": "Git",
    "github": "GitHub",
    "gitlab": "GitLab",
    "html": "HTML",
    "css": "CSS",
    "php": "PHP",
    "r": "R",
    "c": "C",
    "sap": "SAP",
    "crm": "CRM",
    "seo": "SEO",
    "sem": "SEM",
    "rest api": "REST API",
}


def extract_skills(text: str) -> list:
    """
    Extract skills by keyword matching ONLY against the curated ALL_SKILLS dictionary.
    Section text is used as context only — raw section tokens are NOT added unless
    they match a known skill. This prevents OCR noise from polluting skill output.
    """
    # Keyword matching against curated list (case-insensitive, whole-word boundary)
    text_lower = text.lower()
    matched = set()
    for skill in ALL_SKILLS:
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
            canonical = _SKILL_CANONICAL.get(skill.lower())
            if canonical:
                matched.add(canonical)
            elif len(skill.split()) > 1:
                matched.add(skill.title())
            elif len(skill) <= 3:
                matched.add(skill.upper())
            else:
                matched.add(skill.title())

    # Deduplication using canonical keys
    seen_lower = set()
    deduped = []
    for skill_str in matched:
        norm_key = _SKILL_CANONICAL.get(skill_str.lower(), skill_str).lower()
        if norm_key not in seen_lower:
            seen_lower.add(norm_key)
            canonical = _SKILL_CANONICAL.get(skill_str.lower())
            deduped.append(canonical if canonical else skill_str)

    return deduped[:35]


def extract_education(text: str) -> dict:
    """
    Extract education info using section parsing + degree regex + university regex.
    Returns: {degrees: [], universities: [], raw_section: str}
    """
    section_text = _extract_section_text(text, "education")
    search_text = section_text if section_text else text[:2000]

    # ── Degrees ──
    # Anchoring keywords that a real degree string MUST start with
    _DEG_ANCHORS = re.compile(
        r"^\s*(?:bachelor|b\.?s\.?c?|b\.?e\.?|b\.?tech|b\.?a\.?|bsc|bca|bit|be|bs|bcom|"
        r"master|m\.?s\.?c?|m\.?e\.?|m\.?tech|mba|mca|msc|ms|"
        r"ph\.?d|doctor(?:ate)?|associate|diploma|certificate|"
        r"high school|secondary|ssc|hsc|10\+2|intermediate|ncea|"
        r"graduated with ncea)",
        re.IGNORECASE,
    )
    # Words that indicate this is NOT a degree (OCR garbage / addresses / noise)
    _NOT_DEGREE = [
        "kathmandu", "nepal", "maitidevi", "lalitpur", "pokhara",
        "new york", "london", "australia", "street", "lane", "avenue",
        "managed", "developed", "methods", "maintair", "maintained",
        "provided", "responsible", "lorem", "ipsum", "dolor", "consectetur",
    ]

    raw_degrees = []
    seen_degrees = set()
    for match in _DEGREE_RE.finditer(search_text):
        deg = match.group(0).strip()
        # Must start with a real degree anchor word
        if not _DEG_ANCHORS.match(deg):
            continue
        # Reject if it contains obvious address / noise tokens
        if any(nd in deg.lower() for nd in _NOT_DEGREE):
            continue
        # If there are periods or semicolons, take the primary clause
        deg = re.split(r"[\.\;]", deg)[0].strip()
        deg_clean = re.sub(r"\s+", " ", deg)[:100]
        # Remove any email or @ content
        deg_clean = re.sub(r"[\s,]*[\w.+\-]+@.*$", "", deg_clean).strip()
        deg_clean = re.sub(r"[\s,]*@.*$", "", deg_clean).strip()
        # Clean OCR noise
        deg_clean = re.sub(r"^B?Sc\s+Ptora\)\s*", "BSc (Hons) ", deg_clean, flags=re.I)
        # Remove trailing single character noise
        deg_clean = re.sub(r"[\s,|-]+[a-zA-Z]$", "", deg_clean).strip()
        deg_clean = re.sub(r"[\s,|-]+$", "", deg_clean).strip()
        # Minimum quality: at least 8 chars, not all caps noise shorter than 4 tokens
        if deg_clean and len(deg_clean) > 8 and deg_clean.lower() not in seen_degrees:
            seen_degrees.add(deg_clean.lower())
            raw_degrees.append(deg_clean)

    # Consolidate duplicate NCEA / High school qualifications
    degrees = []
    seen_clean = set()
    ncea_school = ""
    ncea_grade = ""
    has_ncea = False

    for d in raw_degrees:
        if "ncea" in d.lower():
            has_ncea = True
            if "excellence" in d.lower():
                ncea_grade = "with Excellence"
            elif "merit" in d.lower() and not ncea_grade:
                ncea_grade = "with Merit"
            m_school = re.search(r"(?:at|,)\s*([A-Z][a-zA-Z\s]+(?:College|School|Academy|High School)[\w\s,]*)", d, re.I)
            if m_school and not ncea_school:
                ncea_school = m_school.group(1).strip()
            continue

        if d.lower() not in seen_clean:
            seen_clean.add(d.lower())
            degrees.append(d)

    if has_ncea:
        parts = ["NCEA Level 3"]
        if ncea_grade:
            parts.append(ncea_grade)
        if ncea_school:
            parts.append(f"— {ncea_school}")
        ncea_entry = " ".join(parts).strip()
        if ncea_entry.lower() not in seen_clean:
            seen_clean.add(ncea_entry.lower())
            degrees.append(ncea_entry)

    # ── Universities / Institutions ──
    # Generic single-word noise to reject
    _GENERIC_WORDS = {"college", "university", "institute", "school", "academy", "campus", "polytechnic"}

    universities = []
    seen_univs = set()
    for match in _UNIV_RE.finditer(search_text):
        univ = match.group(0).strip()
        # Remove any trailing email that regex captured
        univ = re.sub(r"\s+[\w.+\-]+@[\w.\-]+\.[a-zA-Z]{2,6}.*$", "", univ).strip()
        univ = re.sub(r"^(?:in\s+partnership\s+with|partnenthip\s+with|partnership\s+with|affiliated\s+with|in\s+affiliation\s+with|and\s+|at\s+)\s*", "", univ, flags=re.I).strip()
        # Strip leaked degree subject names like "with Artificial Intelligence" from start
        univ = re.sub(r"^(?:with\s+)?(?:Artificial\s+Intelligence|Computer\s+Science|Software\s+Engineering|Data\s+Science|Information\s+Technology|Business\s+Administration|Mechanical\s+Engineering|Electrical\s+Engineering|Civil\s+Engineering)\s*", "", univ, flags=re.I).strip()
        univ = re.sub(r"\s+(?:Current|Semester|GPA|\d{4}).*$", "", univ, flags=re.I).strip()
        # Remove trailing noise like "BOUL UK" → just keep "UK" context
        univ = re.sub(r"\bBOUL\b", "(BCU)", univ)
        univ = re.sub(r"\s+", " ", univ)[:100].strip()
        
        # Clean common OCR typos in university names
        univ = re.sub(r"\bOty\b", "City", univ)
        univ = re.sub(r"\bSurman\b|\bburnmay\b", "Sunway", univ)
        univ = re.sub(r"\bKarthrrendu\b|\bKathrresdu\b|\bKatenandta\b", "Kathmandu", univ)
        univ = re.sub(r"\bGemninghan\b|\bBerningharn\b", "Birmingham", univ)

        # Filter 1: skip if empty or too short (allow known 3+ letter institutions like MIT, NYU, IIT, UCLA)
        if len(univ) < 3:
            continue
        # Filter 2: skip if it contains an email
        if re.search(r"@", univ):
            continue
        # Filter 3: skip if it has long digit sequences (zip, phone)
        if re.search(r"\d{4,}", univ):
            continue
        # Filter 4: skip if it's just a single generic word like "College" or generic word + city like "College, Auckland"
        first_token = univ.split(",")[0].strip().lower()
        if univ.lower().strip() in _GENERIC_WORDS or first_token in _GENERIC_WORDS:
            continue
        # Filter 5: skip if already seen
        if univ.lower() in seen_univs:
            continue
        # Filter 6: skip if this institution name is fully contained inside an already-detected degree
        already_in_degree = any(univ.lower() in d.lower() for d in degrees)
        if already_in_degree:
            continue

        seen_univs.add(univ.lower())
        universities.append(univ)

    return {
        "degrees": degrees[:5],
        "universities": universities[:5],
        "raw_section": section_text[:500] if section_text else "",
    }


def extract_experience(text: str) -> dict:
    """
    Extract structured work experience entries.
    Each entry contains:
      - title: Job title / post (e.g., 'Graphic Designer')
      - company: Company / organization (e.g., 'Aqualine Studios')
      - period: Time duration / date range (e.g., 'Jan 2020 – Dec 2022' or '2020 – Present')
    Also returns job_titles and organizations lists for backward compatibility.
    """
    section_text = _extract_section_text(text, "experience")
    if section_text:
        search_text = section_text
    else:
        # Strip education section so academic text never leaks into experience
        edu_text = _extract_section_text(text, "education")
        search_text = text
        if edu_text and len(edu_text) > 10:
            search_text = search_text.replace(edu_text, "")

    # Date regex components
    _MONTHS = (
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    )
    _PRESENT = r"(?:Present|Current|Ongoing|Now|Till\s*Date|To\s*Date)"
    _SINGLE_DATE = rf"(?:{_MONTHS}\s*[\.,\'-]?\s*\d{{2,4}}|\d{{1,2}}/\d{{2,4}}|\b(?:19|20)\d{{2}}\b)"

    _PERIOD_RANGE_RE = re.compile(
        rf"({_SINGLE_DATE}\s*(?:[\-–—/~]+|\s+to\s+)\s*(?:{_SINGLE_DATE}|{_PRESENT}))",
        re.IGNORECASE,
    )

    _ACTION_VERB_RE = re.compile(
        r"^(?:created|developed|managed|led|assisted|designed|built|provided|maintained|"
        r"responsible|worked|organized|collaborated|implemented|coordinated|directed|"
        r"supervised|analyzed|achieved|conducted|established|spearheaded|oversaw|supported|"
        r"resolved|executed|participated|handled|performed)\b",
        re.IGNORECASE,
    )

    _ADDRESS_KEYWORDS = {
        "street", "lane", "road", "rd", "st", "ave", "avenue", "drive",
        "blvd", "boulevard", "way", "highway", "pkwy", "box", "po"
    }

    _SECTION_NOISE = {
        "experience", "work experience", "employment history", "professional experience",
        "skills", "technical skills", "education", "profile", "contact", "summary",
        "references", "certifications", "projects"
    }

    _ACADEMIC_WORDS = {
        "graduated", "graduate", "graduation", "degree", "bachelor", "master",
        "phd", "diploma", "certificate", "certification", "secondary", "high school",
        "ncea", "gpa", "course", "courses", "curriculum", "syllabus", "subject",
        "geography", "history", "english", "science", "math", "mathematics",
        "physics", "chemistry", "biology", "student", "class", "grade",
        "merit", "excellence", "endorsement", "endorsements", "honors", "honours",
        "academic", "school", "college", "university", "qualification", "studies"
    }

    def _is_academic(s: str) -> bool:
        if not s:
            return False
        tokens = set(re.findall(r"\b[a-zA-Z]+\b", s.lower()))
        first_token = s.strip().split()[0].lower() if s.strip() else ""
        if first_token in _ACADEMIC_WORDS:
            return True
        if tokens & {"ncea", "gpa", "graduated", "degree", "diploma", "bachelor", "master", "endorsement", "excellence", "merit"}:
            return True
        return False

    def _clean_noise(s: str) -> str:
        s = re.sub(r"^[\s•\-\*\+>◆▪►\u2022\u25CF\u25AA\d\.\)]+", "", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def _is_junk(line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) < 2:
            return True
        if stripped.startswith(("+", "-", "*", "•", "–", "—", ">", "▪", "►", "■")):
            return True
        cleaned = _clean_noise(stripped)
        if cleaned.lower() in _SECTION_NOISE or _is_academic(cleaned):
            return True
        if _ACTION_VERB_RE.match(cleaned):
            return True
        # Street address check
        tokens = set(re.findall(r"\b[a-zA-Z]+\b", cleaned.lower()))
        if (tokens & _ADDRESS_KEYWORDS) and re.search(r"\d+", cleaned):
            return True
        # Email, phone, URL check
        if "@" in cleaned or re.search(r"\b(?:http|www|\.com|\.org|\.net)\b", cleaned, re.I):
            return True
        if re.match(r"^\+?\d[\d\s\-\(\)]{7,}\d$", cleaned):
            return True
        # Pure skills/tools check
        if cleaned.lower() in ALL_SKILLS:
            return True
        return False

    def _get_title(cand: str) -> str:
        c = _clean_noise(cand)
        if not c or _is_junk(c) or _is_academic(c):
            return ""
        m = _JOB_TITLE_RE.search(c)
        if m:
            matched_title = m.group(0).strip()
            if _is_academic(matched_title):
                return ""
            if len(c.split()) <= 4 and not _is_academic(c):
                return c
            return matched_title
        return ""

    lines = [l.strip() for l in search_text.split("\n") if l.strip()]
    entries = []
    used_indices = set()

    # Pass 1: Delimited lines (e.g., "Graphic Designer | Aqualine Studios | Jan 2020 - Dec 2022")
    for i, line in enumerate(lines):
        if _is_junk(line) or _is_academic(line):
            continue
        parts = [p.strip() for p in re.split(r"[\s\t]*[|•·][\s\t]*", line) if p.strip()]
        if len(parts) >= 2:
            role_cand, comp_cand, per_cand = "", "", ""
            for p in parts:
                p_clean = _clean_noise(p)
                pm = _PERIOD_RANGE_RE.search(p_clean)
                if pm:
                    per_cand = pm.group(1).strip()
                    p_clean = _PERIOD_RANGE_RE.sub("", p_clean).strip("|-–, ()[]")
                title_match = _get_title(p_clean)
                if title_match and not role_cand:
                    role_cand = title_match
                elif p_clean and not _is_junk(p_clean) and not _is_academic(p_clean) and len(p_clean) >= 3 and not comp_cand:
                    comp_cand = p_clean

            # ONLY add if a real job title is verified
            if role_cand and not _is_academic(role_cand):
                entries.append({
                    "title": role_cand,
                    "company": comp_cand or "",
                    "period": per_cand,
                })
                used_indices.add(i)

    # Pass 2: "Role at Company" (strictly require verified job title and real date)
    for i, line in enumerate(lines):
        if i in used_indices or _is_junk(line) or _is_academic(line):
            continue
        at_match = re.match(
            r"^([A-Z][^@\d]{2,45}?)\s+(?:at|@)\s+([A-Z][^@\d]{2,50})(?:\s*[\(\[]?(.*?)[\)\]]?)?$",
            line,
            re.IGNORECASE,
        )
        if at_match:
            r_c = _get_title(at_match.group(1))
            c_c = at_match.group(2).strip()
            if r_c and not _is_academic(c_c) and not _is_junk(c_c):
                p_c = ""
                pm = _PERIOD_RANGE_RE.search(line)
                if pm:
                    p_c = pm.group(1).strip()
                elif at_match.group(3):
                    # Only accept group(3) if it is actually a date
                    dm = _PERIOD_RANGE_RE.search(at_match.group(3)) or re.search(r"\b(?:19|20)\d{2}\b", at_match.group(3))
                    if dm:
                        p_c = dm.group(0).strip()
                entries.append({
                    "title": r_c,
                    "company": c_c,
                    "period": p_c,
                })
                used_indices.add(i)

    # Pass 3: Multi-line blocks anchored by Date Ranges (Role / Company / Dates)
    for i, line in enumerate(lines):
        if i in used_indices or _is_junk(line) or _is_academic(line):
            continue
        pm = _PERIOD_RANGE_RE.search(line)
        if pm:
            period = pm.group(1).strip()
            remainder = _clean_noise(_PERIOD_RANGE_RE.sub("", line).strip("|-–, ()[]"))

            candidates = []
            if remainder and not _is_junk(remainder) and not _is_academic(remainder) and len(remainder) >= 3:
                candidates.append((i, remainder))

            # Look up to 3 lines backward
            for prev_idx in range(max(0, i - 3), i):
                if prev_idx not in used_indices and not _is_junk(lines[prev_idx]) and not _is_academic(lines[prev_idx]):
                    candidates.append((prev_idx, lines[prev_idx]))
            # Look up to 2 lines forward
            for next_idx in range(i + 1, min(len(lines), i + 3)):
                if next_idx not in used_indices and not _is_junk(lines[next_idx]) and not _is_academic(lines[next_idx]):
                    candidates.append((next_idx, lines[next_idx]))

            role_cand, comp_cand = "", ""
            matched_indices = set()

            # Find verified title first
            for idx, text_cand in candidates:
                t = _get_title(text_cand)
                if t and not role_cand:
                    role_cand = t
                    matched_indices.add(idx)
                    break

            # Find company from remaining candidates
            for idx, text_cand in candidates:
                if idx in matched_indices:
                    continue
                cand_clean = _clean_noise(text_cand)
                if not _is_junk(cand_clean) and not _is_academic(cand_clean) and len(cand_clean) >= 3 and len(cand_clean.split()) <= 6:
                    comp_cand = cand_clean
                    matched_indices.add(idx)
                    break

            # ONLY append if a real verified role is found
            if role_cand and not _is_academic(role_cand):
                entries.append({
                    "title": role_cand,
                    "company": comp_cand or "",
                    "period": period,
                })
                used_indices.add(i)
                used_indices.update(matched_indices)

    # Pass 4: Unmatched job titles — search nearby lines for company and date
    for i, line in enumerate(lines):
        if i in used_indices or _is_junk(line) or _is_academic(line):
            continue
        title = _get_title(line)
        if title and not _is_academic(title):
            comp = ""
            period = ""
            matched_adjs = set()

            # Search nearby lines for company
            for adj in [i + 1, i - 1, i + 2]:
                if 0 <= adj < len(lines) and adj not in used_indices and not _is_junk(lines[adj]) and not _is_academic(lines[adj]):
                    cand = _clean_noise(lines[adj])
                    # Check if candidate has a date inside
                    pm = _PERIOD_RANGE_RE.search(cand) or re.search(r"\b(?:19|20)\d{2}\b", cand)
                    if pm and not period:
                        period = pm.group(0).strip()
                        cand = _PERIOD_RANGE_RE.sub("", cand).strip("|-–, ()[]")

                    if not _get_title(cand) and len(cand) >= 3 and len(cand.split()) <= 6:
                        comp = cand
                        matched_adjs.add(adj)
                        break

            # Search nearby lines for date if not yet found
            if not period:
                for d_adj in range(max(0, i - 3), min(len(lines), i + 4)):
                    if d_adj not in used_indices and not _is_junk(lines[d_adj]) and not _is_academic(lines[d_adj]):
                        pm = _PERIOD_RANGE_RE.search(lines[d_adj])
                        if pm:
                            period = pm.group(1).strip()
                            matched_adjs.add(d_adj)
                            break
                        single_yr = re.search(r"\b(?:19|20)\d{2}\b", lines[d_adj])
                        if single_yr:
                            period = single_yr.group(0).strip()
                            matched_adjs.add(d_adj)
                            break

            # Clean trailing noise/single letters from company
            comp = re.sub(r"[\s,|-]+[a-zA-Z]$", "", comp).strip()
            comp = re.sub(r"[\s,|-]+$", "", comp).strip()

            entries.append({
                "title": title,
                "company": comp,
                "period": period,
            })
            used_indices.add(i)
            used_indices.update(matched_adjs)

    # Deduplicate structured entries and reject OCR garbage titles/companies
    def _looks_like_ocr_noise(s: str) -> bool:
        """Return True if string looks like OCR gibberish, not real text."""
        if not s:
            return True
        # Reject if more than 30% of characters are non-alphanumeric/space
        alpha_ratio = sum(1 for c in s if c.isalpha()) / max(len(s), 1)
        if alpha_ratio < 0.45:
            return True
        # Reject if all tokens are very short (≤2 chars) - noise like 'Af, SD, TT'
        tokens = s.split()
        if tokens and all(len(t) <= 2 for t in tokens):
            return True
        # Reject if the title contains known OCR-noise substrings
        noise_patterns = [r"\bAf,\b", r"\bSD\b", r"\bTT\b", r"\bGINS\b", r"\bDein\b",
                          r"\bDain\b", r"\bmarwerIns\b", r"\brecunicat\b", r"\bwncuaces\b",
                          r"\bAvml\b", r"\bSrophic\b", r"\burxcesen\b"]
        for pat in noise_patterns:
            if re.search(pat, s, re.I):
                return True
        # Reject if title ends with punctuation and starts with non-capital
        if len(s) > 0 and not s[0].isupper() and len(tokens) == 1 and len(s) < 10:
            return True
        return False

    # Extract candidate name from top of text (heuristic: first line, all-caps or title-case)
    _cand_name = ""
    for ln in search_text.split("\n")[:5]:
        ln = ln.strip()
        if ln and 2 <= len(ln.split()) <= 4 and all(w[0].isupper() for w in ln.split() if w and w[0].isalpha()):
            _cand_name = ln.upper()
            break

    # Sentence fragments that are NOT job titles
    _NOT_TITLE_STARTERS = re.compile(
        r"^(?:and\b|to\b|the\b|a\b|an\b|in\b|of\b|for\b|with\b|on\b|aim\b|by\b|or\b|that\b|"
        r"this\b|where\b|while\b|which\b|who\b|how\b|when\b|seeking\b|looking\b|passionate)",
        re.IGNORECASE
    )

    deduped_entries = []
    seen = set()
    for e in entries:
        t = e["title"].strip()
        c = re.sub(r"[\s,|-]+[a-zA-Z]$", "", e["company"].strip()).strip("|-–, ")
        p = e["period"].strip()
        key = (t.lower(), c.lower())

        # Skip if title looks like OCR noise
        if _looks_like_ocr_noise(t):
            continue
        # Skip if title starts with a sentence fragment word
        if _NOT_TITLE_STARTERS.match(t):
            continue
        # Skip if company equals the candidate's own name (e.g. "ML DEVELOPER | NISHAN GAYAK")
        if _cand_name and c.upper() == _cand_name:
            c = ""
        # Skip if company looks like a project date range, not a real company
        if re.match(r"^\w[\w\s]+ \d{4}\s*[-–]\s*\d{4}$", c):
            c = ""
        if key not in seen and t and not _is_academic(t):
            seen.add(key)
            deduped_entries.append({
                "title": t,
                "company": c,
                "period": p,
            })

    # Prepare backward-compatible job_titles and organizations lists
    job_titles = [e["title"] for e in deduped_entries if e["title"]]
    organizations = [e["company"] for e in deduped_entries if e["company"]]

    # Fallback to spaCy ORGs only if no companies found and section_text exists
    if not organizations and section_text:
        nlp = _get_nlp()
        doc = nlp(search_text[:2500])
        seen_orgs = set()
        for ent in doc.ents:
            if ent.label_ == "ORG":
                org = _clean_noise(ent.text)
                if (
                    len(org) > 3
                    and org.lower() not in seen_orgs
                    and not _is_junk(org)
                    and not _is_academic(org)
                    and len(org.split()) <= 5
                ):
                    seen_orgs.add(org.lower())
                    organizations.append(org)

    return {
        "experience_entries": deduped_entries[:8],
        "job_titles": job_titles[:8],
        "organizations": organizations[:8],
        "raw_section": section_text[:500] if section_text else "",
    }


def extract_certifications(text: str) -> list:
    """Extract certifications from the certifications section or inline text."""
    section_text = _extract_section_text(text, "certifications")
    search_text = section_text if section_text else text
    
    cert_patterns = [
        r"\b(?:Machine\s+Learning\s+with\s+Python|Introduction\s+to\s+AI|Responsive\s+Web\s+Design|UI/UX\s+Design(?:\s+Basics)?|Git\s*&\s*GitHub(?:\s+Fundamentals)?|[A-Z][a-zA-Z0-9\s\&\.\-]{3,40}\s*-\s*(?:Coursera|Udemy|freeCodeCamp|edX|IBM|Google|Microsoft))\b[^\n]*",
        r"\b(?:Certified|Certificate(?:\s+in)?|Certification(?:\s+in)?|AWS\s+Certified|PMP|CPA|CISSP|CEH|CompTIA|Oracle|Microsoft|Google|Cisco)\b[^\n]{0,80}",
    ]
    cert_re = re.compile("|".join(cert_patterns), re.IGNORECASE)
    
    # Words/patterns that indicate garbage, not real certifications
    _CERT_NOISE = {
        "nepal", "kathmandu", "maitidevi", "lalitpur", "bhaktapur",
        "contact", "phone", "email", "address", "mobile",
    }

    def _is_valid_cert(c):
        if len(c) < 8 or len(c.split()) < 2:
            return False
        c_strip = re.sub(r'^[^a-zA-Z]+', '', c).strip()
        if len(c_strip) < 8:
            return False
        if "@" in c or re.search(r"\+?\d{7,}", c):
            return False
        if re.search(r"\b(?:contact|http|www|linkedin|github|gralicon|gralcon|gmail|\.com|trees)\b", c, re.I):
            return False
        c_words = set(w.lower() for w in re.findall(r'[a-zA-Z]+', c))
        if c_words & _CERT_NOISE:
            return False
        # Reject if contains tilde (OCR separator noise)
        if "~" in c:
            return False
        words = c.split()
        # Reject if too many very short words (garbled OCR)
        garbled_count = sum(1 for w in words if len(w) <= 2)
        if len(words) > 0 and garbled_count / len(words) > 0.35:
            return False
        # Vowel ratio check: real English text ~38-45% vowels
        letters = re.sub(r'[^a-zA-Z]', '', c)
        if len(letters) >= 8:
            vowel_ratio = sum(1 for ch in letters.lower() if ch in 'aeiou') / len(letters)
            if vowel_ratio < 0.25:
                return False
        # Must contain at least one recognized cert anchor word or provider
        _CERT_ANCHORS = {
            "coursera", "udemy", "freecodecamp", "edx", "ibm", "google",
            "microsoft", "cisco", "oracle", "aws", "comptia",
            "certified", "certificate", "certification",
            "machine", "learning", "design", "python", "javascript",
            "responsive", "introduction", "fundamentals", "git", "github",
        }
        if not (c_words & _CERT_ANCHORS):
            return False
        return True

    found = []
    seen = set()
    for m in cert_re.finditer(search_text):
        c = m.group(0).strip()
        # Strip leading bullet/symbol characters
        c = re.sub(r'^[\-\*\+\s\t©®™•▪►◆]+', '', c).strip()
        if c.lower() not in seen and _is_valid_cert(c):
            seen.add(c.lower())
            found.append(c)
            
    if not found and section_text:
        # Split by newlines for better accuracy (cert names are usually one per line)
        cert_lines = [l.strip() for l in section_text.split("\n") if l.strip()]
        for c in cert_lines:
            c = re.sub(r'^[©®™•\-\*\+\s]+', '', c).strip()
            if c.lower() not in seen and _is_valid_cert(c):
                seen.add(c.lower())
                found.append(c)
        
    return found[:10]


def extract_languages(text: str) -> list:
    """Extract languages (spoken) from resume."""
    section_text = _extract_section_text(text, "languages")
    if not section_text:
        return []
    langs = _split_skills_text(section_text)
    
    # Known human languages for validation
    _KNOWN_LANGUAGES = {
        "english", "hindi", "nepali", "nepal", "spanish", "french", "german",
        "chinese", "mandarin", "cantonese", "japanese", "korean", "arabic",
        "portuguese", "russian", "italian", "dutch", "turkish", "polish",
        "swedish", "danish", "norwegian", "finnish", "greek", "thai",
        "vietnamese", "indonesian", "malay", "tagalog", "bengali", "urdu",
        "tamil", "telugu", "marathi", "gujarati", "punjabi", "kannada",
        "malayalam", "sinhala", "burmese", "khmer", "lao", "tibetan",
        "sanskrit", "pali", "newari", "maithili", "bhojpuri", "awadhi",
        "swahili", "amharic", "yoruba", "igbo", "hausa", "zulu",
        "czech", "slovak", "hungarian", "romanian", "bulgarian", "serbian",
        "croatian", "bosnian", "albanian", "macedonian", "slovenian",
        "ukrainian", "belarusian", "georgian", "armenian", "azerbaijani",
        "persian", "farsi", "pashto", "dari", "kurdish", "hebrew",
        "afrikaans", "catalan", "basque", "galician", "welsh", "irish",
        "scottish", "gaelic", "latvian", "lithuanian", "estonian",
    }
    
    # Common OCR misspellings of language names
    _OCR_LANG_MAP = {
        "exgith": "English", "engih": "English", "engish": "English",
        "englsh": "English", "engilsh": "English",
        "hind": "Hindi", "hinde": "Hindi", "hindl": "Hindi",
        "nepalr": "Nepali", "nepall": "Nepali", "nepa": "Nepali",
        "naive": "Native",  # proficiency level, not a language
    }
    
    _NOISE_LANG = {
        "skills", "soft skills", "technical", "proficiency", "native",
        "professional", "vancuaces", "rececai", "naive",
        # Soft skills that OCR may garble into language section
        "communication", "creativity", "leadership", "teamwork",
        "problem solving", "fast learner", "time management",
        "adaptability", "collaboration", "presentation",
    }
    
    cleaned_langs = []
    for l in langs:
        l_clean = re.sub(r"^[+\-•*©°«\d\s\(\)]+", "", l).strip()
        l_clean = re.sub(r"[\-–—/~]+\s*(?:Native|Professional|Fluent|Proficiency|Intermediate|Basic|Conversational).*$", "", l_clean, flags=re.I).strip()
        # Remove proficiency levels from end
        l_clean = re.sub(r"\s+(?:Native|Professional|Fluent|Intermediate|Basic|Conversational|Proficiency|Prfcercy|Professonal).*$", "", l_clean, flags=re.I).strip()
        
        # Fix OCR artifacts
        if l_clean.lower() in _OCR_LANG_MAP:
            mapped = _OCR_LANG_MAP[l_clean.lower()]
            if mapped == "Native":  # skip proficiency level
                continue
            l_clean = mapped
        
        # "Nepal" in languages section likely means "Nepali"
        if l_clean.lower() == "nepal":
            l_clean = "Nepali"
        
        # Skip noise words and soft skills
        if l_clean.lower() in _NOISE_LANG:
            continue
        if any(kw in l_clean.lower() for kw in ["soft", "skill", "technic", "profic", "solving", "learner", "management", "adaptab", "collabor", "communicat", "creativ", "leadersh", "teamw"]):
            continue
        
        # Validate: MUST be a known language - no looser fallback to avoid OCR noise
        first_word = l_clean.split()[0].lower() if l_clean.split() else ""
        is_known = first_word in _KNOWN_LANGUAGES or l_clean.lower() in _KNOWN_LANGUAGES
        
        if is_known:
            if l_clean not in cleaned_langs:
                cleaned_langs.append(l_clean)
    
    return cleaned_langs[:10]


# ═══════════════════════════════════════════════════════════════════════════════
#  Master extraction function
# ═══════════════════════════════════════════════════════════════════════════════

def extract_all(text: str) -> dict:
    """
    Run the full hybrid information extraction pipeline.
    Returns a structured dictionary of extracted fields.

    This is the primary function called by the Streamlit app and notebook.
    """
    default_empty = {
        "name": None,
        "email": None,
        "phone": None,
        "location": None,
        "links": {"urls": []},
        "skills": [],
        "degrees": [],
        "universities": [],
        "job_titles": [],
        "organizations": [],
        "experience_entries": [],
        "certifications": [],
        "languages": [],
    }

    if not text or not isinstance(text, str) or not text.strip():
        res = dict(default_empty)
        res["error"] = "Empty resume text provided."
        return res

    education = extract_education(text)
    experience = extract_experience(text)

    result = {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "location": extract_location(text),
        "links": extract_urls(text),
        "skills": extract_skills(text),
        "degrees": education["degrees"],
        "universities": education["universities"],
        "experience_entries": experience.get("experience_entries", []),
        "job_titles": experience["job_titles"],
        "organizations": experience["organizations"],
        "certifications": extract_certifications(text),
        "languages": extract_languages(text),
    }

    return result


# Alias for backward-compatibility with notebook and external scripts
extract_resume_info = extract_all

