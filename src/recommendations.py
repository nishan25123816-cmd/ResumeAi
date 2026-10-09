def build_resume_recommendations(extracted: dict, category: str = "Unknown") -> dict:
    """Generate practical recommendations and quick CV health checks."""
    score = 100
    recommendations = []
    strengths = []
    missing_sections = []
    exact_writing_suggestions = []

    name = (extracted or {}).get("name")
    email = (extracted or {}).get("email")
    phone = (extracted or {}).get("phone")
    location = (extracted or {}).get("location")
    skills = (extracted or {}).get("skills") or []
    degrees = (extracted or {}).get("degrees") or []
    experience = (extracted or {}).get("experience_entries") or []
    job_titles = (extracted or {}).get("job_titles") or []
    universities = (extracted or {}).get("universities") or []
    text_lower = " ".join(str(s).lower() for s in skills)

    if name:
        strengths.append("Name/contact header is present.")
    else:
        recommendations.append("Add your full name at the top of the CV so recruiters can identify you quickly.")
        missing_sections.append("Name")
        score -= 18

    if email:
        strengths.append("Professional email is present.")
    else:
        recommendations.append("Add a professional email address in a consistent format, such as name@domain.com.")
        missing_sections.append("Email")
        score -= 18

    if phone:
        strengths.append("Phone number is included.")
    else:
        recommendations.append("Include a contact number so employers can reach you directly.")
        missing_sections.append("Phone")
        score -= 14

    if location:
        strengths.append("Location is available.")
    else:
        recommendations.append("Add your city or location to make your CV easier to shortlist locally.")
        missing_sections.append("Location")
        score -= 10

    if len(skills) >= 6:
        strengths.append("Skills section is reasonably detailed.")
    else:
        recommendations.append("Add more job-relevant skills to match the target role and strengthen keyword relevance.")
        missing_sections.append("Skills")
        score -= 18

    if degrees or universities:
        strengths.append("Education section appears to be present.")
    else:
        recommendations.append("Include education details, especially your degree, institution, and graduation year.")
        missing_sections.append("Education")
        score -= 10

    if experience or job_titles:
        strengths.append("Work experience is visible.")
    else:
        recommendations.append("Add a clear experience section with company names, roles, and dates.")
        missing_sections.append("Experience")
        score -= 20

    if not recommendations:
        recommendations.append("Your CV structure looks strong. Consider adding measurable outcomes and achievements to improve impact.")

    if category in ("Unknown", "HEALTHCARE"):
        recommendations.append("Use role-specific keywords from the target job description to improve category matching and recruiter relevance.")
        score -= 8

    if len(recommendations) > 4:
        recommendations = recommendations[:4]

    keyword_map = {
        "HEALTHCARE": ["patient care", "clinical support", "medical terminology", "health records", "HIPAA", "team collaboration"],
        "ENGINEERING": ["system design", "automation", "python", "testing", "requirements analysis", "optimization"],
        "FINANCE": ["financial analysis", "budgeting", "forecasting", "risk analysis", "excel", "reporting"],
        "SALES": ["client acquisition", "pipeline management", "negotiation", "CRM", "business development", "retention"],
        "IT": ["cloud", "sql", "python", "api", "agile", "project delivery"],
        "DENTIST": ["patient care", "dental procedures", "oral health", "treatment planning", "clinical dentistry", "infection control"],
        "DEFAULT": ["leadership", "communication", "problem solving", "teamwork", "results", "stakeholder management"],
    }

    category_key = category.upper() if category else "DEFAULT"
    keywords = keyword_map.get(category_key, keyword_map["DEFAULT"])

    skill_gap = []
    for kw in keywords:
        if kw.lower() not in text_lower:
            skill_gap.append(kw)
    skill_gap = skill_gap[:5]

    ats_score = max(0, min(100, int(score)))
    job_match_score = max(0, min(100, int((ats_score * 0.7) + (len(skills) * 4))))

    exact_writing_suggestions = [
        "Improved patient care and clinical support by delivering high-quality service in a fast-paced healthcare setting.",
        "Led cross-functional teamwork to improve service quality, patient satisfaction, and operational efficiency.",
        "Delivered measurable results through strong problem-solving, communication, and stakeholder collaboration.",
        "Developed and optimized workflows to improve reliability, performance, and customer experience.",
        "Managed responsibilities with a focus on quality assurance, compliance, and continuous process improvement.",
    ]

    if category_key == "DENTIST":
        exact_writing_suggestions = [
            "Provided comprehensive patient care with a focus on diagnosis, treatment planning, and oral health education.",
            "Maintained infection control standards and delivered high-quality dental procedures in a patient-centered clinic environment.",
            "Improved patient outcomes by combining clinical expertise, communication, and effective treatment coordination.",
        ]
    elif category_key == "ENGINEERING":
        exact_writing_suggestions = [
            "Designed and optimized engineering systems to improve performance, quality, and operational efficiency.",
            "Collaborated with cross-functional teams to deliver scalable solutions using automation, testing, and analytical problem solving.",
            "Improved system reliability and workflow efficiency through continuous optimization and structured engineering processes.",
        ]
    elif category_key == "HEALTHCARE":
        exact_writing_suggestions = [
            "Delivered patient-focused care while maintaining compliance, documentation accuracy, and service quality standards.",
            "Supported clinical operations through effective communication, teamwork, and continuous quality improvement.",
            "Improved healthcare service delivery by coordinating patient support, documentation, and operational efficiency.",
        ]

    return {
        "category": category,
        "score": ats_score,
        "ats_score": ats_score,
        "job_match_score": job_match_score,
        "recommendations": recommendations,
        "keywords": keywords,
        "skill_gap": skill_gap,
        "keyword_gap": skill_gap,
        "missing_sections": missing_sections,
        "strengths": strengths,
        "exact_writing_suggestions": exact_writing_suggestions,
    }
