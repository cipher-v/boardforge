from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

from boardforge.models import CandidateProfile, Preferences


SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "Python": (r"\bpython\b",),
    "Java": (r"\bjava\b",),
    "JavaScript": (r"\bjavascript\b", r"\bnode(?:\.js)?\b"),
    "TypeScript": (r"\btypescript\b",),
    "SQL": (r"\bsql\b",),
    "PostgreSQL": (r"\bpostgres(?:ql)?\b",),
    "FastAPI": (r"\bfastapi\b",),
    "Django": (r"\bdjango\b",),
    "Flask": (r"\bflask\b",),
    "React": (r"\breact(?:\.js)?\b",),
    "PyTorch": (r"\bpytorch\b", r"\btorch\b"),
    "TensorFlow": (r"\btensorflow\b",),
    "scikit-learn": (r"\bscikit[- ]learn\b", r"\bsklearn\b"),
    "Pandas": (r"\bpandas\b",),
    "NumPy": (r"\bnumpy\b",),
    "OpenCV": (r"\bopencv\b",),
    "YOLO": (r"\byolo(?:v?\d+|\w*)?\b",),
    "LLMs": (r"\bllms?\b", r"\blarge language models?\b"),
    "RAG": (r"\brag\b", r"\bretrieval[- ]augmented generation\b"),
    "Embeddings": (r"\bembeddings?\b",),
    "FAISS": (r"\bfaiss\b",),
    "LangChain": (r"\blangchain\b",),
    "Docker": (r"\bdocker\b",),
    "Kubernetes": (r"\bkubernetes\b", r"\bk8s\b"),
    "AWS": (r"\baws\b", r"\bamazon web services\b"),
    "Git": (r"\bgit\b", r"\bgithub\b"),
    "REST APIs": (r"\brest(?:ful)?\s+apis?\b", r"\bapi development\b"),
    "Spark": (r"\b(?:apache\s+)?spark\b",),
    "Airflow": (r"\b(?:apache\s+)?airflow\b",),
}

DOMAIN_RULES: dict[str, tuple[str, ...]] = {
    "LLM / Generative AI": (r"\bgenerative ai\b", r"\bllms?\b", r"\blarge language models?\b"),
    "RAG / Retrieval": (r"\brag\b", r"\bretrieval\b", r"\bsemantic search\b", r"\bembeddings?\b"),
    "Computer Vision": (r"\bcomputer vision\b", r"\bobject detection\b", r"\bsegmentation\b", r"\bopencv\b", r"\byolo"),
    "Backend / APIs": (r"\bbackend\b", r"\bback-end\b", r"\bfastapi\b", r"\bdjango\b", r"\bflask\b", r"\brest(?:ful)?\s+apis?\b"),
    "Data / Analytics": (r"\bdata science\b", r"\bdata engineering\b", r"\betl\b", r"\bpandas\b", r"\bsql\b"),
}

EDUCATION_RE = re.compile(
    r"\b(?:m\.?tech|b\.?tech|m\.?s\.?|b\.?s\.?|bachelor|master|ph\.?d|"
    r"computer science|engineering|university|institute of technology|iit\b)\b",
    re.I,
)


def extract_resume_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.casefold()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        elif ext == ".docx":
            from docx import Document
            document = Document(BytesIO(data))
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        elif ext in {".txt", ".md"}:
            text = data.decode("utf-8", errors="ignore")
        else:
            raise ValueError("supported resume formats are PDF, DOCX and TXT")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not read resume: {exc}") from exc

    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(text) < 40:
        raise ValueError("resume did not contain enough readable text")
    return text


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def extract_candidate_profile(text: str, *, filename: str = "") -> CandidateProfile:
    normalized = " ".join(text.split())

    skills = [
        canonical
        for canonical, patterns in SKILL_ALIASES.items()
        if _matches_any(normalized, patterns)
    ]

    domains = [
        domain
        for domain, patterns in DOMAIN_RULES.items()
        if _matches_any(normalized, patterns)
    ]

    skill_set = set(skills)
    role_signals: list[str] = []
    if (
        skill_set & {"PyTorch", "TensorFlow", "scikit-learn", "LLMs", "RAG", "Embeddings"}
        or re.search(r"\b(machine learning|deep learning|artificial intelligence|ai engineer|ml engineer)\b", normalized, re.I)
    ):
        role_signals.append("AI/ML")
    if (
        skill_set & {"OpenCV", "YOLO"}
        or re.search(r"\b(computer vision|perception|object detection|image segmentation)\b", normalized, re.I)
    ):
        role_signals.append("Computer Vision")
    if (
        skill_set & {"FastAPI", "Django", "Flask", "PostgreSQL", "REST APIs"}
        or re.search(r"\b(back[- ]?end|server[- ]side|api engineer)\b", normalized, re.I)
    ):
        role_signals.append("Backend")
    if (
        skill_set & {"Java", "JavaScript", "TypeScript", "React", "Docker", "Kubernetes"}
        or re.search(r"\bsoftware (?:engineer|developer|engineering)\b", normalized, re.I)
    ):
        role_signals.append("Software Engineering")
    if skill_set & {"Spark", "Airflow"} or re.search(r"\b(data engineer|data engineering|etl)\b", normalized, re.I):
        role_signals.append("Data Engineering")
    if (
        skill_set & {"Pandas", "NumPy", "scikit-learn"}
        and re.search(r"\b(data scientist|data science|statistical|analytics)\b", normalized, re.I)
    ):
        role_signals.append("Data Science")

    education: list[str] = []
    for line in text.splitlines():
        cleaned = " ".join(line.split())
        if cleaned and EDUCATION_RE.search(cleaned):
            education.append(cleaned[:180])
        if len(education) >= 5:
            break

    return CandidateProfile(
        skills=_unique(skills),
        role_signals=_unique(role_signals),
        domains=_unique(domains),
        education=_unique(education),
        resume_filename=Path(filename).name if filename else "",
    )


def apply_candidate_context(preferences: Preferences, profile: CandidateProfile) -> Preferences:
    """Blend resume evidence as soft terms without creating hard role constraints."""
    out = preferences.model_copy(deep=True)
    avoided = {term.casefold() for term in out.avoid_terms}

    soft_terms = profile.skills[:10] + profile.domains[:4]
    for term in soft_terms:
        if term.casefold() in avoided:
            continue
        if not any(existing.casefold() == term.casefold() for existing in out.prefer_terms):
            out.prefer_terms.append(term)
    return out
