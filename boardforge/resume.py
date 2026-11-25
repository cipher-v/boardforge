from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

from boardforge.models import CandidateEvidence, CandidateProfile, Preferences


SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "Python": (r"\bpython\b",),
    "Java": (r"\bjava\b",),
    "JavaScript": (r"\bjavascript\b", r"\bnode\.?js\b"),
    "TypeScript": (r"\btypescript\b",),
    "C++": (r"(?<!\w)c\+\+(?!\w)",),
    "SQL": (r"\bsql\b",),
    "PostgreSQL": (r"\bpostgres(?:ql)?\b",),
    "FastAPI": (r"\bfastapi\b",),
    "Django": (r"\bdjango\b",),
    "Flask": (r"\bflask\b",),
    "React": (r"\breact(?:\.js)?\b",),
    "PyTorch": (r"\bpytorch\b", r"\btorch\b"),
    "TensorFlow": (r"\btensorflow\b",),
    "scikit-learn": (r"\bscikit[- ]learn\b", r"\bsklearn\b"),
    "NLP": (r"\bnlp\b", r"\bnatural language processing\b"),
    "Transformers": (r"\btransformers?\b",),
    "Model Evaluation": (r"\bmodel evaluation\b",),
    "Representation Learning": (r"\brepresentation learning\b",),
    "PyTorch Geometric": (r"\bpytorch geometric\b",),
    "Pandas": (r"\bpandas\b",),
    "NumPy": (r"\bnumpy\b",),
    "OpenCV": (r"\bopencv\b",),
    "YOLO": (r"\byolo(?:v?\d+|\w*)?\b",),
    "LLMs": (r"\bllms?\b", r"\blarge language models?\b"),
    "RAG": (r"\brag\b", r"\bretrieval[- ]augmented generation\b"),
    "Embeddings": (r"\bembeddings?\b",),
    "Vector Search": (r"\bvector search\b",),
    "FAISS": (r"\bfaiss\b",),
    "LangChain": (r"\blangchain\b",),
    "LangGraph": (r"\blanggraph\b",),
    "LlamaIndex": (r"\bllamaindex\b",),
    "Google ADK": (r"\bgoogle adk\b",),
    "Gemini": (r"\bgemini\b",),
    "Vertex AI": (r"\bvertex ai\b",),
    "MCP": (r"\bmcp\b",),
    "Tool/Function Calling": (r"\btool/function calling\b", r"\bfunction calling\b"),
    "Structured Outputs": (r"\bstructured outputs?\b",),
    "LLM Evaluation": (r"\bllm evaluation\b",),
    "Sentence Transformers": (r"\bsentence transformers?\b",),
    "Reranking": (r"\breranking\b",),
    "Whisper": (r"\bwhisper\b",),
    "Tesseract OCR": (r"\btesseract(?: ocr)?\b",),
    "FFmpeg": (r"\bffmpeg\b",),
    "Pydantic": (r"\bpydantic\b",),
    "Docker": (r"\bdocker\b",),
    "Kubernetes": (r"\bkubernetes\b", r"\bk8s\b"),
    "AWS": (r"\baws\b", r"\bamazon web services\b"),
    "Cloud Run": (r"\bcloud run\b",),
    "Cloud Tasks": (r"\bcloud tasks\b",),
    "Firestore": (r"\bfirestore\b",),
    "Modal": (r"\bmodal\b",),
    "Git": (r"\bgit\b", r"\bgithub\b"),
    "GitHub Actions": (r"\bgithub actions\b",),
    "pytest": (r"\bpytest\b",),
    "Ruff": (r"\bruff\b",),
    "Linux": (r"\blinux\b",),
    "GitHub Webhooks/Checks": (r"\bgithub webhooks?/checks\b",),
    "Static Analysis": (r"\bstatic analysis\b",),
    "ASTs": (r"\basts?\b",),
    "Tree-sitter": (r"\btree[- ]sitter\b",),
    "REST APIs": (r"\brest(?:ful)?\s+apis?\b", r"\bapi development\b"),
    "Spark": (r"\b(?:apache\s+)?spark\b",),
    "Airflow": (r"\b(?:apache\s+)?airflow\b",),
}

SECTION_HEADINGS = {
    "summary": "summary",
    "education": "education",
    "open source contributions": "open_source",
    "projects": "projects",
    "publication": "publication",
    "publications": "publication",
    "technical skills": "skills",
    "skills": "skills",
    "experience": "experience",
    "work experience": "experience",
    "achievements": "achievements",
}

ROLE_EVIDENCE_PATTERNS: dict[str, tuple[str, ...]] = {
    "AI/ML": (
        r"\b(?:machine learning|agentic ai|llms?|rag|pytorch|tensorflow|"
        r"scikit[- ]learn|transformers?)\b",
    ),
    "Computer Vision": (
        r"\b(?:computer vision|opencv|object detection|image segmentation|"
        r"tesseract|video q&a)\b",
    ),
    "Backend": (r"\b(?:backend|fastapi|rest(?:ful)? apis?|server[- ]side)\b",),
    "Software Engineering": (
        r"\b(?:software engineering|production fixes?|static analysis|"
        r"regression tests?|github pr|typescript)\b",
    ),
    "Data Engineering": (r"\b(?:data engineering|etl|spark|airflow)\b",),
    "Data Science": (r"\b(?:data science|statistical|analytics)\b",),
}

DOMAIN_RULES: dict[str, tuple[str, ...]] = {
    "LLM / Generative AI": (r"\bgenerative ai\b", r"\bllms?\b", r"\blarge language models?\b"),
    "RAG / Retrieval": (r"\brag\b", r"\bretrieval\b", r"\bsemantic search\b", r"\bembeddings?\b"),
    "Computer Vision": (r"\bcomputer vision\b", r"\bobject detection\b", r"\bsegmentation\b", r"\bopencv\b", r"\byolo"),
    "Backend / APIs": (r"\bbackend\b", r"\bback-end\b", r"\bfastapi\b", r"\bdjango\b", r"\bflask\b", r"\brest(?:ful)?\s+apis?\b"),
    "Data / Analytics": (r"\bdata science\b", r"\bdata engineering\b", r"\betl\b", r"\bpandas\b", r"\banalytics\b"),
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

    # PDF extractors preserve visual line wrapping and commonly split a word as
    # ``Representa-\ntion``. Rejoin only an alphabetic hyphen at a line boundary;
    # ordinary in-line compounds remain unchanged.
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])", "", text)
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


def _evidence_snippet(text: str, patterns: tuple[str, ...], fallback: str) -> str:
    for line in text.splitlines():
        cleaned = " ".join(line.split())
        if cleaned and _matches_any(cleaned, patterns):
            return cleaned[:220]
    return fallback[:220]


def _sectioned_lines(text: str) -> list[tuple[str, str]]:
    section = ""
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        cleaned = " ".join(line.split())
        if not cleaned:
            continue
        heading = SECTION_HEADINGS.get(cleaned.casefold().rstrip(":"))
        if heading:
            section = heading
            continue
        rows.append((section, cleaned))
    return rows


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
    projects: list[str] = []
    experience: list[str] = []
    sectioned = _sectioned_lines(text)
    known_sections = {section for section, _ in sectioned if section}
    structured = bool(known_sections)
    has_education_section = "education" in known_sections
    has_project_section = bool({"projects", "open_source"} & known_sections)
    for section, cleaned in sectioned:
        if (
            (section == "education" or (not has_education_section and not structured))
            and EDUCATION_RE.search(cleaned)
        ):
            education.append(cleaned[:180])
        project_action = re.search(
            r"\b(?:built|developed|implemented|created|designed|shipped|fixed)\b",
            cleaned, re.I,
        )
        project_title = (
            not cleaned.startswith("•")
            and bool(re.search(
                r"\s[-–—]\s|\b(?:source|repository|live demo)(?:\s*\|.*)?$",
                cleaned, re.I,
            ))
            and not re.match(r"^merged pr\b", cleaned, re.I)
        )
        if (
            (section in {"projects", "open_source"} and (
                project_title or project_action
            ))
            or (not has_project_section and not structured and project_action)
        ):
            projects.append(cleaned[:220])
        if (section == "experience" or (not structured and section == "")) and (
            re.search(r"\b(?:intern(?:ship)?|experience|engineer|developer|researcher)\b", cleaned, re.I)
            and not EDUCATION_RE.search(cleaned)
        ):
            experience.append(cleaned[:220])

    education = _unique(education)[:5]
    projects = _unique(projects)[:10]
    experience = _unique(experience)[:5]

    evidence: list[CandidateEvidence] = []
    for canonical, patterns in SKILL_ALIASES.items():
        if canonical in skills:
            evidence.append(CandidateEvidence(
                kind="skill",
                value=canonical,
                snippet=_evidence_snippet(text, patterns, canonical),
                confidence=0.9,
            ))
    for role in role_signals:
        patterns = ROLE_EVIDENCE_PATTERNS.get(role, (re.escape(role),))
        supporting = _evidence_snippet(
            text, patterns, f"No source evidence found for inferred role: {role}",
        )
        evidence.append(CandidateEvidence(
            kind="role", value=role, snippet=supporting, confidence=0.65,
        ))
    for item in education:
        evidence.append(CandidateEvidence(
            kind="education", value=item, snippet=item, confidence=0.85,
        ))
    for item in projects:
        evidence.append(CandidateEvidence(
            kind="project", value=item[:80], snippet=item, confidence=0.8,
        ))
    for item in experience:
        evidence.append(CandidateEvidence(
            kind="experience", value=item[:80], snippet=item, confidence=0.7,
        ))

    return CandidateProfile(
        skills=_unique(skills),
        role_signals=_unique(role_signals),
        domains=_unique(domains),
        education=education,
        projects=projects,
        experience=experience,
        evidence=evidence[:100],
        resume_filename=Path(filename).name if filename else "",
    )


def apply_candidate_context(preferences: Preferences, profile: CandidateProfile) -> Preferences:
    """Blend resume evidence as soft terms without creating hard role constraints."""
    out = preferences.model_copy(deep=True)
    avoided = {term.casefold() for term in out.avoid_terms}

    soft_terms = profile.skills[:10] + profile.domains[:4] + profile.role_signals[:3]
    for term in soft_terms:
        if term.casefold() in avoided:
            continue
        if not any(existing.casefold() == term.casefold() for existing in out.prefer_terms):
            out.prefer_terms.append(term)
    return out
