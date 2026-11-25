from boardforge.models import Preferences
from boardforge.resume import apply_candidate_context, extract_candidate_profile, extract_resume_text
from boardforge.store import Store


SAMPLE = b"""
Tanish Example
M.Tech Computer Science, Example Institute of Technology
Projects
Built a multimodal RAG system using Python, FastAPI, PyTorch, embeddings and FAISS.
Built a computer vision pipeline with OpenCV and YOLO.
Skills: Python, SQL, PostgreSQL, Docker, Git
"""


def test_text_resume_extracts_candidate_profile():
    text = extract_resume_text("resume.txt", SAMPLE)
    profile = extract_candidate_profile(text, filename="resume.txt")
    assert "Python" in profile.skills
    assert "RAG" in profile.skills
    assert "OpenCV" in profile.skills
    assert "AI/ML" in profile.role_signals
    assert "Computer Vision" in profile.role_signals
    assert "Backend" in profile.role_signals
    assert profile.resume_filename == "resume.txt"


def test_resume_context_is_soft_and_does_not_create_hard_roles():
    profile = extract_candidate_profile(extract_resume_text("resume.txt", SAMPLE))
    prefs = apply_candidate_context(Preferences(avoid_terms=["Docker"]), profile)
    assert prefs.roles == []
    assert "Python" in prefs.prefer_terms
    assert "Docker" not in prefs.prefer_terms


def test_candidate_profile_round_trip(tmp_path):
    profile = extract_candidate_profile(extract_resume_text("resume.txt", SAMPLE), filename="resume.txt")
    store = Store(str(tmp_path / "profile.db"))
    store.save_candidate_profile("u", profile)
    loaded = store.load_candidate_profile("u")
    assert loaded is not None
    assert loaded.skills == profile.skills
    assert loaded.role_signals == profile.role_signals


def test_structured_resume_does_not_turn_summary_or_node_research_into_profile_facts():
    text = """
Summary
M.Tech graduate focused on machine learning. Built production systems.
Education
Example Institute of Technology 2024 - 2026
M.Tech in Computer Science & Engineering GPA: 8.0/10
Projects
Graph Research - Streaming Anomaly Detection Source
Developed incremental node representations for streaming graphs.
Backend Service - API Platform Source
Built a Python FastAPI backend.
Technical Skills
Python, TypeScript, C++, Google ADK, LangGraph, Tool/Function Calling, Pydantic, Cloud Run
"""
    profile = extract_candidate_profile(text, filename="resume.txt")

    assert "JavaScript" not in profile.skills
    assert {
        "C++", "Google ADK", "LangGraph", "Tool/Function Calling",
        "Pydantic", "Cloud Run",
    } <= set(profile.skills)
    assert "Data / Analytics" not in profile.domains
    assert all("focused on machine learning" not in item for item in profile.education)
    assert all("focused on machine learning" not in item for item in profile.projects)
    assert profile.education == [
        "Example Institute of Technology 2024 - 2026",
        "M.Tech in Computer Science & Engineering GPA: 8.0/10",
    ]
    assert any("Graph Research" in item for item in profile.projects)
    assert any("Backend Service" in item for item in profile.projects)
    assert profile.experience == []


def test_role_evidence_is_grounded_in_matching_resume_text():
    profile = extract_candidate_profile(extract_resume_text("resume.txt", SAMPLE))
    evidence = {row.value: row.snippet for row in profile.evidence if row.kind == "role"}
    assert "PyTorch" in evidence["AI/ML"] or "RAG" in evidence["AI/ML"]
    assert "OpenCV" in evidence["Computer Vision"] or "computer vision" in evidence["Computer Vision"].lower()
    assert "FastAPI" in evidence["Backend"]


def test_resume_text_rejoins_visual_line_break_hyphenation():
    raw = b"Summary with enough readable content for extraction.\nSkills\nRepresenta-\ntion Learning"
    text = extract_resume_text("resume.txt", raw)
    profile = extract_candidate_profile(text)
    assert "Representation Learning" in profile.skills
    assert "Representa-" not in text
