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
