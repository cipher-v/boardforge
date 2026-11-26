import pytest
from boardforge.matching import rank_jobs, role_evidence
from boardforge.models import Job, Preferences
from boardforge.preferences import effective_search_preferences
from boardforge.resume import apply_candidate_context
from boardforge.models import CandidateProfile

QUERY = "Find entry-level artificial intelligence, machine learning and computer vision engineering roles in India requiring up to one year of experience. Include relevant internships."


def job(title, description, **kwargs):
    return Job(source_kind="greenhouse", source_key="example", external_id=title,
               company="Example", title=title, description=description,
               country="India", location="Bengaluru, India", **kwargs)


@pytest.mark.parametrize("title,description,kind", [
    ("Research Engineering Intern, AI Evaluations", "Run model inference through APIs. Build evaluation pipelines. Strong Python and LLMs.", "role"),
    ("Data Scientist - Intern", "Developing models and implementing algorithms. Familiar with ML concepts and algorithms. Python and SQL.", "adjacent_role"),
    ("Research Intern - AI Evaluations", "Apply social science methods to evaluate language technologies. No technical degree required.", None),
    ("Data Scientist", "Create Excel reports and business dashboards using Python.", None),
    ("Customer Support Engineer", "Python, model inference and machine learning.", None),
])
def test_role_evidence(title, description, kind):
    result = role_evidence(job(title, description), "AI/ML", query=QUERY)
    assert (result.kind if result else None) == kind


def test_partial_skills_and_coverage():
    prefs = apply_candidate_context(effective_search_preferences(Preferences(), QUERY),
                                    CandidateProfile(skills=["Python", "PyTorch", "SQL"]))
    jobs = [job("Research Engineering Intern, AI Evaluations", "Run model inference with TensorFlow and Python. Build evaluation pipelines."),
            job("Data Scientist - Intern", "Developing models using ML concepts and algorithms, Python and R."),
            job("Research Intern - AI Evaluations", "Social science research methods for language technologies."),
            job("Machine Learning Engineer", "Train ML models with TensorFlow.", min_experience_years=4)]
    result = {m.job.title: m for m in rank_jobs(jobs, prefs, query=QUERY, top_k=10)}
    assert set(result) == {jobs[0].title, jobs[1].title}
    assert result[jobs[0].title].status == "needs_verification"
    assert result[jobs[1].title].status == "alternative"


@pytest.mark.parametrize("title", ["SDE I", "Associate SDE", "Software Development Engineer Intern"])
def test_swe_abbreviations(title):
    assert role_evidence(job(title, "Build software services in Java."), "Software Engineering") is not None
