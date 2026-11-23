from boardforge.matching import hard_violations, rank_jobs, role_evidence
from boardforge.models import Job, Preferences


def job(ext: str, title: str, *, country="India", desc="", remote="unknown", min_exp=None):
    return Job(
        source_kind="x",
        source_key="demo",
        external_id=ext,
        title=title,
        company="Demo",
        country=country,
        description=desc,
        remote_policy=remote,
        min_experience_years=min_exp,
    )


def test_hard_constraints_and_rejection_memory():
    p = Preferences(
        roles=["AI/ML"],
        countries=["India"],
        max_required_experience_years=1,
        avoid_terms=["hardware"],
    )
    jobs = [
        job("good", "AI Engineer", desc="Python machine learning"),
        job("us", "AI Engineer", country="USA"),
        job("senior", "ML Engineer", min_exp=3),
        job("hw", "AI Engineer", desc="hardware embedded"),
    ]

    # The only clean candidate is explicitly rejected by user feedback.
    # All other candidates violate hard constraints, so the result is empty.
    ranked = rank_jobs(jobs, p, top_k=10, rejected_ids={"x:demo:good"})
    assert ranked == []


def test_hard_violations_are_detected():
    p = Preferences(
        countries=["India"],
        max_required_experience_years=1,
        avoid_terms=["hardware"],
    )

    assert "country" in hard_violations(job("us", "AI Engineer", country="USA"), p)
    assert "experience" in hard_violations(job("senior", "ML Engineer", min_exp=3), p)
    assert "avoid:hardware" in hard_violations(
        job("hw", "AI Engineer", desc="hardware embedded"),
        p,
    )


def test_clean_match_ranks_before_violation():
    p = Preferences(roles=["Backend"], avoid_terms=["sales"])
    good = job("1", "Backend Engineer", desc="Python APIs")
    bad = job("2", "Backend Engineer", desc="sales automation")
    ranked = rank_jobs([bad, good], p)
    assert [m.job.external_id for m in ranked] == ["1"]
    assert not ranked[0].hard_violations


def test_role_matching_uses_title_phrases_not_substrings():
    assert role_evidence(job("retail", "Retail Engineer"), "AI/ML") is None
    assert role_evidence(job("email", "Email Developer"), "AI/ML") is None
    assert role_evidence(job("llm", "LLM Engineer"), "AI/ML") is not None
    assert role_evidence(job("rag", "RAG Engineer"), "AI/ML") is not None
    assert role_evidence(job("cv", "Computer Vision Engineer"), "Computer Vision") is not None


def test_role_profile_excludes_location_only_matches():
    p = Preferences(roles=["AI/ML"], locations=["Bengaluru"], countries=["India"])
    relevant = job("1", "LLM Engineer")
    relevant.location = "Bengaluru"
    irrelevant = job("2", "UI/UX Designer")
    irrelevant.location = "Bengaluru"
    ranked = rank_jobs([irrelevant, relevant], p)
    assert [match.job.external_id for match in ranked] == ["1"]
    assert ranked[0].evidence[0].kind == "role"


def test_unknown_country_is_excluded_for_country_constraint():
    p = Preferences(countries=["India"])
    unknown = job("unknown", "Software Engineer", country="")
    assert "country_unknown" in hard_violations(unknown, p)
    assert rank_jobs([unknown], p) == []


def test_known_salary_below_minimum_is_excluded():
    p = Preferences(countries=["India"], min_salary_lpa=12)
    underpaid = job("low", "Software Engineer")
    underpaid.salary_currency = "INR"
    underpaid.salary_min = 900_000
    underpaid.salary_period = "year"
    assert "salary" in hard_violations(underpaid, p)


def test_unknown_salary_is_demoted_when_minimum_salary_is_requested():
    p = Preferences(roles=["AI/ML"], countries=["India"], min_salary_lpa=12)
    known = Job(
        source_kind="x", source_key="demo", external_id="known",
        title="AI Engineer", company="Demo", country="India",
        salary_currency="INR", salary_min=1400000, salary_period="year",
    )
    unknown = Job(
        source_kind="x", source_key="demo", external_id="unknown",
        title="AI Engineer", company="Demo", country="India",
    )
    ranked = rank_jobs([unknown, known], p, query="AI")
    assert ranked[0].job.external_id == "known"
    assert ranked[1].status == "needs_verification"
