from boardforge.matching import hard_violations, rank_jobs
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
    ranked = rank_jobs(jobs, p, top_k=10, rejected_ids={"x:demo:good"})
    ids = [m.job.external_id for m in ranked]
    assert "good" not in ids
    assert ranked[0].hard_violations


def test_clean_match_ranks_before_violation():
    p = Preferences(roles=["Backend"], avoid_terms=["sales"])
    good = job("1", "Backend Engineer", desc="Python APIs")
    bad = job("2", "Backend Engineer", desc="sales automation")
    ranked = rank_jobs([bad, good], p)
    assert ranked[0].job.external_id == "1"
    assert not ranked[0].hard_violations
