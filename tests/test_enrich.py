from boardforge.enrich import enrich_job
from boardforge.models import Job


def base(**kwargs):
    data = dict(source_kind="x",source_key="y",external_id="1",title="Engineer",company="Demo")
    data.update(kwargs)
    return Job(**data)


def test_infers_india_remote_and_experience():
    j = enrich_job(base(location="Bengaluru",description="Remote role. 0-2 years experience."))
    assert j.country == "India"
    assert j.remote_policy == "remote"
    assert j.min_experience_years == 0
    assert j.max_experience_years == 2
