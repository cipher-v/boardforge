from boardforge.enrich import enrich_job, infer_country, salary_min_lpa
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


def test_country_inference_uses_location_not_description_substrings():
    j = enrich_job(base(location="London", description="Great candidate capabilities"))
    assert j.country == ""
    assert infer_country("San Mateo, CA") == "USA"
    assert infer_country("Bengaluru, India") == "India"


def test_parses_lpa_and_monthly_inr_salary():
    annual = enrich_job(base(description="Compensation is 8-12 LPA"))
    assert salary_min_lpa(annual) == 8
    assert annual.salary_max == 1_200_000

    monthly = enrich_job(base(description="Stipend: ₹75,000 per month"))
    assert salary_min_lpa(monthly) == 9

    unpaid = enrich_job(base(location="Pune, India", description="Unpaid internship. No stipend."))
    assert salary_min_lpa(unpaid) == 0


def test_country_inference_handles_explicit_global_countries_conservatively():
    assert infer_country("Hybrid - Singapore") == "Singapore"
    assert infer_country("Remote - Argentina") == "Argentina"
    assert infer_country("Lenexa, Kansas") == "USA"
    assert infer_country("Remote - Spain") == "Spain"
    assert infer_country("Remote - Australia; Remote - Japan") == ""
    assert infer_country("Remote - Brazil; Remote - Colombia") == ""
