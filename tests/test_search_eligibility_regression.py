from boardforge.confidence import parse_query_intent
from boardforge.matching import rank_jobs
from boardforge.models import Job, Preferences
from boardforge.planning import tool_names
from boardforge.preferences import deterministic_extract, effective_search_preferences, preserve_search_constraints


QUERY = ("Find entry-level software engineering and backend roles in India suitable for my profile. "
         "I have no formal employment experience. Include roles requiring up to one year of experience, "
         "and clearly flag any uncertain eligibility.")


def test_full_query_preserves_eligibility_without_fake_task_terms():
    p = deterministic_extract(QUERY)
    assert p.countries == ["India"]
    assert p.max_required_experience_years == 1
    assert "Software Engineering" in p.roles
    assert "Backend" in p.roles
    assert not parse_query_intent(p, QUERY).specific_terms
    assert tool_names(QUERY) == ["search_jobs"]


def test_technical_task_terms_are_preserved():
    q = "Find entry-level backend roles in India using Python and distributed systems, up to one year of experience"
    p = deterministic_extract(q)
    assert {"python", "distributed", "systems"} <= parse_query_intent(p, q).specific_terms


def test_refinement_cannot_relax_country_and_experience_cap():
    saved = Preferences()
    refined = effective_search_preferences(saved, "backend roles worldwide up to five years")
    guarded = preserve_search_constraints(saved, QUERY, refined)
    assert guarded.countries == ["India"]
    assert guarded.max_required_experience_years == 1
    assert saved.max_required_experience_years is None


def test_known_violations_excluded_and_unknown_remains_uncertain():
    def job(key, years, country="India"):
        return Job(source_kind="test", source_key="regression", external_id=key,
                   title="Software Engineer", company="Example", country=country,
                   description="Develop Python backend APIs and microservices.",
                   min_experience_years=years)
    p = deterministic_extract(QUERY)
    matches = rank_jobs([job("senior", 7), job("unknown", None), job("junior", 0),
                         job("overseas", 0, "USA")], p, query=QUERY)
    assert [m.job.external_id for m in matches] == ["junior", "unknown"]
    assert matches[0].status == "recommended"
    assert matches[1].status == "needs_verification"
