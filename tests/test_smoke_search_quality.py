from __future__ import annotations

from boardforge.connectors.lever import _text as lever_text
from boardforge.enrich import enrich_job, extract_job_experience, reenrich_job_metadata
from boardforge.matching import rank_jobs
from boardforge.models import Job, Preferences
from boardforge.preferences import deterministic_extract, effective_search_preferences


SMOKE_QUERY = (
    "Find entry-level software engineering and backend roles in India suitable for my profile. "
    "I have no formal employment experience. Include roles requiring up to one year of experience, "
    "and clearly flag any uncertain eligibility."
)


def _job(
    ext: str,
    title: str,
    *,
    company: str = "Demo",
    source_kind: str = "lever",
    location: str = "Bengaluru, India",
    country: str = "India",
    description: str = "Build Python backend services, REST APIs, and microservices.",
    min_exp: float | None = None,
    max_exp: float | None = None,
) -> Job:
    return Job(
        source_kind=source_kind,
        source_key=company.casefold(),
        external_id=ext,
        title=title,
        company=company,
        location=location,
        country=country,
        description=description,
        min_experience_years=min_exp,
        max_experience_years=max_exp,
    )


def test_entry_level_fresher_and_one_year_searches_exclude_conflicting_senior_titles():
    jobs = [
        _job("aiprise-swe1", "Software Engineer I (Bangalore, India)", company="aiprise", source_kind="ashby", min_exp=0.0, max_exp=2.0),
        _job("jr-be", "Junior Backend Engineer", company="acme", min_exp=1.0),
        _job("ambiguous-odp", "Software Engineer - Open Data Platform (ODP)", company="acceldata"),
        _job("ambiguous-devops", "AI Software Engineer (DevOps) - Bangalore, India", company="zimperium"),
        _job("sr-fs", "Senior Full Stack Engineer (Noida, BLR, India)", company="levelai"),
        _job("sr-ai", "Senior AI Engineer – Backend Engineering (Agentic AI)", company="acceldata"),
        _job("principal-be", "Principal Software Engineer — Backend & Infrastructure - Noida / Bengaluru", company="levelai"),
        _job("sr-be", "Senior Backend Engineer - Customer Engineering", company="levelai"),
        _job("staff-be", "Staff Software Engineer- Backend", company="acceldata"),
        _job("lead-ai", "Lead Software Engineer - AI Agents", company="levelai"),
        _job("eng-mgr", "Engineering Manager", company="razorpay"),
        _job("swe3", "Software Engineer III (Bangalore, India)", company="aiprise"),
    ]

    for query in (
        SMOKE_QUERY,
        "Fresher backend jobs in India",
        "Backend roles in India requiring up to one year of experience",
        "Entry-level backend roles in India, avoid senior or lead titles",
    ):
        prefs = effective_search_preferences(Preferences(), query)
        ranked = rank_jobs(jobs, prefs, query=query, top_k=15)
        returned_ids = [m.job.external_id for m in ranked]
        assert "aiprise-swe1" in returned_ids
        assert "ambiguous-odp" in returned_ids
        assert "ambiguous-devops" in returned_ids
        for forbidden in ("sr-fs", "sr-ai", "principal-be", "sr-be", "staff-be", "lead-ai", "eng-mgr", "swe3"):
            assert forbidden not in returned_ids, f"{forbidden} leaked for {query!r}"

        by_id = {m.job.external_id: m for m in ranked}
        assert by_id["jr-be"].status == "recommended"
        if "software engineering" in query.casefold():
            assert by_id["aiprise-swe1"].status == "recommended"
        assert by_id["ambiguous-odp"].status == "needs_verification"
        assert by_id["ambiguous-devops"].status == "needs_verification"


def test_senior_exclusion_not_applied_to_general_or_explicitly_senior_searches():
    jobs = [
        _job("swe1", "Software Engineer I", min_exp=0.0, max_exp=2.0),
        _job("sr-be", "Senior Backend Engineer", min_exp=None),
        _job("staff-be", "Staff Backend Engineer", min_exp=None),
        _job("principal-be", "Principal Software Engineer — Backend & Infrastructure", min_exp=None),
    ]

    # 1. General search (no experience constraint) preserves Senior, Staff, and Principal roles.
    general_q = "Backend software engineer roles in India"
    general_prefs = effective_search_preferences(Preferences(), general_q)
    general_ids = {m.job.external_id for m in rank_jobs(jobs, general_prefs, query=general_q, top_k=10)}
    assert {"swe1", "sr-be", "staff-be", "principal-be"} <= general_ids

    # 2. Multi-year cap (>1 year) without entry-level cue does not trigger the <=1yr title exclusion.
    three_yr_q = "Backend roles in India requiring up to 3 years of experience"
    three_yr_prefs = effective_search_preferences(Preferences(), three_yr_q)
    three_yr_ids = {m.job.external_id for m in rank_jobs(jobs, three_yr_prefs, query=three_yr_q, top_k=10)}
    assert "sr-be" in three_yr_ids

    # 3. Explicitly senior search works even if the user previously saved a <=1 year preference.
    saved_entry = deterministic_extract("From now on, entry-level roles in India")
    assert saved_entry.max_required_experience_years == 1.0
    senior_q = "Senior backend roles in India"
    senior_prefs = effective_search_preferences(saved_entry, senior_q)
    senior_ids = {m.job.external_id for m in rank_jobs(jobs, senior_prefs, query=senior_q, top_k=10)}
    assert "sr-be" in senior_ids


def test_experience_extraction_handles_common_phrasings_written_numbers_and_sections():
    cases = [
        # Written-out numbers and parenthetical confirmations
        ("Required: Minimum of five (5) years of experience as a software engineer.", (5.0, None)),
        ("Required Skills and Experience: At least five (5) years of hands-on experience.", (5.0, None)),
        ("What You'll Bring:\n- Three to five years of backend engineering experience", (3.0, 5.0)),
        # 'minimum of', 'a minimum of', spaced '+', written 'plus', 'more than'
        ("We are seeking a Senior Software Engineer with a minimum of 7 years of experience in distributed systems.", (7.0, None)),
        ("Minimum of 3 years of relevant experience in backend engineering.", (3.0, None)),
        ("Mandatory Qualifications: 6 + Years of overall experience in software systems.", (6.0, None)),
        ("What We're Looking For:\n- 10 plus years in solutions engineering and architecture", (10.0, None)),
        ("*Required: More than 10 years of combined experience in enterprise software.", (10.0, None)),
        # Plain unranged required experience statements & ranges with '+'
        ("Required: 4 years of recent, continuous backend engineering experience.", (4.0, None)),
        ("Required: Possess a total of 8 years of tech sales experience, with a minimum of 6 years closing.", (8.0, None)),
        ("MUST-HAVE:\n- 0-2 years of professional software engineering experience", (0.0, 2.0)),
        ("MUST-HAVE:\n- 4-7+ years of professional software engineering experience", (4.0, 7.0)),
        # Upper-bound-only junior phrasing & 'or' alternative pathways
        ("Open to freshers or engineers with up to 2 years of professional experience.", (0.0, 2.0)),
        ("Required: At least 1 year of backend engineering experience, or 2-3+ years of QA automation experience.", (1.0, None)),
        ("Minimum Experience: 3 years", (3.0, None)),
    ]
    for snippet, expected in cases:
        got = extract_job_experience("Software Engineer", snippet)
        assert got == expected, f"For {snippet!r}, expected {expected}, got {got}"


def test_experience_extraction_ignores_preferred_sections_and_unrelated_numbers():
    # Preferred / Nice-to-have / Desired only -> None
    preferred_only = [
        "MUST-HAVE:\n- Strong Python and PostgreSQL fundamentals\nNICE-TO-HAVE:\n- 2+ years of experience in FinTech or payments",
        "Requirements:\n- Strong Java and Spring Boot skills\nPreferred Qualifications:\n- 3-5 years of experience with Kafka",
        "Desired: 1+ year of customer-facing experience (but not required!)",
        "Build backend APIs. 2+ years of Rust experience is a plus.",
        "Build backend APIs. 3+ years of Kubernetes experience preferred.",
    ]
    for snippet in preferred_only:
        assert extract_job_experience("Software Engineer", snippet) == (None, None), snippet

    # Preferred section higher number must NOT override lower Required section number
    mixed = (
        "Preferred Qualifications:\n- 5+ years of distributed systems experience\n"
        "Minimum Qualifications:\n- 0-2 years of software engineering experience"
    )
    assert extract_job_experience("Software Engineer I", mixed) == (0.0, 2.0)

    # Unrelated numerical statements (roadmap, company history, onboarding, degree, age, vesting, salary) -> None
    unrelated = [
        "Over the next 3 years, we are rebuilding nearly every component of our observability stack.",
        "Named one of Best Startup Employers for two consecutive years and founded 6 years ago.",
        "What you'll do\nWhat success looks like in your first year\nYou'll own backend services.",
        "Over your first year, you learn our platform architecture.",
        "Requirements: 4-year degree in Computer Science or related field; rising senior in summer 2027.",
        "Applicants must be at least 18 years of age. Equity has a 4-year vesting schedule.",
        "Achieving quotas of $1.6M+ per year across 200+ countries.",
        "We don't care about 5+ years of experience; show us systems you have shipped.",
    ]
    for snippet in unrelated:
        assert extract_job_experience("Software Engineer", snippet) == (None, None), snippet

    # Company/persona boilerplate in intro does not override Must-Have section
    persona_then_req = (
        "About the Role\n"
        "Our customers have been doing this work for 15+ years and know the domain deeply.\n"
        "MUST-HAVE:\n"
        "- 5+ years in backend engineering"
    )
    assert extract_job_experience("Backend Lead", persona_then_req) == (5.0, None)


def test_lever_connector_preserves_list_bullet_content_for_experience_enrichment():
    lever_posting = {
        "id": "lever-odp-1",
        "text": "Software Engineer - Open Data Platform (ODP)",
        "descriptionPlain": "About the Role: We’re hiring a software engineer to build ODP.",
        "lists": [
            {
                "text": "What You’ll Do",
                "content": "<li>Build distributed data platform services in Java and Python.</li>",
            },
            {
                "text": "What We’re Looking For",
                "content": (
                    "<li>Minimum of 3 years of backend or distributed systems experience.</li>"
                    "<li>Strong SQL and Linux fundamentals.</li>"
                ),
            },
            {
                "text": "Good to Have",
                "content": "<li>5+ years of Apache Spark committer experience.</li>",
            },
        ],
        "additionalPlain": "Equal opportunity employer.",
    }
    extracted_text = lever_text(lever_posting)
    assert "Minimum of 3 years of backend or distributed systems experience." in extracted_text
    enriched = enrich_job(_job(
        "lever-odp-1",
        lever_posting["text"],
        company="acceldata",
        description=extracted_text,
    ))
    assert enriched.min_experience_years == 3.0
    assert enriched.max_experience_years is None

    # Re-enrichment also updates experience metadata on previously stored jobs
    stale = _job(
        "accel-sr",
        "Senior Software Engineer - Open Data Platform (ODP)",
        company="acceldata",
        description="We are seeking a Senior Software Engineer with a minimum of 7 years of experience in distributed systems.",
        min_exp=None,
    )
    updated, changed, note = reenrich_job_metadata(stale)
    assert changed is True
    assert updated.min_experience_years == 7.0
    assert "exp:None->7.0" in note
