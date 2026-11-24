from __future__ import annotations

from datetime import datetime, timezone

from boardforge.models import Job


def build_corpus_snapshot(
    jobs: list[Job],
    *,
    country: str,
    include_test: bool = False,
) -> dict:
    """Freeze a real active-job subset for independently labeled relevance evals."""
    wanted_country = country.casefold()
    country_jobs = [job for job in jobs if job.country.casefold() == wanted_country]
    excluded_test_jobs = sum(job.source_kind == "test" for job in country_jobs)

    selected = [
        job
        for job in country_jobs
        if include_test or job.source_kind != "test"
    ]
    selected.sort(
        key=lambda job: (
            job.company.casefold(),
            job.title.casefold(),
            job.stable_id,
        )
    )

    return {
        "version": "1.0",
        "kind": "real_corpus_snapshot",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "country": country,
        "excluded_test_jobs": 0 if include_test else excluded_test_jobs,
        "jobs": [job.model_dump(mode="json") for job in selected],
    }
