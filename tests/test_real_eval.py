from boardforge.models import Job
from boardforge.real_eval import build_corpus_snapshot


def _job(ext: str, title: str, company: str, country: str, source_kind: str = "greenhouse") -> Job:
    return Job(
        source_kind=source_kind,
        source_key=company.casefold(),
        external_id=ext,
        title=title,
        company=company,
        country=country,
    )


def test_real_corpus_snapshot_filters_country_and_test_rows():
    payload = build_corpus_snapshot(
        [
            _job("2", "ML Engineer", "Beta", "India"),
            _job("1", "Backend Engineer", "Acme", "India", source_kind="test"),
            _job("3", "ML Engineer", "Gamma", "USA"),
            _job("4", "AI Engineer", "Alpha", "India"),
        ],
        country="India",
    )

    assert payload["country"] == "India"
    assert payload["excluded_test_jobs"] == 1
    assert [job["external_id"] for job in payload["jobs"]] == ["4", "2"]


def test_real_corpus_snapshot_can_include_test_rows_explicitly():
    payload = build_corpus_snapshot(
        [_job("1", "Backend Engineer", "Acme", "India", source_kind="test")],
        country="India",
        include_test=True,
    )

    assert payload["excluded_test_jobs"] == 0
    assert [job["external_id"] for job in payload["jobs"]] == ["1"]
