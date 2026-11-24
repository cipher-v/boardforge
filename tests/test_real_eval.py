import json

import pytest

from boardforge.models import Job
from boardforge.real_eval import build_corpus_snapshot, run_real_eval
from boardforge.semantic import EmbeddingClient


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


class _FakeProvider:
    enabled = True
    model = "fake:2"
    dimensions = 2

    async def embed(self, texts, *, task_type="RETRIEVAL_DOCUMENT"):
        out = []
        for text in texts:
            low = text.casefold()
            if "machine learning" in low or "ml engineer" in low:
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        return out


@pytest.mark.asyncio
async def test_real_eval_runs_against_frozen_corpus(tmp_path):
    ml = _job("ml", "Machine Learning Engineer", "Acme", "India")
    sales = _job("sales", "Sales Executive", "Beta", "India")
    corpus = {
        "captured_at": "2026-01-01T00:00:00Z",
        "country": "India",
        "jobs": [ml.model_dump(mode="json"), sales.model_dump(mode="json")],
    }
    dataset = {
        "version": "test",
        "corpus": {"expected_jobs": 2},
        "cases": [{
            "name": "ml",
            "family": "ai_ml",
            "query": "machine learning engineer India",
            "relevant_ids": [ml.stable_id],
            "forbidden_ids": [sales.stable_id],
        }],
    }
    corpus_path = tmp_path / "corpus.json"
    dataset_path = tmp_path / "dataset.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    report = await run_real_eval(
        corpus_path,
        dataset_path,
        top_k=2,
        client=EmbeddingClient(_FakeProvider()),
    )

    assert report["status"] == "ok"
    assert report["corpus_snapshot"]["jobs"] == 2
    assert report["cases"] == 1
    assert report["macro"]["precision_among_returned"] == 1.0
    assert report["macro"]["top1_accuracy"] == 1.0


@pytest.mark.asyncio
async def test_real_eval_rejects_labels_for_missing_jobs(tmp_path):
    ml = _job("ml", "Machine Learning Engineer", "Acme", "India")
    corpus_path = tmp_path / "corpus.json"
    dataset_path = tmp_path / "dataset.json"
    corpus_path.write_text(
        json.dumps({"jobs": [ml.model_dump(mode="json")]}),
        encoding="utf-8",
    )
    dataset_path.write_text(
        json.dumps({
            "corpus": {"expected_jobs": 1},
            "cases": [{
                "name": "bad",
                "family": "ai_ml",
                "query": "ML",
                "relevant_ids": ["greenhouse:missing:123"],
                "forbidden_ids": [],
            }],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing from the frozen corpus"):
        await run_real_eval(
            corpus_path,
            dataset_path,
            client=EmbeddingClient(_FakeProvider()),
        )
