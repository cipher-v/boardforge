import inspect
import json
from pathlib import Path

from boardforge import holdout_ai_audit
from boardforge.holdout import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "boardforge" / "data"


def _datasets():
    corpus = json.loads((DATA / "holdout_corpus_v1.json").read_text(encoding="utf-8"))
    provisional = json.loads(
        (DATA / "holdout_labels_v1_provisional.json").read_text(encoding="utf-8")
    )
    audited = json.loads(
        (DATA / "holdout_labels_v1_1_ai_audited.json").read_text(encoding="utf-8")
    )
    return corpus, provisional, audited


def test_ai_audit_preserves_frozen_inputs_and_is_not_human_verified():
    corpus, provisional, audited = _datasets()
    assert corpus["records_sha256"] == "d1f7a5bdbf65dceea31b6c5f67143a4c633c20fdaff88b5748b4837524e139e3"
    assert provisional["labels_sha256"] == "95166916abf9fcc1d221df3f94e06bde8c4a65dd952b98872305b4d1e31bb020"
    assert audited["corpus_records_sha256"] == corpus["records_sha256"]
    assert audited["provisional_labels_sha256"] == provisional["labels_sha256"]
    assert audited["verification_status"] == "ai_audited_not_human_verified"
    assert canonical_sha256(audited["cases"]) == audited["labels_sha256"]


def test_audit_scope_covers_all_requested_non_irrelevant_decisions():
    _, _, audited = _datasets()
    scope = audited["annotation_method"]["scope"]
    assert scope["all_provisional_non_irrelevant"] == 511
    assert scope["stratified_provisional_irrelevant_sample"] >= 150
    assert scope["additional_targeted_missed_positive_reviews"] > 0
    assert audited["statistics"]["correction_count"] > 0
    assert all(item["reason"] and item["evidence_snippet"] for item in audited["corrections"])


def test_reported_contamination_examples_are_irrelevant():
    _, _, audited = _datasets()
    cases = {case["name"]: case for case in audited["cases"]}
    vision = cases["computer_vision_perception"]["labels"]
    assert vision["greenhouse:anthropic:5073998008"]["label"] == "irrelevant"
    assert vision["greenhouse:anthropic:5383714008"]["label"] == "irrelevant"
    assert vision["ashby:notion:3979a6f5-2abb-447b-bd3a-0565b06bae24"]["label"] == "irrelevant"
    llm = cases["llm_inference_engineer"]["labels"]
    assert llm["greenhouse:roblox:8072244"]["label"] == "irrelevant"


def test_no_vacancy_and_unknown_eligibility_are_not_conflated():
    _, _, audited = _datasets()
    cases = {case["name"]: case for case in audited["cases"]}
    for case in cases.values():
        assert case["labels"]["greenhouse:anthropic:4980436008"]["label"] == "forbidden"
    india = cases["india_backend_only"]["labels"]
    unknown = india["greenhouse:cloudflare:8401112"]
    assert unknown["label"] == "direct"
    assert unknown["eligibility"] == "unresolved"
    confirmed = india["lever:palantir:10dfc8bc-99ad-4ca2-ab76-853cb90a92c2"]
    assert confirmed["label"] == "forbidden"


def test_semantic_audit_does_not_import_ranking_or_confidence_logic():
    source = inspect.getsource(holdout_ai_audit)
    assert "boardforge.confidence" not in source
    assert "rank_jobs" not in source
    assert "retrieve_and_rank_jobs" not in source
