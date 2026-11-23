from boardforge.comparison import DEFAULT_PATH, run_comparison


def test_v2_benchmark_has_120_cases():
    report = run_comparison(DEFAULT_PATH, top_k=5)
    assert report["cases"] == 120
    assert report["jobs"] >= 20


def test_comparison_contains_expected_variants():
    report = run_comparison(DEFAULT_PATH, top_k=5)
    assert set(report["variants"]) == {
        "lexical_only",
        "constrained_lexical",
        "full_evidence",
        "full_evidence_feedback",
    }


def test_feedback_variant_does_not_regress_forbidden_rate():
    report = run_comparison(DEFAULT_PATH, top_k=5)
    full = report["variants"]["full_evidence"]["macro"]
    feedback = report["variants"]["full_evidence_feedback"]["macro"]
    assert feedback["forbidden_rate"] <= full["forbidden_rate"]
