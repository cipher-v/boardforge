from boardforge.preference_eval import run_preference_eval


def test_preference_extraction_regression_suite():
    report = run_preference_eval()
    assert report["cases"] >= 20
    assert report["exact_case_accuracy"] >= 0.95
    assert report["field_accuracy"] >= 0.98
