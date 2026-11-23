from boardforge.benchmark import run_benchmark


def test_starter_benchmark_has_no_forbidden_recommendations():
    report = run_benchmark()
    assert report["cases"] >= 6
    assert report["macro"]["forbidden_rate"] == 0
    assert report["macro"]["precision_at_k"] >= 0.8
    assert report["macro"]["recall_at_k"] >= 0.8
