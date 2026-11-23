from boardforge.evals import jaccard_at_k, ndcg_at_k, precision_at_k, recall_at_k


def test_ranking_metrics():
    pred = ["a","b","c"]
    rel = {"a","c","d"}
    assert precision_at_k(pred, rel, 3) == 2/3
    assert recall_at_k(pred, rel, 3) == 2/3
    assert 0 < ndcg_at_k(pred, rel, 3) <= 1


def test_paraphrase_stability_metric():
    assert jaccard_at_k(["a","b"], ["a","c"], 2) == 1/3
