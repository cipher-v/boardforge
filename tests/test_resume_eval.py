from boardforge.resume_eval import run_resume_eval


def test_resume_profile_regression_dataset():
    report = run_resume_eval()
    assert report["cases"] >= 8
    assert report["macro"]["skill_recall"] >= 0.95
    assert report["macro"]["role_recall"] >= 0.90
