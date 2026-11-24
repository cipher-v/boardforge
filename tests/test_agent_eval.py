from boardforge.agent_eval import deterministic_tool_plan, run_deterministic_tool_eval


def test_tool_planner_search_only():
    assert deterministic_tool_plan("Find backend roles in India") == ["search_jobs"]


def test_tool_planner_preference_then_search():
    assert deterministic_tool_plan("Don't show QA anymore. Find backend roles.") == [
        "update_preferences",
        "search_jobs",
    ]


def test_tool_planner_feedback_then_search():
    assert deterministic_tool_plan(
        "Reject benchmark:v2:ai-llm-blr and show other AI jobs"
    ) == ["record_feedback", "search_jobs"]


def test_agent_tool_dataset_has_40_cases():
    report = run_deterministic_tool_eval()
    assert report["cases"] == 40
    assert report["macro"]["exact_sequence_accuracy"] >= 0.95
    assert report["macro"]["tool_set_f1"] >= 0.95


def test_tool_planner_negated_show_is_preference_only():
    assert deterministic_tool_plan("Do not show QA roles anymore") == ["update_preferences"]


def test_tool_planner_not_clause_updates_preferences():
    assert deterministic_tool_plan("Not QA, but AI and backend are good") == ["update_preferences"]
