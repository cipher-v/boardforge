import json

from boardforge.connectors.generic import _walk


def test_walk_finds_nested_job_posting():
    payload = {
        "@graph": [
            {"@type": "Organization", "name": "X"},
            {"@type": "JobPosting", "title": "Engineer"},
        ]
    }
    nodes = list(_walk(payload))
    assert any(n.get("@type") == "JobPosting" for n in nodes if isinstance(n, dict))
