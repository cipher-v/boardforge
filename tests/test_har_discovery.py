from boardforge.discovery.har import discover_from_har


def test_discovers_job_feed_from_har():
    har = {
        "log": {
            "entries": [
                {
                    "request":{"url":"https://careers.example.com/api/jobs","method":"GET"},
                    "response":{
                        "content":{
                            "mimeType":"application/json",
                            "text":'''{"data":{"jobs":[
                                {"id":"1","title":"AI Engineer","location":"Bengaluru","description":"Python ML","applyUrl":"https://x/1"},
                                {"id":"2","title":"Backend Engineer","location":"Pune","description":"APIs","applyUrl":"https://x/2"}
                            ]}}'''
                        }
                    }
                }
            ]
        }
    }
    found = discover_from_har(har)
    assert found
    best = found[0]
    assert best.endpoint.endswith("/api/jobs")
    assert best.fields.title == "title"
    assert best.fields.id == "id"
    assert best.response_path == ["data","jobs"]
    assert best.confidence >= 0.55


def test_ignores_authenticated_or_sensitive_har_requests():
    job_payload = '{"jobs":[{"id":"1","title":"Engineer","applyUrl":"https://x/1"}]}'
    har = {
        "log": {
            "entries": [
                {
                    "request": {
                        "url": "https://example.com/api/jobs?access_token=secret",
                        "method": "GET",
                        "headers": [],
                    },
                    "response": {"content": {"mimeType": "application/json", "text": job_payload}},
                },
                {
                    "request": {
                        "url": "https://example.com/private/jobs",
                        "method": "GET",
                        "headers": [{"name": "Authorization", "value": "Bearer secret"}],
                    },
                    "response": {"content": {"mimeType": "application/json", "text": job_payload}},
                },
            ]
        }
    }
    assert discover_from_har(har) == []
