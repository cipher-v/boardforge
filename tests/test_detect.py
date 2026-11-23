from boardforge.connectors.detect import detect_source


def test_detect_greenhouse():
    s = detect_source("https://job-boards.greenhouse.io/skildai-careers")
    assert s.kind == "greenhouse"
    assert s.key == "skildai-careers"


def test_detect_lever():
    s = detect_source("https://jobs.lever.co/trackmind")
    assert s.kind == "lever"
    assert s.key == "trackmind"


def test_detect_ashby():
    s = detect_source("https://jobs.ashbyhq.com/example")
    assert s.kind == "ashby"
    assert s.key == "example"


def test_detect_generic():
    s = detect_source("https://careers.example.com/jobs")
    assert s.kind == "generic"
