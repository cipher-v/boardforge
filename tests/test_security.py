import pytest

from boardforge.security import UnsafeUrlError, has_sensitive_query, validate_public_url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/jobs",
        "http://10.0.0.5/jobs",
        "http://169.254.169.254/latest/meta-data",
        "http://localhost/jobs",
        "file:///etc/passwd",
        "https://user:secret@example.com/jobs",
    ],
)
def test_rejects_non_public_urls_without_dns(url: str):
    with pytest.raises(UnsafeUrlError):
        validate_public_url(url, resolve_dns=False)


def test_accepts_public_https_shape_without_dns():
    assert validate_public_url("https://careers.example.com/jobs", resolve_dns=False)


def test_detects_sensitive_query_parameters():
    assert has_sensitive_query("https://example.com/jobs?access_token=secret")
    assert not has_sensitive_query("https://example.com/jobs?department=engineering")
