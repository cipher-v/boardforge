from concurrent.futures import ThreadPoolExecutor
import os
import threading

from boardforge import beta


class FakeStore:
    def __init__(self):
        self.counts = {}
        self.lock = threading.Lock()

    def consume_beta_usage(self, profile, operation, limit, global_limit):
        with self.lock:
            key = (profile, operation)
            if self.counts.get(key, 0) >= limit:
                return False
            if global_limit is not None and sum(v for (p, op), v in self.counts.items() if op == operation) >= global_limit:
                return False
            self.counts[key] = self.counts.get(key, 0) + 1
            return True


def test_disabled_does_not_touch_store(monkeypatch):
    monkeypatch.delenv("BOARDFORGE_BETA_LIMITS", raising=False)
    assert beta.admit(None, "search")


def test_independent_limits_and_profiles(monkeypatch):
    monkeypatch.setenv("BOARDFORGE_BETA_LIMITS", "1")
    store = FakeStore()
    token = beta.current_profile.set("tester-1")
    try:
        assert all(beta.admit(store, "verification") for _ in range(3))
        assert not beta.admit(store, "verification")
        assert beta.admit(store, "search")
        assert beta.admit(store, "verification", "tester-2")
    finally:
        beta.current_profile.reset(token)
    assert not beta.admit(store, "search")


def test_global_discovery_limit(monkeypatch):
    monkeypatch.setenv("BOARDFORGE_BETA_LIMITS", "1")
    store = FakeStore()
    assert all(beta.admit(store, "discovery", str(i)) for i in range(5))
    assert not beta.admit(store, "discovery", "sixth")


def test_completion_counts_separate(monkeypatch):
    monkeypatch.setenv("BOARDFORGE_BETA_LIMITS", "1")
    store = FakeStore()
    beta.record(store, "search_completed", "tester")
    assert store.counts == {("tester", "search_completed"): 1}
