from boardforge.models import Preferences
from boardforge.preferences import deterministic_extract


def test_incremental_preferences():
    p = deterministic_extract("I want AI or backend jobs in Bangalore. No unpaid roles. Avoid hardware.")
    assert "AI/ML" in p.roles
    assert "Backend" in p.roles
    assert "Bengaluru" in p.locations
    assert p.paid_only is True
    assert "hardware" in p.avoid_terms


def test_worldwide_clears_country_lock():
    p = deterministic_extract("worldwide remote roles", Preferences(countries=["India"]))
    assert p.countries == []
