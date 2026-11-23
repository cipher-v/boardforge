from boardforge.models import Preferences
from boardforge.preferences import deterministic_extract


def test_new_profile_has_no_implicit_country_lock():
    assert Preferences().countries == []


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


def test_negated_role_is_removed_instead_of_added():
    current = Preferences(roles=["QA", "Backend"])
    p = deterministic_extract("Do not show QA anymore", current)
    assert p.roles == ["Backend"]
    assert "qa" in p.avoid_terms


def test_negated_role_does_not_negate_later_positive_clause():
    p = deterministic_extract("I do not want QA, but I want AI and backend roles")
    assert "QA" not in p.roles
    assert "AI/ML" in p.roles
    assert "Backend" in p.roles


def test_salary_and_unpaid_preferences_are_extracted():
    p = deterministic_extract("At least 12 LPA and no unpaid internships")
    assert p.min_salary_lpa == 12
    assert p.paid_only is True
