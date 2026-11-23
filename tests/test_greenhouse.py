from boardforge.connectors.greenhouse import _strip_html


def test_strip_html_decodes_nested_entities():
    assert _strip_html("&amp;lt;p&amp;gt;Hello&amp;lt;/p&amp;gt;") == "Hello"
    assert _strip_html("&amp;amp;lt;b&amp;amp;gt;Nested&amp;amp;lt;/b&amp;amp;gt;") == "Nested"
