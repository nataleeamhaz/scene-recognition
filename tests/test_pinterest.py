from urllib.parse import parse_qs, urlparse

import pinterest


def test_build_authorization_url_contains_required_params(monkeypatch):
    monkeypatch.setattr(pinterest, "PINTEREST_APP_ID", "test-app-id")
    monkeypatch.setattr(pinterest, "PINTEREST_REDIRECT_URI", "http://localhost:8000/callback")

    url = pinterest.build_authorization_url()
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    assert url.startswith(pinterest.PINTEREST_OAUTH_AUTHORIZE_URL)
    assert params["client_id"] == ["test-app-id"]
    assert params["redirect_uri"] == ["http://localhost:8000/callback"]
    assert params["response_type"] == ["code"]
    assert "scope" in params


def test_build_authorization_url_omits_state_when_not_given(monkeypatch):
    monkeypatch.setattr(pinterest, "PINTEREST_APP_ID", "test-app-id")
    url = pinterest.build_authorization_url()
    assert "state" not in parse_qs(urlparse(url).query)


def test_build_authorization_url_includes_state_when_given(monkeypatch):
    monkeypatch.setattr(pinterest, "PINTEREST_APP_ID", "test-app-id")
    url = pinterest.build_authorization_url(state="csrf-token-123")
    params = parse_qs(urlparse(url).query)
    assert params["state"] == ["csrf-token-123"]


def test_search_multiple_terms_dedupes_by_pin_url(monkeypatch):
    def fake_search_pins(query, limit=6):
        return [{"title": query, "image_url": "img", "pin_url": "https://pinterest.com/pin/1/"}]

    monkeypatch.setattr(pinterest, "search_pins", fake_search_pins)
    results = pinterest.search_multiple_terms(["cozy", "modern"])
    assert len(results) == 1


def test_search_multiple_terms_caps_at_three_terms(monkeypatch):
    calls = []

    def fake_search_pins(query, limit=6):
        calls.append(query)
        return []

    monkeypatch.setattr(pinterest, "search_pins", fake_search_pins)
    pinterest.search_multiple_terms(["a", "b", "c", "d", "e"])
    assert calls == ["a", "b", "c"]
