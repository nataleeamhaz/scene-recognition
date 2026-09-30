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


def test_list_boards_returns_empty_without_token(monkeypatch):
    monkeypatch.setattr(pinterest, "PINTEREST_ACCESS_TOKEN", "")
    assert pinterest.list_boards() == []


def test_get_board_pins_returns_empty_without_token(monkeypatch):
    monkeypatch.setattr(pinterest, "PINTEREST_ACCESS_TOKEN", "")
    assert pinterest.get_board_pins("some-board-id") == []


def test_get_board_pins_returns_empty_without_board_id(monkeypatch):
    monkeypatch.setattr(pinterest, "PINTEREST_ACCESS_TOKEN", "test-token")
    assert pinterest.get_board_pins("") == []


def test_parse_pin_item_prefers_400x300_image():
    item = {
        "id": "123",
        "title": "Cozy nook",
        "media": {"images": {"150x150": {"url": "small.jpg"}, "400x300": {"url": "medium.jpg"}}},
    }
    pin = pinterest._parse_pin_item(item)
    assert pin["title"] == "Cozy nook"
    assert pin["image_url"] == "medium.jpg"
    assert pin["pin_url"] == "https://www.pinterest.com/pin/123/"


def test_parse_pin_item_falls_back_to_any_available_image():
    item = {"id": "456", "media": {"images": {"1200x": {"url": "large.jpg"}}}}
    pin = pinterest._parse_pin_item(item, fallback_title="cozy modern")
    assert pin["image_url"] == "large.jpg"
    assert pin["title"] == "cozy modern"
