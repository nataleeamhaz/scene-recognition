import photo_generation


def test_resolve_design_style_maps_known_clip_labels():
    assert photo_generation.resolve_design_style("mid-century modern") == "midcenturymodern"
    assert photo_generation.resolve_design_style("bohemian") == "boho"
    assert photo_generation.resolve_design_style("scandinavian") == "scandinavian"


def test_resolve_design_style_falls_back_for_unknown_label():
    assert photo_generation.resolve_design_style("some_unknown_style") == photo_generation.DEFAULT_DESIGN_STYLE


def test_parse_redesign_response_extracts_images():
    data = {
        "error": "",
        "message": "Successfully generated designs for room.",
        "info": {
            "images": [
                {"uuid": "abc", "width": 1024, "height": 768, "url": "https://prod-files.decor8.ai/a.png"},
            ]
        },
    }
    result = photo_generation._parse_redesign_response(data)
    assert result == [{"url": "https://prod-files.decor8.ai/a.png", "width": 1024, "height": 768}]


def test_parse_redesign_response_skips_entries_without_url():
    data = {"info": {"images": [{"uuid": "abc", "width": 1024, "height": 768}]}}
    assert photo_generation._parse_redesign_response(data) == []


def test_parse_redesign_response_handles_missing_info():
    assert photo_generation._parse_redesign_response({}) == []


def test_generate_redesigned_photos_returns_empty_without_api_key(monkeypatch):
    monkeypatch.setattr(photo_generation, "DECOR8_API_KEY", "")
    assert photo_generation.generate_redesigned_photos("https://example.com/room.jpg", "modern", "livingroom") == []


def test_generate_redesigned_photos_returns_empty_without_image_url(monkeypatch):
    monkeypatch.setattr(photo_generation, "DECOR8_API_KEY", "test-key")
    assert photo_generation.generate_redesigned_photos("", "modern", "livingroom") == []
