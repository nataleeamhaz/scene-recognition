from recommendations import build_prompt


def test_build_prompt_includes_all_facts():
    prompt = build_prompt(
        style="modern",
        mood="cozy",
        colors=["#ffffff", "#8b6043"],
        furniture=[{"label": "couch", "confidence": 0.9}, {"label": "chair", "confidence": 0.8}],
        user_prompt="make it feel bigger",
    )
    assert "modern" in prompt
    assert "cozy" in prompt
    assert "#ffffff" in prompt
    assert "couch" in prompt
    assert "chair" in prompt
    assert "make it feel bigger" in prompt


def test_build_prompt_handles_no_furniture():
    prompt = build_prompt(style="rustic", mood="warm", colors=[], furniture=[], user_prompt="")
    assert "none detected" in prompt
    assert "none extracted" in prompt
    assert "general room improvement" in prompt


def test_build_prompt_requests_ranked_recommendations():
    prompt = build_prompt(style="modern", mood="calm", colors=["#000000"], furniture=[], user_prompt="")
    assert "exactly 5 objects" in prompt
    assert "'rank'" in prompt
    assert "'text'" in prompt
