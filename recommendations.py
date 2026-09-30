"""
Vision-grounded recommendation generation.

CLIP handles style/mood classification and k-means handles the color
palette (see style_analysis.py) — both are closed-vocabulary/fixed-output
and free, so they stay in the pipeline as cheap, deterministic priors. But
neither can see decor/textile items outside their fixed label sets (a
blanket, a rug, a piece of wall art) or describe *where* something is in
the room — so this call also sends the actual room photo to Claude, which
can reference specific visible objects, colors, and positions when making
recommendations. See the cost tradeoff this reopens in TASKS.md.
"""

import base64
import json
import os

import anthropic
from fastapi import HTTPException

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))


def build_prompt(style: str, mood: str, colors: list[str], furniture: list[dict], user_prompt: str) -> str:
    """Build the text prompt from already-classified facts. Pure/testable."""
    furniture_labels = ", ".join(item["label"] for item in furniture) if furniture else "none detected"
    colors_str = ", ".join(colors) if colors else "none extracted"

    return (
        f"Room style (from CLIP): {style}\n"
        f"Room mood (from CLIP): {mood}\n"
        f"Dominant colors (from k-means): {colors_str}\n"
        f"Furniture/objects detected (from YOLO): {furniture_labels}\n"
        f"User's design goal: {user_prompt or 'general room improvement'}\n\n"
        "The attached photo is the actual room these facts were computed from — look at it "
        "directly. The facts above only cover a fixed set of categories, so use the photo to "
        "notice anything they miss (rugs, pillows, blankets, curtains, wall art, clutter, etc.) "
        "and to reason about where things are positioned.\n\n"
        "Return a JSON object with exactly these keys:\n"
        "- existing_strengths (string): what the room already does well\n"
        "- recommendations (array of exactly 5 objects): specific actionable improvements, "
        "each an object with keys 'rank' (integer 1-5, 1 = most important) and 'text' "
        "(string, the improvement itself), ordered by rank ascending. Where the room supports "
        "it, ground recommendations in what you actually see — name the specific item and its "
        "location or color (e.g. 'swap the blue blanket on the sofa for a cream throw' or "
        "'move the armchair from the corner to face the window') rather than generic advice.\n"
        "- pinterest_search_terms (array of 2-4 strings): search phrases for Pinterest inspiration\n\n"
        "Return ONLY valid JSON, no markdown, no explanation."
    )


def generate_recommendations(
    style: str, mood: str, colors: list[str], furniture: list[dict], user_prompt: str, image_bytes: bytes
) -> dict:
    """
    Call Claude (vision) to generate recommendations, strengths, and Pinterest
    search terms — grounded in both the pre-classified style/mood/color/furniture
    facts and the actual room photo.
    """
    prompt = build_prompt(style, mood, colors, furniture, user_prompt)
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    try:
        response = _client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            system=(
                "You are an expert interior designer. "
                "You always respond with valid JSON only — no markdown fences, no prose."
            ),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e}") from e

    raw = response.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "existing_strengths": "",
            "recommendations": [],
            "pinterest_search_terms": [],
            "raw_response": raw,
        }
