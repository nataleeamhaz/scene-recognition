"""
Text-only recommendation generation.

CLIP handles style/mood classification and k-means handles the color
palette (see style_analysis.py) — neither can write free text. This module
covers the two fields that genuinely need a generative model: actionable
recommendations and a note on existing strengths, plus Pinterest search
terms derived from the same context. No image is sent — it's a small,
cheap, text-only call (see cost comparison in TASKS.md).
"""

import json
import os

import anthropic
from fastapi import HTTPException

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))


def build_prompt(style: str, mood: str, colors: list[str], furniture: list[dict], user_prompt: str) -> str:
    """Build the text-only prompt from already-classified facts. Pure/testable."""
    furniture_labels = ", ".join(item["label"] for item in furniture) if furniture else "none detected"
    colors_str = ", ".join(colors) if colors else "none extracted"

    return (
        f"Room style: {style}\n"
        f"Room mood: {mood}\n"
        f"Dominant colors: {colors_str}\n"
        f"Furniture/objects detected: {furniture_labels}\n"
        f"User's design goal: {user_prompt or 'general room improvement'}\n\n"
        "Based on these facts, return a JSON object with exactly these keys:\n"
        "- existing_strengths (string): what the room already does well\n"
        "- recommendations (array of exactly 5 strings): specific actionable improvements\n"
        "- pinterest_search_terms (array of 2-4 strings): search phrases for Pinterest inspiration\n\n"
        "Return ONLY valid JSON, no markdown, no explanation."
    )


def generate_recommendations(
    style: str, mood: str, colors: list[str], furniture: list[dict], user_prompt: str
) -> dict:
    """
    Call Claude (text-only, no image) to generate recommendations, strengths,
    and Pinterest search terms from already-classified style/mood/color facts.
    """
    prompt = build_prompt(style, mood, colors, furniture, user_prompt)

    try:
        response = _client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            system=(
                "You are an expert interior designer. "
                "You always respond with valid JSON only — no markdown fences, no prose."
            ),
            messages=[{"role": "user", "content": prompt}],
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
