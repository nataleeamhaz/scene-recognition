"""
Decor8 AI room redesign client.

Generates a photorealistic "after" photo showing the room's actual layout
(walls, windows, proportions) with a new design style applied — as opposed
to a generic text-to-image room that bears no relation to the user's photo.

Trial-mode note: Decor8's `generate_designs_for_room` endpoint only accepts
a publicly-reachable `input_image_url` — no base64 or file upload (confirmed
against Decor8's own API docs, api-docs.decor8.ai). main.py hosts the
resized room photo at a short-lived temporary URL (see /tmp-image/{id}.jpg)
for Decor8's servers to fetch it from. That means this call only works when
this app itself is reachable from the public internet (a real deployment,
or a local tunnel such as ngrok) — plain localhost cannot be fetched by
Decor8's servers.
"""

import os

import requests

DECOR8_API_KEY = os.getenv("DECOR8_API_KEY", "")
DECOR8_API_BASE = "https://api.decor8.ai"

# style_analysis.py's CLIP STYLE_LABELS -> Decor8's `design_style` enum
# (confirmed against Decor8's own API docs — see photo_generation.py docstring)
STYLE_TO_DESIGN_STYLE = {
    "modern": "modern",
    "minimalist": "minimalist",
    "bohemian": "boho",
    "industrial": "industrial",
    "scandinavian": "scandinavian",
    "farmhouse": "farmhouse",
    "traditional": "traditional",
    "mid-century modern": "midcenturymodern",
    "contemporary": "contemporary",
    "rustic": "rustic",
    "coastal": "coastal",
    "eclectic": "eclectic",
}
DEFAULT_DESIGN_STYLE = "contemporary"

# A curated subset of Decor8's 31 room_type values — the common home rooms
# this app's users are likely to photograph.
ROOM_TYPES = [
    "livingroom",
    "bedroom",
    "kitchen",
    "bathroom",
    "diningroom",
    "office",
    "kidsroom",
    "familyroom",
]


def resolve_design_style(style: str) -> str:
    """Map a CLIP style label to Decor8's design_style enum. Pure/testable."""
    return STYLE_TO_DESIGN_STYLE.get(style, DEFAULT_DESIGN_STYLE)


def _parse_redesign_response(data: dict) -> list[dict]:
    """Extract {url, width, height} entries from Decor8's response JSON. Pure/testable."""
    images = data.get("info", {}).get("images", [])
    return [
        {"url": img.get("url", ""), "width": img.get("width"), "height": img.get("height")}
        for img in images
        if img.get("url")
    ]


def generate_redesigned_photos(image_url: str, style: str, room_type: str, num_images: int = 1) -> list[dict]:
    """
    Call Decor8's room-redesign endpoint with a publicly-reachable photo URL.
    Returns a list of dicts like [{"url": ..., "width": ..., "height": ...}].
    Returns [] gracefully if Decor8 is unavailable, unconfigured, or the
    request fails — never raises.
    """
    if not DECOR8_API_KEY or not image_url:
        return []

    try:
        response = requests.post(
            f"{DECOR8_API_BASE}/generate_designs_for_room",
            headers={
                "Authorization": f"Bearer {DECOR8_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "input_image_url": image_url,
                "room_type": room_type,
                "design_style": resolve_design_style(style),
                "num_images": num_images,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    return _parse_redesign_response(data)
