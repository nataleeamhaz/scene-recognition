"""
Pinterest API client.

Trial-mode note: /v5/search/pins searches your own account's pins only.
Save 20-30 interior design pins to a board in your test account before demoing.
"""

import base64
import os
import requests
from dotenv import load_dotenv

load_dotenv()

PINTEREST_ACCESS_TOKEN = os.getenv("PINTEREST_ACCESS_TOKEN", "")
PINTEREST_APP_ID = os.getenv("PINTEREST_APP_ID", "")
PINTEREST_APP_SECRET = os.getenv("PINTEREST_APP_SECRET", "")
PINTEREST_REDIRECT_URI = os.getenv("PINTEREST_REDIRECT_URI", "http://localhost:8000/callback")
PINTEREST_API_BASE = "https://api.pinterest.com/v5"
PINTEREST_OAUTH_AUTHORIZE_URL = "https://www.pinterest.com/oauth/"
PINTEREST_OAUTH_TOKEN_URL = f"{PINTEREST_API_BASE}/oauth/token"


def build_authorization_url(scope: str = "boards:read,pins:read", state: str = "") -> str:
    """
    Build the Pinterest OAuth authorization URL to redirect a user to.
    They approve access there, then Pinterest redirects back to
    PINTEREST_REDIRECT_URI with a `code` query param.
    """
    params = {
        "client_id": PINTEREST_APP_ID,
        "redirect_uri": PINTEREST_REDIRECT_URI,
        "response_type": "code",
        "scope": scope,
    }
    if state:
        params["state"] = state
    query = "&".join(f"{key}={requests.utils.quote(str(value))}" for key, value in params.items())
    return f"{PINTEREST_OAUTH_AUTHORIZE_URL}?{query}"


def exchange_code_for_token(code: str) -> dict:
    """
    Exchange an OAuth authorization code for an access token.
    Returns Pinterest's raw token response dict, e.g.
    {"access_token": "...", "refresh_token": "...", "expires_in": ..., ...}
    """
    basic_auth = base64.b64encode(f"{PINTEREST_APP_ID}:{PINTEREST_APP_SECRET}".encode()).decode()

    response = requests.post(
        PINTEREST_OAUTH_TOKEN_URL,
        headers={
            "Authorization": f"Basic {basic_auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": PINTEREST_REDIRECT_URI,
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def search_pins(query: str, limit: int = 6) -> list[dict]:
    """
    Search Pinterest pins for the given query.
    Returns a list of dicts with keys: title, image_url, pin_url.
    Returns [] gracefully if Pinterest is unavailable or unconfigured.
    """
    if not PINTEREST_ACCESS_TOKEN:
        return []

    try:
        response = requests.get(
            f"{PINTEREST_API_BASE}/search/pins",
            headers={"Authorization": f"Bearer {PINTEREST_ACCESS_TOKEN}"},
            params={"query": query, "page_size": limit},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    pins = []
    for item in data.get("items", []):
        media = item.get("media", {})
        images = media.get("images", {})
        # prefer 400x300, fall back to any available size
        image_url = ""
        for size_key in ("400x300", "600x", "150x150"):
            if size_key in images:
                image_url = images[size_key].get("url", "")
                break
        if not image_url:
            url_dict = next(iter(images.values()), {})
            image_url = url_dict.get("url", "")

        pin_id = item.get("id", "")
        pins.append(
            {
                "title": item.get("title") or item.get("description") or query,
                "image_url": image_url,
                "pin_url": f"https://www.pinterest.com/pin/{pin_id}/",
            }
        )
    return pins


def search_multiple_terms(terms: list[str], pins_per_term: int = 3) -> list[dict]:
    """
    Search up to 3 terms, deduplicate results by pin_url.
    Returns [] gracefully if Pinterest is unavailable.
    """
    seen_urls: set[str] = set()
    results: list[dict] = []

    for term in terms[:3]:
        for pin in search_pins(term, limit=pins_per_term):
            if pin["pin_url"] not in seen_urls:
                seen_urls.add(pin["pin_url"])
                results.append(pin)

    return results
