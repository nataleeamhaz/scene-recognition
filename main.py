"""
Room Design Recommendation API
"""

import base64
import json
import os

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from feature_extraction import detect_furniture
from pinterest import build_authorization_url, exchange_code_for_token, search_multiple_terms
from preprocessing import assess_image_quality, convert_to_cv2_image, cv2_to_bytes, validate_and_resize
from rate_limit import RateLimiter

load_dotenv()

app = FastAPI(title="Room Design Recommender")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

_anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

# /analyze calls Claude (paid) and Pinterest (quota) per request — limit
# per-IP usage so a public deployment can't be used to run up API costs.
_analyze_rate_limiter = RateLimiter(
    max_requests=int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10")),
    window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "3600")),
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/pinterest/login")
async def pinterest_login():
    """Redirect to Pinterest's OAuth consent screen to obtain an access token."""
    return RedirectResponse(build_authorization_url())


@app.get("/callback")
async def pinterest_callback(code: str | None = None, error: str | None = None):
    """
    Pinterest OAuth redirect target. Exchanges the authorization code for an
    access token and returns it directly — this is a dev-only helper for
    obtaining a token to paste into PINTEREST_ACCESS_TOKEN in .env. It is not
    meant to run as-is in production (the token is shown in the response body
    rather than stored server-side).
    """
    if error:
        raise HTTPException(status_code=400, detail=f"Pinterest authorization failed: {error}")
    if not code:
        raise HTTPException(status_code=400, detail="Missing 'code' query parameter")

    try:
        token_data = exchange_code_for_token(code)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Token exchange failed: {e}") from e

    return {
        "message": "Copy the access_token value into PINTEREST_ACCESS_TOKEN in your .env file.",
        **token_data,
    }


@app.post("/analyze")
async def analyze(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(default=""),
):
    """
    Full pipeline:
      1. Decode image & assess quality (blur/resolution/exposure)
      2. Resize for the vision model
      3. YOLO furniture detection
      4. Claude Vision style analysis
      5. Pinterest inspiration images
    """
    client_ip = request.client.host if request.client else "unknown"
    _analyze_rate_limiter.check(client_ip)

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    # 1. Preprocess
    cv2_image = convert_to_cv2_image(image_bytes)
    if cv2_image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Quality check runs on the original decode — resizing smooths out the
    # high-frequency detail the blur check relies on.
    quality = assess_image_quality(cv2_image)

    cv2_image = validate_and_resize(cv2_image)

    # 2. YOLO furniture detection
    yolo_results = detect_furniture(cv2_image)

    # 3. Claude Vision analysis
    image_bytes_resized = cv2_to_bytes(cv2_image)
    style_analysis = claude_analyze(image_bytes_resized, yolo_results, prompt)

    # 4. Pinterest inspiration
    pinterest_terms = style_analysis.get("pinterest_search_terms", [])
    pinterest_results = search_multiple_terms(pinterest_terms)

    return {
        "quality": quality,
        "detected_furniture": yolo_results,
        "style_analysis": style_analysis,
        "pinterest_results": pinterest_results,
    }


# ---------------------------------------------------------------------------
# Claude Vision helper
# ---------------------------------------------------------------------------


def claude_analyze(image_bytes: bytes, yolo_results: list[dict], user_prompt: str) -> dict:
    """
    Send image + YOLO context + user prompt to Claude Vision.
    Returns parsed JSON dict with style analysis.
    """
    detected_labels = ", ".join(item["label"] for item in yolo_results) if yolo_results else "none detected"

    user_message = (
        f"Computer vision detected the following furniture/objects in the room: {detected_labels}.\n\n"
        f"User's design goal: {user_prompt or 'general room improvement'}\n\n"
        "Analyze this room photo and return a JSON object with exactly these keys:\n"
        "- style (string): current or target interior design style (e.g. 'modern', 'bohemian', 'minimalist')\n"
        "- mood (string): emotional feel of the space (e.g. 'calm', 'energetic', 'cozy')\n"
        "- color_palette (array of strings): 3-5 dominant or recommended colors\n"
        "- existing_strengths (string): what the room already does well\n"
        "- recommendations (array of exactly 5 strings): specific actionable improvements\n"
        "- pinterest_search_terms (array of 2-4 strings): search phrases for Pinterest inspiration\n\n"
        "Return ONLY valid JSON, no markdown, no explanation."
    )

    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    try:
        response = _anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=(
                "You are an expert interior designer and color consultant. "
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
                        {"type": "text", "text": user_message},
                    ],
                }
            ],
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e}") from e

    raw = response.content[0].text.strip()

    # Strip accidental markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Return a graceful fallback so the rest of the response still works
        return {
            "style": "unknown",
            "mood": "unknown",
            "color_palette": [],
            "existing_strengths": "",
            "recommendations": [],
            "pinterest_search_terms": [],
            "raw_response": raw,
        }
