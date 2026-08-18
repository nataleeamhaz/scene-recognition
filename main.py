"""
Room Design Recommendation API
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from feature_extraction import detect_furniture
from pinterest import build_authorization_url, exchange_code_for_token, search_multiple_terms
from preprocessing import assess_image_quality, convert_to_cv2_image, validate_and_resize
from rate_limit import RateLimiter
from recommendations import generate_recommendations
from style_analysis import classify_style_and_mood, extract_color_palette

load_dotenv()

app = FastAPI(title="Room Design Recommender")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# /analyze calls Claude (paid, text-only) and Pinterest (quota) per request —
# limit per-IP usage so a public deployment can't be used to run up API costs.
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
      2. Resize for downstream models
      3. YOLO furniture detection
      4. CLIP style/mood classification + k-means color palette
      5. Claude (text-only) recommendations + Pinterest search terms
      6. Pinterest inspiration images
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

    # 3. CLIP style/mood classification + color palette (local, no API cost)
    style_mood = classify_style_and_mood(cv2_image)
    color_palette = extract_color_palette(cv2_image)

    # 4. Claude (text-only) for recommendations, strengths, Pinterest terms
    generated = generate_recommendations(
        style=style_mood["style"],
        mood=style_mood["mood"],
        colors=color_palette,
        furniture=yolo_results,
        user_prompt=prompt,
    )

    style_analysis = {
        "style": style_mood["style"],
        "mood": style_mood["mood"],
        "color_palette": color_palette,
        **generated,
    }

    # 5. Pinterest inspiration
    pinterest_terms = style_analysis.get("pinterest_search_terms", [])
    pinterest_results = search_multiple_terms(pinterest_terms)

    return {
        "quality": quality,
        "detected_furniture": yolo_results,
        "style_analysis": style_analysis,
        "pinterest_results": pinterest_results,
    }
