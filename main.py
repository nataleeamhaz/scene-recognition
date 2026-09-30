"""
Room Design Recommendation API
"""

import os
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from feature_extraction import detect_furniture
from photo_generation import ROOM_TYPES, generate_redesigned_photos
from pinterest import (
    build_authorization_url,
    exchange_code_for_token,
    get_board_pins,
    list_boards,
    search_multiple_terms,
)
from preprocessing import assess_image_quality, convert_to_cv2_image, cv2_to_bytes, validate_and_resize
from rate_limit import RateLimiter
from recommendations import generate_recommendations
from style_analysis import classify_style_and_mood, extract_color_palette

load_dotenv()

app = FastAPI(title="Room Design Recommender")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# /analyze calls Claude (paid, vision) and Pinterest (quota) per request —
# limit per-IP usage so a public deployment can't be used to run up API costs.
_analyze_rate_limiter = RateLimiter(
    max_requests=int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10")),
    window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "3600")),
)

# /generate-photo calls Decor8 AI at ~$0.20/image — a stricter, separate
# limit than /analyze's, since each call has real per-request cost.
_photo_rate_limiter = RateLimiter(
    max_requests=int(os.getenv("PHOTO_RATE_LIMIT_MAX_REQUESTS", "5")),
    window_seconds=int(os.getenv("PHOTO_RATE_LIMIT_WINDOW_SECONDS", "3600")),
)

# Decor8's redesign endpoint only accepts a publicly-reachable image URL (no
# base64/file upload — see photo_generation.py), so we host the uploaded
# photo ourselves at a short-lived temporary URL for Decor8 to fetch.
# PUBLIC_BASE_URL must be this app's real internet-reachable address (a
# deployed URL, or a local tunnel like ngrok) — plain localhost cannot be
# fetched by Decor8's servers.
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")
_TEMP_IMAGE_TTL_SECONDS = 600
_temp_images: dict[str, tuple[bytes, float]] = {}


def _store_temp_image(image_bytes: bytes) -> str:
    """Store JPEG bytes in memory for temporary public serving; returns its id."""
    now = time.time()
    expired = [key for key, (_, created_at) in _temp_images.items() if now - created_at > _TEMP_IMAGE_TTL_SECONDS]
    for key in expired:
        del _temp_images[key]

    image_id = uuid.uuid4().hex
    _temp_images[image_id] = (image_bytes, now)
    return image_id


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/pinterest/boards")
async def pinterest_boards():
    """List boards on the configured Pinterest account, for the inspiration-source picker."""
    return {"boards": list_boards()}


@app.get("/room-types")
async def room_types():
    """Room type options for the final-photo generator's room-type picker."""
    return {"room_types": ROOM_TYPES}


@app.get("/tmp-image/{image_id}.jpg")
async def tmp_image(image_id: str):
    """
    Short-lived public serving of a just-uploaded room photo, so Decor8's
    servers can fetch it (their redesign endpoint requires a publicly-
    reachable URL — see photo_generation.py). Not meant as general storage:
    entries expire after _TEMP_IMAGE_TTL_SECONDS.
    """
    entry = _temp_images.get(image_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Image not found or expired")
    image_bytes, _ = entry
    return Response(content=image_bytes, media_type="image/jpeg")


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
    board_id: str = Form(default=""),
):
    """
    Full pipeline:
      1. Decode image & assess quality (blur/resolution/exposure)
      2. Resize for downstream models
      3. YOLO furniture detection
      4. CLIP style/mood classification + k-means color palette
      5. Claude (vision) recommendations + Pinterest search terms — sees the
         resized room photo plus the CLIP/k-means/YOLO facts
      6. Pinterest inspiration images — from the selected board if board_id is
         given (the user's own chosen "vision"), else from Haiku's generated
         search terms
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

    # 4. Claude (vision) for recommendations, strengths, Pinterest terms —
    # sees the actual (resized) room photo alongside the pre-classified facts
    generated = generate_recommendations(
        style=style_mood["style"],
        mood=style_mood["mood"],
        colors=color_palette,
        furniture=yolo_results,
        user_prompt=prompt,
        image_bytes=cv2_to_bytes(cv2_image),
    )

    style_analysis = {
        "style": style_mood["style"],
        "mood": style_mood["mood"],
        "color_palette": color_palette,
        **generated,
    }

    # 5. Pinterest inspiration — a linked board takes priority over keyword search
    if board_id:
        pinterest_results = get_board_pins(board_id)
    else:
        pinterest_terms = style_analysis.get("pinterest_search_terms", [])
        pinterest_results = search_multiple_terms(pinterest_terms)

    return {
        "quality": quality,
        "detected_furniture": yolo_results,
        "style_analysis": style_analysis,
        "pinterest_results": pinterest_results,
    }


@app.post("/generate-photo")
async def generate_photo(
    request: Request,
    image: UploadFile = File(...),
    style: str = Form(...),
    room_type: str = Form(...),
):
    """
    Generate a photorealistic "after" photo of the user's actual room in the
    given design style, via Decor8 AI. Separate from /analyze since this is
    a slower (10-30s), real-money ($0.20/image) call the user opts into
    explicitly, after already seeing text recommendations.
    """
    if not PUBLIC_BASE_URL:
        raise HTTPException(
            status_code=503,
            detail="Photo generation isn't configured — PUBLIC_BASE_URL is not set.",
        )

    client_ip = request.client.host if request.client else "unknown"
    _photo_rate_limiter.check(client_ip)

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    cv2_image = convert_to_cv2_image(image_bytes)
    if cv2_image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    cv2_image = validate_and_resize(cv2_image)

    image_id = _store_temp_image(cv2_to_bytes(cv2_image))
    image_url = f"{PUBLIC_BASE_URL}/tmp-image/{image_id}.jpg"

    images = generate_redesigned_photos(image_url=image_url, style=style, room_type=room_type)

    return {"images": images}
