"""
Accepts user images and preprocess
"""

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile

app = FastAPI()


def convert_to_cv2_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into a cv2 (BGR) numpy array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def assess_image_quality(
    cv2_image: np.ndarray,
    blur_threshold: float = 100.0,
    min_dimension: int = 400,
    dark_threshold: float = 40.0,
    bright_threshold: float = 220.0,
) -> dict:
    """
    Run objective quality checks on a decoded image: blur, resolution, exposure.

    Call this on the original decoded image, before validate_and_resize —
    downscaling smooths out the high-frequency detail the blur check relies on.

    Returns a dict with per-check flags/values and a "warnings" list of
    human-readable messages (empty if the image passes all checks).
    """
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    h, w = cv2_image.shape[:2]

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = blur_score < blur_threshold

    is_low_resolution = min(h, w) < min_dimension

    brightness = float(np.mean(gray))
    is_too_dark = brightness < dark_threshold
    is_too_bright = brightness > bright_threshold

    warnings = []
    if is_blurry:
        warnings.append("This photo looks blurry — try holding the camera steady or improving lighting.")
    if is_low_resolution:
        warnings.append(f"Image resolution is low ({w}x{h}) — a higher-resolution photo will give better results.")
    if is_too_dark:
        warnings.append("This photo looks quite dark — try taking it with more light.")
    if is_too_bright:
        warnings.append("This photo looks overexposed — try reducing direct light or flash.")

    return {
        "blur_score": round(blur_score, 2),
        "is_blurry": is_blurry,
        "resolution": {"width": w, "height": h},
        "is_low_resolution": is_low_resolution,
        "brightness": round(brightness, 2),
        "is_too_dark": is_too_dark,
        "is_too_bright": is_too_bright,
        "warnings": warnings,
    }


def validate_and_resize(cv2_image: np.ndarray, max_dimension: int = 1568) -> np.ndarray:
    """
    Resize image if its largest dimension exceeds max_dimension, maintaining aspect ratio.
    Claude Vision recommends images no larger than 1568px on either side.
    """
    h, w = cv2_image.shape[:2]
    largest = max(h, w)
    if largest <= max_dimension:
        return cv2_image
    scale = max_dimension / largest
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(cv2_image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def cv2_to_bytes(cv2_image: np.ndarray) -> bytes:
    """Encode a cv2 (BGR) image back to JPEG bytes."""
    success, buffer = cv2.imencode(".jpg", cv2_image)
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    return buffer.tobytes()


@app.post("/upload")
async def uploadImage(image: UploadFile):
    image_bytes = await image.read()
    image = convert_to_cv2_image(image_bytes)
    return {"received_image": "success"}
