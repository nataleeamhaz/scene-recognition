"""
CLIP-based style/mood classification and color palette extraction.

Replaces the vision half of what used to be a single Claude Vision call:
CLIP (zero-shot) handles style/mood classification, k-means handles the
dominant color palette. Both are local, free, and run on every request
with no per-call API cost. What's left for an LLM (see recommendations.py)
is only the free-text fields CLIP fundamentally can't produce.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

STYLE_LABELS = [
    "modern",
    "minimalist",
    "bohemian",
    "industrial",
    "scandinavian",
    "farmhouse",
    "traditional",
    "mid-century modern",
    "contemporary",
    "rustic",
    "coastal",
    "eclectic",
]

MOOD_LABELS = [
    "calm",
    "cozy",
    "energetic",
    "elegant",
    "playful",
    "serene",
    "dramatic",
    "warm",
    "sterile",
]

_MODEL_NAME = "openai/clip-vit-base-patch32"
_model = CLIPModel.from_pretrained(_MODEL_NAME)
_processor = CLIPProcessor.from_pretrained(_MODEL_NAME)


def pick_best_label(scores: dict[str, float]) -> str:
    """Return the label with the highest score. Pure/testable — no model involved."""
    return max(scores, key=scores.get)


def classify_style_and_mood(cv2_image: np.ndarray) -> dict:
    """
    Zero-shot classify a room photo's design style and mood using CLIP.

    Returns {"style": str, "mood": str}.
    """
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    style_scores = _zero_shot_scores(pil_image, [f"a photo of a {label} styled room" for label in STYLE_LABELS], STYLE_LABELS)
    mood_scores = _zero_shot_scores(pil_image, [f"a {label} feeling room" for label in MOOD_LABELS], MOOD_LABELS)

    return {
        "style": pick_best_label(style_scores),
        "mood": pick_best_label(mood_scores),
    }


def _zero_shot_scores(pil_image: Image.Image, prompts: list[str], labels: list[str]) -> dict[str, float]:
    inputs = _processor(text=prompts, images=pil_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = _model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()
    return dict(zip(labels, probs))


def extract_color_palette(cv2_image: np.ndarray, num_colors: int = 5) -> list[str]:
    """
    Extract the dominant colors from an image via k-means clustering.
    Returns a list of hex color strings (e.g. "#8b6043"), most prominent first.
    """
    pixels = cv2_image.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    counts = np.bincount(labels.flatten())
    order = np.argsort(-counts)

    hex_colors = []
    for idx in order:
        b, g, r = np.round(centers[idx]).astype(int)
        hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return hex_colors
