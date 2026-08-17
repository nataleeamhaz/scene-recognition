"""
YOLO-based furniture detection pipeline.
Model: YOLOv8n (COCO-pretrained, auto-downloads ~6MB on first run)
"""

from ultralytics import YOLO
import numpy as np

# COCO class IDs relevant to interior design / furniture
FURNITURE_CLASSES = {
    56: "chair",
    57: "couch",
    59: "bed",
    
    60: "dining table",
    62: "tv",
    72: "refrigerator",
    74: "clock",
    75: "vase",
    58: "potted plant",
    63: "laptop",
}

# Load model once at module level to avoid reloading per request
_model = YOLO("yolov8n.pt")


def detect_furniture(cv2_image: np.ndarray, confidence_threshold: float = 0.3) -> list[dict]:
    """
    Run YOLO inference on a cv2 (BGR) image and return detected furniture items.

    Args:
        cv2_image: OpenCV image in BGR format
        confidence_threshold: Minimum confidence to include a detection

    Returns:
        List of dicts like [{"label": "chair", "confidence": 0.91}, ...]
        Deduplicated per class (highest confidence kept).
    """
    results = _model(cv2_image, verbose=False)

    best_per_class: dict[str, float] = {}

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id not in FURNITURE_CLASSES:
                continue
            if conf < confidence_threshold:
                continue
            label = FURNITURE_CLASSES[cls_id]
            if label not in best_per_class or conf > best_per_class[label]:
                best_per_class[label] = conf

    detections = [
        {"label": label, "confidence": round(conf, 2)}
        for label, conf in sorted(best_per_class.items(), key=lambda x: -x[1])
    ]
    return detections
