import numpy as np

from preprocessing import assess_image_quality, convert_to_cv2_image, cv2_to_bytes, validate_and_resize


def test_convert_to_cv2_image_roundtrip():
    original = np.random.randint(0, 256, (50, 80, 3), dtype=np.uint8)
    encoded = cv2_to_bytes(original)
    decoded = convert_to_cv2_image(encoded)
    assert decoded.shape == original.shape


def test_convert_to_cv2_image_invalid_bytes_returns_none():
    assert convert_to_cv2_image(b"not an image") is None


def test_validate_and_resize_noop_under_max_dimension():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result = validate_and_resize(image, max_dimension=1568)
    assert result.shape == image.shape


def test_validate_and_resize_scales_down_over_max_dimension():
    image = np.zeros((1000, 2000, 3), dtype=np.uint8)
    result = validate_and_resize(image, max_dimension=1000)
    h, w = result.shape[:2]
    assert max(h, w) == 1000
    assert w / h == 2000 / 1000  # aspect ratio preserved


def test_assess_image_quality_blur_threshold_toggles():
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    lenient = assess_image_quality(image, blur_threshold=0)
    strict = assess_image_quality(image, blur_threshold=1e9)
    assert lenient["is_blurry"] is False
    assert strict["is_blurry"] is True


def test_assess_image_quality_low_resolution():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert assess_image_quality(image, min_dimension=200)["is_low_resolution"] is True
    assert assess_image_quality(image, min_dimension=50)["is_low_resolution"] is False


def test_assess_image_quality_too_dark():
    dark_image = np.zeros((50, 50, 3), dtype=np.uint8)
    result = assess_image_quality(dark_image)
    assert result["is_too_dark"] is True
    assert result["is_too_bright"] is False
    assert result["brightness"] == 0.0


def test_assess_image_quality_too_bright():
    bright_image = np.full((50, 50, 3), 255, dtype=np.uint8)
    result = assess_image_quality(bright_image)
    assert result["is_too_bright"] is True
    assert result["is_too_dark"] is False


def test_assess_image_quality_warnings_populated_when_failing():
    dark_image = np.zeros((50, 50, 3), dtype=np.uint8)
    result = assess_image_quality(dark_image)
    assert len(result["warnings"]) > 0


def test_assess_image_quality_no_warnings_for_good_image():
    good_image = np.random.randint(50, 200, (500, 500, 3), dtype=np.uint8)
    result = assess_image_quality(good_image)
    assert result["warnings"] == []
    assert result["is_blurry"] is False
    assert result["is_low_resolution"] is False
    assert result["is_too_dark"] is False
    assert result["is_too_bright"] is False
