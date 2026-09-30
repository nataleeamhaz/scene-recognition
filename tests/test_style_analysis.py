import re

import numpy as np

from style_analysis import extract_color_palette, pick_best_label


def test_pick_best_label_returns_highest_score():
    scores = {"modern": 0.1, "bohemian": 0.7, "rustic": 0.2}
    assert pick_best_label(scores) == "bohemian"


def test_pick_best_label_single_entry():
    assert pick_best_label({"cozy": 0.99}) == "cozy"


def test_extract_color_palette_returns_requested_count():
    image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    palette = extract_color_palette(image, num_colors=5)
    assert len(palette) == 5


def test_extract_color_palette_returns_valid_hex_strings():
    image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    palette = extract_color_palette(image, num_colors=3)
    for color in palette:
        assert re.fullmatch(r"#[0-9a-f]{6}", color)


def test_extract_color_palette_solid_color_image():
    # A solid blue (BGR) image should cluster to a single color (repeated).
    image = np.full((30, 30, 3), (200, 50, 10), dtype=np.uint8)  # BGR
    palette = extract_color_palette(image, num_colors=2)
    assert palette[0] == "#0a32c8"  # RGB hex from BGR (10, 50, 200)
