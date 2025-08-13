import numpy as np
from src.utils import random_crop, color_jitter, load_image

def test_random_crop():
    img = np.ones((100, 100, 3), dtype=np.uint8)
    crop = random_crop(img, (50, 50))
    assert crop.shape == (50, 50, 3)

def test_color_jitter():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 127
    jittered = color_jitter(img)
    assert jittered.shape == img.shape

def test_load_image(tmp_path):
    # Create a dummy image
    img_path = tmp_path / "dummy.jpg"
    img = np.ones((10, 10, 3), dtype=np.uint8)
    import cv2
    cv2.imwrite(str(img_path), img)
    loaded = load_image(str(img_path))
    assert loaded.shape == img.shape
