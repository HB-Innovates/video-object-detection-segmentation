import numpy as np
from src.detection import YOLOv5Detector

def test_detector_init():
    detector = YOLOv5Detector('dummy_path.pt')
    assert detector.model_path == 'dummy_path.pt'
    assert detector.model is None
