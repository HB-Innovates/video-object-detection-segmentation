import numpy as np
try:
    from src.detection import YOLOv5Detector
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    from detection import YOLOv5Detector

def test_detector_init():
    detector = YOLOv5Detector('dummy_path.pt')
    assert detector.model_path == 'dummy_path.pt'
    assert detector.model is None
