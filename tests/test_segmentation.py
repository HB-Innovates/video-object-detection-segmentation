try:
    from src.segmentation import UNetSegmenter
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    from segmentation import UNetSegmenter

def test_segmenter_init():
    segmenter = UNetSegmenter('dummy_path.pth')
    assert segmenter.model_path == 'dummy_path.pth'
    assert segmenter.model is None
