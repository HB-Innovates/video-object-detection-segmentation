from src.segmentation import UNetSegmenter

def test_segmenter_init():
    segmenter = UNetSegmenter('dummy_path.pth')
    assert segmenter.model_path == 'dummy_path.pth'
    assert segmenter.model is None
