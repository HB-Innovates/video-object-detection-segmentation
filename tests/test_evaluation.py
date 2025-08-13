import numpy as np
import torch
from src.evaluation import calculate_mAP, evaluate_model

def test_calculate_mAP():
    # Dummy predictions and ground truths
    preds = [ [[0,0,10,10]], [[5,5,15,15]] ]
    gts = [ [[0,0,10,10]], [[5,5,15,15]] ]
    mAP = calculate_mAP(preds, gts)
    assert 0.9 < mAP <= 1.0

def test_evaluate_model():
    class DummyModel:
        def eval(self): pass
        def __call__(self, images):
            return [ [0,0,10,10] for _ in images ]
    class DummyLoader:
        def __iter__(self):
            for _ in range(2):
                yield torch.zeros((1,3,10,10)), [ [0,0,10,10] ]
    model = DummyModel()
    loader = DummyLoader()
    device = torch.device('cpu')
    mAP = evaluate_model(model, loader, device)
    assert 0.9 < mAP <= 1.0
