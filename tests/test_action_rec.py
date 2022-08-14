import numpy as np

from fast_action.action_rec import SGN


def test_sgn():
    action_recognizer = SGN("sgn.onnx")

    skeletons = np.random.randn(1, 2, 20, 17, 3).astype(np.float32)
    output = action_recognizer(skeletons, 960, 960)
    assert output.shape == (1, 120)
