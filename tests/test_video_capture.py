import numpy as np

from fast_action import videos


def test_video_capture():
    video = videos.Video(0)

    for img in video:
        break

    assert isinstance(img, np.ndarray)
    assert img.shape[2] == 3
