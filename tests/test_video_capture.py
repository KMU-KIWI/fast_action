import cv2

from fast_action import videos


def test_video_capture():
    img_size = 640
    video = videos.Video(0, img_size=img_size)

    for _ in range(10):
        image = video.read()
        assert image.shape == (1, 3, img_size, img_size)

    image = video.read()
    assert image.shape == (1, 3, img_size, img_size)

    cv2.imshow("test_video_capture", videos.cvimg_from_np(image))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
