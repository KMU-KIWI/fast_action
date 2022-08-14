import cv2

import numpy as np


# from https://github.com/WongKinYiu/yolov7/blob/pose/utils/datasets.py
def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


def cvimg_from_np(img):
    img = img[0, ::-1].transpose(1, 2, 0)  # RGB to BGR, CHW to HWC
    return img.copy()


def npimg_from_cv(img):
    img = img.astype(np.float32)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.expand_dims(img, axis=0)  # add batch dim
    img = np.ascontiguousarray(img)
    return img


class Video:
    """
    Captures frame from camera or video file
    """

    def __init__(self, source, img_size=640, is_rect=False, stride=32):
        self.img_size = img_size
        self.is_rect = is_rect
        self.stride = stride

        self.cap = cv2.VideoCapture(0)

    def read(self):
        """
        Read single frame from camera or file

        make sure output frame matches height, width by adding letterboxes

        Returns:
            numpy array of shape (1, 3, height, width)
        """
        # Capture
        ret, img = self.cap.read()

        # Letterbox
        img, ratio, (dw, dh) = letterbox(
            img, self.img_size, auto=self.is_rect, stride=self.stride
        )

        # Convert
        img = npimg_from_cv(img)

        return img
