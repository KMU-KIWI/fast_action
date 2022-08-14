import cv2

import numpy as np

from fast_action.pose import YoloPose, plot_skeleton_kpts
from fast_action.videos import letterbox, npimg_from_cv


def test_yolo_pose():
    pose_estimator = YoloPose("yolov7-w6-pose.onnx")

    img = cv2.imread("./1.jpg")
    img, _, _ = letterbox(img, 960, auto=False)
    img = npimg_from_cv(img) / 255

    bboxes, skeletons = pose_estimator(img)

    nimg = img[0].transpose(1, 2, 0) * 255
    nimg = nimg.astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    for idx in range(skeletons.shape[1]):
        plot_skeleton_kpts(nimg, skeletons[0, idx].reshape(-1), 3)

    cv2.imshow("test_yolo_pose", nimg)

    cv2.waitKey(0)

    _, num_skeletons, num_joints, _ = skeletons.shape
    assert skeletons.shape == (
        1,
        num_skeletons,
        num_joints,
        3,
    )
