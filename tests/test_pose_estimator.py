import cv2

from fast_action.pose import YoloPose
from fast_action.videos import letterbox, npimg_from_cv


def test_yolo_pose():
    pose_estimator = YoloPose("yolov7-w6-pose.onnx")

    img = cv2.imread("./1.jpg")
    img, _, _ = letterbox(img, 960, auto=False)
    img = npimg_from_cv(img)

    bboxes, skeletons = pose_estimator(img)

    _, num_skeletons, num_joints, _ = skeletons.shape
    assert skeletons.shape == (
        1,
        num_skeletons,
        num_joints,
        3,
    )
