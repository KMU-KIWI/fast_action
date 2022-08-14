import cv2

from fast_action.pose import YoloPose
from fast_action.videos import letterbox, npimg_from_cv


def test_yolo_pose():
    pose_estimator = YoloPose("yolov7-w6-pose.onnx")

    image = cv2.imread("./person.jpg")
    image = letterbox(image, 960, stride=64, auto=True)
    breakpoint()
    image = npimg_from_cv(image)

    skeletons = pose_estimator(image)

    cv2.imshow("dlfkjasdk", skeletons)

    cv2.waitKey(0)

    assert skeletons.shape == (
        1,
        pose_estimator.num_skeletons,
        pose_estimator.num_joints,
        3,
    )
