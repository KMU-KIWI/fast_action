import argparse

from .videos import Video
from .pose import YoloPose
from .action_rec import SGN


def main(args):
    video = Video()
    pose_estimator = YoloPose()
    action_recognizer = SGN()

    while True:
        # get images
        image = video.read()

        # get skeletons from image
        skeletons = pose_estimator(image)

        # draw keypoints to image

        # filter keypoints

        # add keypoints to input

        # preprocess input

        # run action recognition
        logits = action_recognizer(skeletons)

        # display label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    main(args)
