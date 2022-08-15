import argparse

import torch
import torch.nn.functional as F

import numpy as np

import cv2

from videos import Video, npimg_from_cv, cvimg_from_np, letterbox
from pose import YoloPose, plot_skeleton_kpts
from action_rec import sample_frames, SGN
from ntu_labels import labels
from draw import write_topk


def main(args):
    video = Video(args.video)

    pose_estimator = YoloPose(args.pose_onnx_path)
    action_recognizer = SGN(args.action_onnx_path)

    buffer = []
    for img in video:
        img, _, (dw, dh) = letterbox(img, args.img_size, auto=False)
        img = npimg_from_cv(img)

        # get skeletons from image
        bboxes, skeletons = pose_estimator(img)
        if bboxes is not None:
            frame = np.zeros((1, 2, 17, 3))
            frame[:, : skeletons.shape[1]] = skeletons

            if len(buffer) == args.window:
                buffer.pop(0)
            buffer.append(frame)

            img = cvimg_from_np(img)

            for idx in range(skeletons.shape[1]):
                plot_skeleton_kpts(
                    img, skeletons[0, idx].reshape(-1), 3, img_size=args.img_size
                )



            frames = np.zeros((1, 2, args.window, 17, 3))
            frames[:, :, : len(buffer)] = np.stack(buffer, axis=2)

            indices = sample_frames(len(buffer), clip_len=args.num_frames)
            skeletons = frames[:, :, indices].astype(np.float32)

            output = action_recognizer(
                skeletons, args.img_size - 2 * dw, args.img_size - 2 * dh
            )

            output = torch.from_numpy(output)
            probs = F.softmax(output, dim=-1)[0]

            topk = torch.topk(probs, 5)

            write_topk(img, labels, topk.indices, topk.values)

        else:
            img = cvimg_from_np(img)
        cv2.imshow("test_yolo_pose", img)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", type=str, default=0)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--pose_onnx_path", type=str, default="yolov7-w6-pose.onnx")
    parser.add_argument("--action_onnx_path", type=str, default="sgn.onnx")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--num_frames", type=int, default=20)

    args = parser.parse_args()
    main(args)
