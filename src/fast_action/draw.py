import cv2


def write_topk(img, labels, indices, probs):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = [50, 50]

    # Blue color in BGR
    color = (255, 0, 0)

    for idx, prob in zip(indices, probs):
        img = cv2.putText(
            img,
            f"{labels[idx]}: {prob:.2}",
            org,
            fontFace=font,
            fontScale=1,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        org[1] += 25

    return img
