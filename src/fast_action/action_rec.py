import numpy as np

import onnxruntime as ort


class SGN:
    def __init__(self, onnx_path):
        self.ort_sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        self.input_name = self.ort_sess.get_inputs()[0].name

    def __call__(self, skeletons, w, h):
        # N, M, T, V, C
        skeletons[:, :, :, :, 0] = (skeletons[:, :, :, :, 0] - (w / 2)) / (w / 2)
        skeletons[:, :, :, :, 1] = (skeletons[:, :, :, :, 1] - (h / 2)) / (w / 2)

        (output,) = self.ort_sess.run([], {self.input_name: skeletons})

        return output


def sample_frames(
    num_frames, clip_len, num_clips=1, p_interval=(1, 1), float_ok=False, seed=42
):
    """Uniformly sample indices for testing clips.
    Args:
        num_frames (int): The number of frames.
        clip_len (int): The length of the clip.
    """
    np.random.seed(seed)
    if float_ok:
        interval = (num_frames - 1) / clip_len
        offsets = np.arange(clip_len) * interval
        inds = np.concatenate(
            [np.random.rand(clip_len) * interval + offsets for i in range(num_clips)]
        ).astype(np.float32)

    all_inds = []

    for i in range(num_clips):

        old_num_frames = num_frames
        pi = p_interval
        ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
        num_frames = int(ratio * num_frames)
        off = np.random.randint(old_num_frames - num_frames + 1)

        if num_frames < clip_len:
            start_ind = i if num_frames < num_clips else i * num_frames // num_clips
            inds = np.arange(start_ind, start_ind + clip_len)
        elif clip_len <= num_frames < clip_len * 2:
            basic = np.arange(clip_len)
            inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = np.random.randint(bsize)
            inds = bst + offset

        all_inds.append(inds + off)
        num_frames = old_num_frames

    return np.concatenate(all_inds)
