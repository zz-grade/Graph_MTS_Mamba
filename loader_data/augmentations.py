import pywt
import numpy as np
import torch


def changes_correlations(Adj, num_remained):
    return Adj * num_remained


def wavelet_transform(x, weak=True):
    b_g, b_d = pywt.dwt(x, 'db2')
    if weak:
        b_d = b_d + np.random.random(b_d.shape) * 0.1
    else:
        b_g = b_g + np.random.random(b_g.shape) * 0.1

    a_data = pywt.idwt(b_g, b_d, 'db2')
    return a_data


def partial_changes_MTS(x, coeffi, num_maintain):
    x_aug = wavelet_transform(x, coeffi)

    bs, N, time_length = x_aug.shape
    idx = np.random.randint(0, N, size=[bs, num_maintain])
    bat_id = np.expand_dims(np.arange(num_maintain), axis=1)
    x[bat_id, idx] = x_aug[bat_id, idx]

    return x


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(num_segs, size=num_segs[i], replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
        return torch.from_numpy(ret)


def scaling(x, digma=1.1):
    factor = np.random.normal(0, digma, (x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate(ai, axis=1)


def DataTransform(sample, config):
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = scaling(permutation(sample, config.augmentation.max_seg), config.augmentation.jitter_scale_ratio)
    return weak_aug, strong_aug
