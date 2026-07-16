import numpy as np


def pos(rgb, fps):

    N = rgb.shape[0]
    weight = np.zeros(N)

    pulse = np.zeros(N)

    window = int(1.6 * fps)

    for start in range(N - window + 1):

        end = start + window

        C = rgb[start:end].T

        mean_color = np.mean(C, axis=1, keepdims=True)

        Cn = (C / mean_color) - 1

        P = np.array([
            [0, 1, -1],
            [-2, 1, 1],
        ])

        S = P @ Cn

        std1 = np.std(S[0])
        std2 = np.std(S[1])

        if std2 < 1e-8:
            continue
        else:
            alpha = std1 / std2

        h = S[0] + alpha * S[1]

        # h = S[0] + alpha * S[1]
        # I want to compare this against the original Wang et al. 
        # POS equation before we finalize the implementation.
        # There are slightly different formulations in the literature, 
        # and I don't want us to lock in a sign convention without 
        # verifying it.

        h = h - np.mean(h)

        pulse[start:end] += h
        weight[start:end] += 1

    weight[weight == 0] = 1
    pulse /= weight

    return pulse