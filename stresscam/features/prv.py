import numpy as np
from scipy.signal import find_peaks
import pandas as pd


def compute_prv_from_peaks(peaks, fs):
    """
    Compute Pulse Rate Variability (PRV) features from detected peaks.

    Parameters
    ----------
    peaks : ndarray
        Peak indices.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    dict
        Dictionary containing PRV features and intermediate results.
    """

    if len(peaks) < 2:
        raise ValueError("Not enough peaks detected to compute PRV features.")

    peak_times = peaks / fs

    ibi = np.diff(peak_times)

    hr = 60 / ibi

    mean_hr = np.mean(hr)
    mean_ibi = np.mean(ibi)

    sdnn = np.std(ibi, ddof=1)

    rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2))

    nn50 = np.sum(np.abs(np.diff(ibi)) > 0.05)

    pnn50 = 100 * nn50 / len(np.diff(ibi))

    return {
        "Mean HR": mean_hr,
        "Mean IBI": mean_ibi,
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "NN50": nn50,
        "pNN50": pnn50,
        "Peaks": peaks,
        "Peak Times": peak_times,
        "IBI": ibi,
        "HR": hr,
    }


def extract_prv_features(
    pulse,
    fps,
    distance=None,
    prominence=0.001,
):
    """
    Extract PRV features from a recovered rPPG signal.
    """

    if distance is None:
        distance = int(0.5 * fps)

    peaks, _ = find_peaks(
        pulse,
        distance=distance,
        prominence=prominence,
    )

    return compute_prv_from_peaks(
        peaks=peaks,
        fs=fps,
    )

def extract_prv_features_from_bvp(
    bvp,
    fs,
    distance=None,
    prominence=None,
):
    """
    Extract PRV features from a ground-truth BVP signal.

    Parameters
    ----------
    bvp : ndarray
        Ground-truth blood volume pulse signal.
    fs : float
        Sampling frequency (Hz).
    distance : int, optional
        Minimum distance between peaks (samples).
    prominence : float, optional
        Minimum peak prominence.

    Returns
    -------
    dict
        Dictionary containing PRV features and intermediate results.
    """

    if distance is None:
        distance = int(0.5 * fs)

    peak_kwargs = {"distance": distance}

    if prominence is not None:
        peak_kwargs["prominence"] = prominence

    peaks, _ = find_peaks(
        bvp,
        **peak_kwargs,
    )

    return compute_prv_from_peaks(
        peaks=peaks,
        fs=fs,
    )