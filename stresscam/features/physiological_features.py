import numpy as np
from scipy.signal import find_peaks
import cv2
from tqdm.auto import tqdm

from stresscam.preprocessing.face_detection import FaceDetector
from stresscam.preprocessing.skin_segmentation import SkinSegmenter
from stresscam.preprocessing.rgb_extraction import RGBExtractor
from stresscam.rppg.pos import pos


def extract_prv_features(
    pulse,
    fps,
    distance=None,
    prominence=0.001,
):
    """
    Extract pulse rate variability (PRV) features from an rPPG signal.

    Parameters
    ----------
    pulse : ndarray
        Recovered rPPG signal.
    fps : float
        Sampling frequency (frames per second).
    distance : int, optional
        Minimum distance between consecutive peaks (samples).
        If None, defaults to 0.5 * fps (~120 BPM maximum).
    prominence : float, optional
        Minimum peak prominence.

    Returns
    -------
    dict
        Dictionary containing PRV features and intermediate results.
    """

    if distance is None:
        distance = int(0.5 * fps)

    peaks, _ = find_peaks(
        pulse,
        distance=distance,
        prominence=prominence,
    )

    # Need at least two peaks to compute IBI
    if len(peaks) < 2:
        raise ValueError("Not enough peaks detected to compute PRV features.")

    peak_times = peaks / fps

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

def process_trial(video_path, show_progress=True):
    """
    Process a single video trial and extract PRV features.

    Parameters
    ----------
    video_path : str or Path
        Path to the input video.

    Returns
    -------
    dict
        Dictionary containing recovered pulse signal,
        physiological features and intermediate results.
    """

    detector = FaceDetector()
    segmenter = SkinSegmenter()
    extractor = RGBExtractor()

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    rgb_trace = []

    processed = 0
    failed = 0

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if show_progress:
        pbar = tqdm(
            total=n_frames,
            desc=f"Processing {video_path.stem}",
            unit="frame",
            dynamic_ncols=True,
            colour="green",
            leave=False,
        )

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if show_progress:
            pbar.update(1)

        # Face detection
        det_results = detector.detect(frame)

        face = detector.crop(frame, det_results)

        if face is None:
            failed += 1
            continue

        # Skin segmentation
        mask, _ = segmenter.segment(face)

        if mask is None:
            failed += 1
            continue

        # RGB extraction
        rgb = extractor.extract(face, mask)

        rgb_trace.append(rgb)

        processed += 1

    # if show_progress:
    #     pbar.close()

    cap.release()

    rgb_trace = np.asarray(rgb_trace)

    pulse = pos(rgb_trace, fps)

    features = extract_prv_features(
        pulse,
        fps,
    )

    features["Pulse"] = pulse
    features["FPS"] = fps
    features["RGB Trace"] = rgb_trace
    features["Processed Frames"] = processed
    features["Failed Frames"] = failed

    return features

def extract_prv_features_from_bvp(
        bvp,
        bvp_fs,
        target_fs=None,
    ):
    