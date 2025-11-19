from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FrameStats:
    index: int
    timestamp: float
    person_count: int
    moving_objects: int
    motion_magnitude: float


@dataclass
class VideoMetadata:
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int


class VideoProcessingError(RuntimeError):
    pass


def _create_hog_detector() -> cv2.HOGDescriptor:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def _count_people(frame_gray: np.ndarray, hog: cv2.HOGDescriptor) -> int:
    rects, _ = hog.detectMultiScale(
        frame_gray,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    return len(rects)


def _motion_magnitude(prev_gray: Optional[np.ndarray], gray: np.ndarray) -> float:
    if prev_gray is None:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(magnitude))


def _estimate_moving_objects(fg_mask: np.ndarray, min_area: int = 400) -> int:
    if fg_mask is None:
        return 0
    blurred = cv2.medianBlur(fg_mask, 5)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return int(sum(1 for contour in contours if cv2.contourArea(contour) >= min_area))


def extract_frame_stats(
    video_path: Path,
    sample_rate: float = 3.0,
    max_samples: Optional[int] = None,
) -> Tuple[List[FrameStats], VideoMetadata]:
    """Sample frames from the video and compute per-frame statistics."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoProcessingError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Fall back to a reasonable default when FPS is missing from metadata.
    effective_fps = fps if fps > 0 else 24.0
    stride = max(1, int(round(effective_fps / sample_rate))) if sample_rate > 0 else 1

    hog = _create_hog_detector()
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=25, detectShadows=False
    )
    stats: List[FrameStats] = []
    prev_gray: Optional[np.ndarray] = None
    frame_index = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % stride != 0:
                frame_index += 1
                continue

            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            person_count = _count_people(gray, hog)
            fg_mask = bg_subtractor.apply(gray)
            moving_objects = _estimate_moving_objects(fg_mask)
            motion_mag = _motion_magnitude(prev_gray, gray)
            stats.append(
                FrameStats(
                    index=frame_index,
                    timestamp=float(timestamp),
                    person_count=person_count,
                    moving_objects=moving_objects,
                    motion_magnitude=motion_mag,
                )
            )
            prev_gray = gray
            frame_index += 1

            if max_samples and len(stats) >= max_samples:
                break
    finally:
        capture.release()

    duration_seconds = (
        (frame_count / fps) if fps > 0 else (len(stats) / sample_rate if sample_rate > 0 else 0.0)
    )

    metadata = VideoMetadata(
        fps=fps if fps > 0 else effective_fps,
        frame_count=frame_count,
        duration_seconds=float(duration_seconds),
        width=width,
        height=height,
    )
    return stats, metadata


__all__ = ["FrameStats", "VideoMetadata", "extract_frame_stats", "VideoProcessingError"]
