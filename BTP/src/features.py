from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .video_processing import FrameStats, VideoMetadata, extract_frame_stats


@dataclass
class VideoFeatures:
    average_motion: float
    peak_motion: float
    motion_std: float
    crowd_ratio: float
    solo_motion_ratio: float
    motion_burst_ratio: float
    person_presence_ratio: float
    multi_person_ratio: float
    motion_presence_ratio: float
    active_motion_ratio: float
    late_motion_ratio: float
    motion_trend: float
    calm_ratio: float
    median_person_count: float
    mean_person_count: float
    avg_moving_objects: float
    max_moving_objects: float
    duration_seconds: float
    frame_samples: int
    fps_estimate: float


class VideoFeatureExtractor:
    def __init__(self, sample_rate: float = 3.0, max_samples: int | None = None):
        self.sample_rate = sample_rate
        self.max_samples = max_samples

    def extract(self, video_path: Path) -> Tuple[VideoFeatures, List[FrameStats], VideoMetadata]:
        stats, metadata = extract_frame_stats(video_path, self.sample_rate, self.max_samples)
        features = self._aggregate(stats, metadata)
        return features, stats, metadata

    def _aggregate(self, stats: List[FrameStats], metadata: VideoMetadata) -> VideoFeatures:
        if not stats:
            return VideoFeatures(
                average_motion=0.0,
                peak_motion=0.0,
                motion_std=0.0,
                crowd_ratio=0.0,
                solo_motion_ratio=0.0,
                motion_burst_ratio=0.0,
                person_presence_ratio=0.0,
                multi_person_ratio=0.0,
                motion_presence_ratio=0.0,
                active_motion_ratio=0.0,
                late_motion_ratio=0.0,
                motion_trend=0.0,
                calm_ratio=1.0,
                median_person_count=0.0,
                mean_person_count=0.0,
                avg_moving_objects=0.0,
                max_moving_objects=0.0,
                duration_seconds=metadata.duration_seconds,
                frame_samples=0,
                fps_estimate=metadata.fps,
            )

        motions = np.array([s.motion_magnitude for s in stats], dtype=float)
        people = np.array([s.person_count for s in stats], dtype=float)
        moving_objects = np.array([s.moving_objects for s in stats], dtype=float)

        calm_threshold = 1.0
        solo_motion_threshold = 1.2
        burst_threshold = float(np.percentile(motions, 75)) if len(motions) else 0.0
        active_threshold = 0.35

        raw_calm_ratio = float(np.mean(motions < calm_threshold))
        solo_motion_ratio = float(np.mean((people <= 1) & (motions > solo_motion_threshold)))
        crowd_ratio = float(np.mean(people >= 3))
        person_presence_ratio = float(np.mean(people >= 1))
        multi_person_ratio = float(np.mean(people >= 2))
        motion_burst_ratio = float(np.mean(motions >= burst_threshold))
        motion_presence_ratio = float(np.mean(moving_objects >= 1))
        active_motion_ratio = float(np.mean(motions >= active_threshold))
        calm_ratio = float(0.5 * raw_calm_ratio + 0.5 * max(0.0, 1.0 - active_motion_ratio))

        segment = max(1, len(stats) // 3)
        early_motions = motions[:segment]
        late_motions = motions[-segment:]
        late_motion_ratio = (
            float(np.mean(late_motions >= active_threshold)) if len(late_motions) else active_motion_ratio
        )
        motion_trend = (
            float(np.mean(late_motions) - np.mean(early_motions))
            if len(early_motions) and len(late_motions)
            else 0.0
        )

        return VideoFeatures(
            average_motion=float(np.mean(motions)),
            peak_motion=float(np.max(motions)),
            motion_std=float(np.std(motions)),
            crowd_ratio=crowd_ratio,
            solo_motion_ratio=solo_motion_ratio,
            motion_burst_ratio=motion_burst_ratio,
            person_presence_ratio=person_presence_ratio,
            multi_person_ratio=multi_person_ratio,
            motion_presence_ratio=motion_presence_ratio,
            active_motion_ratio=active_motion_ratio,
            late_motion_ratio=late_motion_ratio,
            motion_trend=motion_trend,
            calm_ratio=calm_ratio,
            median_person_count=float(np.median(people)),
            mean_person_count=float(np.mean(people)),
            avg_moving_objects=float(np.mean(moving_objects)),
            max_moving_objects=float(np.max(moving_objects)),
            duration_seconds=metadata.duration_seconds,
            frame_samples=len(stats),
            fps_estimate=metadata.fps,
        )


__all__ = ["VideoFeatures", "VideoFeatureExtractor"]
