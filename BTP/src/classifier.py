from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from .features import VideoFeatures


@dataclass
class ClassificationResult:
    label: str
    scores: Dict[str, float]
    explanation: str


class HeuristicCrimeClassifier:
    classes = ("robbery", "theft", "assault", "normal")

    def classify(self, features: VideoFeatures) -> ClassificationResult:
        if features.frame_samples == 0:
            raise ValueError("No frames were analyzed. Provide a longer clip or adjust sampling.")

        motion_score = self._normalize_motion(features.average_motion, features.peak_motion)
        volatility_score = math.tanh(features.motion_std / 3.5)
        burst_score = min(1.0, features.motion_burst_ratio)
        crowd_score = min(1.0, features.crowd_ratio)
        multi_person_score = min(1.0, features.multi_person_ratio)
        solo_motion_score = min(1.0, features.solo_motion_ratio)
        calm_score = max(0.0, min(1.0, features.calm_ratio))
        presence_score = min(1.0, features.person_presence_ratio)
        motion_presence_score = min(1.0, features.motion_presence_ratio)
        active_motion_score = min(1.0, features.active_motion_ratio)
        late_motion_score = min(1.0, features.late_motion_ratio)
        trend_score = math.tanh(max(0.0, features.motion_trend) / 2.0)
        moving_group_score = min(1.0, features.max_moving_objects / 4.0)

        robbery_score = (
            0.25 * crowd_score
            + 0.2 * multi_person_score
            + 0.2 * moving_group_score
            + 0.15 * burst_score
            + 0.1 * late_motion_score
            + 0.1 * motion_presence_score
        )
        theft_score = (
            0.35 * solo_motion_score
            + 0.25 * active_motion_score
            + 0.15 * late_motion_score
            + 0.15 * (1 - crowd_score)
            + 0.1 * motion_presence_score
        )
        assault_score = (
            0.3 * burst_score
            + 0.2 * motion_score
            + 0.2 * active_motion_score
            + 0.15 * multi_person_score
            + 0.1 * trend_score
            + 0.05 * motion_presence_score
        )
        normal_score = (
            0.3 * calm_score
            + 0.25 * (1 - active_motion_score)
            + 0.2 * (1 - burst_score)
            + 0.15 * (1 - motion_presence_score)
            + 0.1 * (1 - late_motion_score)
        )

        scores = {
            "robbery": robbery_score,
            "theft": theft_score,
            "assault": assault_score,
            "normal": normal_score,
        }
        label = max(scores, key=scores.get)
        explanation = self._build_explanation(
            label,
            motion=motion_score,
            crowd=crowd_score,
            calm=calm_score,
            bursts=burst_score,
            presence=presence_score,
            motion_presence=motion_presence_score,
            active=active_motion_score,
            trend=trend_score,
        )
        return ClassificationResult(label=label, scores=scores, explanation=explanation)

    @staticmethod
    def _normalize_motion(avg_motion: float, peak_motion: float) -> float:
        # Hyperbolic tan keeps values in [0, 1) without requiring dataset calibration.
        avg_component = math.tanh(avg_motion / 6.0)
        peak_component = math.tanh(peak_motion / 10.0)
        return min(1.0, 0.6 * avg_component + 0.4 * peak_component)

    @staticmethod
    def _build_explanation(
        label: str,
        *,
        motion: float,
        crowd: float,
        calm: float,
        bursts: float,
        presence: float,
        motion_presence: float,
        active: float,
        trend: float,
    ) -> str:
        template = {
            "robbery": "Large groups and repeated bursts of motion suggest a coordinated grab.",
            "theft": "Isolated motion while the scene stays sparse matches theft-like activity.",
            "assault": "Aggressive bursts with multiple participants point to an assault pattern.",
            "normal": "Low motion and calm frames dominate the clip, indicating routine activity.",
        }.get(label, "Heuristic classification completed.")
        return (
            f"{template} (motion={motion:.2f}, bursts={bursts:.2f}, crowd={crowd:.2f}, "
            f"people={presence:.2f}, movers={motion_presence:.2f}, active={active:.2f}, "
            f"calm={calm:.2f}, trend={trend:.2f})."
        )


__all__ = ["HeuristicCrimeClassifier", "ClassificationResult"]
