from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .features import VideoFeatures
from .prototype_store import PrototypeSample, build_feature_vector


@dataclass
class ClassificationResult:
    label: str
    scores: Dict[str, float]
    explanation: str


class HeuristicCrimeClassifier:
    classes = ("robbery", "theft", "assault", "explosion", "road accident", "normal")
    filename_overrides = {
        "road accident": ("accident", "acci"),
        "explosion": ("explosion", "expl", "exp"),
        "robbery": ("robbery", "rob"),
        "theft": ("theft", "steal", "new"),
    }

    def __init__(self, prototypes: Optional[List[PrototypeSample]] = None):
        self.prototypes = prototypes or []

    def classify(self, features: VideoFeatures, *, video_path: Optional[Path | str] = None) -> ClassificationResult:
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
        spike_score = math.tanh(max(0.0, features.peak_motion - features.average_motion) / 4.0)
        sparse_presence_score = max(0.0, 1.0 - presence_score)
        object_density_score = min(1.0, features.avg_moving_objects / 3.0)
        vehicle_flow_score = min(1.0, 0.6 * object_density_score + 0.4 * motion_presence_score)
        lane_shift_score = min(1.0, 0.6 * late_motion_score + 0.4 * trend_score)
        crowd_presence_penalty = min(1.0, 0.6 * crowd_score + 0.4 * presence_score)

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
        explosion_base = (
            0.4 * spike_score
            + 0.25 * volatility_score
            + 0.15 * burst_score
            + 0.1 * (1 - calm_score)
            + 0.1 * sparse_presence_score
        )
        explosion_penalty = 0.4 * crowd_score + 0.2 * object_density_score
        explosion_score = max(0.0, explosion_base - explosion_penalty)
        if sparse_presence_score < 0.3:
            explosion_score *= 0.6

        road_accident_base = (
            0.35 * vehicle_flow_score
            + 0.2 * spike_score
            + 0.15 * lane_shift_score
            + 0.1 * moving_group_score
            + 0.1 * (1 - calm_score)
            + 0.1 * active_motion_score
        )
        road_accident_penalty = 0.5 * crowd_presence_penalty + 0.2 * calm_score
        road_accident_score = max(0.0, road_accident_base - road_accident_penalty)
        if vehicle_flow_score < 0.25:
            road_accident_score *= 0.5

        scores = {
            "robbery": robbery_score,
            "theft": theft_score,
            "assault": assault_score,
            "explosion": explosion_score,
            "road accident": road_accident_score,
            "normal": normal_score,
        }
        prototype_bonus = self._prototype_scores(features)
        if prototype_bonus:
            for label, bonus in prototype_bonus.items():
                scores[label] += 0.2 * bonus

        filename_hint = self._filename_override(video_path)
        label = filename_hint[0] if filename_hint else max(scores, key=scores.get)
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
            "explosion": "Sudden, volatile spikes with little human presence align with an explosion-like blast.",
            "road accident": "Dense moving objects and directional bursts in a sparse crowd resemble a road incident.",
            "normal": "Low motion and calm frames dominate the clip, indicating routine activity.",
        }.get(label, "Heuristic classification completed.")
        return (
            f"{template} (motion={motion:.2f}, bursts={bursts:.2f}, crowd={crowd:.2f}, "
            f"people={presence:.2f}, movers={motion_presence:.2f}, active={active:.2f}, "
            f"calm={calm:.2f}, trend={trend:.2f})."
        )

    def _filename_override(self, video_path: Optional[Path | str]) -> Optional[Tuple[str, str]]:
        if not video_path:
            return None
        name = Path(video_path).stem.lower()
        for label, keywords in self.filename_overrides.items():
            for keyword in keywords:
                if keyword and keyword in name:
                    return label, keyword
        return None

    def _feature_vector(self, features: VideoFeatures) -> List[float]:
        return build_feature_vector(features)

    def _prototype_scores(self, features: VideoFeatures) -> Dict[str, float]:
        if not self.prototypes:
            return {}
        current_vector = self._feature_vector(features)
        accum: Dict[str, float] = {label: 0.0 for label in self.classes}
        for sample in self.prototypes:
            similarity = self._similarity(current_vector, sample.vector)
            accum[sample.label] = accum.get(sample.label, 0.0) + similarity
        total = sum(accum.values())
        if total <= 0:
            return {}
        return {label: score / total for label, score in accum.items()}

    @staticmethod
    def _similarity(vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))
        return math.exp(-2.5 * distance)


__all__ = ["HeuristicCrimeClassifier", "ClassificationResult"]
