from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from .features import VideoFeatures

# Fields from VideoFeatures used for similarity comparisons along with scaling factors
FEATURE_VECTOR_FIELDS: Sequence[tuple[str, float]] = (
    ("average_motion", 5.0),
    ("peak_motion", 10.0),
    ("motion_std", 5.0),
    ("crowd_ratio", 1.0),
    ("motion_burst_ratio", 1.0),
    ("person_presence_ratio", 1.0),
    ("active_motion_ratio", 1.0),
    ("late_motion_ratio", 1.0),
    ("avg_moving_objects", 4.0),
    ("max_moving_objects", 6.0),
    ("calm_ratio", 1.0),
    ("multi_person_ratio", 1.0),
)


def build_feature_vector(features: VideoFeatures) -> List[float]:
    """Convert VideoFeatures into a normalized vector for prototype matching."""

    vector: List[float] = []
    for field, scale in FEATURE_VECTOR_FIELDS:
        value = float(getattr(features, field))
        normalized = value / scale if scale else value
        vector.append(max(-2.0, min(2.0, normalized)))
    return vector


@dataclass
class PrototypeSample:
    label: str
    vector: List[float]
    source: str | None = None


class PrototypeStore:
    """Simple JSON-backed store for labeled feature prototypes."""

    def __init__(self, path: Path):
        self.path = path
        self.samples: List[PrototypeSample] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.samples = []
            return
        try:
            payload = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            self.samples = []
            return
        samples: List[PrototypeSample] = []
        for entry in payload if isinstance(payload, list) else []:
            label = entry.get("label")
            vector = entry.get("vector")
            if not isinstance(label, str) or not isinstance(vector, list):
                continue
            samples.append(
                PrototypeSample(
                    label=label,
                    vector=[float(v) for v in vector],
                    source=entry.get("source"),
                )
            )
        self.samples = samples

    def add_sample(self, label: str, features: VideoFeatures, source: Path | None = None) -> None:
        vector = build_feature_vector(features)
        source_name = source.name if source else None
        if source_name:
            self.samples = [s for s in self.samples if s.source != source_name]
        self.samples.append(PrototypeSample(label=label, vector=vector, source=source_name))
        self._save()

    def extend(self, samples: Iterable[PrototypeSample]) -> None:
        self.samples.extend(samples)
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(sample) for sample in self.samples]
        self.path.write_text(json.dumps(payload, indent=2))

    def __iter__(self):
        return iter(self.samples)


__all__ = ["PrototypeStore", "PrototypeSample", "build_feature_vector", "FEATURE_VECTOR_FIELDS"]
