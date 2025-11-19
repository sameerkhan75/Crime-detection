from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.classifier import HeuristicCrimeClassifier
from src.features import VideoFeatureExtractor
from src.file_utils import find_video_file
from src.prototype_store import PrototypeStore
from src.summarizer import VideoSummarizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heuristic crime detection demo")
    parser.add_argument(
        "--video",
        "-v",
        help="Optional path to the video file. Defaults to the first video in the project root or videos/",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=3.0,
        help="Number of frames per second to sample for analysis.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on the number of frames to analyze.",
    )
    parser.add_argument(
        "--dump-stats",
        type=Path,
        help="Optional path to dump raw statistics as JSON.",
    )
    parser.add_argument(
        "--train-label",
        choices=HeuristicCrimeClassifier.classes,
        help="Store the analyzed video as a prototype for the given label.",
    )
    parser.add_argument(
        "--prototype-store",
        type=Path,
        default=Path("trained_samples.json"),
        help="Path to the prototype memory JSON file.",
    )
    return parser.parse_args()


def serialize(features, frame_stats, classification) -> dict[str, Any]:
    return {
        "features": asdict(features),
        "frame_stats": [asdict(frame) for frame in frame_stats],
        "classification": {
            "label": classification.label,
            "scores": classification.scores,
            "explanation": classification.explanation,
        },
    }


def main() -> None:
    args = parse_args()
    video_path = find_video_file(args.video)

    extractor = VideoFeatureExtractor(sample_rate=args.sample_rate, max_samples=args.max_frames)
    features, frame_stats, metadata = extractor.extract(video_path)

    prototype_store = PrototypeStore(args.prototype_store)
    classifier = HeuristicCrimeClassifier(prototypes=list(prototype_store))
    classification = classifier.classify(features, video_path=video_path)

    summarizer = VideoSummarizer()
    summary = summarizer.summarize(classification.label, features, frame_stats)

    print("Video:", video_path)
    print("Metadata:", metadata)
    print()
    print(summary)
    print()
    print("Class scores:")
    for label, score in classification.scores.items():
        print(f"  {label:>7}: {score:.3f}")
    print(classification.explanation)

    if args.dump_stats:
        payload = serialize(features, frame_stats, classification)
        args.dump_stats.parent.mkdir(parents=True, exist_ok=True)
        args.dump_stats.write_text(json.dumps(payload, indent=2))
        print(f"\nDetailed statistics written to {args.dump_stats}")

    if args.train_label:
        prototype_store.add_sample(args.train_label, features, video_path)
        print(f"Stored prototype for label '{args.train_label}' in {args.prototype_store}")


if __name__ == "__main__":
    main()
