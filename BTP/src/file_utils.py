from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def list_video_files(search_dirs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        for ext in VIDEO_EXTENSIONS:
            files.extend(sorted(directory.glob(f"*{ext}")))
    return files


def find_video_file(candidate: Optional[str] = None) -> Path:
    """Return the first matching video file, or raise FileNotFoundError."""

    root = Path(__file__).resolve().parents[1]
    if candidate:
        path = Path(candidate)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"Could not find video file at {path}")
        return path

    default_dirs = [root / "videos", root]
    video_files = list_video_files(default_dirs)
    if not video_files:
        raise FileNotFoundError(
            "No video files were found. Place a video in the project root or the"
            "videos/ directory, then run the analyzer again."
        )
    return video_files[0]


__all__ = ["find_video_file", "list_video_files", "VIDEO_EXTENSIONS"]
