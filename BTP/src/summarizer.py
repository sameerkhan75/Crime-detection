from __future__ import annotations

from typing import Iterable, List

from .features import VideoFeatures
from .video_processing import FrameStats


class VideoSummarizer:
    def summarize(
        self,
        label: str,
        features: VideoFeatures,
        frame_stats: Iterable[FrameStats],
    ) -> str:
        stats_list: List[FrameStats] = list(frame_stats)
        high_motion_events = [
            s for s in stats_list if s.motion_magnitude > features.average_motion * 1.5
        ]
        crowd_events = [s for s in stats_list if s.person_count >= 3]

        summary_lines = [
            f"Predicted class: {label.upper()}.",
            f"Analyzed duration: {features.duration_seconds:.1f}s across {features.frame_samples} sampled frames.",
            f"Average detected people per frame: {features.mean_person_count:.1f} (median {features.median_person_count:.1f}).",
            f"Moving objects per frame: avg {features.avg_moving_objects:.1f} (max {features.max_moving_objects:.0f}).",
            f"Motion: avg {features.average_motion:.2f}, peak {features.peak_motion:.2f}, calm ratio {features.calm_ratio:.2f}.",
            f"Active motion ratio: {features.active_motion_ratio:.2f}, late-scene activity: {features.late_motion_ratio:.2f}.",
            f"Frames with crowds (>=3 people): {len(crowd_events)}.",
            f"Frames with spikes in motion: {len(high_motion_events)}.",
        ]

        summary_lines.append(
            self._describe_activity(label, features, len(high_motion_events), len(crowd_events))
        )
        timeline_lines = self._describe_timeline(high_motion_events, crowd_events)
        if timeline_lines:
            summary_lines.append("Notable moments:")
            summary_lines.extend(f"  - {line}" for line in timeline_lines)

        segment_lines = self._describe_segments(stats_list)
        if segment_lines:
            summary_lines.append("Segment view:")
            summary_lines.extend(f"  - {line}" for line in segment_lines)

        return "\n".join(summary_lines)

    @staticmethod
    def _describe_activity(
        label: str,
        features: VideoFeatures,
        high_motion_events: int,
        crowd_events: int,
    ) -> str:
        bursts = features.motion_burst_ratio
        presence = features.person_presence_ratio
        crowd_ratio = features.crowd_ratio
        calm = features.calm_ratio
        movers = features.motion_presence_ratio
        late = features.late_motion_ratio
        trend = features.motion_trend

        if label == "robbery":
            return (
                "Robbery indicators: large groups stay together while motion repeatedly spikes."
            )
        if label == "theft":
            return "Theft indicators: isolated or brief bursts of motion while the scene stays sparse."
        if label == "assault":
            return "Assault indicators: energetic bursts with multiple people involved."
        if label == "explosion":
            return (
                "Explosion indicators: violent motion spikes with volatile movement despite little human presence."
            )
        if label == "road accident":
            return (
                "Road-accident indicators: clusters of moving objects and directional bursts on an otherwise open scene."
            )

        if bursts < 0.2 and calm > 0.7 and movers < 0.3:
            return "Normal activity indicators: calm frames dominate and only light movement occurs."
        if presence < 0.2:
            return "Mostly empty scene: very few people detected and motion stays limited."
        if crowd_ratio < 0.1 and high_motion_events <= 3:
            return "Light traffic: brief motion spikes but little crowding overall."
        if crowd_events > 0:
            return "Passing groups appear, yet motion quickly settles back down."
        if late > 0.5 or trend > 0.2:
            return "Motion escalates toward the end of the clip."
        return "Scene stays balanced with modest movement and no persistent crowds."

    @staticmethod
    def _describe_timeline(
        high_motion_events: List[FrameStats],
        crowd_events: List[FrameStats],
    ) -> List[str]:
        timeline: List[str] = []
        if high_motion_events:
            top_motion = sorted(high_motion_events, key=lambda s: s.motion_magnitude, reverse=True)[:3]
            for event in top_motion:
                timeline.append(
                    f"Spike in motion around t={VideoSummarizer._format_time(event.timestamp)} "
                    f"(level {event.motion_magnitude:.2f})."
                )

        if crowd_events:
            first = min(crowd_events, key=lambda s: s.timestamp)
            last = max(crowd_events, key=lambda s: s.timestamp)
            if first == last:
                timeline.append(
                    f"Crowd detected near t={VideoSummarizer._format_time(first.timestamp)} "
                    f"with ~{first.person_count} people."
                )
            else:
                timeline.append(
                    f"Crowd present between t={VideoSummarizer._format_time(first.timestamp)} "
                    f"and t={VideoSummarizer._format_time(last.timestamp)}."
                )
        max_motion = max((event.motion_magnitude for event in high_motion_events), default=0.0)
        if max_motion > 0:
            mover_surges = [
                s
                for s in high_motion_events
                if s.moving_objects >= 1 and s.motion_magnitude >= 0.8 * max_motion
            ]
            for event in mover_surges:
                timeline.append(
                    f"Visible movers detected near t={VideoSummarizer._format_time(event.timestamp)} "
                    f"(~{event.moving_objects} active regions)."
                )
        return timeline

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:0.1f}s"
        minutes = int(seconds // 60)
        remainder = seconds % 60
        return f"{minutes:02d}:{remainder:04.1f}"

    @staticmethod
    def _describe_segments(stats_list: List[FrameStats]) -> List[str]:
        if not stats_list:
            return []
        length = len(stats_list)
        chunk = max(1, length // 3)
        segments = []
        labels = ["Early", "Middle", "Final"]
        for idx, label in enumerate(labels):
            start = idx * chunk
            end = (idx + 1) * chunk if idx < 2 else length
            segment = stats_list[start:end]
            if not segment:
                continue
            motions = [s.motion_magnitude for s in segment]
            movers = [s.moving_objects for s in segment]
            avg_motion = sum(motions) / len(motions)
            max_motion = max(motions)
            avg_movers = sum(movers) / len(movers)
            max_movers = max(movers)
            description = VideoSummarizer._segment_sentence(
                label, segment[0].timestamp, segment[-1].timestamp, avg_motion, max_motion, avg_movers, max_movers
            )
            segments.append(description)
        return segments

    @staticmethod
    def _segment_sentence(
        label: str,
        start: float,
        end: float,
        avg_motion: float,
        max_motion: float,
        avg_movers: float,
        max_movers: int,
    ) -> str:
        if avg_motion < 0.3:
            motion_phrase = "mostly still"
        elif avg_motion < 0.8:
            motion_phrase = "showing steady movement"
        else:
            motion_phrase = "highly energetic"

        if avg_movers < 0.5:
            mover_phrase = "almost no visible actors"
        elif avg_movers < 1.5:
            mover_phrase = "one active subject"
        else:
            mover_phrase = f"up to {max_movers} moving subjects"

        return (
            f"{label} phase ({VideoSummarizer._format_time(start)}-{VideoSummarizer._format_time(end)}): "
            f"{motion_phrase} with {mover_phrase} (avg motion {avg_motion:.2f}, peak {max_motion:.2f})."
        )


__all__ = ["VideoSummarizer"]
