"""Anomaly detection for identifying unusual security-relevant behaviors."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from raksha.models import AnomalyType, BoundingBox, Detection, SecurityEvent, Severity


@dataclass
class _TrackedSubject:
    """Internal record for a subject being monitored for anomalies."""

    track_id: int
    positions: list[tuple[float, float]] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)

    def add_position(self, cx: float, cy: float, ts: float | None = None) -> None:
        self.positions.append((cx, cy))
        self.timestamps.append(ts if ts is not None else time.time())

    @property
    def duration(self) -> float:
        if not self.timestamps:
            return 0.0
        return self.timestamps[-1] - self.first_seen

    @property
    def displacement(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        p0 = np.array(self.positions[0])
        p1 = np.array(self.positions[-1])
        return float(np.linalg.norm(p1 - p0))

    @property
    def recent_speed(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        n = min(5, len(self.positions))
        positions = self.positions[-n:]
        times = self.timestamps[-n:]
        dt = times[-1] - times[0]
        if dt <= 0:
            return 0.0
        dist = sum(
            float(np.linalg.norm(np.array(positions[i + 1]) - np.array(positions[i])))
            for i in range(len(positions) - 1)
        )
        return dist / dt


@dataclass
class AnomalyDetector:
    """Detects unusual behaviors from tracked detections.

    Supported anomalies:
    - **Loitering**: a subject stays in a small area for an extended period.
    - **Running**: a subject moves faster than the configured speed threshold.
    - **Crowd forming**: the number of detections in a region exceeds a limit.
    - **Perimeter breach**: a detection appears inside a restricted bounding box.
    """

    loiter_time_threshold: float = 30.0  # seconds
    loiter_displacement_threshold: float = 50.0  # pixels
    run_speed_threshold: float = 200.0  # pixels per second
    crowd_count_threshold: int = 5
    crowd_radius: float = 150.0  # pixels

    _subjects: dict[int, _TrackedSubject] = field(default_factory=dict, repr=False)
    _restricted_zones: list[BoundingBox] = field(default_factory=list, repr=False)

    def set_restricted_zones(self, zones: list[BoundingBox]) -> None:
        """Define bounding boxes for restricted perimeter zones."""
        self._restricted_zones = list(zones)

    def analyze(
        self,
        detections: list[Detection],
        current_time: float | None = None,
    ) -> list[SecurityEvent]:
        """Analyze a set of detections for anomalous behavior.

        Args:
            detections: Current frame detections (should include ``track_id``).
            current_time: Override timestamp for deterministic testing.

        Returns:
            List of security events for any anomalies detected this frame.
        """
        ts = current_time if current_time is not None else time.time()
        events: list[SecurityEvent] = []

        # Update subject histories
        for det in detections:
            tid = det.track_id if det.track_id is not None else hash(det.detection_id)
            if tid not in self._subjects:
                self._subjects[tid] = _TrackedSubject(track_id=tid, first_seen=ts)
            self._subjects[tid].add_position(*det.bbox.center, ts)

        # Check each anomaly type
        events.extend(self._check_loitering(detections, ts))
        events.extend(self._check_running(detections, ts))
        events.extend(self._check_crowd(detections, ts))
        events.extend(self._check_perimeter(detections, ts))

        # Prune stale subjects (not seen for 60 s)
        stale = [
            k
            for k, v in self._subjects.items()
            if v.timestamps and ts - v.timestamps[-1] > 60
        ]
        for k in stale:
            del self._subjects[k]

        return events

    # ------------------------------------------------------------------
    # Individual anomaly checks
    # ------------------------------------------------------------------

    def _check_loitering(
        self, detections: list[Detection], ts: float
    ) -> list[SecurityEvent]:
        events: list[SecurityEvent] = []
        for det in detections:
            tid = det.track_id if det.track_id is not None else hash(det.detection_id)
            subj = self._subjects.get(tid)
            if subj is None:
                continue
            if (
                subj.duration >= self.loiter_time_threshold
                and subj.displacement < self.loiter_displacement_threshold
            ):
                events.append(
                    SecurityEvent(
                        event_type="anomaly",
                        anomaly_type=AnomalyType.LOITERING,
                        severity=Severity.WARNING,
                        description=(
                            f"Subject {tid} loitering for {subj.duration:.0f}s "
                            f"with displacement {subj.displacement:.1f}px"
                        ),
                        detections=[det],
                    )
                )
        return events

    def _check_running(
        self, detections: list[Detection], ts: float
    ) -> list[SecurityEvent]:
        events: list[SecurityEvent] = []
        for det in detections:
            tid = det.track_id if det.track_id is not None else hash(det.detection_id)
            subj = self._subjects.get(tid)
            if subj is None:
                continue
            if subj.recent_speed > self.run_speed_threshold:
                events.append(
                    SecurityEvent(
                        event_type="anomaly",
                        anomaly_type=AnomalyType.RUNNING,
                        severity=Severity.WARNING,
                        description=(
                            f"Subject {tid} running at {subj.recent_speed:.0f} px/s"
                        ),
                        detections=[det],
                    )
                )
        return events

    def _check_crowd(
        self, detections: list[Detection], ts: float
    ) -> list[SecurityEvent]:
        if len(detections) < self.crowd_count_threshold:
            return []

        centers = np.array([d.bbox.center for d in detections])
        events: list[SecurityEvent] = []
        visited: set[int] = set()

        for i, c in enumerate(centers):
            if i in visited:
                continue
            dists = np.linalg.norm(centers - c, axis=1)
            cluster_mask = dists < self.crowd_radius
            cluster_indices = list(np.where(cluster_mask)[0])

            if len(cluster_indices) >= self.crowd_count_threshold:
                visited.update(cluster_indices)
                cluster_dets = [detections[j] for j in cluster_indices]
                events.append(
                    SecurityEvent(
                        event_type="anomaly",
                        anomaly_type=AnomalyType.CROWD_FORMING,
                        severity=Severity.CRITICAL,
                        description=(
                            f"Crowd of {len(cluster_indices)} detected near "
                            f"({c[0]:.0f}, {c[1]:.0f})"
                        ),
                        detections=cluster_dets,
                    )
                )
        return events

    def _check_perimeter(
        self, detections: list[Detection], ts: float
    ) -> list[SecurityEvent]:
        events: list[SecurityEvent] = []
        for det in detections:
            cx, cy = det.bbox.center
            for zone in self._restricted_zones:
                if (
                    zone.x <= cx <= zone.x + zone.width
                    and zone.y <= cy <= zone.y + zone.height
                ):
                    events.append(
                        SecurityEvent(
                            event_type="anomaly",
                            anomaly_type=AnomalyType.PERIMETER_BREACH,
                            severity=Severity.EMERGENCY,
                            description=(
                                f"Perimeter breach by track {det.track_id} "
                                f"at ({cx:.0f}, {cy:.0f})"
                            ),
                            detections=[det],
                        )
                    )
                    break  # one event per detection
        return events

    def reset(self) -> None:
        """Clear all tracked subject histories."""
        self._subjects.clear()
