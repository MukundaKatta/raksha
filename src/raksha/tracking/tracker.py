"""Multi-object tracking using Kalman filters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from raksha.models import BoundingBox, Detection


@dataclass
class _KalmanTrack:
    """Single-object Kalman filter track.

    State vector: [x, y, w, h, vx, vy, vw, vh]
    """

    track_id: int
    age: int = 0
    hits: int = 0
    misses: int = 0

    _state: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _P: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    # Constant-velocity transition matrix (8x8)
    _F: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _H: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _Q: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _R: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def init_state(self, bbox: BoundingBox) -> None:
        cx, cy = bbox.center
        self._state = np.array(
            [cx, cy, bbox.width, bbox.height, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self._P = np.eye(8) * 100.0
        self._P[4:, 4:] *= 10.0  # higher uncertainty for velocities

        self._F = np.eye(8)
        for i in range(4):
            self._F[i, i + 4] = 1.0  # dt = 1 frame

        self._H = np.eye(4, 8)

        self._Q = np.eye(8) * 1.0
        self._Q[4:, 4:] *= 4.0

        self._R = np.eye(4) * 10.0

    def predict(self) -> BoundingBox:
        self._state = self._F @ self._state
        self._P = self._F @ self._P @ self._F.T + self._Q
        self.age += 1
        return self._state_to_bbox()

    def update(self, bbox: BoundingBox) -> BoundingBox:
        cx, cy = bbox.center
        z = np.array([cx, cy, bbox.width, bbox.height])
        y = z - self._H @ self._state
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._state = self._state + K @ y
        self._P = (np.eye(8) - K @ self._H) @ self._P
        self.hits += 1
        self.misses = 0
        return self._state_to_bbox()

    def _state_to_bbox(self) -> BoundingBox:
        cx, cy, w, h = self._state[:4]
        w = max(1.0, w)
        h = max(1.0, h)
        return BoundingBox(x=cx - w / 2, y=cy - h / 2, width=w, height=h)

    @property
    def bbox(self) -> BoundingBox:
        return self._state_to_bbox()


@dataclass
class ObjectTracker:
    """Multi-object tracker using Kalman filters and Hungarian assignment.

    Each detection is associated with an existing track or spawns a new one.
    Tracks that are not matched for ``max_misses`` consecutive frames are
    removed.
    """

    max_misses: int = 5
    iou_threshold: float = 0.25
    min_hits: int = 2

    _tracks: list[_KalmanTrack] = field(default_factory=list, repr=False)
    _next_id: int = field(default=1, init=False, repr=False)

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Match detections to existing tracks and return updated detections.

        Each returned detection has its ``track_id`` set.
        """
        # Predict step for all tracks
        for track in self._tracks:
            track.predict()

        if not self._tracks or not detections:
            # No tracks to match -- start new tracks from detections
            for det in detections:
                self._create_track(det)
            return detections

        # Build cost matrix (negative IoU so we can minimise)
        cost = np.zeros((len(self._tracks), len(detections)))
        for t, track in enumerate(self._tracks):
            for d, det in enumerate(detections):
                cost[t, d] = 1.0 - track.bbox.iou(det.bbox)

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        for t, d in zip(row_ind, col_ind):
            if cost[t, d] <= 1.0 - self.iou_threshold:
                self._tracks[t].update(detections[d].bbox)
                detections[d].track_id = self._tracks[t].track_id
                matched_tracks.add(t)
                matched_dets.add(d)

        # Handle unmatched tracks
        for t in range(len(self._tracks)):
            if t not in matched_tracks:
                self._tracks[t].misses += 1

        # Create new tracks for unmatched detections
        for d in range(len(detections)):
            if d not in matched_dets:
                self._create_track(detections[d])

        # Remove dead tracks
        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]

        return detections

    def _create_track(self, detection: Detection) -> None:
        track = _KalmanTrack(track_id=self._next_id)
        track.init_state(detection.bbox)
        track.hits = 1
        self._tracks.append(track)
        detection.track_id = self._next_id
        self._next_id += 1

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        """Remove all tracks and reset ID counter."""
        self._tracks.clear()
        self._next_id = 1
