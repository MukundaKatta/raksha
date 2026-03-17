"""Tests for multi-object tracking."""

from raksha.models import BoundingBox, Detection
from raksha.tracking.tracker import ObjectTracker


class TestObjectTracker:
    def _make_detection(self, x: float, y: float, frame_id: int = 0) -> Detection:
        return Detection(
            frame_id=frame_id,
            bbox=BoundingBox(x=x, y=y, width=40, height=80),
            confidence=0.9,
        )

    def test_creates_tracks(self):
        tracker = ObjectTracker()
        dets = [self._make_detection(100, 100), self._make_detection(300, 200)]
        result = tracker.update(dets)
        assert all(d.track_id is not None for d in result)
        assert tracker.active_tracks == 2

    def test_maintains_track_ids(self):
        tracker = ObjectTracker()
        d1 = [self._make_detection(100, 100)]
        tracker.update(d1)
        first_id = d1[0].track_id

        # Same position next frame
        d2 = [self._make_detection(102, 101, frame_id=1)]
        tracker.update(d2)
        assert d2[0].track_id == first_id

    def test_removes_stale_tracks(self):
        tracker = ObjectTracker(max_misses=2)
        dets = [self._make_detection(100, 100)]
        tracker.update(dets)

        # Send empty detections to age out the track
        for _ in range(4):
            tracker.update([])

        assert tracker.active_tracks == 0

    def test_reset(self):
        tracker = ObjectTracker()
        tracker.update([self._make_detection(100, 100)])
        tracker.reset()
        assert tracker.active_tracks == 0
