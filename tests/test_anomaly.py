"""Tests for anomaly detection."""

from raksha.detection.anomaly import AnomalyDetector
from raksha.models import AnomalyType, BoundingBox, Detection


class TestAnomalyDetector:
    def _make_detection(
        self, track_id: int, x: float, y: float, frame_id: int = 0
    ) -> Detection:
        return Detection(
            frame_id=frame_id,
            bbox=BoundingBox(x=x, y=y, width=40, height=80),
            confidence=0.9,
            track_id=track_id,
        )

    def test_no_anomaly_initially(self):
        ad = AnomalyDetector()
        dets = [self._make_detection(1, 100, 100)]
        events = ad.analyze(dets, current_time=0.0)
        assert events == []

    def test_loitering_detected(self):
        ad = AnomalyDetector(loiter_time_threshold=5.0, loiter_displacement_threshold=20.0)
        # Subject stays in roughly the same spot for 10 seconds
        for t in range(12):
            dets = [self._make_detection(1, 100 + t * 0.5, 100)]
            events = ad.analyze(dets, current_time=float(t))
        # By t=10 the subject has been loitering for > 5s
        loiter_events = [
            e for e in events if e.anomaly_type == AnomalyType.LOITERING
        ]
        assert len(loiter_events) > 0

    def test_running_detected(self):
        ad = AnomalyDetector(run_speed_threshold=50.0)
        # Subject moves fast
        for t in range(5):
            dets = [self._make_detection(1, 100 + t * 100, 100)]
            events = ad.analyze(dets, current_time=float(t))
        running_events = [
            e for e in events if e.anomaly_type == AnomalyType.RUNNING
        ]
        assert len(running_events) > 0

    def test_crowd_forming(self):
        ad = AnomalyDetector(crowd_count_threshold=3, crowd_radius=100.0)
        dets = [
            self._make_detection(i, 100 + i * 10, 100) for i in range(5)
        ]
        events = ad.analyze(dets, current_time=0.0)
        crowd_events = [
            e for e in events if e.anomaly_type == AnomalyType.CROWD_FORMING
        ]
        assert len(crowd_events) > 0

    def test_perimeter_breach(self):
        ad = AnomalyDetector()
        restricted = BoundingBox(x=50, y=50, width=100, height=100)
        ad.set_restricted_zones([restricted])
        # Detection inside restricted zone
        dets = [self._make_detection(1, 70, 70)]
        events = ad.analyze(dets, current_time=0.0)
        breach_events = [
            e for e in events if e.anomaly_type == AnomalyType.PERIMETER_BREACH
        ]
        assert len(breach_events) > 0

    def test_reset(self):
        ad = AnomalyDetector()
        dets = [self._make_detection(1, 100, 100)]
        ad.analyze(dets, current_time=0.0)
        ad.reset()
        assert len(ad._subjects) == 0
