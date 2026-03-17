"""Tests for pydantic data models."""

from datetime import datetime

from raksha.models import (
    Alert,
    AnomalyType,
    BoundingBox,
    Detection,
    Frame,
    SecurityEvent,
    Severity,
)


class TestBoundingBox:
    def test_center(self):
        box = BoundingBox(x=10, y=20, width=100, height=50)
        assert box.center == (60.0, 45.0)

    def test_area(self):
        box = BoundingBox(x=0, y=0, width=10, height=20)
        assert box.area == 200.0

    def test_iou_identical(self):
        box = BoundingBox(x=0, y=0, width=10, height=10)
        assert box.iou(box) == 1.0

    def test_iou_no_overlap(self):
        a = BoundingBox(x=0, y=0, width=10, height=10)
        b = BoundingBox(x=20, y=20, width=10, height=10)
        assert a.iou(b) == 0.0

    def test_iou_partial_overlap(self):
        a = BoundingBox(x=0, y=0, width=10, height=10)
        b = BoundingBox(x=5, y=5, width=10, height=10)
        iou = a.iou(b)
        assert 0.0 < iou < 1.0


class TestFrame:
    def test_creation(self):
        f = Frame(frame_id=0, width=640, height=480)
        assert f.shape == (480, 640)
        assert f.data is None


class TestDetection:
    def test_creation(self):
        d = Detection(
            frame_id=1,
            bbox=BoundingBox(x=10, y=10, width=50, height=100),
            confidence=0.9,
        )
        assert d.label == "person"
        assert d.track_id is None
        assert len(d.detection_id) == 12


class TestSecurityEvent:
    def test_defaults(self):
        e = SecurityEvent(event_type="motion")
        assert e.severity == Severity.INFO
        assert e.anomaly_type is None
        assert e.detections == []

    def test_with_anomaly(self):
        e = SecurityEvent(
            event_type="anomaly",
            anomaly_type=AnomalyType.LOITERING,
            severity=Severity.WARNING,
        )
        assert e.anomaly_type == AnomalyType.LOITERING


class TestAlert:
    def test_defaults(self):
        a = Alert(severity=Severity.CRITICAL, title="Test", description="A test alert")
        assert not a.acknowledged
        assert not a.notified
