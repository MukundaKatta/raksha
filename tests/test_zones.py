"""Tests for zone management."""

from raksha.models import BoundingBox, Detection, Severity, ZoneType
from raksha.tracking.zones import ZoneManager


class TestZoneManager:
    def _make_detection(self, x: float, y: float) -> Detection:
        return Detection(
            frame_id=0,
            bbox=BoundingBox(x=x, y=y, width=20, height=40),
            confidence=0.9,
        )

    def test_add_and_list_zones(self):
        zm = ZoneManager()
        zm.add_zone("lobby", ZoneType.MONITORED, BoundingBox(x=0, y=0, width=200, height=200))
        zm.add_zone("vault", ZoneType.RESTRICTED, BoundingBox(x=300, y=300, width=100, height=100))
        assert len(zm.zones) == 2

    def test_restricted_entry_violation(self):
        zm = ZoneManager()
        zm.add_zone(
            "vault",
            ZoneType.RESTRICTED,
            BoundingBox(x=100, y=100, width=100, height=100),
            severity=Severity.EMERGENCY,
        )
        # Detection center is at (120, 130) which is inside the vault
        det = self._make_detection(110, 110)
        violations = zm.check_detections([det])
        assert len(violations) == 1
        assert violations[0].violation_type == "restricted_entry"
        assert violations[0].severity == Severity.EMERGENCY

    def test_no_violation_outside_zone(self):
        zm = ZoneManager()
        zm.add_zone("vault", ZoneType.RESTRICTED, BoundingBox(x=100, y=100, width=50, height=50))
        det = self._make_detection(0, 0)  # center at (10, 20), outside vault
        violations = zm.check_detections([det])
        assert violations == []

    def test_overcrowding(self):
        zm = ZoneManager()
        zm.add_zone(
            "corridor",
            ZoneType.MONITORED,
            BoundingBox(x=0, y=0, width=500, height=500),
            max_occupancy=2,
        )
        dets = [self._make_detection(10 + i * 30, 10) for i in range(4)]
        violations = zm.check_detections(dets)
        overcrowd = [v for v in violations if v.violation_type == "overcrowding"]
        assert len(overcrowd) > 0

    def test_remove_zone(self):
        zm = ZoneManager()
        zm.add_zone("temp", ZoneType.MONITORED, BoundingBox(x=0, y=0, width=10, height=10))
        assert zm.remove_zone("temp") is True
        assert len(zm.zones) == 0
        assert zm.remove_zone("nonexistent") is False

    def test_get_restricted_bounds(self):
        zm = ZoneManager()
        zm.add_zone("a", ZoneType.RESTRICTED, BoundingBox(x=0, y=0, width=10, height=10))
        zm.add_zone("b", ZoneType.MONITORED, BoundingBox(x=20, y=20, width=10, height=10))
        restricted = zm.get_restricted_bounds()
        assert len(restricted) == 1
