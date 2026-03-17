"""Tests for the alert system."""

from datetime import datetime

from raksha.models import AnomalyType, SecurityEvent, Severity
from raksha.tracking.alerts import AlertSystem


class TestAlertSystem:
    def test_creates_alert_for_sufficient_severity(self):
        sys = AlertSystem(min_severity=Severity.WARNING)
        event = SecurityEvent(
            event_type="anomaly",
            severity=Severity.CRITICAL,
            description="Test critical event",
        )
        alert = sys.process_event(event)
        assert alert is not None
        assert alert.severity == Severity.CRITICAL

    def test_ignores_low_severity(self):
        sys = AlertSystem(min_severity=Severity.WARNING)
        event = SecurityEvent(
            event_type="motion",
            severity=Severity.INFO,
            description="Minor motion",
        )
        alert = sys.process_event(event)
        assert alert is None

    def test_cooldown(self):
        sys = AlertSystem(min_severity=Severity.WARNING, cooldown_seconds=60)
        ts = datetime(2025, 1, 1, 12, 0, 0)
        e1 = SecurityEvent(
            event_type="anomaly",
            anomaly_type=AnomalyType.LOITERING,
            severity=Severity.WARNING,
            timestamp=ts,
            description="first",
        )
        e2 = SecurityEvent(
            event_type="anomaly",
            anomaly_type=AnomalyType.LOITERING,
            severity=Severity.WARNING,
            timestamp=ts,  # same timestamp
            description="second",
        )
        a1 = sys.process_event(e1)
        a2 = sys.process_event(e2)
        assert a1 is not None
        assert a2 is None  # suppressed by cooldown

    def test_acknowledge(self):
        sys = AlertSystem(min_severity=Severity.INFO)
        event = SecurityEvent(event_type="test", severity=Severity.WARNING, description="t")
        alert = sys.process_event(event)
        assert alert is not None
        assert len(sys.unacknowledged) == 1
        sys.acknowledge(alert.alert_id)
        assert len(sys.unacknowledged) == 0

    def test_callback(self):
        captured: list = []
        sys = AlertSystem(min_severity=Severity.WARNING)
        sys.register_callback(Severity.WARNING, lambda a: captured.append(a))
        event = SecurityEvent(event_type="test", severity=Severity.WARNING, description="cb")
        sys.process_event(event)
        assert len(captured) == 1
