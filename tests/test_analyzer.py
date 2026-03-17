"""Tests for analyzer modules (timeline, heatmap, stats)."""

from datetime import datetime, timedelta

import numpy as np

from raksha.analyzer.heatmap import ActivityHeatmap
from raksha.analyzer.stats import SecurityStats
from raksha.analyzer.timeline import EventTimeline
from raksha.models import (
    AnomalyType,
    BoundingBox,
    Detection,
    SecurityEvent,
    Severity,
)


class TestEventTimeline:
    def test_record_and_count(self):
        tl = EventTimeline()
        tl.record(SecurityEvent(event_type="motion", description="m1"))
        tl.record(SecurityEvent(event_type="anomaly", description="a1"))
        assert tl.count == 2

    def test_query_by_severity(self):
        tl = EventTimeline()
        tl.record(SecurityEvent(event_type="a", severity=Severity.INFO, description="i"))
        tl.record(
            SecurityEvent(event_type="b", severity=Severity.CRITICAL, description="c")
        )
        result = tl.query(severity=Severity.CRITICAL)
        assert len(result) == 1
        assert result[0].severity == Severity.CRITICAL

    def test_query_by_anomaly_type(self):
        tl = EventTimeline()
        tl.record(
            SecurityEvent(
                event_type="anomaly",
                anomaly_type=AnomalyType.RUNNING,
                description="r",
            )
        )
        tl.record(SecurityEvent(event_type="motion", description="m"))
        result = tl.query(anomaly_type=AnomalyType.RUNNING)
        assert len(result) == 1

    def test_clear(self):
        tl = EventTimeline()
        tl.record(SecurityEvent(event_type="x", description="x"))
        tl.clear()
        assert tl.count == 0


class TestActivityHeatmap:
    def test_update_and_total(self):
        hm = ActivityHeatmap(width=100, height=100)
        det = Detection(
            frame_id=0,
            bbox=BoundingBox(x=40, y=40, width=20, height=20),
            confidence=0.9,
        )
        hm.update([det])
        assert hm.total_detections == 1

    def test_heatmap_shape(self):
        hm = ActivityHeatmap(width=200, height=150)
        result = hm.get_heatmap()
        assert result.shape == (150, 200)

    def test_normalized_range(self):
        hm = ActivityHeatmap(width=100, height=100, blur_sigma=5.0)
        det = Detection(
            frame_id=0,
            bbox=BoundingBox(x=50, y=50, width=10, height=10),
            confidence=0.9,
        )
        hm.update([det])
        result = hm.get_heatmap(normalize=True)
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_reset(self):
        hm = ActivityHeatmap(width=50, height=50)
        det = Detection(
            frame_id=0,
            bbox=BoundingBox(x=10, y=10, width=10, height=10),
            confidence=0.9,
        )
        hm.update([det])
        hm.reset()
        assert hm.total_detections == 0


class TestSecurityStats:
    def _make_events(self) -> list[SecurityEvent]:
        base = datetime(2025, 1, 6, 10, 0, 0)  # Monday
        return [
            SecurityEvent(
                event_type="anomaly",
                anomaly_type=AnomalyType.LOITERING,
                severity=Severity.WARNING,
                timestamp=base + timedelta(hours=i),
                description=f"event {i}",
            )
            for i in range(5)
        ]

    def test_hourly_counts(self):
        stats = SecurityStats()
        stats.ingest(self._make_events())
        hc = stats.hourly_counts()
        assert sum(hc.values()) == 5

    def test_daily_counts(self):
        stats = SecurityStats()
        stats.ingest(self._make_events())
        dc = stats.daily_counts()
        assert sum(dc.values()) == 5

    def test_weekly_counts(self):
        stats = SecurityStats()
        stats.ingest(self._make_events())
        wc = stats.weekly_counts()
        assert wc["Monday"] == 5

    def test_severity_distribution(self):
        stats = SecurityStats()
        stats.ingest(self._make_events())
        sd = stats.severity_distribution()
        assert sd["warning"] == 5

    def test_peak_hour(self):
        stats = SecurityStats()
        stats.ingest(self._make_events())
        assert stats.peak_hour() is not None

    def test_summary(self):
        stats = SecurityStats()
        stats.ingest(self._make_events())
        s = stats.summary()
        assert s["total_events"] == 5
        assert "severity_distribution" in s
