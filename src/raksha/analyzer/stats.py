"""Security statistics with hourly, daily, and weekly patterns."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from raksha.models import AnomalyType, SecurityEvent, Severity


@dataclass
class SecurityStats:
    """Computes aggregate statistics from security events.

    Provides hourly, daily, and weekly breakdowns of event counts,
    severity distributions, and anomaly type frequencies.
    """

    _events: list[SecurityEvent] = field(default_factory=list, repr=False)

    def ingest(self, events: list[SecurityEvent]) -> None:
        """Add events to the stats engine."""
        self._events.extend(events)

    def hourly_counts(self) -> dict[int, int]:
        """Event counts bucketed by hour of day (0-23)."""
        counter: dict[int, int] = defaultdict(int)
        for e in self._events:
            counter[e.timestamp.hour] += 1
        return dict(sorted(counter.items()))

    def daily_counts(self) -> dict[str, int]:
        """Event counts bucketed by calendar date (YYYY-MM-DD)."""
        counter: dict[str, int] = defaultdict(int)
        for e in self._events:
            counter[e.timestamp.strftime("%Y-%m-%d")] += 1
        return dict(sorted(counter.items()))

    def weekly_counts(self) -> dict[str, int]:
        """Event counts bucketed by day of week (Monday-Sunday)."""
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        counter: dict[str, int] = {d: 0 for d in days}
        for e in self._events:
            counter[days[e.timestamp.weekday()]] += 1
        return counter

    def severity_distribution(self) -> dict[str, int]:
        """Count of events per severity level."""
        counter: Counter[str] = Counter()
        for e in self._events:
            counter[e.severity.value] += 1
        return dict(counter)

    def anomaly_distribution(self) -> dict[str, int]:
        """Count of events per anomaly type."""
        counter: Counter[str] = Counter()
        for e in self._events:
            if e.anomaly_type is not None:
                counter[e.anomaly_type.value] += 1
        return dict(counter)

    def peak_hour(self) -> int | None:
        """Return the hour with the most events, or None if empty."""
        hc = self.hourly_counts()
        if not hc:
            return None
        return max(hc, key=hc.get)  # type: ignore[arg-type]

    def total_events(self) -> int:
        return len(self._events)

    def summary(self) -> dict:
        """Return a summary dict suitable for reporting."""
        return {
            "total_events": self.total_events(),
            "severity_distribution": self.severity_distribution(),
            "anomaly_distribution": self.anomaly_distribution(),
            "peak_hour": self.peak_hour(),
            "hourly_counts": self.hourly_counts(),
            "weekly_counts": self.weekly_counts(),
        }

    def reset(self) -> None:
        self._events.clear()
