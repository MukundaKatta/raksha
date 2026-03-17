"""Event timeline for recording and querying security events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from raksha.models import AnomalyType, SecurityEvent, Severity


@dataclass
class EventTimeline:
    """Chronological record of all security events.

    Provides filtering by time range, severity, and anomaly type.
    """

    _events: list[SecurityEvent] = field(default_factory=list, repr=False)

    def record(self, event: SecurityEvent) -> None:
        """Add an event to the timeline."""
        self._events.append(event)
        self._events.sort(key=lambda e: e.timestamp)

    def record_many(self, events: list[SecurityEvent]) -> None:
        for e in events:
            self._events.append(e)
        self._events.sort(key=lambda e: e.timestamp)

    def query(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        severity: Optional[Severity] = None,
        anomaly_type: Optional[AnomalyType] = None,
        event_type: Optional[str] = None,
        zone_name: Optional[str] = None,
    ) -> list[SecurityEvent]:
        """Return events matching the given filters."""
        results = self._events

        if start is not None:
            results = [e for e in results if e.timestamp >= start]
        if end is not None:
            results = [e for e in results if e.timestamp <= end]
        if severity is not None:
            results = [e for e in results if e.severity == severity]
        if anomaly_type is not None:
            results = [e for e in results if e.anomaly_type == anomaly_type]
        if event_type is not None:
            results = [e for e in results if e.event_type == event_type]
        if zone_name is not None:
            results = [e for e in results if e.zone_name == zone_name]

        return results

    def recent(self, seconds: float = 300) -> list[SecurityEvent]:
        """Return events from the last *seconds* seconds."""
        cutoff = datetime.now() - timedelta(seconds=seconds)
        return [e for e in self._events if e.timestamp >= cutoff]

    @property
    def count(self) -> int:
        return len(self._events)

    @property
    def all_events(self) -> list[SecurityEvent]:
        return list(self._events)

    def clear(self) -> None:
        self._events.clear()
