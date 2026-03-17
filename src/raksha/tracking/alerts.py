"""Alert system with severity levels and notification support."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from raksha.models import Alert, SecurityEvent, Severity


@dataclass
class AlertSystem:
    """Manages alert generation, filtering, and notification.

    Alerts are created from security events when their severity meets or
    exceeds the configured minimum.  Notification callbacks can be registered
    per severity level.
    """

    min_severity: Severity = Severity.WARNING
    cooldown_seconds: float = 30.0  # suppress duplicate alerts within window
    console: Console = field(default_factory=Console, repr=False)

    _alerts: list[Alert] = field(default_factory=list, repr=False)
    _callbacks: dict[Severity, list[Callable[[Alert], None]]] = field(
        default_factory=lambda: {s: [] for s in Severity}, repr=False
    )
    _last_alert_key: dict[str, float] = field(default_factory=dict, repr=False)

    _SEVERITY_ORDER: dict[Severity, int] = field(
        default_factory=lambda: {
            Severity.INFO: 0,
            Severity.WARNING: 1,
            Severity.CRITICAL: 2,
            Severity.EMERGENCY: 3,
        },
        init=False,
        repr=False,
    )

    def register_callback(
        self, severity: Severity, callback: Callable[[Alert], None]
    ) -> None:
        """Register a notification callback for a given severity level."""
        self._callbacks[severity].append(callback)

    def process_event(self, event: SecurityEvent) -> Optional[Alert]:
        """Create an alert from a security event if severity is sufficient.

        Returns the alert if one was created, otherwise ``None``.
        """
        if self._SEVERITY_ORDER[event.severity] < self._SEVERITY_ORDER[self.min_severity]:
            return None

        # Cooldown check
        key = f"{event.event_type}:{event.anomaly_type}"
        now = event.timestamp.timestamp()
        last = self._last_alert_key.get(key, 0.0)
        if now - last < self.cooldown_seconds:
            return None
        self._last_alert_key[key] = now

        alert = Alert(
            severity=event.severity,
            title=self._build_title(event),
            description=event.description,
            event=event,
        )

        self._alerts.append(alert)
        self._notify(alert)
        return alert

    def display_alert(self, alert: Alert) -> None:
        """Print an alert to the console using rich formatting."""
        color_map = {
            Severity.INFO: "blue",
            Severity.WARNING: "yellow",
            Severity.CRITICAL: "red",
            Severity.EMERGENCY: "bold red",
        }
        color = color_map.get(alert.severity, "white")
        title = Text(f"[{alert.severity.value.upper()}] {alert.title}")
        self.console.print(
            Panel(
                alert.description,
                title=str(title),
                border_style=color,
            )
        )

    def acknowledge(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged.  Returns True if found."""
        for a in self._alerts:
            if a.alert_id == alert_id:
                a.acknowledged = True
                return True
        return False

    @property
    def unacknowledged(self) -> list[Alert]:
        return [a for a in self._alerts if not a.acknowledged]

    @property
    def all_alerts(self) -> list[Alert]:
        return list(self._alerts)

    def _build_title(self, event: SecurityEvent) -> str:
        if event.anomaly_type:
            return f"{event.anomaly_type.value.replace('_', ' ').title()} Detected"
        return event.event_type.replace("_", " ").title()

    def _notify(self, alert: Alert) -> None:
        for cb in self._callbacks.get(alert.severity, []):
            cb(alert)
        alert.notified = True

    def reset(self) -> None:
        self._alerts.clear()
        self._last_alert_key.clear()
