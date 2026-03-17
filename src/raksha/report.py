"""Report generation from security events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.text import Text

from raksha.analyzer.stats import SecurityStats
from raksha.analyzer.timeline import EventTimeline
from raksha.models import SecurityEvent, Severity


@dataclass
class SecurityReport:
    """Generates human-readable security reports from recorded events."""

    timeline: EventTimeline
    console: Console = field(default_factory=Console, repr=False)

    def generate(
        self,
        title: str = "RAKSHA Security Report",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> str:
        """Generate a text report and return it as a string.

        Also prints a rich-formatted version to the console.
        """
        events = self.timeline.query(start=start, end=end)
        stats = SecurityStats()
        stats.ingest(events)
        summary = stats.summary()

        lines: list[str] = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  {title}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if start:
            lines.append(f"  From:      {start.strftime('%Y-%m-%d %H:%M:%S')}")
        if end:
            lines.append(f"  To:        {end.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Total events: {summary['total_events']}")
        lines.append("")

        # Severity breakdown
        lines.append("  Severity Distribution:")
        for sev, cnt in summary["severity_distribution"].items():
            lines.append(f"    {sev.upper():12s} {cnt}")
        lines.append("")

        # Anomaly breakdown
        if summary["anomaly_distribution"]:
            lines.append("  Anomaly Types:")
            for anom, cnt in summary["anomaly_distribution"].items():
                lines.append(f"    {anom:20s} {cnt}")
            lines.append("")

        # Peak hour
        if summary["peak_hour"] is not None:
            lines.append(f"  Peak activity hour: {summary['peak_hour']:02d}:00")
        lines.append(f"{'=' * 60}")

        report_text = "\n".join(lines)

        # Rich console output
        self._print_rich(events, summary, title)

        return report_text

    def _print_rich(
        self, events: list[SecurityEvent], summary: dict, title: str
    ) -> None:
        self.console.print(f"\n[bold]{title}[/bold]\n")

        table = Table(title="Event Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Total Events", str(summary["total_events"]))
        for sev, cnt in summary["severity_distribution"].items():
            table.add_row(f"  {sev}", str(cnt))
        if summary["peak_hour"] is not None:
            table.add_row("Peak Hour", f"{summary['peak_hour']:02d}:00")
        self.console.print(table)

        if events:
            recent_table = Table(title="Recent Events (last 10)")
            recent_table.add_column("Time", style="dim")
            recent_table.add_column("Type")
            recent_table.add_column("Severity")
            recent_table.add_column("Description")

            color_map = {
                Severity.INFO: "blue",
                Severity.WARNING: "yellow",
                Severity.CRITICAL: "red",
                Severity.EMERGENCY: "bold red",
            }

            for evt in events[-10:]:
                color = color_map.get(evt.severity, "white")
                recent_table.add_row(
                    evt.timestamp.strftime("%H:%M:%S"),
                    evt.event_type,
                    f"[{color}]{evt.severity.value}[/{color}]",
                    evt.description[:60],
                )
            self.console.print(recent_table)
