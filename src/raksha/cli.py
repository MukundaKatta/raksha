"""CLI entry point for the RAKSHA security system."""

from __future__ import annotations

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table

from raksha import __version__

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="raksha")
def cli() -> None:
    """RAKSHA -- AI Security Camera with threat detection."""


@cli.command()
@click.option("--duration", default=30, help="Simulation duration in frames.")
@click.option("--persons", default=3, help="Number of simulated persons.")
@click.option("--width", default=640, help="Frame width.")
@click.option("--height", default=480, help="Frame height.")
@click.option("--seed", default=42, help="Random seed for reproducibility.")
def simulate(duration: int, persons: int, width: int, height: int, seed: int) -> None:
    """Run the scene simulator and process detections through the pipeline."""
    from raksha.analyzer.heatmap import ActivityHeatmap
    from raksha.analyzer.stats import SecurityStats
    from raksha.analyzer.timeline import EventTimeline
    from raksha.detection.anomaly import AnomalyDetector
    from raksha.detection.motion import MotionDetector
    from raksha.report import SecurityReport
    from raksha.simulator import SceneSimulator
    from raksha.tracking.alerts import AlertSystem
    from raksha.tracking.tracker import ObjectTracker

    console.print(f"[bold green]RAKSHA Simulator[/bold green] v{__version__}")
    console.print(f"  Frames: {duration}, Persons: {persons}, Seed: {seed}\n")

    sim = SceneSimulator(
        width=width, height=height, num_persons=persons, seed=seed
    )
    motion_det = MotionDetector()
    tracker = ObjectTracker()
    anomaly_det = AnomalyDetector()
    alert_sys = AlertSystem()
    timeline = EventTimeline()
    heatmap = ActivityHeatmap(width=width, height=height)
    stats = SecurityStats()

    alert_sys.register_callback(
        alert_sys.min_severity,
        lambda a: console.print(
            f"  [yellow]ALERT[/yellow] [{a.severity.value}] {a.title}: {a.description}"
        ),
    )

    for i in range(duration):
        frame, gt_detections = sim.step()

        # Motion detection on raw frame
        if frame.data is not None:
            motion_dets = motion_det.detect(frame.data, frame_id=frame.frame_id)
        else:
            motion_dets = gt_detections

        # Track objects
        tracked = tracker.update(gt_detections)

        # Anomaly analysis
        events = anomaly_det.analyze(tracked)
        for evt in events:
            timeline.record(evt)
            alert_sys.process_event(evt)

        # Update analytics
        heatmap.update(tracked)
        stats.ingest(events)

        if (i + 1) % 10 == 0:
            console.print(
                f"  Frame {i + 1}/{duration} | "
                f"Tracks: {tracker.active_tracks} | "
                f"Events: {timeline.count}"
            )

    console.print(f"\n[bold]Simulation complete.[/bold]")
    console.print(f"  Total detections accumulated: {heatmap.total_detections}")
    console.print(f"  Total security events: {timeline.count}")
    console.print(f"  Unacknowledged alerts: {len(alert_sys.unacknowledged)}")

    # Print report
    report = SecurityReport(timeline=timeline, console=console)
    report.generate()


@cli.command()
@click.option("--period", type=click.Choice(["hourly", "daily", "weekly"]), default="daily")
def report(period: str) -> None:
    """Generate a security report (requires recorded events)."""
    from raksha.analyzer.timeline import EventTimeline
    from raksha.report import SecurityReport

    timeline = EventTimeline()
    console.print("[dim]No recorded events found. Run 'raksha simulate' first.[/dim]")
    rep = SecurityReport(timeline=timeline, console=console)
    rep.generate(title=f"RAKSHA {period.title()} Report")


@cli.command()
def heatmap() -> None:
    """Display activity heatmap information."""
    console.print("[dim]Heatmap requires a running simulation or recorded data.[/dim]")
    console.print("Run 'raksha simulate' to generate heatmap data.")


@cli.command()
def status() -> None:
    """Show system status."""
    table = Table(title="RAKSHA System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_row("Motion Detector", "Ready")
    table.add_row("Person Detector", "Ready")
    table.add_row("Anomaly Detector", "Ready")
    table.add_row("Object Tracker", "Ready")
    table.add_row("Alert System", "Ready")
    table.add_row("Analytics", "Ready")
    console.print(table)


if __name__ == "__main__":
    cli()
