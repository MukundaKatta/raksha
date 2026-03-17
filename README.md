# RAKSHA

AI-powered security camera system with intelligent threat detection, multi-object tracking, and behavioral anomaly analysis.

## Features

- **Motion Detection** -- Adaptive frame differencing with configurable sensitivity thresholds
- **Person Detection** -- CNN-based human detection with confidence scoring
- **Anomaly Detection** -- Identifies suspicious behaviors: loitering, running, crowd forming, perimeter breach
- **Multi-Object Tracking** -- Kalman filter-based tracker maintaining object identities across frames
- **Zone Management** -- Define restricted and monitored areas with custom alert rules
- **Alert System** -- Severity-based notifications (INFO, WARNING, CRITICAL, EMERGENCY)
- **Analytics** -- Event timelines, activity heatmaps, and hourly/daily/weekly pattern statistics
- **Simulation** -- Built-in scene simulator for testing without a live camera feed
- **Reporting** -- Generate security reports from recorded events

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Run the simulator with default settings
raksha simulate --duration 60

# Monitor a specific zone configuration
raksha monitor --zones config/zones.json

# Generate a report from recorded events
raksha report --period daily

# Show activity heatmap
raksha heatmap --output heatmap.png
```

## Architecture

```
src/raksha/
  models.py         -- Pydantic data models (Frame, Detection, SecurityEvent, Alert)
  simulator.py      -- Scene simulator generating synthetic security footage
  report.py         -- Report generation from security events
  cli.py            -- Click-based CLI entry point
  detection/
    motion.py       -- MotionDetector with adaptive thresholds
    person.py       -- PersonDetector CNN model
    anomaly.py      -- AnomalyDetector for behavioral analysis
  tracking/
    tracker.py      -- ObjectTracker with Kalman filter
    zones.py        -- ZoneManager for restricted/monitored areas
    alerts.py       -- AlertSystem with severity levels
  analyzer/
    timeline.py     -- EventTimeline for security event recording
    heatmap.py      -- ActivityHeatmap for traffic visualization
    stats.py        -- SecurityStats with temporal patterns
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
