"""Zone management for restricted and monitored areas."""

from __future__ import annotations

from dataclasses import dataclass, field

from raksha.models import BoundingBox, Detection, Severity, ZoneType


@dataclass
class Zone:
    """A named security zone."""

    name: str
    zone_type: ZoneType
    bounds: BoundingBox
    severity: Severity = Severity.WARNING
    max_occupancy: int | None = None

    def contains_point(self, x: float, y: float) -> bool:
        return (
            self.bounds.x <= x <= self.bounds.x + self.bounds.width
            and self.bounds.y <= y <= self.bounds.y + self.bounds.height
        )

    def contains_detection(self, detection: Detection) -> bool:
        cx, cy = detection.bbox.center
        return self.contains_point(cx, cy)


@dataclass
class ZoneViolation:
    """Record of a zone rule being violated."""

    zone: Zone
    detection: Detection
    violation_type: str
    severity: Severity


@dataclass
class ZoneManager:
    """Manages security zones and checks detections against zone rules."""

    _zones: list[Zone] = field(default_factory=list)

    def add_zone(
        self,
        name: str,
        zone_type: ZoneType,
        bounds: BoundingBox,
        severity: Severity = Severity.WARNING,
        max_occupancy: int | None = None,
    ) -> Zone:
        """Register a new security zone."""
        zone = Zone(
            name=name,
            zone_type=zone_type,
            bounds=bounds,
            severity=severity,
            max_occupancy=max_occupancy,
        )
        self._zones.append(zone)
        return zone

    def remove_zone(self, name: str) -> bool:
        """Remove a zone by name.  Returns True if found."""
        before = len(self._zones)
        self._zones = [z for z in self._zones if z.name != name]
        return len(self._zones) < before

    def check_detections(self, detections: list[Detection]) -> list[ZoneViolation]:
        """Check all detections against zone rules and return violations."""
        violations: list[ZoneViolation] = []

        for zone in self._zones:
            inside = [d for d in detections if zone.contains_detection(d)]

            if zone.zone_type == ZoneType.RESTRICTED:
                for det in inside:
                    violations.append(
                        ZoneViolation(
                            zone=zone,
                            detection=det,
                            violation_type="restricted_entry",
                            severity=zone.severity,
                        )
                    )

            if zone.max_occupancy is not None and len(inside) > zone.max_occupancy:
                for det in inside:
                    violations.append(
                        ZoneViolation(
                            zone=zone,
                            detection=det,
                            violation_type="overcrowding",
                            severity=Severity.CRITICAL,
                        )
                    )

        return violations

    @property
    def zones(self) -> list[Zone]:
        return list(self._zones)

    def get_restricted_bounds(self) -> list[BoundingBox]:
        """Return bounding boxes for all restricted zones."""
        return [z.bounds for z in self._zones if z.zone_type == ZoneType.RESTRICTED]

    def reset(self) -> None:
        self._zones.clear()
