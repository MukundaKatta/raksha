"""Scene simulator for testing the RAKSHA pipeline without a live camera."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from raksha.models import BoundingBox, Detection, Frame


@dataclass
class _SimulatedPerson:
    """A person moving through the simulated scene."""

    person_id: int
    x: float
    y: float
    vx: float
    vy: float
    width: float = 40.0
    height: float = 80.0

    def step(self, dt: float = 1.0, noise: float = 2.0) -> None:
        self.x += self.vx * dt + random.gauss(0, noise)
        self.y += self.vy * dt + random.gauss(0, noise)


@dataclass
class SceneSimulator:
    """Generates synthetic frames with moving persons.

    The simulator creates a configurable number of people that move across
    a virtual scene.  It produces both raw frames (numpy arrays with
    simple rectangular blobs) and ground-truth detections.
    """

    width: int = 640
    height: int = 480
    num_persons: int = 3
    fps: float = 10.0
    seed: int | None = None

    _persons: list[_SimulatedPerson] = field(default_factory=list, repr=False)
    _frame_count: int = field(default=0, init=False)
    _start_time: datetime = field(default_factory=datetime.now, repr=False)
    _rng: random.Random = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._spawn_persons()

    def _spawn_persons(self) -> None:
        self._persons.clear()
        for i in range(self.num_persons):
            self._persons.append(
                _SimulatedPerson(
                    person_id=i + 1,
                    x=self._rng.uniform(50, self.width - 50),
                    y=self._rng.uniform(50, self.height - 50),
                    vx=self._rng.uniform(-5, 5),
                    vy=self._rng.uniform(-3, 3),
                )
            )

    def step(self) -> tuple[Frame, list[Detection]]:
        """Advance the simulation by one frame.

        Returns:
            A tuple of (Frame, ground-truth detections).
        """
        dt = 1.0 / self.fps
        self._frame_count += 1
        timestamp = self._start_time + timedelta(seconds=self._frame_count * dt)

        # Move persons
        for p in self._persons:
            p.step(dt)
            # Bounce off edges
            if p.x < 0 or p.x + p.width > self.width:
                p.vx *= -1
                p.x = np.clip(p.x, 0, self.width - p.width)
            if p.y < 0 or p.y + p.height > self.height:
                p.vy *= -1
                p.y = np.clip(p.y, 0, self.height - p.height)

        # Render frame
        frame_data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Add slight background noise
        frame_data += self._rng.randint(5, 15)

        detections: list[Detection] = []
        for p in self._persons:
            x0 = max(0, int(p.x))
            y0 = max(0, int(p.y))
            x1 = min(self.width, int(p.x + p.width))
            y1 = min(self.height, int(p.y + p.height))
            # Draw a bright rectangle for the person
            color = [
                150 + self._rng.randint(0, 50),
                100 + self._rng.randint(0, 50),
                80 + self._rng.randint(0, 50),
            ]
            frame_data[y0:y1, x0:x1] = color

            detections.append(
                Detection(
                    frame_id=self._frame_count,
                    timestamp=timestamp,
                    bbox=BoundingBox(
                        x=float(x0),
                        y=float(y0),
                        width=float(x1 - x0),
                        height=float(y1 - y0),
                    ),
                    label="person",
                    confidence=0.95 + self._rng.uniform(-0.05, 0.05),
                )
            )

        frame = Frame(
            frame_id=self._frame_count,
            timestamp=timestamp,
            width=self.width,
            height=self.height,
            data=frame_data,
        )

        return frame, detections

    def run(self, num_frames: int) -> list[tuple[Frame, list[Detection]]]:
        """Run the simulation for *num_frames* frames."""
        return [self.step() for _ in range(num_frames)]

    def reset(self) -> None:
        self._frame_count = 0
        self._start_time = datetime.now()
        self._rng = random.Random(self.seed)
        self._spawn_persons()
