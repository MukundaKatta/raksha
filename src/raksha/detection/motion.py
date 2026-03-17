"""Motion detection using frame differencing with adaptive thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter, label

from raksha.models import BoundingBox, Detection


@dataclass
class MotionDetector:
    """Detects motion between consecutive frames using adaptive frame differencing.

    The detector maintains a running background model and adapts its threshold
    based on recent activity levels.  When the difference between the current
    frame and the background exceeds the threshold, connected regions are
    extracted and returned as detections.
    """

    base_threshold: float = 25.0
    min_area: int = 500
    learning_rate: float = 0.05
    gaussian_sigma: float = 1.5
    adaptive_window: int = 50

    # Internal state
    _background: np.ndarray | None = field(default=None, repr=False)
    _threshold: float = field(default=0.0, init=False, repr=False)
    _recent_magnitudes: list[float] = field(default_factory=list, repr=False)
    _frame_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._threshold = self.base_threshold

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to single-channel grayscale float."""
        if frame.ndim == 3:
            # Weighted luminance conversion
            return np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(
                np.float32
            )
        return frame.astype(np.float32)

    def _update_threshold(self, magnitude: float) -> None:
        """Adapt the detection threshold using recent motion magnitudes."""
        self._recent_magnitudes.append(magnitude)
        if len(self._recent_magnitudes) > self.adaptive_window:
            self._recent_magnitudes.pop(0)

        if len(self._recent_magnitudes) >= 5:
            mean_mag = np.mean(self._recent_magnitudes)
            std_mag = np.std(self._recent_magnitudes)
            self._threshold = max(
                self.base_threshold * 0.5,
                min(self.base_threshold * 2.0, mean_mag + 2.0 * std_mag),
            )

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> list[Detection]:
        """Detect motion regions in the given frame.

        Args:
            frame: Input image as a numpy array (H, W) or (H, W, C).
            frame_id: Identifier for the current frame.

        Returns:
            List of Detection objects for each motion region found.
        """
        gray = self._to_gray(frame)
        gray = gaussian_filter(gray, sigma=self.gaussian_sigma)
        self._frame_count += 1

        if self._background is None:
            self._background = gray.copy()
            return []

        # Frame differencing
        diff = np.abs(gray - self._background)
        magnitude = float(np.mean(diff))
        self._update_threshold(magnitude)

        # Binary motion mask
        motion_mask = (diff > self._threshold).astype(np.uint8)

        # Connected component analysis
        labelled, num_features = label(motion_mask)
        detections: list[Detection] = []

        for i in range(1, num_features + 1):
            component = labelled == i
            coords = np.argwhere(component)

            if len(coords) < self.min_area:
                continue

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            w = int(x_max - x_min + 1)
            h = int(y_max - y_min + 1)

            # Confidence based on how strongly the region differs
            region_diff = diff[component]
            confidence = float(
                np.clip(np.mean(region_diff) / (self._threshold * 3), 0.0, 1.0)
            )

            detections.append(
                Detection(
                    frame_id=frame_id,
                    bbox=BoundingBox(x=float(x_min), y=float(y_min), width=float(w), height=float(h)),
                    label="motion",
                    confidence=confidence,
                )
            )

        # Update background model
        self._background = (
            (1 - self.learning_rate) * self._background + self.learning_rate * gray
        )

        return detections

    def reset(self) -> None:
        """Reset the detector state."""
        self._background = None
        self._threshold = self.base_threshold
        self._recent_magnitudes.clear()
        self._frame_count = 0
