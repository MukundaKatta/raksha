"""Activity heatmap showing high-traffic areas."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from raksha.models import Detection


@dataclass
class ActivityHeatmap:
    """Accumulates detection positions into a spatial heatmap.

    The heatmap is a 2-D array where each cell counts the number of
    detections whose center falls into that cell.  A Gaussian blur is
    applied when the heatmap is retrieved for visualisation.
    """

    width: int = 640
    height: int = 480
    blur_sigma: float = 15.0
    decay_rate: float = 0.0  # per-frame multiplicative decay (0 = no decay)

    _grid: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _total_detections: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._grid = np.zeros((self.height, self.width), dtype=np.float64)

    def update(self, detections: list[Detection]) -> None:
        """Add detection positions to the heatmap."""
        if self.decay_rate > 0:
            self._grid *= 1.0 - self.decay_rate

        for det in detections:
            cx, cy = det.bbox.center
            ix = int(np.clip(cx, 0, self.width - 1))
            iy = int(np.clip(cy, 0, self.height - 1))
            self._grid[iy, ix] += 1.0
            self._total_detections += 1

    def get_heatmap(self, normalize: bool = True) -> np.ndarray:
        """Return the blurred heatmap array.

        Args:
            normalize: If True, scale values to [0, 1].
        """
        blurred = gaussian_filter(self._grid, sigma=self.blur_sigma)
        if normalize and blurred.max() > 0:
            blurred = blurred / blurred.max()
        return blurred

    def get_hotspots(self, threshold: float = 0.7, top_n: int = 5) -> list[tuple[int, int, float]]:
        """Return the top-N hotspot coordinates above *threshold*.

        Returns list of (x, y, intensity) tuples.
        """
        hm = self.get_heatmap(normalize=True)
        coords = np.argwhere(hm >= threshold)
        if len(coords) == 0:
            return []

        intensities = [(int(c[1]), int(c[0]), float(hm[c[0], c[1]])) for c in coords]
        intensities.sort(key=lambda t: t[2], reverse=True)
        return intensities[:top_n]

    @property
    def total_detections(self) -> int:
        return self._total_detections

    def reset(self) -> None:
        self._grid = np.zeros((self.height, self.width), dtype=np.float64)
        self._total_detections = 0
