"""CNN-based person detection module."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from raksha.models import BoundingBox, Detection


class _PersonCNN(nn.Module):
    """Lightweight CNN for person/non-person classification on image patches.

    This network operates on fixed-size 64x64 patches and outputs a confidence
    score indicating whether a person is present.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@dataclass
class PersonDetector:
    """Detects persons in a frame using a CNN classifier.

    The detector takes candidate regions (e.g. from motion detection) and
    classifies each one.  If no candidates are provided it uses a sliding
    window approach.
    """

    confidence_threshold: float = 0.5
    patch_size: int = 64
    stride: int = 32
    device: str = "cpu"

    _model: _PersonCNN = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._model = _PersonCNN()
        self._model.to(self.device)
        self._model.eval()

    def _preprocess_patch(self, patch: np.ndarray) -> torch.Tensor:
        """Resize a patch to the expected input size and normalise."""
        from scipy.ndimage import zoom

        if patch.ndim == 2:
            patch = np.stack([patch] * 3, axis=-1)

        h, w = patch.shape[:2]
        scale_h = self.patch_size / h
        scale_w = self.patch_size / w
        resized = zoom(patch, (scale_h, scale_w, 1), order=1)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def detect(
        self,
        frame: np.ndarray,
        candidates: list[BoundingBox] | None = None,
        frame_id: int = 0,
    ) -> list[Detection]:
        """Detect persons in a frame.

        Args:
            frame: Input image array (H, W, C) or (H, W).
            candidates: Optional bounding boxes to classify.  When ``None``
                a sliding window is used instead.
            frame_id: Identifier for the current frame.

        Returns:
            List of detections where the CNN predicts a person.
        """
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)

        h, w = frame.shape[:2]

        if candidates is None:
            candidates = self._sliding_window(w, h)

        detections: list[Detection] = []

        with torch.no_grad():
            for bbox in candidates:
                x0 = max(0, int(bbox.x))
                y0 = max(0, int(bbox.y))
                x1 = min(w, int(bbox.x + bbox.width))
                y1 = min(h, int(bbox.y + bbox.height))

                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue

                patch = frame[y0:y1, x0:x1]
                tensor = self._preprocess_patch(patch)
                score = float(self._model(tensor).item())

                if score >= self.confidence_threshold:
                    detections.append(
                        Detection(
                            frame_id=frame_id,
                            bbox=bbox,
                            label="person",
                            confidence=score,
                        )
                    )

        return detections

    def _sliding_window(self, width: int, height: int) -> list[BoundingBox]:
        """Generate candidate bounding boxes via sliding window."""
        boxes: list[BoundingBox] = []
        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                boxes.append(
                    BoundingBox(
                        x=float(x),
                        y=float(y),
                        width=float(self.patch_size),
                        height=float(self.patch_size),
                    )
                )
        return boxes

    def load_weights(self, path: str) -> None:
        """Load pre-trained model weights."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()
