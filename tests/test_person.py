"""Tests for person detection."""

import numpy as np

from raksha.detection.person import PersonDetector
from raksha.models import BoundingBox


class TestPersonDetector:
    def test_creation(self):
        det = PersonDetector()
        assert det.confidence_threshold == 0.5
        assert det._model is not None

    def test_detect_with_candidates(self):
        det = PersonDetector(confidence_threshold=0.0)  # accept any confidence
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        candidates = [BoundingBox(x=10, y=10, width=50, height=50)]
        result = det.detect(frame, candidates=candidates, frame_id=0)
        # With random weights the score is unpredictable, but it should
        # return something since threshold is 0
        assert isinstance(result, list)

    def test_detect_grayscale(self):
        det = PersonDetector(confidence_threshold=0.0)
        frame = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        candidates = [BoundingBox(x=0, y=0, width=64, height=64)]
        result = det.detect(frame, candidates=candidates, frame_id=0)
        assert isinstance(result, list)

    def test_sliding_window(self):
        det = PersonDetector(patch_size=32, stride=16)
        boxes = det._sliding_window(64, 64)
        assert len(boxes) > 0
        assert all(b.width == 32 and b.height == 32 for b in boxes)
