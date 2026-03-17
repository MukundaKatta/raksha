"""Tests for motion detection."""

import numpy as np

from raksha.detection.motion import MotionDetector


class TestMotionDetector:
    def test_no_motion_on_first_frame(self):
        det = MotionDetector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = det.detect(frame, frame_id=0)
        assert result == []

    def test_no_motion_on_identical_frames(self):
        det = MotionDetector()
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        det.detect(frame, frame_id=0)
        result = det.detect(frame.copy(), frame_id=1)
        assert result == []

    def test_detects_motion(self):
        det = MotionDetector(base_threshold=10, min_area=10)
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        det.detect(frame1, frame_id=0)

        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2[30:60, 30:60] = 200  # bright rectangle
        result = det.detect(frame2, frame_id=1)
        assert len(result) >= 1
        assert result[0].label == "motion"
        assert result[0].confidence > 0.0

    def test_grayscale_input(self):
        det = MotionDetector(base_threshold=10, min_area=10)
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        det.detect(frame1, frame_id=0)

        frame2 = np.zeros((100, 100), dtype=np.uint8)
        frame2[20:50, 20:50] = 200
        result = det.detect(frame2, frame_id=1)
        assert len(result) >= 1

    def test_reset(self):
        det = MotionDetector()
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        det.detect(frame)
        det.reset()
        assert det._background is None
        assert det._frame_count == 0
