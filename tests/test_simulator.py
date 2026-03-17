"""Tests for the scene simulator."""

from raksha.simulator import SceneSimulator


class TestSceneSimulator:
    def test_step_returns_frame_and_detections(self):
        sim = SceneSimulator(num_persons=2, seed=42)
        frame, dets = sim.step()
        assert frame.data is not None
        assert frame.data.shape == (480, 640, 3)
        assert len(dets) == 2
        assert all(d.label == "person" for d in dets)

    def test_run_multiple_frames(self):
        sim = SceneSimulator(num_persons=1, seed=0)
        results = sim.run(10)
        assert len(results) == 10
        for frame, dets in results:
            assert frame.frame_id > 0
            assert len(dets) == 1

    def test_deterministic_with_seed(self):
        sim1 = SceneSimulator(num_persons=2, seed=123)
        sim2 = SceneSimulator(num_persons=2, seed=123)
        _, d1 = sim1.step()
        _, d2 = sim2.step()
        assert d1[0].bbox.x == d2[0].bbox.x
        assert d1[0].bbox.y == d2[0].bbox.y

    def test_reset(self):
        sim = SceneSimulator(num_persons=1, seed=42)
        sim.run(5)
        sim.reset()
        frame, _ = sim.step()
        assert frame.frame_id == 1
