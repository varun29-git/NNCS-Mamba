import unittest

import numpy as np

from stl_monitor import STLSpec, evaluate_stabilization_stl


class TestSTLMonitor(unittest.TestCase):
    def test_satisfying_stabilization_trajectory(self):
        states = np.zeros((20, 12), dtype=float)
        states[:, 4] = np.linspace(1.4, 1.0, 20)
        states[-8:, 4] = 1.0
        actions = np.ones((19, 4), dtype=float) * 0.1
        spec = STLSpec(goal_position=np.array([0.0, 0.0, 1.0]))
        result = evaluate_stabilization_stl(states, actions, spec, np.zeros(4), np.ones(4))
        self.assertGreaterEqual(result["stl_robustness"], 0.0)
        self.assertEqual(result["stl_satisfied"], 1.0)

    def test_violating_altitude_trajectory(self):
        states = np.zeros((20, 12), dtype=float)
        states[:, 4] = -0.2
        actions = np.ones((19, 4), dtype=float) * 0.1
        spec = STLSpec(goal_position=np.array([0.0, 0.0, 1.0]))
        result = evaluate_stabilization_stl(states, actions, spec, np.zeros(4), np.ones(4))
        self.assertLess(result["stl_robustness"], 0.0)
        self.assertEqual(result["stl_satisfied"], 0.0)

    def test_violating_input_bounds(self):
        states = np.zeros((20, 12), dtype=float)
        states[:, 4] = 1.0
        actions = np.ones((19, 4), dtype=float) * 2.0
        spec = STLSpec(goal_position=np.array([0.0, 0.0, 1.0]))
        result = evaluate_stabilization_stl(states, actions, spec, np.zeros(4), np.ones(4))
        self.assertLess(result["input_safety_robustness"], 0.0)


if __name__ == "__main__":
    unittest.main()
