import unittest

import numpy as np

from nncs_mamba.models.controller import ConstantController
from nncs_mamba.rollout import run_controller_rollout
from nncs_mamba.safe_control_gym_config import ACTION_LABELS, STATE_LABELS


class FakeActionSpace:
    def __init__(self):
        self.low = np.zeros(len(ACTION_LABELS), dtype=np.float32)
        self.high = np.ones(len(ACTION_LABELS), dtype=np.float32)


class FakeEnv:
    """A tiny environment used to test rollout code without PyBullet."""

    def __init__(self, done_after=None):
        self.action_space = FakeActionSpace()
        self.done_after = done_after
        self.step_count = 0
        self.actions = []

    def reset(self):
        self.step_count = 0
        self.actions = []
        return np.zeros(len(STATE_LABELS), dtype=np.float32), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.actions.append(action.copy())
        self.step_count += 1

        obs = np.zeros(len(STATE_LABELS), dtype=np.float32)
        obs[0] = float(self.step_count)
        reward = -float(np.sum(action))
        done = self.done_after is not None and self.step_count >= self.done_after
        return obs, reward, done, {"step_count": self.step_count}


class TestControllerContract(unittest.TestCase):
    def test_constant_controller_forward_sequence_shapes(self):
        controller = ConstantController(action=np.array([0.1, 0.2, 0.3, 0.4]), value=0.25)
        states = np.zeros((2, 5, len(STATE_LABELS)), dtype=np.float32)

        action_pred, value_pred = controller.forward_sequence(states)

        self.assertEqual(action_pred.shape, (2, 5, len(ACTION_LABELS)))
        self.assertEqual(value_pred.shape, (2, 5, 1))
        np.testing.assert_allclose(action_pred[0, 0], np.array([0.1, 0.2, 0.3, 0.4]), rtol=1e-6)
        np.testing.assert_allclose(value_pred[:, :, 0], 0.25, rtol=1e-6)

    def test_constant_controller_accepts_single_sequence(self):
        controller = ConstantController(action=0.5, value=-0.1)
        states = np.zeros((5, len(STATE_LABELS)), dtype=np.float32)

        action_pred, value_pred = controller.forward_sequence(states)

        self.assertEqual(action_pred.shape, (1, 5, len(ACTION_LABELS)))
        self.assertEqual(value_pred.shape, (1, 5, 1))
        np.testing.assert_allclose(action_pred, 0.5, rtol=1e-6)
        np.testing.assert_allclose(value_pred, -0.1, rtol=1e-6)

    def test_step_shapes_and_cache_counter(self):
        controller = ConstantController(action=np.full(len(ACTION_LABELS), 0.2), value=-0.5)
        cache = controller.initial_cache(batch_size=1)

        action, value, next_cache = controller.step(np.zeros(len(STATE_LABELS), dtype=np.float32), cache)

        self.assertEqual(action.shape, (len(ACTION_LABELS),))
        self.assertEqual(value.shape, (1,))
        self.assertEqual(int(cache["steps"][0]), 0)
        self.assertEqual(int(next_cache["steps"][0]), 1)

    def test_rollout_clips_actions_and_collects_value(self):
        env = FakeEnv()
        controller = ConstantController(action=np.full(len(ACTION_LABELS), 2.0), value=0.125)

        rollout = run_controller_rollout(env, controller, steps=3)

        self.assertEqual(rollout.states.shape, (4, len(STATE_LABELS)))
        self.assertEqual(rollout.actions.shape, (3, len(ACTION_LABELS)))
        self.assertEqual(rollout.values.shape, (3, 1))
        self.assertEqual(rollout.rewards.shape, (3,))
        self.assertFalse(rollout.done)
        np.testing.assert_allclose(rollout.actions, 1.0, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(env.actions), 1.0, rtol=1e-6)
        np.testing.assert_allclose(rollout.values, 0.125, rtol=1e-6)
        self.assertAlmostEqual(rollout.total_reward, -12.0)

    def test_rollout_stops_on_done(self):
        env = FakeEnv(done_after=2)
        controller = ConstantController(action=np.full(len(ACTION_LABELS), 0.25), value=0.0)

        rollout = run_controller_rollout(env, controller, steps=5)

        self.assertEqual(rollout.states.shape, (3, len(STATE_LABELS)))
        self.assertEqual(rollout.actions.shape, (2, len(ACTION_LABELS)))
        self.assertTrue(rollout.done)
        self.assertEqual(rollout.final_info["step_count"], 2)


if __name__ == "__main__":
    unittest.main()
