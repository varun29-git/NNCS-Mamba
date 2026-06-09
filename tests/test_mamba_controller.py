import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from nncs_mamba.safe_control_gym_config import ACTION_LABELS, STATE_LABELS


if torch is not None:
    from nncs_mamba.models.controller import ACTION_PRED_KEY, VALUE_PRED_KEY
    from nncs_mamba.models.mamba import DualHeadMambaController, MambaControllerConfig
    from nncs_mamba.rollout import run_controller_rollout


class FakeActionSpace:
    def __init__(self):
        self.low = np.zeros(len(ACTION_LABELS), dtype=np.float32)
        self.high = np.ones(len(ACTION_LABELS), dtype=np.float32)


class FakeEnv:
    def __init__(self):
        self.action_space = FakeActionSpace()
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.zeros(len(STATE_LABELS), dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        obs = np.zeros(len(STATE_LABELS), dtype=np.float32)
        obs[0] = float(self.step_count)
        reward = -float(np.sum(np.asarray(action, dtype=np.float32)))
        return obs, reward, False, {"step_count": self.step_count}


@unittest.skipIf(torch is None, "Torch is not installed.")
class TestDualHeadMambaController(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(7)
        config = MambaControllerConfig(
            d_model=16,
            d_state=4,
            n_layers=2,
            head_hidden_dim=24,
            dropout=0.0,
        )
        model = DualHeadMambaController(config=config)
        model.eval()
        return model

    def test_forward_sequence_shapes(self):
        model = self.make_model()
        states = np.zeros((2, 5, len(STATE_LABELS)), dtype=np.float32)

        action_pred, value_pred = model.forward_sequence(states)

        self.assertEqual(action_pred.shape, (2, 5, len(ACTION_LABELS)))
        self.assertEqual(value_pred.shape, (2, 5, 1))
        self.assertTrue(np.all(np.isfinite(action_pred)))
        self.assertTrue(np.all(np.isfinite(value_pred)))

    def test_torch_forward_returns_named_outputs(self):
        model = self.make_model()
        states = torch.zeros(2, 5, len(STATE_LABELS))

        outputs = model(states)

        self.assertEqual(set(outputs), {ACTION_PRED_KEY, VALUE_PRED_KEY})
        self.assertEqual(tuple(outputs[ACTION_PRED_KEY].shape), (2, 5, len(ACTION_LABELS)))
        self.assertEqual(tuple(outputs[VALUE_PRED_KEY].shape), (2, 5, 1))

    def test_cached_step_matches_sequence_path(self):
        model = self.make_model()
        states = np.random.default_rng(3).normal(size=(1, 6, len(STATE_LABELS))).astype(np.float32)
        sequence_actions, sequence_values = model.forward_sequence(states)

        cache = model.initial_cache(batch_size=1)
        step_actions = []
        step_values = []
        for step_idx in range(states.shape[1]):
            action, value, cache = model.step(states[0, step_idx], cache)
            step_actions.append(action)
            step_values.append(value)

        np.testing.assert_allclose(np.asarray(step_actions), sequence_actions[0], atol=1e-5)
        np.testing.assert_allclose(np.asarray(step_values), sequence_values[0], atol=1e-5)
        self.assertEqual(int(cache["steps"][0].item()), states.shape[1])

    def test_forward_torch_keeps_gradients(self):
        model = self.make_model()
        model.train()
        states = torch.randn(2, 4, len(STATE_LABELS))

        action_pred, value_pred = model.forward_torch(states)
        loss = action_pred.square().mean() + value_pred.square().mean()
        loss.backward()

        grads = [param.grad for param in model.parameters() if param.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(grad).all().item() for grad in grads))

    def test_rollout_accepts_mamba_controller(self):
        model = self.make_model()
        rollout = run_controller_rollout(FakeEnv(), model, steps=3)

        self.assertEqual(rollout.states.shape, (4, len(STATE_LABELS)))
        self.assertEqual(rollout.actions.shape, (3, len(ACTION_LABELS)))
        self.assertEqual(rollout.values.shape, (3, 1))
        self.assertTrue(np.all(np.isfinite(rollout.actions)))
        self.assertTrue(np.all(np.isfinite(rollout.values)))


if __name__ == "__main__":
    unittest.main()
