"""Measure the first dual-head Mamba controller on CPU.

This script does not start Safe-Control-Gym and does not use a GPU. It only
checks the neural controller core:

- parameter count;
- full-sequence output shapes;
- cached online step latency.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nncs_mamba.models.mamba import DualHeadMambaController, MambaControllerConfig
from nncs_mamba.safe_control_gym_config import STATE_LABELS


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parameter_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the dual-head Mamba controller core on CPU.")
    parser.add_argument("--steps", type=positive_int, default=1000)
    parser.add_argument("--warmup", type=positive_int, default=100)
    parser.add_argument("--sequence-len", type=positive_int, default=300)
    parser.add_argument("--d-model", type=positive_int, default=64)
    parser.add_argument("--d-state", type=positive_int, default=8)
    parser.add_argument("--layers", type=positive_int, default=2)
    parser.add_argument("--threads", type=positive_int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(7)

    config = MambaControllerConfig(
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.layers,
    )
    model = DualHeadMambaController(config=config)
    model.eval()

    sequence = np.zeros((1, args.sequence_len, len(STATE_LABELS)), dtype=np.float32)
    action_pred, value_pred = model.forward_sequence(sequence)

    obs = np.zeros(len(STATE_LABELS), dtype=np.float32)
    cache = model.initial_cache(batch_size=1)
    for _ in range(args.warmup):
        _, _, cache = model.step(obs, cache)

    latencies_ms = []
    for _ in range(args.steps):
        start_ns = time.perf_counter_ns()
        _, _, cache = model.step(obs, cache)
        end_ns = time.perf_counter_ns()
        latencies_ms.append((end_ns - start_ns) / 1_000_000.0)

    print("model: DualHeadMambaController")
    print(f"parameters: {parameter_count(model)}")
    print(f"sequence_action_shape: {action_pred.shape}")
    print(f"sequence_value_shape: {value_pred.shape}")
    print(f"cached_step_mean_ms: {float(np.mean(latencies_ms)):.6f}")
    print(f"cached_step_p95_ms:  {percentile(latencies_ms, 95):.6f}")
    print(f"cached_step_p99_ms:  {percentile(latencies_ms, 99):.6f}")
    print(f"cached_steps_recorded: {int(cache['steps'][0].item())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
