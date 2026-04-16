import numpy as np
import torch
from collections import deque
from torch.utils.data import DataLoader, TensorDataset
from drone_env import DronePlant, DroneExpertController, CHECKPOINTS, CHECKPOINT_RADIUS
from mamba_learner import MambaController


def _suffix_eventually(signal: np.ndarray) -> np.ndarray:
    """Robustness of F phi from every time index onward."""
    out = np.empty_like(signal)
    best = -np.inf
    for idx in range(len(signal) - 1, -1, -1):
        best = max(best, signal[idx])
        out[idx] = best
    return out


def _suffix_always(signal: np.ndarray) -> np.ndarray:
    """Robustness of G phi from every time index onward."""
    out = np.empty_like(signal)
    worst = np.inf
    for idx in range(len(signal) - 1, -1, -1):
        worst = min(worst, signal[idx])
        out[idx] = worst
    return out


def _bounded_suffix_eventually(signal: np.ndarray, latest_idx: int) -> float:
    """Robustness of F_[0, latest_idx] phi at the initial time."""
    if len(signal) == 0:
        return -np.inf
    capped_idx = min(len(signal) - 1, latest_idx)
    return float(np.max(signal[:capped_idx + 1]))


def _sequential_visit_robustness(visit_signals, latest_indices=None) -> float:
    """
    Robustness of the nested formula:
        F visit_A AND F visit_B AND F visit_C in order,
    implemented as F(A AND F(B AND F C)) in discrete time.

    When latest_indices is provided, each visit must happen by that absolute index.
    """
    if not visit_signals:
        return -np.inf

    if latest_indices is None:
        nested = visit_signals[-1]
        for signal in reversed(visit_signals[:-1]):
            nested = np.minimum(signal, _suffix_eventually(nested))
        return float(np.max(nested))

    if len(latest_indices) != len(visit_signals):
        raise ValueError("latest_indices must match the number of visit signals")

    nested_value = _bounded_suffix_eventually(visit_signals[-1], latest_indices[-1])
    for signal, latest_idx in zip(reversed(visit_signals[:-1]), reversed(latest_indices[:-1])):
        capped_idx = min(len(signal) - 1, latest_idx)
        candidates = np.minimum(signal[:capped_idx + 1], nested_value)
        nested_value = float(np.max(candidates)) if len(candidates) > 0 else -np.inf
    return nested_value


def check_stl_score(trajectory_y, s_ov=25.0, s_st=2.0, checkpoint_deadlines=None) -> float:
    """
    Evaluates a trajectory with discrete-time STL-style robustness:

    1. G safety: position distance from origin < s_ov and |roll|, |pitch| < 1.0
    2. F(A and F(B and F(C))) with optional absolute time deadlines
    3. F G docked: eventually always within s_st of origin and speed < 0.5

    Returns a continuous robustness degree (positive = satisfied, negative = violated).
    """
    trajectory_y = np.asarray(trajectory_y)
    positions = trajectory_y[:, 0:3]
    rolls = trajectory_y[:, 3]
    pitches = trajectory_y[:, 4]
    speeds = np.linalg.norm(trajectory_y[:, 6:9], axis=1)

    safety_signal = np.minimum.reduce([
        s_ov - np.linalg.norm(positions, axis=1),
        1.0 - np.abs(rolls),
        1.0 - np.abs(pitches),
    ])
    safety_robustness = float(np.min(safety_signal))

    visit_radius = 2.0
    visit_signals = [
        visit_radius - np.linalg.norm(positions - checkpoint, axis=1)
        for checkpoint in CHECKPOINTS[:3]
    ]
    sequential_robustness = _sequential_visit_robustness(
        visit_signals,
        latest_indices=checkpoint_deadlines,
    )

    dock_signal = np.minimum(
        s_st - np.linalg.norm(positions, axis=1),
        0.5 - speeds,
    )
    stabilization_robustness = float(np.max(_suffix_always(dock_signal)))

    return min(safety_robustness, sequential_robustness, stabilization_robustness)


def plot_trajectory(mamba_traj, expert_traj, cycle, filename="trajectory_comparison.png"):
    """
    Enhanced 4-panel trajectory comparison with checkpoint markers.
    """
    import matplotlib.pyplot as plt

    mamba_pos = np.array([y[0:3] for y in mamba_traj])
    expert_pos = np.array([y[0:3] for y in expert_traj])
    
    steps = np.arange(len(mamba_pos))
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    labels = ['X', 'Y', 'Z']
    
    # Subplots 0-2: X, Y, Z position over time
    for i in range(3):
        axes[i].plot(steps, mamba_pos[:, i], label='Mamba', color='#2196F3', linewidth=1.5)
        axes[i].plot(steps, expert_pos[:, i], label='Expert', color='#FF9800', linestyle='--', linewidth=1.5)
        
        # Draw checkpoint lines
        for j, (cp, name) in enumerate(zip(CHECKPOINTS, ['A', 'B', 'C', 'Dock'])):
            axes[i].axhline(y=cp[i], color='gray', linestyle=':', alpha=0.5)
            axes[i].annotate(f'{name}', xy=(0, cp[i]), fontsize=8, color='gray', alpha=0.7)
        
        axes[i].set_ylabel(f'{labels[i]} Position (m)')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)
        
    # Subplot 3: Distance to current sequential target
    # Track which checkpoint Mamba is "working toward"
    mamba_dists = []
    cp_idx = 0
    for pos in mamba_pos:
        if cp_idx < 3 and np.linalg.norm(pos - CHECKPOINTS[cp_idx]) < 2.0:
            cp_idx += 1
        target = CHECKPOINTS[min(cp_idx, 3)]
        mamba_dists.append(np.linalg.norm(pos - target))
    
    expert_dists = []
    cp_idx = 0
    for pos in expert_pos:
        if cp_idx < 3 and np.linalg.norm(pos - CHECKPOINTS[cp_idx]) < 2.0:
            cp_idx += 1
        target = CHECKPOINTS[min(cp_idx, 3)]
        expert_dists.append(np.linalg.norm(pos - target))
    
    axes[3].plot(steps, mamba_dists, label='Mamba → Target', color='#2196F3', linewidth=1.5)
    axes[3].plot(steps, expert_dists, label='Expert → Target', color='#FF9800', linestyle='--', linewidth=1.5)
    axes[3].axhline(y=CHECKPOINT_RADIUS, color='green', linestyle=':', alpha=0.6, label='Visit Threshold')
    axes[3].set_ylabel('Distance to Current Target (m)')
    axes[3].set_xlabel('Time Steps')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle(f'CEGIS Iteration {cycle}: Mamba vs Expert (A→B→C→Dock)', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"-> Trajectory plot saved: {filename}")


def falsify_cem(mamba_ctrl, plant, expert_ctrl, num_generations=3, pop_size=60, 
                elite_frac=0.2, seq_steps=300, dt=0.1):
    """
    CEM Falsifier adapted for the multi-checkpoint environment.
    Samples initial states spread across the wider 3D checkpoint volume.
    """
    mamba_ctrl.eval()
    failed_inits = []
    
    # Initial distribution centered near typical spawn region
    mu = np.zeros(12)
    mu[0:3] = [3.0, 0.0, 15.0]  # Near checkpoint A's altitude
    sigma = np.ones(12) * 5.0
    sigma[3:6] = 0.3  # Keep attitudes reasonable
    
    for gen in range(num_generations):
        samples = []
        scores = []
        
        for _ in range(pop_size):
            mock_state = np.random.normal(mu, sigma)
            mock_state[0:3] = np.clip(mock_state[0:3], -15, 15)
            mock_state[2] = np.clip(mock_state[2], 5, 22)  # Keep z positive and high
            mock_state[3:6] = np.clip(mock_state[3:6], -0.8, 0.8)
            
            plant.reset()
            plant.state = mock_state.copy()
            plant.time = 0.0
            mamba_ctrl.reset()
            
            y = plant.state.copy()
            traj = [y]
            for _ in range(seq_steps):
                u = mamba_ctrl.forward(y)
                y = plant.step(u, dt)
                traj.append(y)
                
            score = check_stl_score(traj)
            samples.append(mock_state)
            scores.append(score)
            
            if score < 0.0:
                failed_inits.append(mock_state.copy())
                
        # CEM elite update
        sorted_indices = np.argsort(scores)
        n_elites = max(1, int(pop_size * elite_frac))
        elites = np.array(samples)[sorted_indices[:n_elites]]
        
        mu = np.mean(elites, axis=0)
        sigma = np.std(elites, axis=0) + 0.1
            
    # Deduplicate
    unique_fails = []
    for f in failed_inits:
        if not any(np.allclose(f, uf, atol=1.0) for uf in unique_fails):
            unique_fails.append(f)
            
    return unique_fails


def fix_and_merge(failed_inits, expert_ctrl, plant, dataset, seq_steps=300, dt=0.1):
    """
    Deploy expert from failure states to generate corrective demonstrations.
    Expert has privileged access to plant mass for gain compensation.
    """
    latest_demonstrations = []
    for init_state in failed_inits:
        plant.reset()
        plant.state = init_state.copy()
        plant.time = 0.0
        expert_ctrl.reset()
        expert_ctrl.set_plant_ref(plant)  # Give expert privileged mass access
        
        obs_seq, act_seq = [], []
        y = plant.state.copy()
        for _ in range(seq_steps):
            u = expert_ctrl.compute_action(y)
            obs_seq.append(y)
            act_seq.append(u)
            y = plant.step(u, dt)
            
        demo = (np.array(obs_seq), np.array(act_seq))
        latest_demonstrations.append(demo)
        dataset.append(demo)
        
    return latest_demonstrations


def generate_expert_demonstrations(
    plant,
    expert_ctrl,
    num_trajectories=2000,
    seq_steps=300,
    dt=0.1,
    initial_states=None,
):
    """
    Generate expert trajectories for baseline imitation training.
    """
    dataset = []
    for idx in range(num_trajectories):
        if initial_states is None:
            y = plant.reset()
        else:
            plant.reset()
            plant.state = initial_states[idx].copy()
            plant.time = 0.0
            y = plant.state.copy()

        expert_ctrl.reset()
        expert_ctrl.set_plant_ref(plant)
        obs_seq, act_seq = [], []
        for _ in range(seq_steps):
            u = expert_ctrl.compute_action(y)
            obs_seq.append(y)
            act_seq.append(u)
            y = plant.step(u, dt)

        dataset.append((np.asarray(obs_seq, dtype=np.float32), np.asarray(act_seq, dtype=np.float32)))
    return dataset


def build_sequence_loader(raw_list, batch_size=32, shuffle=False, num_workers=0, pin_memory=False):
    if not raw_list:
        return None
    obs = torch.tensor(np.stack([x[0] for x in raw_list]), dtype=torch.float32)
    act = torch.tensor(np.stack([x[1] for x in raw_list]), dtype=torch.float32)
    return DataLoader(
        TensorDataset(obs, act),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def prepare_dataloaders(
    dataset,
    latest_demos,
    batch_size=32,
    val_split=0.15,
    num_workers=0,
    pin_memory=False,
    shuffle=True,
):
    """
    Build train/val DataLoaders with 50/50 oversampling of latest failure demonstrations.
    """
    total_data = list(dataset)
    np.random.shuffle(total_data)
    
    split_idx = int(len(total_data) * (1 - val_split))
    train_raw = total_data[:split_idx]
    val_raw = total_data[split_idx:]
    
    def to_tensors(raw_list):
        if not raw_list:
            return None, None
        o = torch.tensor(np.stack([x[0] for x in raw_list]), dtype=torch.float32)
        a = torch.tensor(np.stack([x[1] for x in raw_list]), dtype=torch.float32)
        return o, a

    train_obs, train_act = to_tensors(train_raw)
    val_obs, val_act = to_tensors(val_raw)
    
    # Oversample latest demonstrations
    if latest_demos and train_obs is not None:
        ld_obs, ld_act = to_tensors(latest_demos)
        if ld_obs is not None:
            repeat_count = max(1, len(train_raw) // max(1, len(latest_demos)))
            train_obs = torch.cat([train_obs, ld_obs.repeat(repeat_count, 1, 1)], dim=0)
            train_act = torch.cat([train_act, ld_act.repeat(repeat_count, 1, 1)], dim=0)
        
    train_loader = DataLoader(
        TensorDataset(train_obs, train_act),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_obs is not None:
        val_loader = DataLoader(
            TensorDataset(val_obs, val_act),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    
    return train_loader, val_loader


def summarize_rollout(trajectory_y, checkpoint_radius=CHECKPOINT_RADIUS, dock_radius=2.0):
    positions = np.asarray(trajectory_y)[:, 0:3]
    checkpoints_reached = 0
    for checkpoint in CHECKPOINTS[:3]:
        if np.any(np.linalg.norm(positions - checkpoint, axis=1) < checkpoint_radius):
            checkpoints_reached += 1
        else:
            break

    final_distance = float(np.linalg.norm(positions[-1] - CHECKPOINTS[-1]))
    docked = bool(final_distance <= dock_radius)
    return {
        "checkpoints_reached": checkpoints_reached,
        "checkpoint_order_success": float(checkpoints_reached == 3),
        "docking_success": float(docked),
        "final_distance": final_distance,
    }


def run_validation_rollouts(
    mamba_ctrl,
    plant,
    num_missions=5,
    seq_steps=300,
    dt=0.1,
    initial_states=None,
):
    metrics = {
        "checkpoint_order_success_rate": 0.0,
        "docking_success_rate": 0.0,
        "avg_final_distance": 0.0,
        "avg_checkpoints_reached": 0.0,
    }
    if num_missions <= 0:
        return metrics

    rollout_summaries = []
    for mission_idx in range(num_missions):
        if initial_states is None:
            y = plant.reset()
        else:
            plant.reset()
            plant.state = initial_states[mission_idx].copy()
            plant.time = 0.0
            y = plant.state.copy()

        mamba_ctrl.reset()
        traj = [y]
        for _ in range(seq_steps):
            u = mamba_ctrl.forward(y)
            y = plant.step(u, dt)
            traj.append(y)

        rollout_summaries.append(summarize_rollout(traj))

    metrics["checkpoint_order_success_rate"] = float(np.mean([x["checkpoint_order_success"] for x in rollout_summaries]))
    metrics["docking_success_rate"] = float(np.mean([x["docking_success"] for x in rollout_summaries]))
    metrics["avg_final_distance"] = float(np.mean([x["final_distance"] for x in rollout_summaries]))
    metrics["avg_checkpoints_reached"] = float(np.mean([x["checkpoints_reached"] for x in rollout_summaries]))
    return metrics


def build_cegis_framework():
    D_OBS, D_ACT = 12, 4
    SEQ_STEPS = 300  # 30 seconds — enough for A→B→C→Dock with mass decay
    DT = 0.1
    MAX_ITER = 10
    
    mamba_ctrl = MambaController(obs_dim=D_OBS, action_dim=D_ACT, d_model=128, d_state=16, num_layers=3, lr=3e-4)
    drone_plant = DronePlant(m0=2.5, fuel_rate=0.02, m_min=1.0)
    expert_ctrl = DroneExpertController()
    expert_ctrl.set_plant_ref(drone_plant)
    dataset = deque(maxlen=2000)
    
    # Generate baseline demonstrations
    print("Generating baseline expert demonstrations (A→B→C→Dock)...")
    init_states = [drone_plant.reset() for _ in range(20)]
    latest_demos = fix_and_merge(init_states, expert_ctrl, drone_plant, dataset, seq_steps=SEQ_STEPS, dt=DT)
    
    print("================ CEGIS LOOP STARTED ================")
    print(f"Checkpoints: A={list(CHECKPOINTS[0])}, B={list(CHECKPOINTS[1])}, C={list(CHECKPOINTS[2])}, Dock={list(CHECKPOINTS[3])}")
    print(f"Mass: {drone_plant.m0}kg → {drone_plant.m_min}kg (burn rate: {drone_plant.fuel_rate} kg/s)")
    print(f"Sequence length: {SEQ_STEPS} steps ({SEQ_STEPS * DT}s)")
    
    for cycle in range(1, MAX_ITER + 1):
        print(f"\n{'='*50}")
        print(f"--- Iteration {cycle}/{MAX_ITER} ---")
        print(f"{'='*50}")
        
        train_loader, val_loader = prepare_dataloaders(
            dataset,
            latest_demos,
            batch_size=32,
            num_workers=0,
            pin_memory=mamba_ctrl.device.type == "cuda",
        )
        
        epochs = 5 if cycle == 1 else min(15, max(3, len(latest_demos)))
        metrics = mamba_ctrl.update(
            train_loader,
            val_loader,
            epochs=epochs,
            fit_normalizer=cycle == 1 and not mamba_ctrl.normalizer_fitted,
        )
        print(
            f"-> Train MSE: {metrics['train_loss']:.4f} | "
            f"Val MSE: {metrics['val_loss']:.4f} | "
            f"Dataset: {len(dataset)}"
        )
        
        # Falsification
        print("-> Running CEM Falsifier (multi-checkpoint + mass decay)...")
        failures = falsify_cem(mamba_ctrl, drone_plant, expert_ctrl,
                              num_generations=1, pop_size=20, seq_steps=SEQ_STEPS, dt=DT)
        num_fails = len(failures)
        
        coverage = (1.0 - min(1.0, num_fails / 60.0)) * 100
        print(f"-> Failures: {num_fails} | Safety Coverage: {coverage:.1f}%")
        
        # Generate comparison plot
        test_state = failures[0] if num_fails > 0 else drone_plant.reset()
        
        # Mamba trajectory
        drone_plant.reset()
        drone_plant.state = test_state.copy()
        drone_plant.time = 0.0
        mamba_ctrl.reset()
        m_traj = []
        y = drone_plant.state.copy()
        for _ in range(SEQ_STEPS):
            u = mamba_ctrl.forward(y)
            y = drone_plant.step(u, DT)
            m_traj.append(y)
            
        # Expert trajectory (same init)
        drone_plant.reset()
        drone_plant.state = test_state.copy()
        drone_plant.time = 0.0
        expert_ctrl.reset()
        expert_ctrl.set_plant_ref(drone_plant)
        e_traj = []
        y = drone_plant.state.copy()
        for _ in range(SEQ_STEPS):
            u = expert_ctrl.compute_action(y)
            y = drone_plant.step(u, DT)
            e_traj.append(y)
            
        plot_trajectory(m_traj, e_traj, cycle, filename=f"iter_{cycle}_comparison.png")
        
        if num_fails == 0:
            print(f"\n✔️  CEGIS Converged after {cycle} iterations!")
            print("    Mamba has learned: sequential checkpoints, mass compensation, and wind prediction.")
            break
            
        print(f"-> Deploying expert to {num_fails} failure states...")
        latest_demos = fix_and_merge(failures, expert_ctrl, drone_plant, dataset, seq_steps=SEQ_STEPS, dt=DT)


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    build_cegis_framework()
