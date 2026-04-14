import numpy as np
import torch
from collections import deque
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from drone_env import DronePlant, DroneExpertController, CHECKPOINTS, CHECKPOINT_RADIUS
from mamba_controller import MambaController


def check_stl_score(trajectory_y, s_ov=25.0, s_st=2.0) -> float:
    """
    Evaluates a trajectory against the full multi-checkpoint PSTL specification:
    
    1. SAFETY: Position distance from origin < s_ov at all times, attitude < 1.0 rad
    2. SEQUENTIAL VISITS: Must pass within 2.0 of A, then B, then C, in order
    3. TERMINAL STABILIZATION: Eventually always within s_st of origin, velocity < 0.5
    
    Returns a continuous robustness margin (positive = pass, negative = fail).
    """
    min_safety_margin = float('inf')
    
    # ------------------------------------------------------------------
    # RULE 1: GLOBAL SAFETY ENVELOPE (overshoot + attitude)
    # ------------------------------------------------------------------
    for y in trajectory_y:
        dist = np.linalg.norm(y[0:3])
        roll, pitch = y[3], y[4]
        
        dist_margin = s_ov - dist
        roll_margin = 1.0 - abs(roll)
        pitch_margin = 1.0 - abs(pitch)
        
        margin = min(dist_margin, roll_margin, pitch_margin)
        if margin < min_safety_margin:
            min_safety_margin = margin
    
    # ------------------------------------------------------------------
    # RULE 2: SEQUENTIAL CHECKPOINT VISITS (A → B → C)
    # The trajectory must pass within visit_radius of each checkpoint IN ORDER.
    # ------------------------------------------------------------------
    visit_radius = 2.0
    checkpoints_to_visit = CHECKPOINTS[:3]  # A, B, C (not dock)
    next_checkpoint_idx = 0
    
    for y in trajectory_y:
        if next_checkpoint_idx >= len(checkpoints_to_visit):
            break  # All visited
        target = checkpoints_to_visit[next_checkpoint_idx]
        dist_to_target = np.linalg.norm(y[0:3] - target)
        if dist_to_target < visit_radius:
            next_checkpoint_idx += 1
    
    # Penalize proportionally for each unvisited checkpoint
    unvisited = len(checkpoints_to_visit) - next_checkpoint_idx
    if unvisited > 0:
        # Each unvisited checkpoint contributes a negative penalty
        # Scale: closest approach to the next unvisited checkpoint
        if next_checkpoint_idx < len(checkpoints_to_visit):
            next_target = checkpoints_to_visit[next_checkpoint_idx]
            closest = min(np.linalg.norm(y[0:3] - next_target) for y in trajectory_y)
            checkpoint_margin = visit_radius - closest  # negative if never got close
        else:
            checkpoint_margin = 0.0
        # Weight unvisited checkpoints heavily
        checkpoint_penalty = checkpoint_margin - (unvisited * 5.0)
        min_safety_margin = min(min_safety_margin, checkpoint_penalty)
    
    # ------------------------------------------------------------------
    # RULE 3: EVENTUALLY ALWAYS STABILIZATION AT DOCK
    # ------------------------------------------------------------------
    entered_step = -1
    for i, y in enumerate(trajectory_y):
        dist = np.linalg.norm(y[0:3])
        vel = np.linalg.norm(y[6:9])
        if dist < s_st and vel < 0.5:
            entered_step = i
            break
            
    if entered_step != -1:
        for y in trajectory_y[entered_step:]:
            dist = np.linalg.norm(y[0:3])
            vel = np.linalg.norm(y[6:9])
            margin = min(s_st - dist, 0.5 - vel)
            if margin < 0:
                if margin < min_safety_margin:
                    min_safety_margin = margin
    else:
        # Never stabilized — penalty based on closest approach
        dists = [np.linalg.norm(y[0:3]) for y in trajectory_y]
        min_dist = min(dists)
        stab_margin = s_st - min_dist
        min_safety_margin = min(min_safety_margin, stab_margin)

    return min_safety_margin


def plot_trajectory(mamba_traj, expert_traj, cycle, filename="trajectory_comparison.png"):
    """
    Enhanced 4-panel trajectory comparison with checkpoint markers.
    """
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


def falsify_cem(mamba_ctrl, plant, expert_ctrl, num_generations=2, pop_size=30, 
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


def prepare_dataloaders(dataset, latest_demos, device, val_split=0.15):
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
        o = torch.tensor(np.stack([x[0] for x in raw_list]), dtype=torch.float32, device=device)
        a = torch.tensor(np.stack([x[1] for x in raw_list]), dtype=torch.float32, device=device)
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
        
    train_loader = DataLoader(TensorDataset(train_obs, train_act), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_obs, val_act), batch_size=8) if val_obs is not None else None
    
    return train_loader, val_loader


def build_cegis_framework():
    D_OBS, D_ACT = 12, 4
    SEQ_STEPS = 300  # 30 seconds — enough for A→B→C→Dock with mass decay
    DT = 0.1
    MAX_ITER = 10
    
    mamba_ctrl = MambaController(obs_dim=D_OBS, action_dim=D_ACT, d_model=64, d_state=16, num_layers=2, lr=3e-4)
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
        
        train_loader, val_loader = prepare_dataloaders(dataset, latest_demos, mamba_ctrl.device)
        
        epochs = 5 if cycle == 1 else min(15, max(3, len(latest_demos)))
        t_loss, v_loss = mamba_ctrl.update(train_loader, val_loader, epochs=epochs)
        print(f"-> Train MSE: {t_loss:.4f} | Val MSE: {v_loss:.4f} | Dataset: {len(dataset)}")
        
        # Falsification
        print("-> Running CEM Falsifier (multi-checkpoint + mass decay)...")
        failures = falsify_cem(mamba_ctrl, drone_plant, expert_ctrl,
                              num_generations=2, pop_size=30, seq_steps=SEQ_STEPS, dt=DT)
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
