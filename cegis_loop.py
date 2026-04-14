import numpy as np
import torch
from collections import deque
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from drone_env import DronePlant, DroneExpertController
from mamba_controller import MambaController

def check_stl_score(trajectory_y, s_ov=15.0, s_st=2.0) -> float:
    """
    Returns a continuous 'robustness' margin metric.
    No more flat -100/-50 cliffs. Instead, we return a deviation-based score
    to give the CEM falsifier a smooth gradient to optimize against.
    """
    min_safety_margin = float('inf')
    
    # RULE 1: OVERSHOOT AND ATTITUDE ENVELOPES
    for y in trajectory_y:
        dist = np.linalg.norm(y[0:3])
        roll, pitch = y[3], y[4]
        
        dist_margin = s_ov - dist
        roll_margin = 1.0 - abs(roll)
        pitch_margin = 1.0 - abs(pitch)
        
        # We take the minimum margin. If any is < 0, it's a failure.
        margin = min(dist_margin, roll_margin, pitch_margin)
        if margin < min_safety_margin:
            min_safety_margin = margin
            
    # RULE 2: EVENTUALLY ALWAYS STABILIZATION
    # We find if/when the drone entered the stabilization ball.
    entered_step = -1
    for i, y in enumerate(trajectory_y):
        dist = np.linalg.norm(y[0:3])
        vel = np.linalg.norm(y[6:9])
        if dist < s_st and vel < 0.5:
            entered_step = i
            break
            
    if entered_step != -1:
        # Check 'Always' after entering.
        for y in trajectory_y[entered_step:]:
            dist = np.linalg.norm(y[0:3])
            vel = np.linalg.norm(y[6:9])
            # Instead of -100, we penalize by how far it left the ball
            margin = min(s_st - dist, 0.5 - vel)
            if margin < 0:
                # Continuous penalty based on deviation
                if margin < min_safety_margin:
                    min_safety_margin = margin
    else:
        # Never entered. Penalty is based on the closest it ever got.
        dists = [np.linalg.norm(y[0:3]) for y in trajectory_y]
        min_dist = min(dists)
        min_safety_margin = min(min_safety_margin, s_st - min_dist)

    return min_safety_margin

def plot_trajectory(mamba_traj, expert_traj, cycle, filename="trajectory_comparison.png"):
    """
    Saves a comparison plot of Mamba vs Expert trajectories.
    """
    mamba_pos = np.array([y[0:3] for y in mamba_traj])
    expert_pos = np.array([y[0:3] for y in expert_traj])
    
    steps = np.arange(len(mamba_pos))
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    labels = ['X', 'Y', 'Z']
    
    for i in range(3):
        axes[i].plot(steps, mamba_pos[:, i], label='Mamba', color='blue', alpha=0.8)
        axes[i].plot(steps, expert_pos[:, i], label='Expert', color='orange', linestyle='--', alpha=0.8)
        axes[i].set_ylabel(f'{labels[i]} Position')
        axes[i].legend()
        axes[i].grid(True)
        
    axes[2].set_xlabel('Time Steps')
    plt.suptitle(f'CEGIS Iteration {cycle}: Mamba vs Expert Tracking')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"-> Trajectory plot saved as {filename}")

def falsify_cem(mamba_ctrl, plant, num_generations=2, pop_size=30, elite_frac=0.2, seq_steps=150, dt=0.1):
    mamba_ctrl.eval()
    failed_inits = []
    
    mu = np.zeros(12)
    mu[2] = 12.0 # Target z
    sigma = np.ones(12) * 6.0
    sigma[3:6] = 0.4
    
    for gen in range(num_generations):
        samples = []
        scores = []
        
        for _ in range(pop_size):
            mock_state = np.random.normal(mu, sigma)
            # Clip for physical realism
            mock_state[0:3] = np.clip(mock_state[0:3], -15, 15)
            mock_state[3:6] = np.clip(mock_state[3:6], -1.2, 1.2)
            
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
                
        # CEM update
        sorted_indices = np.argsort(scores)
        elites = np.array(samples)[sorted_indices[:int(pop_size * elite_frac)]]
        
        mu = np.mean(elites, axis=0)
        sigma = np.std(elites, axis=0) + 0.1
            
    # Deduplicate
    unique_fails = []
    for f in failed_inits:
        if not any(np.allclose(f, uf, atol=1.0) for uf in unique_fails):
            unique_fails.append(f)
            
    return unique_fails

def fix_and_merge(failed_inits, expert_ctrl, plant, dataset, seq_steps=150, dt=0.1):
    latest_demonstrations = []
    for init_state in failed_inits:
        plant.reset()
        plant.state = init_state.copy()
        plant.time = 0.0
        expert_ctrl.reset()
        
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
    Implements 50/50 oversampling for new failures and a validation split.
    """
    total_data = list(dataset)
    np.random.shuffle(total_data)
    
    split_idx = int(len(total_data) * (1 - val_split))
    train_raw = total_data[:split_idx]
    val_raw = total_data[split_idx:]
    
    def to_tensors(raw_list):
        if not raw_list: return None, None
        o = torch.tensor(np.stack([x[0] for x in raw_list]), dtype=torch.float32, device=device)
        a = torch.tensor(np.stack([x[1] for x in raw_list]), dtype=torch.float32, device=device)
        return o, a

    train_obs, train_act = to_tensors(train_raw)
    val_obs, val_act = to_tensors(val_raw)
    
    # Oversample latest demonstrations in training set
    if latest_demos:
        ld_obs, ld_act = to_tensors(latest_demos)
        # Calculate how many times to repeat ld to match ~50% of the train set
        repeat_count = max(1, len(train_raw) // len(latest_demos))
        train_obs = torch.cat([train_obs, ld_obs.repeat(repeat_count, 1, 1)], dim=0)
        train_act = torch.cat([train_act, ld_act.repeat(repeat_count, 1, 1)], dim=0)
        
    train_loader = DataLoader(TensorDataset(train_obs, train_act), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_obs, val_act), batch_size=16) if val_obs is not None else None
    
    return train_loader, val_loader

def build_cegis_framework():
    D_OBS, D_ACT = 12, 4
    SEQ_STEPS, DT, MAX_ITER = 150, 0.1, 10
    
    mamba_ctrl = MambaController(obs_dim=D_OBS, action_dim=D_ACT, d_model=64, d_state=16, num_layers=2, lr=3e-4)
    drone_plant = DronePlant()
    expert_ctrl = DroneExpertController()
    dataset = deque(maxlen=2000)
    
    print("Generating baseline tracking data...")
    init_states = [drone_plant.reset() for _ in range(20)]
    latest_demos = fix_and_merge(init_states, expert_ctrl, drone_plant, dataset, seq_steps=SEQ_STEPS, dt=DT)
    
    print("================ CEGIS LOOP STARTED ================")
    for cycle in range(1, MAX_ITER + 1):
        print(f"\n--- Iteration {cycle}/{MAX_ITER} ---")
        
        train_loader, val_loader = prepare_dataloaders(dataset, latest_demos, mamba_ctrl.device)
        
        epochs = 5 if cycle == 1 else min(15, max(3, len(latest_demos)))
        t_loss, v_loss = mamba_ctrl.update(train_loader, val_loader, epochs=epochs)
        print(f"-> Train MSE: {t_loss:.4f} | Val MSE: {v_loss:.4f}")
        
        # Falsification
        print("-> Running CEM Falsifier...")
        failures = falsify_cem(mamba_ctrl, drone_plant, num_generations=2, pop_size=30, seq_steps=SEQ_STEPS, dt=DT)
        num_fails = len(failures)
        
        coverage = (1.0 - min(1.0, num_fails / 60.0)) * 100
        print(f"-> Failures Found: {num_fails} | Coverage: {coverage:.2f}%")
        
        # Plot one of the failures (or a success if no failures)
        test_state = failures[0] if num_fails > 0 else drone_plant.reset()
        
        # Get Mamba traj
        drone_plant.reset(); drone_plant.state = test_state.copy(); mamba_ctrl.reset()
        m_traj = []
        y = drone_plant.state.copy()
        for _ in range(SEQ_STEPS):
            u = mamba_ctrl.forward(y)
            y = drone_plant.step(u, DT)
            m_traj.append(y)
            
        # Get Expert traj
        drone_plant.reset(); drone_plant.state = test_state.copy(); expert_ctrl.reset()
        e_traj = []
        y = drone_plant.state.copy()
        for _ in range(SEQ_STEPS):
            u = expert_ctrl.compute_action(y)
            y = drone_plant.step(u, DT)
            e_traj.append(y)
            
        plot_trajectory(m_traj, e_traj, cycle, filename=f"iter_{cycle}_comparison.png")
        
        if num_fails == 0:
            print("\n✔️ CEGIS Converged!")
            break
            
        latest_demos = fix_and_merge(failures, expert_ctrl, drone_plant, dataset, seq_steps=SEQ_STEPS, dt=DT)

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    build_cegis_framework()
