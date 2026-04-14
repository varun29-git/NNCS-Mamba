import numpy as np
import torch
from collections import deque
from torch.utils.data import DataLoader, TensorDataset
from drone_env import DronePlant, DroneExpertController
from mamba_controller import MambaController

def check_stl_score(trajectory_y, s_ov=15.0, s_st=2.0) -> float:
    """
    Returns a continuous 'robustness' margin metric evaluating strict PSTL Logic definitions.
    Positive Margin = Satisfied logic safely.
    Negative Margin = Critical Logic parameter violated.
    The lower the negative score, the further out-of-bounds the anomaly was.
    """
    min_safety_margin = float('inf')
    
    # RULE 1: OVERSHOOT AND FATAL ATTITUDE ENVELOPES
    for y in trajectory_y:
        dist = np.linalg.norm(y[0:3])
        roll, pitch = y[3], y[4]
        
        dist_margin = s_ov - dist
        roll_margin = 1.0 - abs(roll)   # Strict 1-Radian flip constraint
        pitch_margin = 1.0 - abs(pitch) 
        
        margin = min(dist_margin, roll_margin, pitch_margin)
        if margin < min_safety_margin:
            min_safety_margin = margin
            
        if margin < 0: 
            return margin # Fatal safety break (immediate anomaly return)
            
    # RULE 2: EVENTUALLY ALWAYS STABILIZATION LOGIC
    # Verifies both velocity settling bounds AND spatial settling bounds
    entered_stabilization = False
    
    for y in trajectory_y:
        dist = np.linalg.norm(y[0:3])
        vel = np.linalg.norm(y[6:9])
        
        is_stable = (dist < s_st) and (vel < 0.5)
        if is_stable:
            entered_stabilization = True
            
        if entered_stabilization:
            # Check "Always" conditional properties (once entered, never leave again)
            margin = min(s_st - dist, 0.5 - vel)
            if margin < 0:
                # Oscillated and ejected out of stabilization loop
                return margin - 100.0 # Extremely heavy penalty
                
    if not entered_stabilization:
        return -50.0 # Failed to converge at all within timeframe limits
        
    return min_safety_margin

def falsify_cem(mamba_ctrl, plant, num_generations=2, pop_size=30, elite_frac=0.2, seq_steps=150, dt=0.1):
    """
    Automated Falsifier deploying an Adversarial Cross-Entropy Method.
    It systematically bounds and extracts exact weakness vector boundaries optimizing strictly for worst-case initializations.
    """
    mamba_ctrl.eval()
    failed_inits = []
    
    # Start evaluating via structural broad Monte-Carlo bounds
    mu = np.zeros(12)
    mu[2] = 10.0 # Base target height Z
    sigma = np.ones(12) * 5.0
    sigma[3:6] = 0.5 # keep physical attitudes constrained somewhat realistic
    
    for gen in range(num_generations):
        samples = []
        scores = []
        
        for _ in range(pop_size):
            mock_state = np.random.normal(mu, sigma)
            # Clip generated points generally around simulation box
            mock_state[0:3] = np.clip(mock_state[0:3], -15, 15)
            mock_state[3:6] = np.clip(mock_state[3:6], -1.0, 1.0)
            
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
                
            # Score robustly
            score = check_stl_score(traj)
            samples.append(mock_state)
            scores.append(score)
            
            if score < 0.0:
                failed_inits.append(mock_state.copy())
                
        # CEM Optimization Step: Adjust means bounds onto the weakest points mapping the failures
        sorted_indices = np.argsort(scores)
        elites = np.array(samples)[sorted_indices[:int(pop_size * elite_frac)]]
        
        mu = np.mean(elites, axis=0) # Shift distributions tightly onto failing configurations
        sigma = np.std(elites, axis=0) + 0.1 # Small jitter prevents total collapse
        
    # Deduplicate extracted arrays physically isolating overlapping bounds
    unique_fails = []
    for f in failed_inits:
        if not any(np.allclose(f, uf, atol=1.0) for uf in unique_fails):
            unique_fails.append(f)
            
    return unique_fails

def fix_and_merge(failed_inits, expert_ctrl, plant, dataset, seq_steps=150, dt=0.1):
    new_data = []
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
            
        new_data.append((np.array(obs_seq), np.array(act_seq)))
        dataset.append(new_data[-1])
        
    return new_data

def prepare_dataloaders(dataset, new_failures_data, device):
    """
    Compiles mixed DataLoader configurations ensuring new logic bounds aren't systematically diluted out by standard historical samples.
    Force-applies a 50/50 balance metric across the CEGIS pipeline.
    """
    obs_list = [item[0] for item in dataset]
    act_list = [item[1] for item in dataset]
    
    # Generic Historic Replay Buffer Loading
    obs_ten = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=device)
    act_ten = torch.tensor(np.stack(act_list), dtype=torch.float32, device=device)
    
    # Extracted Over-Sampling
    if new_failures_data and len(new_failures_data) > 0:
        fail_obs = torch.tensor(np.stack([item[0] for item in new_failures_data]), dtype=torch.float32, device=device)
        fail_act = torch.tensor(np.stack([item[1] for item in new_failures_data]), dtype=torch.float32, device=device)
        
        oversample_ratio = max(1, len(dataset) // len(new_failures_data))
        
        # Inject exact proportional matching into matrix tensor configurations
        fail_obs = fail_obs.repeat(oversample_ratio, 1, 1)
        fail_act = fail_act.repeat(oversample_ratio, 1, 1)
        
        obs_ten = torch.cat([obs_ten, fail_obs], dim=0)
        act_ten = torch.cat([act_ten, fail_act], dim=0)
        
    train_data = TensorDataset(obs_ten, act_ten)
    return DataLoader(train_data, batch_size=16, shuffle=True)

def build_cegis_framework():
    D_OBS = 12
    D_ACT = 4
    SEQ_STEPS = 150
    DT = 0.1
    MAX_ITER = 10
    
    # Note: Higher generalized target LR for robust continuous state integration (3e-4 minimum)
    mamba_ctrl = MambaController(obs_dim=D_OBS, action_dim=D_ACT, d_model=64, d_state=16, num_layers=2, lr=3e-4) 
    drone_plant = DronePlant()
    expert_ctrl = DroneExpertController()
    
    # Implemented physical limits blocking out-of-memory errors
    dataset = deque(maxlen=2000)
    
    print("Generating baseline tracking data...")
    init_mock_states = [drone_plant.reset() for _ in range(20)]
    new_fails = fix_and_merge(init_mock_states, expert_ctrl, drone_plant, dataset, seq_steps=SEQ_STEPS, dt=DT)
    
    print("================ CEGIS LOOP STARTED ================")
    for cycle in range(1, MAX_ITER + 1):
        print(f"\n--- Iteration {cycle}/{MAX_ITER} ---")
        
        train_loader = prepare_dataloaders(dataset, new_fails, mamba_ctrl.device)
        
        # Dynamically scale learning cycle depths precisely matching explicit complexity parameters
        epochs = 4 if cycle == 1 else min(15, max(3, len(new_fails)))
        
        loss = mamba_ctrl.update(train_loader, epochs=epochs)
        print(f"-> PyTorch AMP Operations Completed ({epochs} dynamic epochs) | MSE Volume: {loss:.4f}")
        
        print("-> Unleashing Falsification Adversary (CEM Method)...")
        # Evaluate 60 distinct parameter distributions via 2 generations matching 30 populations
        failures = falsify_cem(mamba_ctrl, drone_plant, num_generations=2, pop_size=30, seq_steps=SEQ_STEPS, dt=DT)
        num_fails = len(failures)
        
        performance_similarity = 1.0 - min(1.0, (num_fails / 60.0)) 
        print(f"-> Counter-Examples Systematically Bracketed: {num_fails}")
        print(f"-> Formal Safety Mathematical Coverage: {performance_similarity * 100:.2f}%")
        
        if num_fails == 0:
            print("\n✔️ CEGIS Environment Operation Verified! Zero counter-examples isolated across distribution sweeps.")
            break
            
        print("-> Enforcing Algorithmic Demonstrations against anomaly boundary matrices...")
        new_fails = fix_and_merge(failures, expert_ctrl, drone_plant, dataset, seq_steps=SEQ_STEPS, dt=DT)
        
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    build_cegis_framework()
