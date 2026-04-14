import numpy as np
import torch
from drone_env import DronePlant, DroneExpertController
from mamba_controller import MambaController

def check_stl(trajectory_y, s_ov=15.0, s_st=2.0) -> bool:
    """
    Evaluates a trajectory against the specified parametric Spatial/Temporal Logic bounds.
    """
    total_steps = len(trajectory_y)
    
    # Evaluate Overshoot constraints (Distance MUST NOT exceed 15 units at ANY point)
    for y in trajectory_y:
        dist = np.linalg.norm(y[0:3]) # L2 distance of [x, y, z] to origin
        if dist > s_ov:
            return False
            
    # Evaluate Transient Time / Stabilization constraints (Must stay within 2.0 eventually)
    # We slice out the last 20% of the trajectory (e.g. 30 steps if total is 150)
    transient_period = int(0.2 * total_steps)
    if transient_period == 0:
        transient_period = 1
        
    tail_trajectory = trajectory_y[-transient_period:]
    for y in tail_trajectory:
        dist = np.linalg.norm(y[0:3])
        if dist > s_st:
            return False
            
    return True

def falsify(mamba_ctrl, plant, num_tests=30, seq_steps=150, dt=0.1):
    """
    Executes parallel tracking missions under heavy "edge-case" environments.
    Returns: List of initial states that violated the STL specifications.
    """
    failed_inits = []
    
    # We temporarily switch to eval mode for offline neural control tracking
    mamba_ctrl.eval()
    
    for _ in range(num_tests):
        # Generate an extreme edge-case initialization point:
        # e.g., Spawns far horizontally, and with massive random velocity
        mock_state = np.zeros(12)
        mock_state[0:3] = np.random.uniform(-12, 12, size=3)
        mock_state[2] += 12 # z height is around 12-24
        mock_state[6:9] = np.random.uniform(-5, 5, size=3) # Extreme velocity
        
        # Manually reset and spawn
        plant.reset()
        plant.state = mock_state.copy()
        plant.time = 0.0
        
        mamba_ctrl.reset()
        
        # Rollout closed-loop sequence
        y = plant.state.copy()
        trajectory_y = [y]
        
        for _ in range(seq_steps):
            u = mamba_ctrl.forward(y)
            y = plant.step(u, dt)
            trajectory_y.append(y)
            
        # Check formal rules:
        if not check_stl(trajectory_y):
            failed_inits.append(mock_state.copy())
            
    return failed_inits
    
def fix_and_merge(failed_inits, expert_ctrl, plant, dataset, seq_steps=150, dt=0.1):
    """
    Operates Expert Controller precisely at points of model failure to extract corrective data.
    """
    for init_state in failed_inits:
        plant.reset()
        plant.state = init_state.copy()
        plant.time = 0.0
        
        expert_ctrl.reset()
        
        obs_seq = []
        act_seq = []
        
        y = plant.state.copy()
        for _ in range(seq_steps):
            u = expert_ctrl.compute_action(y)
            obs_seq.append(y)
            act_seq.append(u)
            y = plant.step(u, dt)
            
        dataset.append((np.array(obs_seq), np.array(act_seq)))

def build_cegis_framework():
    # Model Setup
    D_OBS = 12
    D_ACT = 4
    SEQ_STEPS = 150
    DT = 0.1
    MAX_ITER = 10
    
    # Environment Setup
    mamba_ctrl = MambaController(obs_dim=D_OBS, action_dim=D_ACT, d_model=64, d_state=16, num_layers=2)
    drone_plant = DronePlant()
    expert_ctrl = DroneExpertController()
    dataset = []
    
    # 0. Generate tiny random Initial baseline (~5 demonstrations)
    print("Generating baseline tracking data...")
    init_mock_states = [drone_plant.reset() for _ in range(5)]
    fix_and_merge(init_mock_states, expert_ctrl, drone_plant, dataset, seq_steps=SEQ_STEPS, dt=DT)
    
    print("================ CEGIS LOOP STARTED ================")
    # 1. Main Procedure Loop
    for cycle in range(1, MAX_ITER + 1):
        print(f"\n--- Iteration {cycle}/{MAX_ITER} ---")
        
        # Step 1: Update Mamba via extracted Data Pools
        loss = mamba_ctrl.update(dataset)
        print(f"-> Mamba Updated. Dataset Size: {len(dataset)} | Average MSE: {loss:.4f}")
        
        # Step 2: Systematically trace failures within bounds
        failures = falsify(mamba_ctrl, drone_plant, num_tests=30, seq_steps=SEQ_STEPS, dt=DT)
        num_fails = len(failures)
        
        # Log Similarity Profile
        performance_similarity = 1.0 - (num_fails / 30.0)
        print(f"-> Counter-Examples (Failures) Found: {num_fails}/30")
        print(f"-> Performance Similarity metric: {performance_similarity * 100:.2f}%")
        
        # Step 3/4: Loop controls
        if num_fails == 0:
            print("\n✔️ CEGIS Loop Completed! No further counter-examples discovered.")
            break
            
        print("-> Fixing models by deploying Expert to failure nodes...")
        fix_and_merge(failures, expert_ctrl, drone_plant, dataset, seq_steps=SEQ_STEPS, dt=DT)
        
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    build_cegis_framework()
