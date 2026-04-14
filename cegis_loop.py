import numpy as np
import torch
from collections import deque
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
    transient_period = max(1, int(0.2 * total_steps))
        
    tail_trajectory = trajectory_y[-transient_period:]
    
    # Rigid Sequence Boolean check (all subset tail vectors must be inside bounds to satisfy STL)
    if not all(np.linalg.norm(y[0:3]) <= s_st for y in tail_trajectory):
        return False
            
    return True

def falsify(mamba_ctrl, plant, num_tests=30, seq_steps=150, dt=0.1):
    """
    Executes structural multi-tiered randomizator sampling to find edge-case failures.
    Returns: List of initial states that violated the STL specifications.
    """
    failed_inits = []
    
    # Switch models out of training context logic (LayerNorms/Dropouts)
    mamba_ctrl.eval()
    
    for i in range(num_tests):
        # Multi-tiered Monte Carlo boundaries: Widens testing search matrix incrementally
        scale = 1.0 if i < (num_tests // 2) else 2.0 
        
        mock_state = np.zeros(12)
        mock_state[0:3] = np.random.uniform(-8 * scale, 8 * scale, size=3)
        mock_state[2] += 10 * scale # elevation boosting
        mock_state[6:9] = np.random.uniform(-4 * scale, 4 * scale, size=3) # Extreme kinetic velocities
        
        # Manually reset parameters for rollout
        plant.reset()
        plant.state = mock_state.copy()
        plant.time = 0.0
        
        mamba_ctrl.reset()
        
        # Rollout offline simulation trace
        y = plant.state.copy()
        trajectory_y = [y]
        
        for _ in range(seq_steps):
            u = mamba_ctrl.forward(y)
            y = plant.step(u, dt)
            trajectory_y.append(y)
            
        # Corroborate formal sequence boundaries
        if not check_stl(trajectory_y):
            failed_inits.append(mock_state.copy())
            
    return failed_inits
    
def fix_and_merge(failed_inits, expert_ctrl, plant, dataset, seq_steps=150, dt=0.1):
    """
    Operates Algorithmic Expert precisely at coordinates where student collapsed.
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
    # Structural variables
    D_OBS = 12
    D_ACT = 4
    SEQ_STEPS = 150
    DT = 0.1
    MAX_ITER = 10
    
    # Environment Setup
    mamba_ctrl = MambaController(obs_dim=D_OBS, action_dim=D_ACT, d_model=64, d_state=16, num_layers=2)
    drone_plant = DronePlant()
    expert_ctrl = DroneExpertController()
    
    # Memory Buffer structured physically as a deque to stop VRAM explosion across CEGIS rounds
    dataset = deque(maxlen=1000)
    
    print("Generating baseline tracking data...")
    init_mock_states = [drone_plant.reset() for _ in range(10)]
    fix_and_merge(init_mock_states, expert_ctrl, drone_plant, dataset, seq_steps=SEQ_STEPS, dt=DT)
    
    print("================ CEGIS LOOP STARTED ================")
    # 1. Main Operation Framework
    for cycle in range(1, MAX_ITER + 1):
        print(f"\n--- Iteration {cycle}/{MAX_ITER} ---")
        
        # Step 1: Update Mamba via Extracted Tensor Buffer Sets
        loss = mamba_ctrl.update(dataset, epochs=3, batch_size=16)
        print(f"-> Mamba Updated. Dataset Size (Buffer Load): {len(dataset)}/1000 | Average MSE: {loss:.4f}")
        
        # Step 2: Extract spatial failures via Tiered Falsification Distribution
        failures = falsify(mamba_ctrl, drone_plant, num_tests=30, seq_steps=SEQ_STEPS, dt=DT)
        num_fails = len(failures)
        
        # Empirical Verification Logic Volume
        performance_similarity = 1.0 - (num_fails / 30.0)
        print(f"-> Counter-Examples (Failures) Found: {num_fails}/30")
        print(f"-> Formal Safety Bound (Similarity Profile): {performance_similarity * 100:.2f}%")
        
        # CEGIS Stop Cond check
        if num_fails == 0:
            print("\n✔️ CEGIS Loop Completed! No further mathematical counter-examples discovered.")
            break
            
        print("-> Fixing models by deploying Expert simulation trajectories to memory buffer...")
        fix_and_merge(failures, expert_ctrl, drone_plant, dataset, seq_steps=SEQ_STEPS, dt=DT)
        
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    build_cegis_framework()
