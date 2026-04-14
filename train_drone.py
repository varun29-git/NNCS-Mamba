import numpy as np
import torch
from drone_env import DronePlant, DroneExpertController
from mamba_learner import MambaController

def evaluate_drone():
    # 12D observation space, 4D continuous action
    mamba_ctrl = MambaController(obs_dim=12, action_dim=4, d_model=64, d_state=16, num_layers=2)
    drone_plant = DronePlant()
    expert_ctrl = DroneExpertController()
    
    print("Collecting sequential expert demonstrations for 3D Docking...")
    # Generate 50 trajectories
    dataset = []
    dt = 0.1
    seq_steps = 150 # 15 seconds per trajectory
    
    for _ in range(50):
        y = drone_plant.reset()
        expert_ctrl.reset()
        
        obs_seq = []
        act_seq = []
        
        for _ in range(seq_steps):
            u = expert_ctrl.compute_action(y)
            obs_seq.append(y)
            act_seq.append(u)
            
            y = drone_plant.step(u, dt)
            
        dataset.append((np.array(obs_seq), np.array(act_seq)))
        
    print(f"Collected {len(dataset)} trajectories. Sequence length: {seq_steps} steps.")
    print("Training Mamba sequence model...")
    # Train Mamba natively across sequence representations
    for epoch in range(1, 21):
        loss = mamba_ctrl.update(dataset)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Sequence MSE Loss: {loss:.4f}")
            
    print("\n--- ZERO-SHOT CLOSED-LOOP EVALUATION ---")
    print("Mamba evaluates online tracking with hidden state persistence.")
    
    y = drone_plant.reset()
    mamba_ctrl.reset() # init InferenceParams and memory
    
    # We will log the progress directly
    print(f"Time:  0.0s | Initial Position: z={y[2]:.2f}")
    
    for t in range(seq_steps):
        # We process inputs autoregressively, one strictly by one!
        # Memoryless controllers would fail to remember the Pre-Dock checkpoint
        u = mamba_ctrl.forward(y)
        y = drone_plant.step(u, dt)
        
        if (t+1) % 30 == 0:
            time_val = (t+1) * dt
            # Print current position z-axis (which goes from ~10.0 to 5.0 (PreDock) to 0.0 (Dock))
            print(f"Time: {time_val:4.1f}s | Action: az={u[2]:.2f} | Current pos: [x={y[0]:.2f}, y={y[1]:.2f}, z={y[2]:.2f}]")

if __name__ == "__main__":
    # Ensure reproducibility for tests
    np.random.seed(42)
    torch.manual_seed(42)
    evaluate_drone()
