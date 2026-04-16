import numpy as np
import torch
import sys
import os

# Add parent directory to path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from drone_env import DronePlant, DroneExpertController, CHECKPOINTS
from mamba_learner import MambaController
from cegis_loop import falsify_cem, plot_trajectory

def hunt():
    checkpoint_path = "cegis_cycle_6.pt"
    print(f"[*] Starting Adversarial Hunt for {checkpoint_path}...")
    
    controller = MambaController(obs_dim=12, action_dim=4, d_model=64, d_state=16, num_layers=2)
    controller.load_checkpoint(checkpoint_path)
    controller.eval()
    
    plant = DronePlant()
    expert = DroneExpertController()
    
    # Run CEM Falsifier with more generations and population for deep hunt
    print("[*] Running CEM Falsifier (Hunting for absolute worst case)...")
    failures = falsify_cem(
        controller, plant, expert,
        num_generations=3, 
        pop_size=40, 
        seq_steps=300, 
        dt=0.1
    )
    
    if not failures:
        print("[!] No failures found by CEM. Model might be quite robust from standard spawn distribution.")
        return
        
    print(f"[*] Found {len(failures)} unique failures.")
    
    # Pick the "worst" failure (lowest STL score potentially, but falsify_cem just returns a list)
    # We'll just take the first one and plot it
    worst_init = failures[0]
    
    print("[*] Plotting worst-case trajectory...")
    plant.reset()
    plant.state = worst_init.copy()
    plant.time = 0.0
    controller.reset()
    
    m_traj = []
    y = plant.state.copy()
    for _ in range(300):
        with torch.no_grad():
            u = controller.forward(y)
        y = plant.step(u, 0.1)
        m_traj.append(y)
        
    # Expert from same init
    plant.reset()
    plant.state = worst_init.copy()
    plant.time = 0.0
    expert.reset()
    expert.set_plant_ref(plant)
    e_traj = []
    y = plant.state.copy()
    for _ in range(300):
        u = expert.compute_action(y)
        y = plant.step(u, 0.1)
        e_traj.append(y)
        
    plot_trajectory(m_traj, e_traj, "adversarial_hunt", filename="worst_case_failure.png")
    print("[*] Adversarial plot saved as 'worst_case_failure.png'")

if __name__ == "__main__":
    hunt()
