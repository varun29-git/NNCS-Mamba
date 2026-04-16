import numpy as np
import torch
import argparse
from pathlib import Path
import sys
import os

# Add parent directory to path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from drone_env import DronePlant, DroneExpertController, CHECKPOINTS, CHECKPOINT_RADIUS
from mamba_learner import MambaController
from cegis_loop import check_stl_score

def analyze_missions(checkpoint_path, num_missions=100, seq_steps=300, dt=0.1):
    print(f"[*] Analyzing {num_missions} missions for checkpoint: {checkpoint_path}")
    
    controller = MambaController(obs_dim=12, action_dim=4, d_model=64, d_state=16, num_layers=2)
    try:
        controller.load_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"[!] Failed to load checkpoint: {e}")
        return None
        
    controller.eval()
    plant = DronePlant()
    
    # Milestone counters
    milestones = {
        "A": 0,
        "B": 0,
        "C": 0,
        "Dock": 0
    }
    
    # Termination reasons
    reasons = {
        "Safety (Overshoot)": 0,
        "Attitude (Instability)": 0,
        "Timed Out": 0,
        "Success": 0
    }
    
    # Track "Drop-off" points (where it failed)
    drop_off = {
        "Before A": 0,
        "After A, Before B": 0,
        "After B, Before C": 0,
        "After C, Before Dock": 0
    }
    
    for i in range(num_missions):
        y = plant.reset()
        controller.reset()
        
        reached = [False, False, False, False] # A, B, C, Dock
        termination = "Timed Out"
        
        for t in range(seq_steps):
            with torch.no_grad():
                u = controller.forward(y)
            y = plant.step(u, dt)
            
            pos = y[0:3]
            att = y[3:5] # roll, pitch
            
            # Check milestones sequentially
            if not reached[0] and np.linalg.norm(pos - CHECKPOINTS[0]) < CHECKPOINT_RADIUS:
                reached[0] = True
            elif reached[0] and not reached[1] and np.linalg.norm(pos - CHECKPOINTS[1]) < CHECKPOINT_RADIUS:
                reached[1] = True
            elif reached[1] and not reached[2] and np.linalg.norm(pos - CHECKPOINTS[2]) < CHECKPOINT_RADIUS:
                reached[2] = True
            elif reached[2] and not reached[3] and np.linalg.norm(pos - CHECKPOINTS[3]) < CHECKPOINT_RADIUS:
                reached[3] = True
                termination = "Success"
                break
                
            # Check safety
            if np.linalg.norm(pos) > 25.0:
                termination = "Safety (Overshoot)"
                break
            if np.any(np.abs(att) > 1.0):
                termination = "Attitude (Instability)"
                break
        
        # Log milestones
        if reached[0]: milestones["A"] += 1
        if reached[1]: milestones["B"] += 1
        if reached[2]: milestones["C"] += 1
        if reached[3]: milestones["Dock"] += 1
        
        # Log termination
        reasons[termination] += 1
        
        # Log drop-off
        if not reached[0]:
            drop_off["Before A"] += 1
        elif not reached[1]:
            drop_off["After A, Before B"] += 1
        elif not reached[2]:
            drop_off["After B, Before C"] += 1
        elif not reached[3]:
            drop_off["After C, Before Dock"] += 1
            
        if (i+1) % 20 == 0:
            print(f"    ...processed {i+1}/{num_missions}")
            
    return {
        "milestones": milestones,
        "reasons": reasons,
        "drop_off": drop_off
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--missions", type=int, default=100)
    args = parser.parse_args()
    
    results = analyze_missions(args.checkpoint, num_missions=args.missions)
    
    if results:
        print("\n" + "="*40)
        print("          DIAGNOSTIC REPORT")
        print("="*40)
        
        print("\n--- MILESTONE SUCCESS RATES ---")
        for k, v in results["milestones"].items():
            print(f"  Reached {k:4}: {v:3} ({v/args.missions:4.0%})")
            
        print("\n--- DROP-OFF POINTS (Where it failed) ---")
        for k, v in results["drop_off"].items():
            print(f"  {k:20}: {v:3} ({v/args.missions:4.0%})")
            
        print("\n--- TERMINATION REASONS ---")
        for k, v in results["reasons"].items():
            print(f"  {k:20}: {v:3} ({v/args.missions:4.0%})")
        print("="*40)

if __name__ == "__main__":
    main()
