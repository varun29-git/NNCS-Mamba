import numpy as np
from mamba_learner import MambaController

def test_mamba_training():
    obs_dim = 3
    action_dim = 2
    
    # Initialize Mamba Controller
    controller = MambaController(obs_dim=obs_dim, action_dim=action_dim, d_model=32, d_state=8, num_layers=2)
    
    # Generate some dummy data (e.g., 5 trajectories of length 20)
    dataset = []
    for _ in range(5):
        seq_len = 20
        obs_seq = np.random.randn(seq_len, obs_dim)
        # Dummy "expert": just a basic linear combination
        act_seq = obs_seq[:, :2] * 0.5 + 0.1 
        dataset.append((obs_seq, act_seq))
        
    print("Training Mamba Controller...")
    
    # Train for a few epochs
    for epoch in range(10):
        metrics = controller.update(dataset)
        print(f"Epoch {epoch+1}, Loss: {metrics['train_loss']:.4f}")
        
    print("Testing autoregressive forward pass...")
    controller.reset()
    for t in range(5):
        action = controller.forward(dataset[0][0][t])
        print(f"Step {t}, Action: {action}")
        
if __name__ == "__main__":
    test_mamba_training()
