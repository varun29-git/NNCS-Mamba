import numpy as np
from abstract_env import Plant, ExpertController

class SimpleLinearPlant(Plant):
    """
    A simple 2D linear system:
    x_dot = A*x + B*u
    """
    def __init__(self):
        # A simple stable system
        self.A = np.array([[-0.5, 0.1], 
                           [-0.1, -0.5]])
        self.B = np.array([[1.0, 0.0],
                           [0.0, 1.0]])
        self.state = np.array([1.0, 1.0])
        
    def reset(self) -> np.ndarray:
        self.state = np.array([1.0, 1.0]) + np.random.randn(2) * 0.1
        return self.state.copy()
        
    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        # Euler integration
        x_dot = self.A @ self.state + self.B @ u
        self.state = self.state + x_dot * dt
        return self.state.copy()

class ProportionalExpert(ExpertController):
    """
    A simple P-controller as an expert targeting origin stabilization.
    u = -Kp * y
    """
    def __init__(self):
        self.Kp = np.array([[2.0, 0.0],
                            [0.0, 2.0]])
                            
    def reset(self) -> None:
        pass
        
    def compute_action(self, y: np.ndarray) -> np.ndarray:
        return -self.Kp @ y

if __name__ == "__main__":
    from mamba_controller import MambaController
    
    plant = SimpleLinearPlant()
    expert = ProportionalExpert()
    mamba_ctrl = MambaController(obs_dim=2, action_dim=2, d_model=16, d_state=4, num_layers=1)
    
    # 1. Collect expert trajectories
    print("Collecting expert trajectories...")
    dataset = []
    dt = 0.1
    for _ in range(50):
        y = plant.reset()
        expert.reset()
        
        obs_seq = []
        act_seq = []
        
        for _ in range(30):
            u = expert.compute_action(y)
            obs_seq.append(y)
            act_seq.append(u)
            y = plant.step(u, dt)
            
        dataset.append((np.array(obs_seq), np.array(act_seq)))
        
    # 2. Train Mamba Controller
    print("Training Mamba model on expert trajectories...")
    for epoch in range(20):
        loss = mamba_ctrl.update(dataset)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
    # 3. Test Closed-Loop Runtime of Mamba
    print("Evaluating closed loop performance...")
    y = plant.reset()
    mamba_ctrl.reset()
    
    print(f"Initial state: {y}")
    for step_idx in range(5):
        u = mamba_ctrl.forward(y)
        y = plant.step(u, dt)
        print(f"Step {step_idx}: action={u}, new_state={y}")
