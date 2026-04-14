import numpy as np
import math
from abstract_env import Plant, ExpertController

class DronePlant(Plant):
    """
    12-dimensional continuous linear drone environment:
    State: [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
    (Simplified linearized model around hover to allow tractable high-level imitation)
    Action: [ax_cmd, ay_cmd, az_cmd, yaw_rate_cmd]
    """
    def __init__(self):
        self.state = np.zeros(12)
        self.time = 0.0
        self.gravity = 9.81
        
    def reset(self) -> np.ndarray:
        # Random initial 3D position [x, y, z], zero orientation & velocities
        self.state = np.zeros(12)
        self.state[0:3] = np.random.uniform(-10, 10, size=3) 
        self.state[2] += 10 # z roughly around +10 to +20 range
        self.time = 0.0
        return self.state.copy()
        
    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        # Physical actuator clipping constraints (Model bounded outputs)
        u = np.clip(u, -20.0, 20.0)
        ax_cmd, ay_cmd, az_cmd, yaw_rate_cmd = u
        
        # Periodic Wind Disturbance (Period = 5s)
        # Note: Wind force is NOT part of the observation! Mamba must infer it.
        wind_gust = 0.0
        if math.sin(2 * math.pi * self.time / 5.0) > 0.8:
            wind_gust = 5.0 # Sudden lateral push in x
            
        # Extract derivatives
        x, y, z = self.state[0:3]
        phi, theta, psi = self.state[3:6]
        vx, vy, vz = self.state[6:9]
        p, q, r = self.state[9:12]
        
        dx = vx
        dy = vy
        dz = vz
        dphi = p
        dtheta = q
        dpsi = r
        
        dvx = self.gravity * theta + wind_gust + ax_cmd
        dvy = -self.gravity * phi + ay_cmd
        dvz = az_cmd
        
        dp = -10.0 * phi  # Inner controller approximations
        dq = -10.0 * theta
        dr = yaw_rate_cmd
        
        # Realism: Inject minor continuous process noise into physical derivatives
        # e.g., representing aerodynamic turbulence, engine variability, IMU instability
        noise = np.random.normal(0, 0.1, size=12)
        
        derivatives = np.array([
            dx, dy, dz, dphi, dtheta, dpsi,
            dvx, dvy, dvz, dp, dq, dr
        ]) + noise
        
        self.state = self.state + derivatives * dt
        self.time += dt
        
        return self.state.copy()

class DroneExpertController(ExpertController):
    """
    A demonstrator demonstrating Sequential Checkpoint Navigation.
    Serves as an optimized geometrical sequence tracker transitioning from PRE_DOCK -> DOCK.
    """
    def __init__(self):
        self.mode = "PRE_DOCK"
        self.pre_dock_pt = np.array([0.0, 0.0, 5.0])
        self.dock_pt = np.array([0.0, 0.0, 0.0])
        
    def reset(self) -> None:
        self.mode = "PRE_DOCK"
        
    def compute_action(self, y: np.ndarray) -> np.ndarray:
        pos = y[0:3]
        vel = y[6:9]
        
        if self.mode == "PRE_DOCK":
            target = self.pre_dock_pt
            if np.linalg.norm(pos - self.pre_dock_pt) < 1.0:
                self.mode = "DOCK"
        else:
            target = self.dock_pt
            
        # PD mathematical path control to target position
        error = target - pos
        v_target = 1.0 * error
        acc_cmd = 3.0 * (v_target - vel) # Stiff aggressive gain bounds
        
        # Enforce realistic bounds on expert output so student doesn't try to learn infinity
        acc_cmd = np.clip(acc_cmd, -15.0, 15.0)
        
        return np.array([acc_cmd[0], acc_cmd[1], acc_cmd[2], 0.0])
