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
        self.state[2] += 10 # z roughly around 10
        self.time = 0.0
        return self.state.copy()
        
    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        # u = [ax_cmd, ay_cmd, az_cmd, yaw_rate_cmd]
        # x, y, z, phi, theta, psi, vx, vy, vz, p, q, r
        
        ax_cmd, ay_cmd, az_cmd, yaw_rate_cmd = u
        
        # Periodic Wind Disturbance (Period = 5s)
        # Note: Wind force is NOT part of the observation! Mamba must infer it.
        wind_gust = 0.0
        if math.sin(2 * math.pi * self.time / 5.0) > 0.8:
            wind_gust = 5.0 # Sudden lateral push in x
            
        # Simplified linear dynamics
        # Derivatives
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
        
        # Accelerations
        dvx = self.gravity * theta + wind_gust + ax_cmd
        dvy = -self.gravity * phi + ay_cmd
        dvz = az_cmd
        
        # Assuming inner cascade controller tracks angular velocities fast
        dp = -10.0 * phi  # simple restoring force mimicking inner controller
        dq = -10.0 * theta
        dr = yaw_rate_cmd
        
        derivatives = np.array([
            dx, dy, dz, dphi, dtheta, dpsi,
            dvx, dvy, dvz, dp, dq, dr
        ])
        
        self.state = self.state + derivatives * dt
        self.time += dt
        
        return self.state.copy()

class DroneExpertController(ExpertController):
    """
    A demonstrator demonstrating Sequential Checkpoint Navigation.
    It switches modes internally from PRE_DOCK -> DOCK when it arrives at [0, 0, 5].
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
            
        # PD control to target position
        error = target - pos
        v_target = 1.0 * error
        acc_cmd = 2.0 * (v_target - vel)
        
        # Limit the accelerations to prevent unrealistic values
        acc_cmd = np.clip(acc_cmd, -10.0, 10.0)
        
        # Return action [ax, ay, az, yaw_cmd=0]
        return np.array([acc_cmd[0], acc_cmd[1], acc_cmd[2], 0.0])
