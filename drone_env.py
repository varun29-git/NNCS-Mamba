import numpy as np
import math
from abstract_env import Plant, ExpertController

# ==============================================================================
# Checkpoint definitions (shared between Expert and STL checker)
# ==============================================================================
CHECKPOINT_A = np.array([8.0, 0.0, 12.0])
CHECKPOINT_B = np.array([-5.0, 6.0, 7.0])
CHECKPOINT_C = np.array([3.0, -5.0, 3.0])
DOCK_ORIGIN  = np.array([0.0, 0.0, 0.0])
CHECKPOINTS  = [CHECKPOINT_A, CHECKPOINT_B, CHECKPOINT_C, DOCK_ORIGIN]
CHECKPOINT_RADIUS = 1.5  # proximity threshold to "visit" a checkpoint


class DronePlant(Plant):
    """
    12-dimensional continuous linear drone with TWO hidden dynamics:
    
    1. Periodic Wind Gust (5s period, hidden from observation)
    2. Time-Varying Mass ("leaky fuel tank"):
       - m(t) = max(m_min, m0 - fuel_rate * t)
       - Same thrust F produces acceleration a = F / m(t)
       - Mass is NOT in the observation vector
    
    State: [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
    Action: [Fx_cmd, Fy_cmd, Fz_cmd, yaw_rate_cmd]  (force commands, not accelerations)
    """
    def __init__(self, m0: float = 2.5, fuel_rate: float = 0.02, m_min: float = 1.0):
        self.state = np.zeros(12)
        self.time = 0.0
        self.gravity = 9.81
        
        # Mass dynamics (hidden from observation)
        self.m0 = m0
        self.fuel_rate = fuel_rate
        self.m_min = m_min
        self.mass = m0
        
    @property
    def current_mass(self) -> float:
        """Returns the current mass. Privileged — NOT exposed in observation."""
        return max(self.m_min, self.m0 - self.fuel_rate * self.time)
        
    def reset(self) -> np.ndarray:
        self.state = np.zeros(12)
        self.state[0:3] = np.random.uniform(-10, 10, size=3)
        self.state[2] += 12  # z roughly around +12 to +22 (near checkpoint A height)
        self.time = 0.0
        self.mass = self.m0
        return self.state.copy()
        
    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        # Physical actuator clipping — these are FORCE commands now
        u = np.clip(u, -20.0, 20.0)
        Fx_cmd, Fy_cmd, Fz_cmd, yaw_rate_cmd = u
        
        # Update mass (fuel consumption)
        self.mass = self.current_mass
        
        # Convert force to acceleration: a = F / m(t)
        # This is the key hidden dynamic — same input produces different outputs over time
        ax_cmd = Fx_cmd / self.mass
        ay_cmd = Fy_cmd / self.mass
        az_cmd = Fz_cmd / self.mass
        
        # Periodic Wind Disturbance (Period = 5s, hidden from observation)
        wind_gust = 0.0
        if math.sin(2 * math.pi * self.time / 5.0) > 0.8:
            wind_gust = 5.0 / self.mass  # Wind force also affected by mass
            
        # Extract state derivatives
        x, y, z = self.state[0:3]
        phi, theta, psi = self.state[3:6]
        vx, vy, vz = self.state[6:9]
        p, q, r = self.state[9:12]
        
        dx, dy, dz = vx, vy, vz
        dphi, dtheta, dpsi = p, q, r
        
        dvx = self.gravity * theta + wind_gust + ax_cmd
        dvy = -self.gravity * phi + ay_cmd
        dvz = az_cmd
        
        dp = -10.0 * phi
        dq = -10.0 * theta
        dr = yaw_rate_cmd
        
        # Process noise (aerodynamic turbulence, IMU jitter)
        noise = np.random.normal(0, 0.05, size=12)
        
        derivatives = np.array([
            dx, dy, dz, dphi, dtheta, dpsi,
            dvx, dvy, dvz, dp, dq, dr
        ]) + noise
        
        self.state = self.state + derivatives * dt
        self.time += dt
        
        return self.state.copy()


class DroneExpertController(ExpertController):
    """
    Sequential Checkpoint Expert: A → B → C → DOCK
    
    The expert has PRIVILEGED access to:
    - Which checkpoint is next (internal phase counter)
    - The current mass of the drone (for gain scaling)
    
    Mamba must reconstruct BOTH of these from pure observation history.
    """
    PHASES = ["GOTO_A", "GOTO_B", "GOTO_C", "DOCK"]
    
    def __init__(self):
        self.phase_idx = 0
        self._plant_ref = None  # Will be set externally for mass access
        
    def reset(self) -> None:
        self.phase_idx = 0
        
    def set_plant_ref(self, plant: DronePlant):
        """Give the expert privileged access to the plant's hidden state (mass)."""
        self._plant_ref = plant
        
    def compute_action(self, y: np.ndarray) -> np.ndarray:
        pos = y[0:3]
        vel = y[6:9]
        
        # Determine current target from phase
        target = CHECKPOINTS[self.phase_idx]
        
        # Check if we've reached the current checkpoint → advance phase
        if self.phase_idx < 3 and np.linalg.norm(pos - target) < CHECKPOINT_RADIUS:
            self.phase_idx += 1
            target = CHECKPOINTS[self.phase_idx]
        
        # PD control: position error → desired velocity → desired acceleration
        error = target - pos
        v_target = 1.5 * error  # Proportional position gain
        acc_desired = 3.0 * (v_target - vel)  # Derivative velocity gain
        
        # Convert acceleration back to FORCE using privileged mass knowledge
        # Expert knows the true mass, so its control is always well-calibrated
        mass = self._plant_ref.mass if self._plant_ref is not None else 2.0
        force_cmd = acc_desired * mass
        
        # Clip to actuator limits
        force_cmd = np.clip(force_cmd, -15.0, 15.0)
        
        return np.array([force_cmd[0], force_cmd[1], force_cmd[2], 0.0])
