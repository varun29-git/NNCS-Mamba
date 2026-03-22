from abc import ABC, abstractmethod
import numpy as np
from typing import Any, List, Tuple

class Plant(ABC):
    """
    Abstract Base Class for a Continuous-Time System (Plant).
    
    The dynamics are defined as:
        x_dot(t) = f(x(t), u(t))
        y(t) = zeta(x(t))
        
    where x is the state, u is the control input, and y is the observation/output.
    """
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Initializes the state x to an initial condition and returns the initial observation y.
        
        Returns:
            np.ndarray: Initial observation y(0).
        """
        pass
    
    @abstractmethod
    def step(self, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Integrates the dynamics over a time step dt using control input u.
        
        Args:
            u (np.ndarray): Control input signal at the current time.
            dt (float): Time duration for integration.
            
        Returns:
            np.ndarray: The new observation y(t + dt) after integration.
        """
        pass

class ExpertController(ABC):
    """
    Abstract Base Class for the Nominal Controller (Teacher).
    
    A discrete-time controller with internal state z:
        z_{k+1} = f_c(z_k, y_k)
        u_k = v(z_k, y_k)
        
    where y_k is the observation at step k, and u_k is the control action.
    """
    
    @abstractmethod
    def reset(self) -> None:
        """
        Clears or initializes the internal state z_0.
        """
        pass
    
    @abstractmethod
    def compute_action(self, y: np.ndarray) -> np.ndarray:
        """
        Takes an observation y and returns the control input u based on the expert policy.
        
        Args:
            y (np.ndarray): The current observation from the plant.
            
        Returns:
            np.ndarray: The optimal control input u.
        """
        pass

class LearnerController(ABC):
    """
    Abstract Base Class for the Neural Network Controller (Student).
    
    Designed for state-space models (like Mamba/S4) that maintain hidden states
    over time for sequential processing.
    """
    
    @abstractmethod
    def reset(self) -> None:
        """
        Clears the hidden state for a new trajectory.
        """
        pass
    
    @abstractmethod
    def forward(self, y: np.ndarray) -> np.ndarray:
        """
        Computes the action during online simulation or inference.
        
        Args:
            y (np.ndarray): Current observation.
            
        Returns:
            np.ndarray: Computed control action u.
        """
        pass
        
    @abstractmethod
    def update(self, dataset: List[Tuple[np.ndarray, np.ndarray]]) -> Any:
        """
        Defines the interface for training the model using a dataset.
        
        Args:
            dataset (List[Tuple[np.ndarray, np.ndarray]]): A collection of (observation, expert_action)
                pairs used for imitation learning.
                
        Returns:
            Any: Training metrics (e.g., loss values).
        """
        pass
