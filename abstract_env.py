from abc import ABC, abstractmethod
import numpy as np
from typing import Any, List, Tuple


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

# ------------------------------------------------------------------------------
