"""
Force Encoder Module for Force-Aware Diffusion Policy
Encodes 6D force/torque measurements (fx, fy, fz, mx, my, mz) into feature representations.
"""

import torch
import torch.nn as nn


class ForceEncoder(nn.Module):
    """
    MLP encoder for force/torque data.
    Architecture: Linear -> Swish -> Linear
    """
    
    def __init__(self, input_dim: int = 6, output_dim: int = 256, hidden_dim: int = 128):
        """
        Args:
            input_dim: Dimension of input force/torque data (default: 6 for fx,fy,fz,mx,my,mz)
            output_dim: Dimension of output features (should match RGB encoder output)
            hidden_dim: Dimension of hidden layer
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # Swish activation
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, force_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of force encoder.
        
        Args:
            force_data: Force/torque measurements of shape (batch_size, force_dim)
                       or (batch_size, n_obs_steps, force_dim)
        
        Returns:
            Encoded force features of shape (batch_size, output_dim)
            or (batch_size, n_obs_steps, output_dim)
        """
        return self.network(force_data)
    
    def output_shape(self):
        """Returns the output feature dimension."""
        return (self.output_dim,)

