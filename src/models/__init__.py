"""
Mathematical Decay Models Module

This module implements the core mathematical models for human engagement decay:

1. Stretched Exponential: f(x) = exp(-x^γ)
   - Describes heterogeneous relaxation processes
   - Common in disordered systems and human behavior

2. Power-Law with Cutoff: f(x) = (1 + x)^(-γ)
   - Scale-free decay with natural regularization
   - Captures heavy-tailed behavior

3. Weibull Distribution: f(x) = exp(-(x/λ)^κ)
   - Flexible failure-time distribution
   - κ < 1: decreasing hazard, κ > 1: increasing hazard

4. Double Exponential: f(x) = A·exp(-x/τ₁) + (1-A)·exp(-x/τ₂)
   - Two-timescale decay process
   - Fast initial decay + slow residual decay

Each model implements:
    - Forward evaluation
    - Gradient computation (for optimization)
    - Parameter bounds and constraints
    - Log-likelihood functions
    - Information criteria (AIC, BIC)
"""

from src.models.base import DecayModel, DecayModelRegistry
from src.models.stretched_exponential import StretchedExponentialModel
from src.models.power_law import PowerLawModel
from src.models.weibull import WeibullModel
from src.models.double_exponential import DoubleExponentialModel
from src.models.mechanistic import MechanisticSDEModel
from src.models.motivation import MotivationScalingModel

__all__ = [
    "DecayModel",
    "DecayModelRegistry",
    "StretchedExponentialModel",
    "PowerLawModel",
    "WeibullModel",
    "DoubleExponentialModel",
    "MechanisticSDEModel",
    "MotivationScalingModel",
]
