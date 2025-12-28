"""
Power-Law Decay Model with Cutoff.

The power-law function with regularization:
    f(x) = (1 + x)^(-γ)

Where:
    - x = t/τ is the rescaled time
    - γ > 0: decay exponent
    - The +1 regularizes behavior at x=0

This model captures:
    - Scale-free decay dynamics
    - Heavy-tailed behavior (slower than exponential)
    - Attention dynamics in social systems

The regularization (1+x) vs pure x^(-γ):
    - Finite at t=0
    - Smooth transition from initial plateau
    - Better numerical stability

Physical interpretation:
    - Power-law memory in reinforcement
    - Lévy-flight attention patterns
    - Self-organized criticality

References:
    - Barabási, A.-L. (2005). Nature 435, 207-211
    - Malmgren, R.D. et al. (2008). Science 325, 1696-1700
"""

from typing import ClassVar
import numpy as np
from numpy.typing import NDArray

from src.models.base import DecayModel, DecayModelRegistry


@DecayModelRegistry.register
class PowerLawModel(DecayModel):
    """
    Power-law decay model with regularization at origin.

    Model:
        E(t)/E₀ = (1 + t/τ)^(-γ)

    Parameters:
        tau (τ): Characteristic timescale
        gamma (γ): Decay exponent (γ > 0)
        E0: Initial engagement level

    For large t:
        E(t) ~ t^(-γ)

    Half-life:
        t_{1/2} = τ · (2^(1/γ) - 1)

    Example:
        >>> model = PowerLawModel()
        >>> t = np.linspace(0, 100, 1000)
        >>> E = model.evaluate(t, tau=10.0, gamma=1.5)
    """

    name: ClassVar[str] = "power_law"
    description: ClassVar[str] = "Power-law decay: (1 + t/τ)^(-γ)"

    @property
    def parameter_names(self) -> list[str]:
        return ["tau", "gamma", "E0"]

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "tau": (1e-6, 1e6),     # Characteristic time
            "gamma": (0.01, 10.0),  # Decay exponent
            "E0": (0.01, 10.0)      # Initial value
        }

    def evaluate(
        self,
        t: NDArray[np.float64],
        tau: float = 1.0,
        gamma: float = 1.0,
        E0: float = 1.0
    ) -> NDArray[np.float64]:
        """
        Evaluate the power-law decay at given time points.

        Args:
            t: Time points
            tau: Characteristic timescale
            gamma: Decay exponent
            E0: Initial engagement level

        Returns:
            Engagement values E(t)
        """
        t = np.asarray(t, dtype=np.float64)
        x = 1.0 + t / tau
        return E0 * np.power(x, -gamma)

    def gradient(
        self,
        t: NDArray[np.float64],
        tau: float = 1.0,
        gamma: float = 1.0,
        E0: float = 1.0
    ) -> dict[str, NDArray[np.float64]]:
        """
        Compute analytical gradients for each parameter.

        ∂E/∂τ = E · γ · (t/τ²) / (1 + t/τ)
        ∂E/∂γ = -E · ln(1 + t/τ)
        ∂E/∂E0 = (1 + t/τ)^(-γ)
        """
        t = np.asarray(t, dtype=np.float64)
        x = 1.0 + t / tau
        E = E0 * np.power(x, -gamma)

        # Gradient w.r.t. tau
        dE_dtau = E * gamma * (t / tau**2) / x

        # Gradient w.r.t. gamma
        dE_dgamma = -E * np.log(x)

        # Gradient w.r.t. E0
        dE_dE0 = np.power(x, -gamma)

        return {
            "tau": dE_dtau,
            "gamma": dE_dgamma,
            "E0": dE_dE0
        }

    def _get_initial_params(
        self,
        t: NDArray[np.float64],
        E: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Generate initial parameter estimates.

        Uses log-log regression for power-law exponent.
        """
        # Estimate E0 from early values
        E0_init = np.max(E[:max(1, len(E) // 10)])

        # Estimate gamma from log-log slope
        E_norm = np.clip(E / E0_init, 1e-10, 1)
        valid = (t > 0) & (E_norm > 0.01)

        if np.sum(valid) > 10:
            # For large t: E/E0 ≈ (t/τ)^(-γ)
            # ln(E/E0) ≈ -γ·ln(t) + γ·ln(τ)
            log_t = np.log(t[valid])
            log_E = np.log(E_norm[valid])
            slope, intercept = np.polyfit(log_t, log_E, 1)
            gamma_init = np.clip(-slope, 0.1, 5.0)
            tau_init = np.clip(np.exp(intercept / gamma_init), 0.1, 1000)
        else:
            gamma_init = 1.0
            tau_init = np.median(t)

        return np.array([tau_init, gamma_init, E0_init])

    def half_life(self, tau: float, gamma: float) -> float:
        """
        Compute the half-life of engagement.

        t_{1/2} = τ · (2^(1/γ) - 1)

        Time at which E(t) = E₀/2.
        """
        return tau * (np.power(2, 1/gamma) - 1)

    def effective_rate(
        self,
        t: NDArray[np.float64],
        tau: float,
        gamma: float
    ) -> NDArray[np.float64]:
        """
        Compute the instantaneous decay rate.

        k(t) = -d(ln E)/dt = γ / (τ + t)

        Shows hyperbolic decay of the rate itself.
        """
        t = np.asarray(t, dtype=np.float64)
        return gamma / (tau + t)

    @staticmethod
    def universal_form(
        x: NDArray[np.float64],
        gamma: float = 1.0
    ) -> NDArray[np.float64]:
        """
        The universal scaling function f(x).

        After time normalization t → t/τ(α), all curves
        should collapse onto this single function.

        Args:
            x: Normalized time t/τ(α)
            gamma: Universal decay exponent

        Returns:
            f(x) = (1 + x)^(-γ)
        """
        x = np.asarray(x, dtype=np.float64)
        return np.power(1.0 + x, -gamma)


@DecayModelRegistry.register
class PurePowerLawModel(DecayModel):
    """
    Pure power-law decay (for comparison).

    Model:
        E(t)/E₀ = (t/τ)^(-γ)  for t > t_min
        E(t) = E₀             for t ≤ t_min

    This is the scale-free form without regularization,
    requiring a lower cutoff to handle t=0.
    """

    name: ClassVar[str] = "pure_power_law"
    description: ClassVar[str] = "Pure power-law: (t/τ)^(-γ) with lower cutoff"

    @property
    def parameter_names(self) -> list[str]:
        return ["tau", "gamma", "E0", "t_min"]

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "tau": (1e-6, 1e6),
            "gamma": (0.01, 10.0),
            "E0": (0.01, 10.0),
            "t_min": (1e-6, 100.0)
        }

    def evaluate(
        self,
        t: NDArray[np.float64],
        tau: float = 1.0,
        gamma: float = 1.0,
        E0: float = 1.0,
        t_min: float = 1.0
    ) -> NDArray[np.float64]:
        """Evaluate pure power-law with lower cutoff."""
        t = np.asarray(t, dtype=np.float64)
        result = np.where(
            t > t_min,
            E0 * np.power(t / tau, -gamma),
            E0
        )
        return result

    def gradient(
        self,
        t: NDArray[np.float64],
        tau: float = 1.0,
        gamma: float = 1.0,
        E0: float = 1.0,
        t_min: float = 1.0
    ) -> dict[str, NDArray[np.float64]]:
        """Compute gradients (analytically where possible)."""
        t = np.asarray(t, dtype=np.float64)
        mask = t > t_min
        E = self.evaluate(t, tau, gamma, E0, t_min)

        dE_dtau = np.where(mask, E * gamma / tau, 0.0)
        dE_dgamma = np.where(mask, -E * np.log(np.maximum(t / tau, 1e-10)), 0.0)
        dE_dE0 = np.where(mask, np.power(t / tau, -gamma), 1.0)
        dE_dt_min = np.zeros_like(t)  # Discontinuous

        return {
            "tau": dE_dtau,
            "gamma": dE_dgamma,
            "E0": dE_dE0,
            "t_min": dE_dt_min
        }
