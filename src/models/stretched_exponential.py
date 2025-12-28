"""
Stretched Exponential (Kohlrausch-Williams-Watts) Decay Model.

The stretched exponential function:
    f(x) = exp(-x^γ)

Where:
    - x = t/τ is the rescaled time
    - γ ∈ (0, 1]: stretching exponent
    - γ = 1 recovers simple exponential decay
    - γ < 1 indicates heterogeneous relaxation

This model is ubiquitous in:
    - Disordered systems (glasses, polymers)
    - Relaxation processes with distributed timescales
    - Human behavioral dynamics

Physical interpretation:
    - Arises from a distribution of relaxation times
    - P(τ) follows a specific distribution
    - Captures "slowing down" of decay at long times

References:
    - Kohlrausch, R. (1854). Ann. Phys. 167, 56-82
    - Williams, G. & Watts, D.C. (1970). Trans. Faraday Soc. 66, 80-85
"""

from typing import ClassVar
import numpy as np
from numpy.typing import NDArray

from src.models.base import DecayModel, DecayModelRegistry


@DecayModelRegistry.register
class StretchedExponentialModel(DecayModel):
    """
    Stretched exponential decay model.

    Model:
        E(t)/E₀ = exp(-(t/τ)^γ)

    Parameters:
        tau (τ): Characteristic decay timescale
        gamma (γ): Stretching exponent (0 < γ ≤ 1)
        E0: Initial engagement level (optional, default=1)

    The mean relaxation time is:
        <τ> = (τ/γ) · Γ(1/γ)

    Where Γ is the gamma function.

    Example:
        >>> model = StretchedExponentialModel()
        >>> t = np.linspace(0, 100, 1000)
        >>> E = model.evaluate(t, tau=10.0, gamma=0.7)
    """

    name: ClassVar[str] = "stretched_exponential"
    description: ClassVar[str] = "Stretched exponential (KWW) decay: exp(-(t/τ)^γ)"

    @property
    def parameter_names(self) -> list[str]:
        return ["tau", "gamma", "E0"]

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "tau": (1e-6, 1e6),    # Characteristic time
            "gamma": (0.01, 2.0),   # Stretching exponent (allow > 1 for compressed)
            "E0": (0.01, 10.0)      # Initial value
        }

    def evaluate(
        self,
        t: NDArray[np.float64],
        tau: float = 1.0,
        gamma: float = 0.5,
        E0: float = 1.0
    ) -> NDArray[np.float64]:
        """
        Evaluate the stretched exponential at given time points.

        Args:
            t: Time points (can be raw or normalized)
            tau: Characteristic decay time
            gamma: Stretching exponent
            E0: Initial engagement level

        Returns:
            Engagement values E(t)
        """
        # Protect against numerical issues
        t = np.asarray(t, dtype=np.float64)
        x = np.clip(t / tau, 0, 500)  # Prevent overflow

        return E0 * np.exp(-np.power(x, gamma))

    def gradient(
        self,
        t: NDArray[np.float64],
        tau: float = 1.0,
        gamma: float = 0.5,
        E0: float = 1.0
    ) -> dict[str, NDArray[np.float64]]:
        """
        Compute analytical gradients for each parameter.

        ∂E/∂τ = E · γ · (t/τ)^γ / τ
        ∂E/∂γ = -E · (t/τ)^γ · ln(t/τ)
        ∂E/∂E0 = exp(-(t/τ)^γ)
        """
        t = np.asarray(t, dtype=np.float64)
        x = np.clip(t / tau, 1e-10, 500)  # Avoid log(0)
        x_gamma = np.power(x, gamma)
        E = E0 * np.exp(-x_gamma)

        # Gradient w.r.t. tau
        dE_dtau = E * gamma * x_gamma / tau

        # Gradient w.r.t. gamma
        log_x = np.log(x)
        dE_dgamma = -E * x_gamma * log_x

        # Gradient w.r.t. E0
        dE_dE0 = np.exp(-x_gamma)

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
        Generate intelligent initial parameter estimates.

        Uses the half-life method and log-log regression.
        """
        # Estimate E0 from early values
        E0_init = np.max(E[:max(1, len(E) // 10)])

        # Normalize and find half-decay point
        E_norm = np.clip(E / E0_init, 1e-10, 1)
        half_idx = np.argmin(np.abs(E_norm - 0.5))
        tau_init = t[half_idx] if half_idx > 0 else t[len(t) // 2]

        # Estimate gamma from log-log slope
        # ln(-ln(E/E0)) = γ·ln(t/τ)
        valid = (E_norm > 0.01) & (E_norm < 0.99)
        if np.sum(valid) > 10:
            y = np.log(-np.log(E_norm[valid]))
            x = np.log(t[valid] / tau_init)
            valid_x = np.isfinite(x) & np.isfinite(y)
            if np.sum(valid_x) > 5:
                gamma_init = np.clip(
                    np.polyfit(x[valid_x], y[valid_x], 1)[0],
                    0.1, 1.5
                )
            else:
                gamma_init = 0.6
        else:
            gamma_init = 0.6

        return np.array([tau_init, gamma_init, E0_init])

    def mean_relaxation_time(self, tau: float, gamma: float) -> float:
        """
        Compute the mean relaxation time <τ>.

        <τ> = (τ/γ) · Γ(1/γ)

        Where Γ is the gamma function.
        """
        from scipy.special import gamma as gamma_func
        return (tau / gamma) * gamma_func(1 / gamma)

    def effective_rate(
        self,
        t: NDArray[np.float64],
        tau: float,
        gamma: float
    ) -> NDArray[np.float64]:
        """
        Compute the instantaneous decay rate.

        k(t) = -d(ln E)/dt = (γ/τ) · (t/τ)^(γ-1)

        This shows the rate decreases with time for γ < 1.
        """
        t = np.asarray(t, dtype=np.float64)
        x = np.clip(t / tau, 1e-10, 500)
        return (gamma / tau) * np.power(x, gamma - 1)

    @staticmethod
    def universal_form(
        x: NDArray[np.float64],
        gamma: float = 0.5
    ) -> NDArray[np.float64]:
        """
        The universal scaling function f(x).

        After time normalization t → t/τ(α), all curves
        should collapse onto this single function.

        Args:
            x: Normalized time t/τ(α)
            gamma: Universal stretching exponent

        Returns:
            f(x) = exp(-x^γ)
        """
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, 0, 500)
        return np.exp(-np.power(x, gamma))
