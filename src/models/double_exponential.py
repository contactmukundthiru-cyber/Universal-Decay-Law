"""
Double Exponential Decay Model.

The bi-exponential (double exponential) function:
    f(t) = A·exp(-t/τ₁) + (1-A)·exp(-t/τ₂)

Where:
    - τ₁ < τ₂: Fast and slow decay timescales
    - A ∈ (0, 1): Relative amplitude of fast component
    - (1-A): Relative amplitude of slow component

This model captures:
    - Two distinct decay processes
    - Fast initial dropout + slow residual engagement
    - Casual vs committed user dynamics
    - Short-term vs long-term motivation

Physical interpretation:
    - Two subpopulations with different decay rates
    - Mixture of intrinsically and extrinsically motivated users
    - Fast process: novelty-driven, reward-seeking
    - Slow process: habit-formed, intrinsically motivated

Applications:
    - Customer churn with multiple segments
    - Learning decay (procedural vs declarative memory)
    - Social media engagement (viral vs organic)

References:
    - Johnston, W.A. & Dark, V.J. (1986). Annu. Rev. Psychol. 37, 43-75
    - Fiedler, S. & Glöckner, A. (2012). Front. Psychol. 3, 472
"""

from typing import ClassVar
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from src.models.base import DecayModel, DecayModelRegistry


@DecayModelRegistry.register
class DoubleExponentialModel(DecayModel):
    """
    Double exponential (bi-exponential) decay model.

    Model:
        E(t)/E₀ = A·exp(-t/τ₁) + (1-A)·exp(-t/τ₂)

    Parameters:
        tau1: Fast decay timescale (τ₁)
        tau2: Slow decay timescale (τ₂ > τ₁)
        amplitude: Relative weight of fast component (A ∈ (0,1))
        E0: Initial engagement level

    Constraint: τ₂ > τ₁ (enforced during fitting)

    Mean lifetime:
        <τ> = A·τ₁ + (1-A)·τ₂

    Example:
        >>> model = DoubleExponentialModel()
        >>> t = np.linspace(0, 100, 1000)
        >>> E = model.evaluate(t, tau1=5.0, tau2=50.0, amplitude=0.7)
    """

    name: ClassVar[str] = "double_exponential"
    description: ClassVar[str] = "Bi-exponential: A·exp(-t/τ₁) + (1-A)·exp(-t/τ₂)"

    @property
    def parameter_names(self) -> list[str]:
        return ["tau1", "tau2", "amplitude", "E0"]

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "tau1": (1e-6, 1e5),      # Fast timescale
            "tau2": (1e-5, 1e6),      # Slow timescale
            "amplitude": (0.01, 0.99),  # Weight of fast component
            "E0": (0.01, 10.0)         # Initial value
        }

    def evaluate(
        self,
        t: NDArray[np.float64],
        tau1: float = 1.0,
        tau2: float = 10.0,
        amplitude: float = 0.5,
        E0: float = 1.0
    ) -> NDArray[np.float64]:
        """
        Evaluate the double exponential at given time points.

        Args:
            t: Time points
            tau1: Fast decay timescale
            tau2: Slow decay timescale
            amplitude: Weight of fast component
            E0: Initial engagement level

        Returns:
            Engagement values E(t)
        """
        t = np.asarray(t, dtype=np.float64)

        # Ensure tau2 > tau1 for interpretability
        if tau2 < tau1:
            tau1, tau2 = tau2, tau1
            amplitude = 1 - amplitude

        # Prevent numerical overflow
        exp1 = np.exp(-np.clip(t / tau1, 0, 500))
        exp2 = np.exp(-np.clip(t / tau2, 0, 500))

        return E0 * (amplitude * exp1 + (1 - amplitude) * exp2)

    def gradient(
        self,
        t: NDArray[np.float64],
        tau1: float = 1.0,
        tau2: float = 10.0,
        amplitude: float = 0.5,
        E0: float = 1.0
    ) -> dict[str, NDArray[np.float64]]:
        """
        Compute analytical gradients.

        ∂E/∂τ₁ = E₀ · A · (t/τ₁²) · exp(-t/τ₁)
        ∂E/∂τ₂ = E₀ · (1-A) · (t/τ₂²) · exp(-t/τ₂)
        ∂E/∂A = E₀ · [exp(-t/τ₁) - exp(-t/τ₂)]
        ∂E/∂E₀ = A·exp(-t/τ₁) + (1-A)·exp(-t/τ₂)
        """
        t = np.asarray(t, dtype=np.float64)

        exp1 = np.exp(-np.clip(t / tau1, 0, 500))
        exp2 = np.exp(-np.clip(t / tau2, 0, 500))

        dE_dtau1 = E0 * amplitude * (t / tau1**2) * exp1
        dE_dtau2 = E0 * (1 - amplitude) * (t / tau2**2) * exp2
        dE_damplitude = E0 * (exp1 - exp2)
        dE_dE0 = amplitude * exp1 + (1 - amplitude) * exp2

        return {
            "tau1": dE_dtau1,
            "tau2": dE_dtau2,
            "amplitude": dE_damplitude,
            "E0": dE_dE0
        }

    def _get_initial_params(
        self,
        t: NDArray[np.float64],
        E: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Generate initial parameter estimates.

        Uses a heuristic approach based on early and late decay.
        """
        # Estimate E0
        E0_init = np.max(E[:max(1, len(E) // 10)])
        E_norm = np.clip(E / E0_init, 1e-10, 1)

        # Find two characteristic times from decay pattern
        # Early decay rate
        early_idx = len(t) // 5
        if early_idx > 1 and E_norm[early_idx] > 0.1:
            tau1_init = -t[early_idx] / np.log(E_norm[early_idx])
        else:
            tau1_init = t[len(t) // 10] if len(t) > 10 else 1.0

        # Late decay rate
        late_idx = 4 * len(t) // 5
        if late_idx < len(t) and E_norm[late_idx] > 0.01:
            tau2_init = -t[late_idx] / np.log(E_norm[late_idx])
        else:
            tau2_init = t[-1] / 2 if len(t) > 0 else 10.0

        # Ensure ordering
        tau1_init = np.clip(tau1_init, 0.1, 1000)
        tau2_init = np.clip(tau2_init, tau1_init * 2, 10000)

        # Estimate amplitude from early/late ratio
        amplitude_init = 0.7  # Default: most decay is fast

        return np.array([tau1_init, tau2_init, amplitude_init, E0_init])

    def mean_lifetime(
        self,
        tau1: float,
        tau2: float,
        amplitude: float
    ) -> float:
        """
        Compute the mean lifetime.

        <τ> = A·τ₁ + (1-A)·τ₂
        """
        return amplitude * tau1 + (1 - amplitude) * tau2

    def effective_single_exponential(
        self,
        tau1: float,
        tau2: float,
        amplitude: float
    ) -> float:
        """
        Compute effective single-exponential timescale.

        The timescale τ_eff such that a single exponential
        with this rate has the same area under curve.

        τ_eff = A·τ₁ + (1-A)·τ₂ = <τ>
        """
        return self.mean_lifetime(tau1, tau2, amplitude)

    def fast_fraction_at_time(
        self,
        t: float,
        tau1: float,
        tau2: float,
        amplitude: float
    ) -> float:
        """
        Compute fraction of remaining engagement from fast component.

        f(t) = A·exp(-t/τ₁) / [A·exp(-t/τ₁) + (1-A)·exp(-t/τ₂)]

        At t=0: f(0) = A
        As t→∞: f(t) → 0 (slow component dominates)
        """
        exp1 = np.exp(-t / tau1)
        exp2 = np.exp(-t / tau2)
        numerator = amplitude * exp1
        denominator = numerator + (1 - amplitude) * exp2
        return numerator / denominator if denominator > 0 else 0.0

    def half_lives(
        self,
        tau1: float,
        tau2: float,
        amplitude: float
    ) -> dict[str, float]:
        """
        Compute half-lives for the composite and individual components.

        Returns:
            Dictionary with 'fast', 'slow', and 'composite' half-lives.
        """
        from scipy.optimize import brentq

        # Individual component half-lives
        t_half_fast = tau1 * np.log(2)
        t_half_slow = tau2 * np.log(2)

        # Composite half-life (numerical)
        def residual(t):
            return self.evaluate(np.array([t]), tau1, tau2, amplitude, 1.0)[0] - 0.5

        try:
            t_half_composite = brentq(residual, 0, 10 * max(tau1, tau2))
        except Exception:
            t_half_composite = amplitude * t_half_fast + (1 - amplitude) * t_half_slow

        return {
            "fast": t_half_fast,
            "slow": t_half_slow,
            "composite": t_half_composite
        }

    def decompose(
        self,
        t: NDArray[np.float64],
        tau1: float,
        tau2: float,
        amplitude: float,
        E0: float = 1.0
    ) -> dict[str, NDArray[np.float64]]:
        """
        Decompose the signal into fast and slow components.

        Args:
            t: Time points
            tau1, tau2, amplitude, E0: Model parameters

        Returns:
            Dictionary with 'fast', 'slow', and 'total' components.
        """
        t = np.asarray(t, dtype=np.float64)

        fast = E0 * amplitude * np.exp(-t / tau1)
        slow = E0 * (1 - amplitude) * np.exp(-t / tau2)
        total = fast + slow

        return {
            "fast": fast,
            "slow": slow,
            "total": total
        }

    @staticmethod
    def universal_form(
        x: NDArray[np.float64],
        amplitude: float = 0.6,
        ratio: float = 10.0
    ) -> NDArray[np.float64]:
        """
        The universal scaling function f(x).

        For the double exponential, we parameterize by:
        - amplitude: weight of fast component
        - ratio: τ₂/τ₁ ratio

        The universal form is then:
            f(x) = A·exp(-x) + (1-A)·exp(-x/ratio)

        Args:
            x: Normalized time t/τ₁(α)
            amplitude: Universal amplitude parameter
            ratio: Universal timescale ratio

        Returns:
            f(x)
        """
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, 0, 500)
        return amplitude * np.exp(-x) + (1 - amplitude) * np.exp(-x / ratio)
