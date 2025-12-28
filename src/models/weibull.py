"""
Weibull Decay Model.

The Weibull distribution survival function:
    f(x) = exp(-(x/λ)^κ)

Where:
    - x = t is time
    - λ > 0: scale parameter (characteristic time)
    - κ > 0: shape parameter

Shape parameter interpretation:
    - κ < 1: Decreasing hazard rate (early failures, then stabilization)
    - κ = 1: Constant hazard rate (exponential decay)
    - κ > 1: Increasing hazard rate (wear-out, aging)

This model is fundamental in:
    - Reliability engineering (failure time analysis)
    - Survival analysis
    - Customer churn modeling
    - Human behavioral persistence

The Weibull is equivalent to the stretched exponential
when viewed appropriately, but parameterization differs.

References:
    - Weibull, W. (1951). J. Appl. Mech. 18, 293-297
    - Rinne, H. (2008). The Weibull Distribution: A Handbook
"""

from typing import ClassVar
import numpy as np
from numpy.typing import NDArray
from scipy.special import gamma as gamma_func

from src.models.base import DecayModel, DecayModelRegistry


@DecayModelRegistry.register
class WeibullModel(DecayModel):
    """
    Weibull decay model (survival function parameterization).

    Model:
        E(t)/E₀ = exp(-(t/λ)^κ)

    Parameters:
        lambda_ (λ): Scale parameter (characteristic life)
        kappa (κ): Shape parameter
        E0: Initial engagement level

    Statistics:
        Mean: E[T] = λ · Γ(1 + 1/κ)
        Variance: Var[T] = λ² · [Γ(1 + 2/κ) - Γ²(1 + 1/κ)]
        Mode: λ · ((κ-1)/κ)^(1/κ) for κ > 1, else 0

    Example:
        >>> model = WeibullModel()
        >>> t = np.linspace(0, 100, 1000)
        >>> E = model.evaluate(t, lambda_=20.0, kappa=0.8)
    """

    name: ClassVar[str] = "weibull"
    description: ClassVar[str] = "Weibull survival function: exp(-(t/λ)^κ)"

    @property
    def parameter_names(self) -> list[str]:
        return ["lambda_", "kappa", "E0"]

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "lambda_": (1e-6, 1e6),  # Scale parameter
            "kappa": (0.01, 5.0),     # Shape parameter
            "E0": (0.01, 10.0)        # Initial value
        }

    def evaluate(
        self,
        t: NDArray[np.float64],
        lambda_: float = 1.0,
        kappa: float = 1.0,
        E0: float = 1.0
    ) -> NDArray[np.float64]:
        """
        Evaluate the Weibull survival function.

        Args:
            t: Time points
            lambda_: Scale parameter (characteristic life)
            kappa: Shape parameter
            E0: Initial engagement level

        Returns:
            Engagement values E(t)
        """
        t = np.asarray(t, dtype=np.float64)
        x = np.clip(t / lambda_, 0, 500)  # Prevent overflow
        return E0 * np.exp(-np.power(x, kappa))

    def gradient(
        self,
        t: NDArray[np.float64],
        lambda_: float = 1.0,
        kappa: float = 1.0,
        E0: float = 1.0
    ) -> dict[str, NDArray[np.float64]]:
        """
        Compute analytical gradients.

        ∂E/∂λ = E · κ · (t/λ)^κ / λ
        ∂E/∂κ = -E · (t/λ)^κ · ln(t/λ)
        ∂E/∂E0 = exp(-(t/λ)^κ)
        """
        t = np.asarray(t, dtype=np.float64)
        x = np.clip(t / lambda_, 1e-10, 500)
        x_kappa = np.power(x, kappa)
        E = E0 * np.exp(-x_kappa)

        # Gradient w.r.t. lambda_
        dE_dlambda = E * kappa * x_kappa / lambda_

        # Gradient w.r.t. kappa
        log_x = np.log(x)
        dE_dkappa = -E * x_kappa * log_x

        # Gradient w.r.t. E0
        dE_dE0 = np.exp(-x_kappa)

        return {
            "lambda_": dE_dlambda,
            "kappa": dE_dkappa,
            "E0": dE_dE0
        }

    def _get_initial_params(
        self,
        t: NDArray[np.float64],
        E: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Generate initial parameter estimates.

        Uses percentile method for robust initialization.
        """
        # Estimate E0
        E0_init = np.max(E[:max(1, len(E) // 10)])
        E_norm = np.clip(E / E0_init, 1e-10, 1)

        # Use 63.2% quantile (characteristic life)
        target_quantile = 1 - np.exp(-1)  # ≈ 0.632
        try:
            lambda_idx = np.argmin(np.abs(E_norm - target_quantile))
            lambda_init = t[lambda_idx] if lambda_idx > 0 else t[len(t) // 3]
        except Exception:
            lambda_init = np.median(t)

        # Estimate kappa from shape of decay
        # Using log-log transformation: ln(-ln(E/E0)) = κ·ln(t/λ)
        valid = (E_norm > 0.01) & (E_norm < 0.99)
        if np.sum(valid) > 10:
            try:
                y = np.log(-np.log(E_norm[valid]))
                x = np.log(t[valid] / lambda_init)
                valid_xy = np.isfinite(x) & np.isfinite(y)
                if np.sum(valid_xy) > 5:
                    kappa_init = np.clip(
                        np.polyfit(x[valid_xy], y[valid_xy], 1)[0],
                        0.2, 3.0
                    )
                else:
                    kappa_init = 1.0
            except Exception:
                kappa_init = 1.0
        else:
            kappa_init = 1.0

        return np.array([lambda_init, kappa_init, E0_init])

    def mean_lifetime(self, lambda_: float, kappa: float) -> float:
        """
        Compute the mean lifetime (expected value).

        E[T] = λ · Γ(1 + 1/κ)
        """
        return lambda_ * gamma_func(1 + 1/kappa)

    def variance(self, lambda_: float, kappa: float) -> float:
        """
        Compute the variance of lifetime.

        Var[T] = λ² · [Γ(1 + 2/κ) - Γ²(1 + 1/κ)]
        """
        g1 = gamma_func(1 + 1/kappa)
        g2 = gamma_func(1 + 2/kappa)
        return lambda_**2 * (g2 - g1**2)

    def hazard_rate(
        self,
        t: NDArray[np.float64],
        lambda_: float,
        kappa: float
    ) -> NDArray[np.float64]:
        """
        Compute the hazard (failure) rate.

        h(t) = (κ/λ) · (t/λ)^(κ-1)

        Interpretation:
            - κ < 1: Decreasing hazard (infant mortality)
            - κ = 1: Constant hazard (random failures)
            - κ > 1: Increasing hazard (wear-out)
        """
        t = np.asarray(t, dtype=np.float64)
        t = np.maximum(t, 1e-10)
        return (kappa / lambda_) * np.power(t / lambda_, kappa - 1)

    def mode(self, lambda_: float, kappa: float) -> float:
        """
        Compute the mode (most likely failure time).

        Mode = λ · ((κ-1)/κ)^(1/κ) for κ > 1
             = 0                  for κ ≤ 1
        """
        if kappa <= 1:
            return 0.0
        return lambda_ * np.power((kappa - 1) / kappa, 1 / kappa)

    def percentile(
        self,
        p: float,
        lambda_: float,
        kappa: float
    ) -> float:
        """
        Compute the p-th percentile of the distribution.

        t_p = λ · (-ln(1-p))^(1/κ)

        Common percentiles:
            - B10 (p=0.10): Early life failures
            - Median (p=0.50): Central tendency
            - B63.2 (p=0.632): Characteristic life

        Args:
            p: Percentile (0 < p < 1)
            lambda_: Scale parameter
            kappa: Shape parameter

        Returns:
            Time at which 100p% of population has "failed"
        """
        if not 0 < p < 1:
            raise ValueError("Percentile must be between 0 and 1")
        return lambda_ * np.power(-np.log(1 - p), 1 / kappa)

    @staticmethod
    def universal_form(
        x: NDArray[np.float64],
        kappa: float = 0.7
    ) -> NDArray[np.float64]:
        """
        The universal scaling function f(x).

        After time normalization t → t/τ(α), all curves
        should collapse onto this single function.

        Note: With x = t/λ already normalized, this is simply
        the standard Weibull with unit scale.

        Args:
            x: Normalized time t/τ(α)
            kappa: Universal shape parameter

        Returns:
            f(x) = exp(-x^κ)
        """
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, 0, 500)
        return np.exp(-np.power(x, kappa))
