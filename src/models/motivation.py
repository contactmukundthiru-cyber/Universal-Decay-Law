"""
Motivation Parameter and Timescale Scaling Model.

This module implements the core scaling relationship:
    τ(α) = τ₀ · α^(-β)

Where:
    - α: Intrinsic/extrinsic motivation balance (0 < α < ∞)
    - τ(α): Characteristic decay timescale
    - τ₀: Base timescale
    - β: Scaling exponent

Higher α (more intrinsic motivation) → slower decay → larger τ
Lower α (more extrinsic motivation) → faster decay → smaller τ

Theoretical basis:
    1. Self-Determination Theory (Deci & Ryan)
    2. Motivational Crowding Effect
    3. Overjustification Effect
    4. Intrinsic Motivation stability

The motivation parameter α is estimated from:
    - Extrinsic signals: badges, points, likes, streaks, notifications
    - Intrinsic proxies: self-initiated behavior, consistency, long-term stability

References:
    - Deci, E.L. & Ryan, R.M. (1985). Intrinsic Motivation and Self-Determination
    - Frey, B.S. & Jegen, R. (2001). J. Econ. Surv. 15, 589-611
    - Gneezy, U. et al. (2011). J. Econ. Perspect. 25, 191-210
"""

from dataclasses import dataclass, field
from typing import ClassVar, Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, curve_fit
from scipy.stats import pearsonr, spearmanr


@dataclass
class MotivationEstimate:
    """
    Estimated motivation parameters for a user/behavior.

    Attributes:
        alpha: Intrinsic/extrinsic motivation balance
        alpha_std: Uncertainty in alpha estimate
        intrinsic_score: Raw intrinsic motivation score (0-1)
        extrinsic_score: Raw extrinsic motivation score (0-1)
        components: Dictionary of individual indicator values
        confidence: Overall confidence in the estimate (0-1)
    """
    alpha: float
    alpha_std: float = 0.0
    intrinsic_score: float = 0.5
    extrinsic_score: float = 0.5
    components: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5


@dataclass
class ScalingFitResult:
    """
    Results from fitting the τ(α) scaling relationship.

    Attributes:
        tau0: Base timescale
        beta: Scaling exponent
        tau0_err: Uncertainty in tau0
        beta_err: Uncertainty in beta
        r_squared: Coefficient of determination
        correlation: Pearson correlation coefficient
        p_value: Statistical significance
    """
    tau0: float
    beta: float
    tau0_err: float = 0.0
    beta_err: float = 0.0
    r_squared: float = 0.0
    correlation: float = 0.0
    p_value: float = 1.0


class MotivationScalingModel:
    """
    Model for the motivation-timescale scaling relationship.

    Core equation:
        τ(α) = τ₀ · α^(-β)

    In log-log space:
        log(τ) = log(τ₀) - β · log(α)

    This is a linear relationship with slope -β.

    Example:
        >>> model = MotivationScalingModel()
        >>> alpha = np.array([0.5, 1.0, 2.0, 4.0])
        >>> tau = model.predict_tau(alpha, tau0=10.0, beta=0.5)
        >>> # tau ≈ [14.14, 10.0, 7.07, 5.0]
    """

    name: ClassVar[str] = "motivation_scaling"
    description: ClassVar[str] = "Timescale-motivation scaling: τ(α) = τ₀·α^(-β)"

    @staticmethod
    def predict_tau(
        alpha: NDArray[np.float64],
        tau0: float,
        beta: float
    ) -> NDArray[np.float64]:
        """
        Predict decay timescale from motivation parameter.

        τ(α) = τ₀ · α^(-β)

        Args:
            alpha: Motivation balance parameters
            tau0: Base timescale
            beta: Scaling exponent

        Returns:
            Predicted timescales
        """
        alpha = np.asarray(alpha, dtype=np.float64)
        alpha = np.maximum(alpha, 1e-10)  # Prevent division issues
        return tau0 * np.power(alpha, -beta)

    @staticmethod
    def predict_alpha_from_tau(
        tau: NDArray[np.float64],
        tau0: float,
        beta: float
    ) -> NDArray[np.float64]:
        """
        Inverse: predict motivation from observed timescale.

        α = (τ/τ₀)^(-1/β)

        Args:
            tau: Observed timescales
            tau0: Base timescale
            beta: Scaling exponent

        Returns:
            Inferred motivation parameters
        """
        tau = np.asarray(tau, dtype=np.float64)
        tau = np.maximum(tau, 1e-10)
        return np.power(tau / tau0, -1 / beta)

    def fit(
        self,
        alpha: NDArray[np.float64],
        tau: NDArray[np.float64],
        weights: Optional[NDArray[np.float64]] = None
    ) -> ScalingFitResult:
        """
        Fit the scaling relationship τ(α) = τ₀ · α^(-β).

        Uses weighted least squares in log-log space.

        Args:
            alpha: Motivation parameters
            tau: Observed timescales
            weights: Optional weights for each observation

        Returns:
            ScalingFitResult with fitted parameters
        """
        alpha = np.asarray(alpha, dtype=np.float64)
        tau = np.asarray(tau, dtype=np.float64)

        # Filter valid data
        valid = (alpha > 0) & (tau > 0) & np.isfinite(alpha) & np.isfinite(tau)
        alpha = alpha[valid]
        tau = tau[valid]

        if len(alpha) < 3:
            return ScalingFitResult(tau0=1.0, beta=0.5)

        # Transform to log-log space
        log_alpha = np.log(alpha)
        log_tau = np.log(tau)

        # Weighted linear regression
        if weights is not None:
            weights = weights[valid]
            W = np.diag(weights)
        else:
            W = np.eye(len(alpha))

        # Design matrix: [1, log_alpha]
        X = np.column_stack([np.ones_like(log_alpha), log_alpha])

        # Weighted least squares: (X'WX)^(-1) X'Wy
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ log_tau

        try:
            params = np.linalg.solve(XtWX, XtWy)
            log_tau0 = params[0]
            neg_beta = params[1]

            tau0 = np.exp(log_tau0)
            beta = -neg_beta

            # Residuals and R²
            log_tau_pred = X @ params
            residuals = log_tau - log_tau_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((log_tau - np.mean(log_tau))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Parameter uncertainties
            try:
                sigma2 = ss_res / (len(alpha) - 2)
                cov = sigma2 * np.linalg.inv(XtWX)
                log_tau0_err = np.sqrt(cov[0, 0])
                beta_err = np.sqrt(cov[1, 1])
                tau0_err = tau0 * log_tau0_err  # Delta method
            except Exception:
                tau0_err = 0.0
                beta_err = 0.0

            # Correlation
            corr, p_value = pearsonr(log_alpha, log_tau)

        except Exception:
            tau0 = np.median(tau)
            beta = 0.5
            tau0_err = 0.0
            beta_err = 0.0
            r_squared = 0.0
            corr = 0.0
            p_value = 1.0

        return ScalingFitResult(
            tau0=tau0,
            beta=beta,
            tau0_err=tau0_err,
            beta_err=beta_err,
            r_squared=r_squared,
            correlation=corr,
            p_value=p_value
        )


class MotivationEstimator:
    """
    Estimate motivation parameter α from behavioral indicators.

    The motivation balance α = I/E where:
        - I: Intrinsic motivation score
        - E: Extrinsic motivation score

    Intrinsic indicators (positive):
        - Self-initiated activity (not triggered by notifications)
        - Consistent timing patterns (routine-based)
        - Long-term engagement without external rewards
        - Activity during non-peak hours
        - Depth over breadth of engagement

    Extrinsic indicators (positive):
        - Response to notifications/reminders
        - Streak-driven behavior
        - Badge/point accumulation patterns
        - Social comparison behavior
        - Responsiveness to gamification

    The estimator uses a Bayesian latent variable model to
    infer α from observed behavioral signals.
    """

    def __init__(self):
        # Prior parameters
        self.alpha_prior_mean = 1.0
        self.alpha_prior_std = 0.5

        # Indicator weights (can be learned from data)
        self.intrinsic_weights = {
            "self_initiated_ratio": 1.0,
            "timing_consistency": 0.8,
            "long_term_stability": 1.2,
            "off_peak_ratio": 0.5,
            "depth_over_breadth": 0.7
        }

        self.extrinsic_weights = {
            "notification_response_rate": 1.0,
            "streak_sensitivity": 0.9,
            "badge_pursuit_score": 0.8,
            "social_comparison_index": 0.6,
            "gamification_engagement": 0.7
        }

    def estimate_from_indicators(
        self,
        indicators: dict[str, float]
    ) -> MotivationEstimate:
        """
        Estimate α from behavioral indicators.

        Args:
            indicators: Dictionary of indicator values (0-1 scale)

        Returns:
            MotivationEstimate with α and confidence
        """
        intrinsic_score = 0.0
        intrinsic_weight_sum = 0.0
        extrinsic_score = 0.0
        extrinsic_weight_sum = 0.0

        components = {}

        # Compute intrinsic score
        for key, weight in self.intrinsic_weights.items():
            if key in indicators:
                value = np.clip(indicators[key], 0, 1)
                intrinsic_score += weight * value
                intrinsic_weight_sum += weight
                components[f"intrinsic_{key}"] = value

        # Compute extrinsic score
        for key, weight in self.extrinsic_weights.items():
            if key in indicators:
                value = np.clip(indicators[key], 0, 1)
                extrinsic_score += weight * value
                extrinsic_weight_sum += weight
                components[f"extrinsic_{key}"] = value

        # Normalize
        if intrinsic_weight_sum > 0:
            intrinsic_score /= intrinsic_weight_sum
        else:
            intrinsic_score = 0.5

        if extrinsic_weight_sum > 0:
            extrinsic_score /= extrinsic_weight_sum
        else:
            extrinsic_score = 0.5

        # Compute α = I / E (with regularization)
        epsilon = 0.1
        alpha = (intrinsic_score + epsilon) / (extrinsic_score + epsilon)

        # Confidence based on number of indicators
        n_indicators = len([k for k in indicators if k in
                          list(self.intrinsic_weights.keys()) +
                          list(self.extrinsic_weights.keys())])
        max_indicators = len(self.intrinsic_weights) + len(self.extrinsic_weights)
        confidence = n_indicators / max_indicators

        # Estimate uncertainty
        alpha_std = self.alpha_prior_std * (1 - confidence) + 0.1 * confidence

        return MotivationEstimate(
            alpha=alpha,
            alpha_std=alpha_std,
            intrinsic_score=intrinsic_score,
            extrinsic_score=extrinsic_score,
            components=components,
            confidence=confidence
        )

    def estimate_from_timeseries(
        self,
        timestamps: NDArray[np.float64],
        engagement: NDArray[np.float64],
        external_triggers: Optional[NDArray[np.bool_]] = None
    ) -> MotivationEstimate:
        """
        Estimate α from engagement time series.

        Args:
            timestamps: Unix timestamps of activities
            engagement: Engagement intensity values
            external_triggers: Boolean mask for externally-triggered activities

        Returns:
            MotivationEstimate
        """
        indicators = {}

        # Self-initiated ratio
        if external_triggers is not None:
            indicators["self_initiated_ratio"] = 1 - np.mean(external_triggers)
        else:
            indicators["self_initiated_ratio"] = 0.5

        # Timing consistency (entropy-based)
        if len(timestamps) > 10:
            hours = (timestamps % 86400) / 3600  # Hour of day
            hist, _ = np.histogram(hours, bins=24, density=True)
            hist = hist + 1e-10  # Prevent log(0)
            entropy = -np.sum(hist * np.log(hist))
            max_entropy = np.log(24)
            consistency = 1 - (entropy / max_entropy)
            indicators["timing_consistency"] = consistency

        # Long-term stability (decay of activity variance)
        if len(engagement) > 30:
            early = engagement[:len(engagement)//3]
            late = engagement[2*len(engagement)//3:]
            early_cv = np.std(early) / (np.mean(early) + 1e-10)
            late_cv = np.std(late) / (np.mean(late) + 1e-10)
            stability = 1 / (1 + abs(early_cv - late_cv))
            indicators["long_term_stability"] = stability

        return self.estimate_from_indicators(indicators)


class BayesianMotivationModel:
    """
    Bayesian latent variable model for motivation estimation.

    Uses PyMC for inference. The model structure:

    α ~ LogNormal(μ_α, σ_α)  [prior on motivation]
    τ = τ₀ · α^(-β)          [scaling relationship]
    E(t) = E₀ · f(t/τ)       [decay model]
    y ~ Normal(E(t), σ_y)    [observations]

    Inference yields posterior P(α | data).
    """

    def __init__(
        self,
        decay_model_type: str = "stretched_exponential",
        alpha_prior_mu: float = 0.0,
        alpha_prior_sigma: float = 1.0,
        tau0: float = 10.0,
        beta: float = 0.5
    ):
        self.decay_model_type = decay_model_type
        self.alpha_prior_mu = alpha_prior_mu
        self.alpha_prior_sigma = alpha_prior_sigma
        self.tau0 = tau0
        self.beta = beta

    def fit(
        self,
        t: NDArray[np.float64],
        E: NDArray[np.float64],
        draws: int = 1000,
        tune: int = 500,
        chains: int = 2
    ) -> dict:
        """
        Fit the Bayesian model using MCMC.

        Args:
            t: Time points
            E: Observed engagement
            draws: Number of posterior samples
            tune: Number of tuning samples
            chains: Number of MCMC chains

        Returns:
            Dictionary with posterior samples and diagnostics
        """
        try:
            import pymc as pm
            import arviz as az
        except ImportError:
            # Fallback to simple estimation
            return self._simple_estimate(t, E)

        with pm.Model() as model:
            # Prior on log(α)
            log_alpha = pm.Normal("log_alpha",
                                 mu=self.alpha_prior_mu,
                                 sigma=self.alpha_prior_sigma)
            alpha = pm.Deterministic("alpha", pm.math.exp(log_alpha))

            # Derived timescale
            tau = pm.Deterministic("tau",
                                   self.tau0 * pm.math.pow(alpha, -self.beta))

            # Decay model parameters
            if self.decay_model_type == "stretched_exponential":
                gamma = pm.Beta("gamma", alpha=2, beta=2)
                x = t / tau
                E_pred = pm.math.exp(-pm.math.pow(x, gamma))
            else:
                # Default exponential
                E_pred = pm.math.exp(-t / tau)

            # Normalize
            E0 = pm.HalfNormal("E0", sigma=1)
            E_model = E0 * E_pred

            # Likelihood
            sigma_y = pm.HalfNormal("sigma_y", sigma=0.1)
            pm.Normal("obs", mu=E_model, sigma=sigma_y, observed=E)

            # Sample
            trace = pm.sample(draws=draws, tune=tune, chains=chains,
                            return_inferencedata=True, progressbar=False)

        # Extract results
        posterior = trace.posterior
        results = {
            "alpha_mean": float(posterior["alpha"].mean()),
            "alpha_std": float(posterior["alpha"].std()),
            "alpha_samples": posterior["alpha"].values.flatten(),
            "tau_mean": float(posterior["tau"].mean()),
            "tau_std": float(posterior["tau"].std()),
            "diagnostics": az.summary(trace),
            "trace": trace
        }

        return results

    def _simple_estimate(
        self,
        t: NDArray[np.float64],
        E: NDArray[np.float64]
    ) -> dict:
        """
        Simple fallback estimation when PyMC is not available.

        Uses curve fitting to estimate tau, then invert for alpha.
        """
        from src.models.stretched_exponential import StretchedExponentialModel

        model = StretchedExponentialModel()
        fit_result = model.fit(t, E)

        if fit_result.converged:
            tau_est = fit_result.parameters.get("tau", self.tau0)
            alpha_est = MotivationScalingModel.predict_alpha_from_tau(
                np.array([tau_est]), self.tau0, self.beta
            )[0]
        else:
            alpha_est = 1.0
            tau_est = self.tau0

        return {
            "alpha_mean": alpha_est,
            "alpha_std": 0.2,
            "alpha_samples": np.array([alpha_est]),
            "tau_mean": tau_est,
            "tau_std": tau_est * 0.1,
            "diagnostics": None,
            "trace": None
        }
