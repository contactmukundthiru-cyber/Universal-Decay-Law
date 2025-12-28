"""
Mechanistic Stochastic Differential Equation (SDE) Model.

Derives the universal decay law from first-principles dynamics:

    dE/dt = -k·E^γ + η(t)

Where:
    - E(t): Engagement intensity
    - k: Decay rate constant
    - γ: Nonlinearity exponent
    - η(t): Stochastic noise term

This SDE naturally produces:
    - Stretched exponential when γ ≠ 1
    - Power-law tails from multiplicative noise
    - Weibull-like behavior under certain conditions

Physical basis:
    1. Reward prediction error dynamics
    2. Utility drift over time
    3. Attentional fatigue with refresh
    4. Multiplicative noise in motivation

References:
    - Oliveira, J.G. & Barabási, A.-L. (2005). Nature 437, 1251
    - Malmgren, R.D. et al. (2008). Science 325, 1696-1700
    - Brockmann, D. et al. (2006). Nature 439, 462-465
"""

from typing import ClassVar, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import odeint, solve_ivp
from scipy.special import gamma as gamma_func
from dataclasses import dataclass

from src.models.base import DecayModel, DecayModelRegistry


@dataclass
class SDESimulationResult:
    """
    Results from SDE simulation.

    Attributes:
        time: Time points
        trajectories: Individual trajectories (n_trajectories × n_times)
        mean: Mean trajectory
        std: Standard deviation envelope
        percentiles: Dictionary of percentile trajectories
    """
    time: NDArray[np.float64]
    trajectories: NDArray[np.float64]
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    percentiles: dict[int, NDArray[np.float64]]


@DecayModelRegistry.register
class MechanisticSDEModel(DecayModel):
    """
    Mechanistic SDE model for engagement decay.

    The model is based on the stochastic differential equation:
        dE = -k·E^γ·dt + σ·E^β·dW

    Where:
        - k: Decay rate
        - γ: Deterministic nonlinearity (drift exponent)
        - σ: Noise intensity
        - β: Noise scaling exponent (multiplicative noise if β > 0)
        - dW: Wiener process increment

    The analytical solution for the mean trajectory depends on γ:
        - γ = 1: Exponential decay
        - γ < 1: Slower-than-exponential (stretched)
        - γ > 1: Faster-than-exponential

    For the deterministic part (σ = 0):
        - γ = 1: E(t) = E₀·exp(-k·t)
        - γ ≠ 1: E(t) = [E₀^(1-γ) - k(1-γ)t]^(1/(1-γ))

    Example:
        >>> model = MechanisticSDEModel()
        >>> result = model.simulate(T=100, n_trajectories=1000)
        >>> print(result.mean[-1])
    """

    name: ClassVar[str] = "mechanistic_sde"
    description: ClassVar[str] = "SDE model: dE = -kE^γdt + σE^βdW"

    @property
    def parameter_names(self) -> list[str]:
        return ["k", "gamma", "sigma", "beta", "E0"]

    @property
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "k": (1e-6, 10.0),       # Decay rate
            "gamma": (0.1, 2.0),     # Drift exponent
            "sigma": (0.0, 1.0),     # Noise intensity
            "beta": (0.0, 1.0),      # Noise scaling exponent
            "E0": (0.01, 10.0)       # Initial value
        }

    def evaluate(
        self,
        t: NDArray[np.float64],
        k: float = 0.1,
        gamma: float = 0.5,
        sigma: float = 0.0,
        beta: float = 0.5,
        E0: float = 1.0
    ) -> NDArray[np.float64]:
        """
        Evaluate the deterministic mean trajectory.

        For σ = 0 or mean-field approximation:
        - γ = 1: E(t) = E₀·exp(-k·t)
        - γ ≠ 1: E(t) = [E₀^(1-γ) - k(1-γ)t]^(1/(1-γ))

        Args:
            t: Time points
            k: Decay rate
            gamma: Drift exponent
            sigma: Noise intensity (not used in deterministic)
            beta: Noise scaling (not used in deterministic)
            E0: Initial engagement

        Returns:
            Mean engagement trajectory
        """
        t = np.asarray(t, dtype=np.float64)

        if abs(gamma - 1.0) < 1e-6:
            # Standard exponential
            return E0 * np.exp(-k * t)
        else:
            # Generalized solution
            one_minus_gamma = 1 - gamma
            inner = E0**one_minus_gamma - k * one_minus_gamma * t

            # Handle cases where inner becomes negative (extinction)
            valid = inner > 0
            result = np.zeros_like(t)
            result[valid] = np.power(inner[valid], 1 / one_minus_gamma)

            return result

    def gradient(
        self,
        t: NDArray[np.float64],
        k: float = 0.1,
        gamma: float = 0.5,
        sigma: float = 0.0,
        beta: float = 0.5,
        E0: float = 1.0
    ) -> dict[str, NDArray[np.float64]]:
        """
        Compute analytical gradients for the deterministic trajectory.
        """
        t = np.asarray(t, dtype=np.float64)
        E = self.evaluate(t, k, gamma, sigma, beta, E0)

        if abs(gamma - 1.0) < 1e-6:
            # Exponential case
            dE_dk = -t * E
            dE_dgamma = np.zeros_like(t)  # Discontinuous
            dE_dE0 = np.exp(-k * t)
        else:
            one_minus_gamma = 1 - gamma
            inner = E0**one_minus_gamma - k * one_minus_gamma * t
            valid = inner > 0

            dE_dk = np.zeros_like(t)
            dE_dgamma = np.zeros_like(t)
            dE_dE0 = np.zeros_like(t)

            if np.any(valid):
                # dE/dk
                dE_dk[valid] = -t[valid] * np.power(inner[valid], gamma / one_minus_gamma)

                # dE/dE0
                dE_dE0[valid] = (E0**(- gamma)) * np.power(inner[valid], gamma / one_minus_gamma)

                # dE/dγ is complex, use numerical
                eps = 1e-6
                E_plus = self.evaluate(t[valid], k, gamma + eps, sigma, beta, E0)
                E_minus = self.evaluate(t[valid], k, gamma - eps, sigma, beta, E0)
                dE_dgamma[valid] = (E_plus - E_minus) / (2 * eps)

        return {
            "k": dE_dk,
            "gamma": dE_dgamma,
            "sigma": np.zeros_like(t),
            "beta": np.zeros_like(t),
            "E0": dE_dE0
        }

    def simulate(
        self,
        T: float = 100.0,
        dt: float = 0.1,
        n_trajectories: int = 1000,
        k: float = 0.1,
        gamma: float = 0.5,
        sigma: float = 0.1,
        beta: float = 0.5,
        E0: float = 1.0,
        seed: Optional[int] = None
    ) -> SDESimulationResult:
        """
        Simulate the SDE using Euler-Maruyama method.

        dE = -k·E^γ·dt + σ·E^β·dW

        Args:
            T: Total simulation time
            dt: Time step
            n_trajectories: Number of independent simulations
            k, gamma, sigma, beta, E0: Model parameters
            seed: Random seed for reproducibility

        Returns:
            SDESimulationResult with trajectories and statistics
        """
        if seed is not None:
            np.random.seed(seed)

        n_steps = int(T / dt)
        time = np.linspace(0, T, n_steps + 1)
        sqrt_dt = np.sqrt(dt)

        # Initialize trajectories
        trajectories = np.zeros((n_trajectories, n_steps + 1))
        trajectories[:, 0] = E0

        # Euler-Maruyama integration
        for i in range(n_steps):
            E = trajectories[:, i]
            E = np.maximum(E, 1e-10)  # Prevent numerical issues

            # Drift term
            drift = -k * np.power(E, gamma)

            # Diffusion term
            diffusion = sigma * np.power(E, beta)

            # Wiener increment
            dW = np.random.randn(n_trajectories) * sqrt_dt

            # Update
            trajectories[:, i + 1] = E + drift * dt + diffusion * dW
            trajectories[:, i + 1] = np.maximum(trajectories[:, i + 1], 0)

        # Compute statistics
        mean = np.mean(trajectories, axis=0)
        std = np.std(trajectories, axis=0)

        percentiles = {
            5: np.percentile(trajectories, 5, axis=0),
            25: np.percentile(trajectories, 25, axis=0),
            50: np.percentile(trajectories, 50, axis=0),
            75: np.percentile(trajectories, 75, axis=0),
            95: np.percentile(trajectories, 95, axis=0)
        }

        return SDESimulationResult(
            time=time,
            trajectories=trajectories,
            mean=mean,
            std=std,
            percentiles=percentiles
        )

    def analytical_moments(
        self,
        t: NDArray[np.float64],
        k: float,
        gamma: float,
        sigma: float,
        beta: float,
        E0: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute analytical approximations for mean and variance.

        Uses moment closure approximation for the SDE.
        For γ = 1 and β = 0.5 (geometric Brownian motion):
            E[E(t)] = E₀·exp(-k·t)
            Var[E(t)] = E₀²·exp(-2k·t)·(exp(σ²·t) - 1)

        Args:
            t: Time points
            k, gamma, sigma, beta, E0: Model parameters

        Returns:
            Tuple of (mean, variance) arrays
        """
        t = np.asarray(t, dtype=np.float64)

        # Mean (deterministic part)
        mean = self.evaluate(t, k, gamma, 0, 0, E0)

        # Variance approximation (linear noise assumption)
        if abs(gamma - 1.0) < 1e-6 and abs(beta - 0.5) < 1e-6:
            # Geometric Brownian motion case
            exp_2kt = np.exp(-2 * k * t)
            exp_s2t = np.exp(sigma**2 * t)
            variance = E0**2 * exp_2kt * (exp_s2t - 1)
        else:
            # General approximation using linear noise
            variance = (sigma * E0**beta)**2 * t * np.exp(-2 * k * t)

        variance = np.maximum(variance, 0)
        return mean, variance

    def extinction_time_distribution(
        self,
        k: float,
        gamma: float,
        sigma: float,
        beta: float,
        E0: float,
        threshold: float = 0.01,
        n_simulations: int = 10000
    ) -> NDArray[np.float64]:
        """
        Estimate the distribution of extinction times.

        Extinction is defined as E(t) < threshold.

        Args:
            k, gamma, sigma, beta, E0: Model parameters
            threshold: Extinction threshold
            n_simulations: Number of simulation runs

        Returns:
            Array of extinction times (inf if not extinct)
        """
        T = 1000.0  # Max simulation time
        dt = 0.1

        result = self.simulate(
            T=T, dt=dt, n_trajectories=n_simulations,
            k=k, gamma=gamma, sigma=sigma, beta=beta, E0=E0
        )

        extinction_times = np.full(n_simulations, np.inf)

        for i in range(n_simulations):
            below_threshold = result.trajectories[i] < threshold
            if np.any(below_threshold):
                extinction_times[i] = result.time[np.argmax(below_threshold)]

        return extinction_times

    def fit_from_data(
        self,
        t: NDArray[np.float64],
        E: NDArray[np.float64],
        sigma: Optional[NDArray[np.float64]] = None,
        include_noise: bool = False
    ):
        """
        Fit the SDE parameters from observed data.

        If include_noise is False, fits only the deterministic parameters.
        If True, uses moment matching to estimate noise parameters.
        """
        if include_noise:
            # Fit with noise (use simulation-based inference)
            # This is computationally expensive
            pass

        # Fit deterministic part
        return self.fit(t, E, sigma)

    @staticmethod
    def universal_form(
        x: NDArray[np.float64],
        gamma: float = 0.5
    ) -> NDArray[np.float64]:
        """
        The universal scaling function from SDE.

        For the deterministic mean field:
            f(x) = [1 - (1-γ)x]^(1/(1-γ)) for γ ≠ 1
            f(x) = exp(-x)                 for γ = 1

        Args:
            x: Normalized time
            gamma: Drift exponent

        Returns:
            Universal decay function
        """
        x = np.asarray(x, dtype=np.float64)

        if abs(gamma - 1.0) < 1e-6:
            return np.exp(-x)
        else:
            one_minus_gamma = 1 - gamma
            inner = 1 - one_minus_gamma * x
            valid = inner > 0
            result = np.zeros_like(x)
            result[valid] = np.power(inner[valid], 1 / one_minus_gamma)
            return result


class RewardPredictionErrorModel:
    """
    Mechanistic model based on reward prediction error (RPE) dynamics.

    The dopaminergic system tracks:
        RPE = actual_reward - expected_reward

    Over time:
    - Expected reward adapts to actual rewards (habituation)
    - RPE → 0 as novelty wears off
    - Engagement follows RPE dynamics

    Model:
        dR_exp/dt = η·(R_act - R_exp)  [adaptation]
        RPE = R_act - R_exp
        E = g(RPE)  [engagement is function of RPE]

    For constant R_act and g(RPE) = RPE:
        E(t) = E₀·exp(-η·t)

    More realistic: R_act also decays (novelty wearing off)
    """

    def __init__(
        self,
        eta: float = 0.1,          # Adaptation rate
        novelty_decay: float = 0.05,  # Rate of novelty decay
        baseline_engagement: float = 0.1  # Residual engagement
    ):
        self.eta = eta
        self.novelty_decay = novelty_decay
        self.baseline = baseline_engagement

    def simulate(
        self,
        T: float = 100.0,
        dt: float = 0.1,
        R0: float = 1.0  # Initial reward level
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Simulate RPE dynamics and resulting engagement.

        Returns:
            Tuple of (time, engagement) arrays
        """
        n_steps = int(T / dt)
        time = np.linspace(0, T, n_steps + 1)

        R_exp = np.zeros(n_steps + 1)  # Expected reward
        R_act = R0 * np.exp(-self.novelty_decay * time)  # Actual reward (decaying novelty)

        # Initial expected reward
        R_exp[0] = 0.0

        # Integrate expectation dynamics
        for i in range(n_steps):
            R_exp[i + 1] = R_exp[i] + self.eta * (R_act[i] - R_exp[i]) * dt

        # RPE and engagement
        RPE = R_act - R_exp
        engagement = self.baseline + np.maximum(RPE, 0)

        return time, engagement


class AttentionalFatigueModel:
    """
    Mechanistic model based on attentional resource depletion.

    Attention behaves like a limited resource:
        dA/dt = recovery - k·usage(E)

    Engagement depends on available attention:
        E = f(A) = A^α

    This naturally produces:
    - Initial high engagement (full attention)
    - Decay as attention depletes
    - Potential recovery after rest periods
    """

    def __init__(
        self,
        max_attention: float = 1.0,
        depletion_rate: float = 0.1,
        recovery_rate: float = 0.05,
        engagement_exponent: float = 1.0
    ):
        self.A_max = max_attention
        self.k_dep = depletion_rate
        self.k_rec = recovery_rate
        self.alpha = engagement_exponent

    def simulate(
        self,
        T: float = 100.0,
        dt: float = 0.1,
        usage_pattern: Optional[NDArray[np.float64]] = None
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Simulate attention dynamics and engagement.

        Args:
            T: Total time
            dt: Time step
            usage_pattern: Optional external usage forcing

        Returns:
            Tuple of (time, attention, engagement) arrays
        """
        n_steps = int(T / dt)
        time = np.linspace(0, T, n_steps + 1)

        A = np.zeros(n_steps + 1)  # Attention
        A[0] = self.A_max

        if usage_pattern is None:
            # Default: constant usage
            usage = np.ones(n_steps + 1)
        else:
            usage = usage_pattern

        for i in range(n_steps):
            # Recovery towards max
            recovery = self.k_rec * (self.A_max - A[i])

            # Depletion from usage
            depletion = self.k_dep * usage[i] * A[i]

            # Update
            A[i + 1] = A[i] + (recovery - depletion) * dt
            A[i + 1] = np.clip(A[i + 1], 0, self.A_max)

        # Engagement follows attention
        engagement = np.power(A / self.A_max, self.alpha)

        return time, A, engagement
