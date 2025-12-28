"""
Base class for decay models.

This module provides the abstract base class and registry for all decay models,
ensuring consistent interface and enabling model comparison.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional, Type, TypeVar
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, differential_evolution
from scipy.stats import kstest


@dataclass
class FitResult:
    """
    Container for model fitting results.

    Attributes:
        parameters: Dictionary of fitted parameter values
        parameter_errors: Standard errors of parameters (from Hessian)
        log_likelihood: Log-likelihood at optimum
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        ks_statistic: Kolmogorov-Smirnov statistic
        ks_pvalue: KS test p-value
        r_squared: Coefficient of determination
        residuals: Fit residuals
        converged: Whether optimization converged
        n_iterations: Number of optimization iterations
        n_observations: Number of data points used
    """
    parameters: dict[str, float]
    parameter_errors: dict[str, float] = field(default_factory=dict)
    log_likelihood: float = 0.0
    aic: float = float("inf")
    bic: float = float("inf")
    ks_statistic: float = 1.0
    ks_pvalue: float = 0.0
    r_squared: float = 0.0
    residuals: Optional[NDArray[np.float64]] = None
    converged: bool = False
    n_iterations: int = 0
    n_observations: int = 0


@dataclass
class ModelComparison:
    """
    Container for model comparison results.

    Attributes:
        model_name: Name of the model
        fit_result: Fitting results
        delta_aic: AIC difference from best model
        delta_bic: BIC difference from best model
        akaike_weight: Akaike weight (probability)
        ranking: Rank among compared models
    """
    model_name: str
    fit_result: FitResult
    delta_aic: float = 0.0
    delta_bic: float = 0.0
    akaike_weight: float = 0.0
    ranking: int = 0


T = TypeVar("T", bound="DecayModel")


class DecayModel(ABC):
    """
    Abstract base class for decay models.

    All decay models must inherit from this class and implement:
        - evaluate(): Compute model predictions
        - parameter_names: Names of model parameters
        - parameter_bounds: Valid ranges for parameters
        - n_parameters: Number of free parameters

    The base class provides:
        - fit(): Maximum likelihood parameter estimation
        - log_likelihood(): Compute log-likelihood
        - aic(), bic(): Information criteria
        - residuals(): Compute fit residuals

    Example:
        >>> model = StretchedExponentialModel()
        >>> result = model.fit(t, E)
        >>> predictions = model.evaluate(t_new, **result.parameters)
    """

    # Class attributes to be overridden by subclasses
    name: ClassVar[str] = "BaseDecayModel"
    description: ClassVar[str] = "Abstract base class for decay models"

    @property
    @abstractmethod
    def parameter_names(self) -> list[str]:
        """Names of model parameters."""
        pass

    @property
    @abstractmethod
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Valid bounds for each parameter as (min, max) tuples."""
        pass

    @property
    def n_parameters(self) -> int:
        """Number of free parameters in the model."""
        return len(self.parameter_names)

    @abstractmethod
    def evaluate(
        self,
        t: NDArray[np.float64],
        **params: float
    ) -> NDArray[np.float64]:
        """
        Evaluate the decay model at given time points.

        Args:
            t: Time points (normalized by τ(α) for universal analysis)
            **params: Model parameters

        Returns:
            Model predictions E(t)/E₀
        """
        pass

    @abstractmethod
    def gradient(
        self,
        t: NDArray[np.float64],
        **params: float
    ) -> dict[str, NDArray[np.float64]]:
        """
        Compute gradients of the model with respect to parameters.

        Args:
            t: Time points
            **params: Model parameters

        Returns:
            Dictionary mapping parameter names to gradient arrays
        """
        pass

    def log_likelihood(
        self,
        t: NDArray[np.float64],
        E: NDArray[np.float64],
        sigma: Optional[NDArray[np.float64]] = None,
        **params: float
    ) -> float:
        """
        Compute Gaussian log-likelihood.

        Assumes errors are normally distributed with variance σ².

        Args:
            t: Time points
            E: Observed engagement values
            sigma: Measurement uncertainties (optional)
            **params: Model parameters

        Returns:
            Log-likelihood value
        """
        predictions = self.evaluate(t, **params)
        residuals = E - predictions

        if sigma is None:
            # Estimate σ from residuals
            sigma_est = np.std(residuals)
            if sigma_est < 1e-10:
                sigma_est = 1e-10
            sigma = np.full_like(E, sigma_est)

        # Gaussian log-likelihood
        n = len(E)
        ll = -0.5 * n * np.log(2 * np.pi)
        ll -= np.sum(np.log(sigma))
        ll -= 0.5 * np.sum((residuals / sigma) ** 2)

        return ll

    def negative_log_likelihood(
        self,
        param_vector: NDArray[np.float64],
        t: NDArray[np.float64],
        E: NDArray[np.float64],
        sigma: Optional[NDArray[np.float64]] = None
    ) -> float:
        """
        Negative log-likelihood for optimization (minimization).

        Args:
            param_vector: Parameter values as array
            t: Time points
            E: Observed engagement values
            sigma: Measurement uncertainties

        Returns:
            Negative log-likelihood (for minimization)
        """
        params = dict(zip(self.parameter_names, param_vector))
        return -self.log_likelihood(t, E, sigma, **params)

    def aic(self, log_likelihood: float, n_observations: int) -> float:
        """
        Compute Akaike Information Criterion.

        AIC = 2k - 2ln(L)

        With small-sample correction (AICc):
        AICc = AIC + (2k² + 2k)/(n - k - 1)

        Args:
            log_likelihood: Log-likelihood at optimum
            n_observations: Number of data points

        Returns:
            AICc value
        """
        k = self.n_parameters
        n = n_observations

        aic = 2 * k - 2 * log_likelihood

        # Small-sample correction
        if n > k + 1:
            aic += (2 * k * k + 2 * k) / (n - k - 1)

        return aic

    def bic(self, log_likelihood: float, n_observations: int) -> float:
        """
        Compute Bayesian Information Criterion.

        BIC = k·ln(n) - 2ln(L)

        Args:
            log_likelihood: Log-likelihood at optimum
            n_observations: Number of data points

        Returns:
            BIC value
        """
        k = self.n_parameters
        return k * np.log(n_observations) - 2 * log_likelihood

    def _get_initial_params(
        self,
        t: NDArray[np.float64],
        E: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Generate initial parameter estimates.

        Override in subclasses for model-specific initialization.

        Args:
            t: Time points
            E: Observed engagement values

        Returns:
            Initial parameter vector
        """
        # Default: middle of bounds
        initial = []
        for name in self.parameter_names:
            low, high = self.parameter_bounds[name]
            if np.isinf(low):
                low = -10.0
            if np.isinf(high):
                high = 10.0
            initial.append((low + high) / 2)
        return np.array(initial)

    def fit(
        self,
        t: NDArray[np.float64],
        E: NDArray[np.float64],
        sigma: Optional[NDArray[np.float64]] = None,
        method: str = "L-BFGS-B",
        global_optimization: bool = False,
        **kwargs: Any
    ) -> FitResult:
        """
        Fit model parameters using maximum likelihood estimation.

        Args:
            t: Time points
            E: Observed engagement values (normalized)
            sigma: Measurement uncertainties (optional)
            method: Optimization method for scipy.optimize.minimize
            global_optimization: Use differential evolution for global search
            **kwargs: Additional arguments for optimizer

        Returns:
            FitResult containing optimized parameters and diagnostics
        """
        # Prepare bounds
        bounds = [self.parameter_bounds[name] for name in self.parameter_names]

        # Initial guess
        x0 = self._get_initial_params(t, E)

        # Optimization
        if global_optimization:
            result = differential_evolution(
                self.negative_log_likelihood,
                bounds,
                args=(t, E, sigma),
                seed=42,
                **kwargs
            )
        else:
            result = minimize(
                self.negative_log_likelihood,
                x0,
                args=(t, E, sigma),
                method=method,
                bounds=bounds,
                **kwargs
            )

        # Extract parameters
        params = dict(zip(self.parameter_names, result.x))

        # Compute metrics
        ll = -result.fun
        n = len(E)
        aic_val = self.aic(ll, n)
        bic_val = self.bic(ll, n)

        # Compute residuals
        predictions = self.evaluate(t, **params)
        residuals = E - predictions

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((E - np.mean(E)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # KS test
        try:
            ks_stat, ks_pval = kstest(residuals, "norm")
        except Exception:
            ks_stat, ks_pval = 1.0, 0.0

        # Parameter errors (from Hessian inverse)
        param_errors = self._compute_parameter_errors(result, t, E, sigma)

        return FitResult(
            parameters=params,
            parameter_errors=param_errors,
            log_likelihood=ll,
            aic=aic_val,
            bic=bic_val,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            r_squared=r_squared,
            residuals=residuals,
            converged=result.success,
            n_iterations=result.nit if hasattr(result, "nit") else 0,
            n_observations=n
        )

    def _compute_parameter_errors(
        self,
        opt_result: Any,
        t: NDArray[np.float64],
        E: NDArray[np.float64],
        sigma: Optional[NDArray[np.float64]]
    ) -> dict[str, float]:
        """
        Estimate parameter uncertainties from Hessian.

        Uses finite differences to approximate the Hessian and
        inverts it to get the covariance matrix.
        """
        from scipy.optimize import approx_fprime

        try:
            # Compute Hessian numerically
            eps = 1e-5
            n_params = len(self.parameter_names)
            hessian = np.zeros((n_params, n_params))

            def grad_i(x, i):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                return (
                    self.negative_log_likelihood(x_plus, t, E, sigma) -
                    self.negative_log_likelihood(x_minus, t, E, sigma)
                ) / (2 * eps)

            for i in range(n_params):
                for j in range(n_params):
                    x_plus = opt_result.x.copy()
                    x_minus = opt_result.x.copy()
                    x_plus[j] += eps
                    x_minus[j] -= eps
                    hessian[i, j] = (grad_i(x_plus, i) - grad_i(x_minus, i)) / (2 * eps)

            # Covariance matrix = inverse of Hessian
            cov = np.linalg.inv(hessian)
            errors = np.sqrt(np.diag(cov))

            return dict(zip(self.parameter_names, errors))
        except Exception:
            # Return zeros if Hessian inversion fails
            return {name: 0.0 for name in self.parameter_names}


class DecayModelRegistry:
    """
    Registry for decay models.

    Provides centralized access to all available decay models
    and utilities for model selection and comparison.

    Example:
        >>> registry = DecayModelRegistry()
        >>> registry.register(StretchedExponentialModel)
        >>> model = registry.get("stretched_exponential")
        >>> comparison = registry.compare_all(t, E)
    """

    _models: ClassVar[dict[str, Type[DecayModel]]] = {}

    @classmethod
    def register(cls, model_class: Type[DecayModel]) -> Type[DecayModel]:
        """
        Register a decay model class.

        Can be used as a decorator:
            @DecayModelRegistry.register
            class MyModel(DecayModel):
                ...
        """
        cls._models[model_class.name] = model_class
        return model_class

    @classmethod
    def get(cls, name: str) -> DecayModel:
        """Get an instance of a registered model by name."""
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Unknown model: {name}. Available: {available}")
        return cls._models[name]()

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model names."""
        return list(cls._models.keys())

    @classmethod
    def compare_all(
        cls,
        t: NDArray[np.float64],
        E: NDArray[np.float64],
        sigma: Optional[NDArray[np.float64]] = None,
        **fit_kwargs: Any
    ) -> list[ModelComparison]:
        """
        Fit all registered models and compare them.

        Args:
            t: Time points
            E: Observed engagement values
            sigma: Measurement uncertainties
            **fit_kwargs: Arguments passed to fit()

        Returns:
            List of ModelComparison objects sorted by AIC
        """
        results = []

        for name, model_class in cls._models.items():
            model = model_class()
            try:
                fit_result = model.fit(t, E, sigma, **fit_kwargs)
                results.append(ModelComparison(
                    model_name=name,
                    fit_result=fit_result
                ))
            except Exception as e:
                # Model fitting failed
                results.append(ModelComparison(
                    model_name=name,
                    fit_result=FitResult(
                        parameters={},
                        converged=False
                    )
                ))

        # Compute deltas and weights
        if results:
            min_aic = min(r.fit_result.aic for r in results if r.fit_result.converged)
            min_bic = min(r.fit_result.bic for r in results if r.fit_result.converged)

            # Akaike weights
            exp_deltas = []
            for r in results:
                if r.fit_result.converged:
                    r.delta_aic = r.fit_result.aic - min_aic
                    r.delta_bic = r.fit_result.bic - min_bic
                    exp_deltas.append((r, np.exp(-0.5 * r.delta_aic)))
                else:
                    r.delta_aic = float("inf")
                    r.delta_bic = float("inf")

            total_weight = sum(w for _, w in exp_deltas)
            for r, w in exp_deltas:
                r.akaike_weight = w / total_weight if total_weight > 0 else 0.0

            # Sort and rank
            results.sort(key=lambda r: r.fit_result.aic)
            for i, r in enumerate(results):
                r.ranking = i + 1

        return results
