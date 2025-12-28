"""
Model Fitting Module.

Provides:
    - Single-user decay model fitting
    - Batch fitting for multiple users
    - Cross-validation within fitting
    - Comprehensive fitting pipeline
"""

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, List, Dict
import numpy as np
from numpy.typing import NDArray
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from src.models.base import DecayModel, DecayModelRegistry, FitResult, ModelComparison
from src.data.loader import ProcessedUser, UnifiedDataset
from src.analysis.preprocessing import MinimalPreprocessor, PreprocessedUser


@dataclass
class UserFitResult:
    """
    Complete fitting results for a single user.

    Attributes:
        user_id: User identifier
        platform: Source platform
        best_model: Name of best-fitting model
        fit_results: Dictionary of model name -> FitResult
        model_comparisons: List of ModelComparison objects
        preprocessed: Preprocessing result
        estimated_tau: Estimated characteristic timescale
        estimated_alpha: Estimated motivation parameter (if computed)
    """
    user_id: str
    platform: str
    best_model: str
    fit_results: Dict[str, FitResult]
    model_comparisons: List[ModelComparison]
    preprocessed: Optional[PreprocessedUser] = None
    estimated_tau: float = 0.0
    estimated_alpha: float = 1.0


class DecayFitter:
    """
    Fit decay models to engagement data.

    Supports multiple model types and selection criteria.

    Example:
        >>> fitter = DecayFitter(models=["stretched_exponential", "power_law"])
        >>> result = fitter.fit(time, engagement)
        >>> print(f"Best model: {result['best_model']}")
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        selection_criterion: str = "aic",
        global_optimization: bool = False
    ):
        """
        Initialize fitter.

        Args:
            models: List of model names to fit (None for all registered)
            selection_criterion: "aic" or "bic" for model selection
            global_optimization: Use global optimization (slower but more robust)
        """
        if models is None:
            self.models = DecayModelRegistry.list_models()
        else:
            self.models = models

        self.selection_criterion = selection_criterion
        self.global_optimization = global_optimization

    def fit(
        self,
        t: NDArray[np.float64],
        E: NDArray[np.float64],
        sigma: Optional[NDArray[np.float64]] = None
    ) -> Dict[str, Any]:
        """
        Fit all models and select best.

        Args:
            t: Time array
            E: Engagement array
            sigma: Measurement uncertainties

        Returns:
            Dictionary with fit results and best model
        """
        t = np.asarray(t, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)

        results = {}
        comparisons = []

        for model_name in self.models:
            try:
                model = DecayModelRegistry.get(model_name)
                fit_result = model.fit(
                    t, E, sigma,
                    global_optimization=self.global_optimization
                )
                results[model_name] = fit_result
                comparisons.append(ModelComparison(
                    model_name=model_name,
                    fit_result=fit_result
                ))
            except Exception as e:
                warnings.warn(f"Failed to fit {model_name}: {e}")

        # Compute comparison metrics
        if comparisons:
            if self.selection_criterion == "aic":
                key_fn = lambda c: c.fit_result.aic
            else:
                key_fn = lambda c: c.fit_result.bic

            valid = [c for c in comparisons if c.fit_result.converged]
            if valid:
                min_val = min(key_fn(c) for c in valid)
                for c in valid:
                    c.delta_aic = c.fit_result.aic - min_val
                    c.delta_bic = c.fit_result.bic - min_val

                # Akaike weights
                exp_deltas = [np.exp(-0.5 * c.delta_aic) for c in valid]
                total = sum(exp_deltas)
                for c, w in zip(valid, exp_deltas):
                    c.akaike_weight = w / total if total > 0 else 0

                # Rank
                valid.sort(key=key_fn)
                for i, c in enumerate(valid):
                    c.ranking = i + 1

                best_model = valid[0].model_name
            else:
                best_model = self.models[0] if self.models else ""
        else:
            best_model = ""

        return {
            "fit_results": results,
            "comparisons": comparisons,
            "best_model": best_model
        }

    def fit_single_model(
        self,
        model_name: str,
        t: NDArray[np.float64],
        E: NDArray[np.float64],
        sigma: Optional[NDArray[np.float64]] = None
    ) -> FitResult:
        """
        Fit a single model.

        Args:
            model_name: Name of model to fit
            t: Time array
            E: Engagement array
            sigma: Measurement uncertainties

        Returns:
            FitResult
        """
        model = DecayModelRegistry.get(model_name)
        return model.fit(t, E, sigma, global_optimization=self.global_optimization)


class BatchFitter:
    """
    Fit models to multiple users in batch.

    Supports parallel processing for efficiency.

    Example:
        >>> batch_fitter = BatchFitter(n_workers=4)
        >>> results = batch_fitter.fit_dataset(dataset)
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        selection_criterion: str = "aic",
        n_workers: int = 1,
        progress_callback: Optional[callable] = None
    ):
        """
        Initialize batch fitter.

        Args:
            models: List of model names
            selection_criterion: Model selection criterion
            n_workers: Number of parallel workers
            progress_callback: Optional callback(current, total)
        """
        self.fitter = DecayFitter(models, selection_criterion)
        self.n_workers = n_workers
        self.progress_callback = progress_callback

    def fit_dataset(
        self,
        dataset: UnifiedDataset,
        platform: Optional[str] = None
    ) -> List[UserFitResult]:
        """
        Fit models to all users in dataset.

        Args:
            dataset: UnifiedDataset to process
            platform: Filter by platform

        Returns:
            List of UserFitResult
        """
        users = list(dataset.iter_users(platform))

        if self.n_workers > 1:
            return self._fit_parallel(users)
        else:
            return self._fit_sequential(users)

    def _fit_sequential(self, users: List[ProcessedUser]) -> List[UserFitResult]:
        """Sequential fitting."""
        results = []
        total = len(users)

        for i, user in enumerate(users):
            result = self._fit_user(user)
            results.append(result)

            if self.progress_callback:
                self.progress_callback(i + 1, total)

        return results

    def _fit_parallel(self, users: List[ProcessedUser]) -> List[UserFitResult]:
        """Parallel fitting using ProcessPoolExecutor."""
        results = []
        total = len(users)

        # Note: For true parallel processing, would need to serialize models
        # Using ThreadPoolExecutor for simplicity
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(self._fit_user, user): user for user in users}

            completed = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    user = futures[future]
                    warnings.warn(f"Failed to fit user {user.user_id}: {e}")

                completed += 1
                if self.progress_callback:
                    self.progress_callback(completed, total)

        return results

    def _fit_user(self, user: ProcessedUser) -> UserFitResult:
        """Fit models for a single user."""
        fit_output = self.fitter.fit(user.time, user.engagement)

        # Extract tau from best model
        best_model = fit_output["best_model"]
        best_result = fit_output["fit_results"].get(best_model)
        estimated_tau = 0.0
        if best_result and best_result.parameters:
            estimated_tau = best_result.parameters.get(
                "tau",
                best_result.parameters.get("lambda_", 0.0)
            )

        return UserFitResult(
            user_id=user.user_id,
            platform=user.platform,
            best_model=best_model,
            fit_results=fit_output["fit_results"],
            model_comparisons=fit_output["comparisons"],
            estimated_tau=estimated_tau
        )


@dataclass
class PipelineConfig:
    """
    Configuration for fitting pipeline.

    IMPORTANT: Defaults are chosen for SCIENTIFIC RIGOR, not convenience.
    See Nature reviewer concerns about max normalization and outlier clipping.
    """
    # Preprocessing - SCIENTIFICALLY RIGOROUS DEFAULTS
    # engagement_norm: "none" or "initial" - NEVER use "max" (forces decay pattern)
    engagement_norm: str = "none"  # Changed from "max" - see Nature reviewer critique
    # outlier_method: None = preserve all data, or "detect_only" for transparency
    outlier_method: Optional[str] = None  # Changed from "iqr" - outlier clipping manipulates data
    outlier_threshold: float = 5.0  # More permissive if used (was 3.0)
    outlier_strategy: str = "none"  # Changed from "clip" - preserve data integrity
    min_valid_points: int = 10

    # Fitting
    models: Optional[List[str]] = None
    selection_criterion: str = "aic"
    global_optimization: bool = False

    # Batch processing
    n_workers: int = 1

    def __post_init__(self):
        """Warn about potentially problematic configurations."""
        if self.engagement_norm == "max":
            warnings.warn(
                "engagement_norm='max' forces peak-then-decay pattern and may create artifacts. "
                "Consider using 'none' or 'initial' instead.",
                UserWarning
            )
        if self.outlier_strategy == "clip":
            warnings.warn(
                "outlier_strategy='clip' modifies data and reduces natural variance. "
                "Consider using 'none' and handling outliers with robust regression.",
                UserWarning
            )


class FittingPipeline:
    """
    Complete fitting pipeline from raw data to results.

    Combines preprocessing and model fitting in a single pipeline.

    Example:
        >>> pipeline = FittingPipeline()
        >>> results = pipeline.run(dataset)
        >>> summary = pipeline.summarize(results)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()

        # Use MinimalPreprocessor with scientifically rigorous settings
        from src.analysis.preprocessing import NormalizationMethod

        # Map config to new preprocessing settings
        norm_mapping = {
            "none": NormalizationMethod.NONE,
            "initial": NormalizationMethod.INITIAL,
            "mean_initial": NormalizationMethod.MEAN_INITIAL,
            "zscore": NormalizationMethod.ZSCORE,
            "max": NormalizationMethod.NONE,  # MAX is not supported - use NONE instead
        }

        normalization = norm_mapping.get(
            self.config.engagement_norm,
            NormalizationMethod.NONE
        )

        self.preprocessor = MinimalPreprocessor(
            normalization=normalization,
            detect_outliers=self.config.outlier_method is not None,
            min_observations=self.config.min_valid_points
        )

        self.fitter = DecayFitter(
            models=self.config.models,
            selection_criterion=self.config.selection_criterion,
            global_optimization=self.config.global_optimization
        )

    def run(
        self,
        dataset: UnifiedDataset,
        platform: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> List[UserFitResult]:
        """
        Run complete fitting pipeline on dataset.

        Args:
            dataset: Dataset to process
            platform: Filter by platform
            progress_callback: Progress callback(current, total)

        Returns:
            List of UserFitResult
        """
        results = []
        users = list(dataset.iter_users(platform))
        total = len(users)

        for i, user in enumerate(users):
            # Preprocess using new MinimalPreprocessor
            prep_result = self.preprocessor.process(
                user_id=user.user_id,
                platform=user.platform,
                time=user.time,
                engagement=user.engagement
            )

            # MinimalPreprocessor returns None if data is unusable
            if prep_result is None:
                continue

            if len(prep_result.engagement) < self.config.min_valid_points:
                continue

            # Fit models
            fit_output = self.fitter.fit(prep_result.time, prep_result.engagement)

            # Extract tau
            best_model = fit_output["best_model"]
            best_result = fit_output["fit_results"].get(best_model)
            estimated_tau = 0.0
            if best_result and best_result.parameters:
                estimated_tau = best_result.parameters.get(
                    "tau",
                    best_result.parameters.get("lambda_", 0.0)
                )

            results.append(UserFitResult(
                user_id=user.user_id,
                platform=user.platform,
                best_model=best_model,
                fit_results=fit_output["fit_results"],
                model_comparisons=fit_output["comparisons"],
                preprocessed=prep_result,
                estimated_tau=estimated_tau
            ))

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def summarize(self, results: List[UserFitResult]) -> Dict[str, Any]:
        """
        Summarize fitting results.

        Args:
            results: List of UserFitResult

        Returns:
            Summary statistics dictionary
        """
        if not results:
            return {"n_users": 0}

        # Model selection counts
        model_counts = {}
        for r in results:
            model_counts[r.best_model] = model_counts.get(r.best_model, 0) + 1

        # Parameter distributions
        taus = [r.estimated_tau for r in results if r.estimated_tau > 0]
        gammas = []
        for r in results:
            if r.best_model and r.fit_results.get(r.best_model):
                params = r.fit_results[r.best_model].parameters
                gamma = params.get("gamma", params.get("kappa"))
                if gamma is not None:
                    gammas.append(gamma)

        # Platform breakdown
        platform_stats = {}
        for r in results:
            if r.platform not in platform_stats:
                platform_stats[r.platform] = {"count": 0, "taus": []}
            platform_stats[r.platform]["count"] += 1
            if r.estimated_tau > 0:
                platform_stats[r.platform]["taus"].append(r.estimated_tau)

        for platform in platform_stats:
            taus_p = platform_stats[platform]["taus"]
            platform_stats[platform]["tau_mean"] = np.mean(taus_p) if taus_p else 0
            platform_stats[platform]["tau_std"] = np.std(taus_p) if taus_p else 0

        return {
            "n_users": len(results),
            "model_selection": model_counts,
            "tau_mean": np.mean(taus) if taus else 0,
            "tau_std": np.std(taus) if taus else 0,
            "tau_median": np.median(taus) if taus else 0,
            "gamma_mean": np.mean(gammas) if gammas else 0,
            "gamma_std": np.std(gammas) if gammas else 0,
            "platform_stats": platform_stats
        }
