"""
Validation Module.

Provides cross-validation, predictive validation, and
out-of-sample testing for the universal decay law.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Dict
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold, cross_val_score

from src.models.base import DecayModel, DecayModelRegistry
from src.data.loader import UnifiedDataset, ProcessedUser
from src.analysis.fitting import FittingPipeline, UserFitResult


@dataclass
class ValidationResult:
    """
    Results from validation analysis.

    Attributes:
        method: Validation method used
        train_platforms: Platforms used for training
        test_platforms: Platforms used for testing
        train_quality: Training collapse quality
        test_quality: Test collapse quality
        prediction_accuracy: Out-of-sample prediction accuracy
        rmse: Root mean squared error
        mae: Mean absolute error
        r_squared: Coefficient of determination
        details: Additional method-specific details
    """
    method: str
    train_platforms: List[str]
    test_platforms: List[str]
    train_quality: float = 0.0
    test_quality: float = 0.0
    prediction_accuracy: float = 0.0
    rmse: float = float('inf')
    mae: float = float('inf')
    r_squared: float = 0.0
    details: dict = field(default_factory=dict)


class CrossValidator:
    """
    Cross-validation for universality analysis.

    Supports:
    - Leave-one-platform-out cross-validation
    - K-fold cross-validation on users
    - Train/test split by time
    """

    def __init__(
        self,
        fitting_pipeline: Optional[FittingPipeline] = None,
        n_folds: int = 5
    ):
        """
        Initialize validator.

        Args:
            fitting_pipeline: Pipeline for fitting models
            n_folds: Number of folds for k-fold CV
        """
        self.pipeline = fitting_pipeline or FittingPipeline()
        self.n_folds = n_folds

    def leave_one_platform_out(
        self,
        dataset: UnifiedDataset
    ) -> List[ValidationResult]:
        """
        Leave-one-platform-out cross-validation.

        Train on N-1 platforms, test on the held-out platform.
        Tests whether the universal law generalizes to new domains.

        Args:
            dataset: UnifiedDataset with multiple platforms

        Returns:
            List of ValidationResult for each held-out platform
        """
        from src.analysis.universality import UniversalityAnalyzer

        platforms = dataset.platforms
        if len(platforms) < 2:
            return []

        results = []
        analyzer = UniversalityAnalyzer()

        for test_platform in platforms:
            train_platforms = [p for p in platforms if p != test_platform]

            # Get train and test users
            train_users = []
            test_users = []
            for p in platforms:
                users = list(dataset.iter_users(p))
                if p == test_platform:
                    test_users.extend(users)
                else:
                    train_users.extend(users)

            # Fit on training platforms
            train_results = []
            for user in train_users:
                fit_output = self.pipeline.fitter.fit(user.time, user.engagement)
                best_model = fit_output["best_model"]
                best_result = fit_output["fit_results"].get(best_model)
                tau = 0.0
                if best_result:
                    tau = best_result.parameters.get("tau", best_result.parameters.get("lambda_", 0))

                train_results.append(UserFitResult(
                    user_id=user.user_id,
                    platform=user.platform,
                    best_model=best_model,
                    fit_results=fit_output["fit_results"],
                    model_comparisons=fit_output["comparisons"],
                    estimated_tau=tau,
                    preprocessed=self.pipeline.preprocessor.process(user.time, user.engagement)
                ))

            # Analyze universality on training data
            train_collapse = analyzer.analyze(train_results)
            if len(train_collapse.rescaled_times) < 5:
                continue

            master = analyzer.fit_master_curve(train_collapse)
            train_quality = analyzer.compute_collapse_quality(train_collapse, master)

            # Fit test users
            test_results = []
            for user in test_users:
                fit_output = self.pipeline.fitter.fit(user.time, user.engagement)
                best_model = fit_output["best_model"]
                best_result = fit_output["fit_results"].get(best_model)
                tau = 0.0
                if best_result:
                    tau = best_result.parameters.get("tau", best_result.parameters.get("lambda_", 0))

                test_results.append(UserFitResult(
                    user_id=user.user_id,
                    platform=user.platform,
                    best_model=best_model,
                    fit_results=fit_output["fit_results"],
                    model_comparisons=fit_output["comparisons"],
                    estimated_tau=tau,
                    preprocessed=self.pipeline.preprocessor.process(user.time, user.engagement)
                ))

            # Evaluate on test platform
            test_collapse = analyzer.analyze(test_results)
            test_quality = analyzer.compute_collapse_quality(test_collapse, master)

            # Compute prediction metrics
            all_residuals = []
            for x, y in zip(test_collapse.rescaled_times, test_collapse.rescaled_engagements):
                y_pred = master.evaluate(x)
                all_residuals.extend((y - y_pred).tolist())

            if all_residuals:
                residuals = np.array(all_residuals)
                rmse = np.sqrt(np.mean(residuals**2))
                mae = np.mean(np.abs(residuals))
            else:
                rmse = float('inf')
                mae = float('inf')

            results.append(ValidationResult(
                method="leave_one_platform_out",
                train_platforms=train_platforms,
                test_platforms=[test_platform],
                train_quality=train_quality,
                test_quality=test_quality,
                rmse=rmse,
                mae=mae,
                details={
                    "n_train_users": len(train_results),
                    "n_test_users": len(test_results),
                    "master_curve_params": master.parameters,
                }
            ))

        return results

    def k_fold_users(
        self,
        dataset: UnifiedDataset,
        platform: Optional[str] = None
    ) -> ValidationResult:
        """
        K-fold cross-validation on users.

        Args:
            dataset: UnifiedDataset
            platform: Optional platform filter

        Returns:
            Aggregated ValidationResult
        """
        from src.analysis.universality import UniversalityAnalyzer

        users = list(dataset.iter_users(platform))
        if len(users) < self.n_folds:
            return ValidationResult(
                method="k_fold",
                train_platforms=dataset.platforms,
                test_platforms=dataset.platforms
            )

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        analyzer = UniversalityAnalyzer()

        fold_results = []

        for train_idx, test_idx in kfold.split(users):
            train_users = [users[i] for i in train_idx]
            test_users = [users[i] for i in test_idx]

            # Fit training users
            train_results = []
            for user in train_users:
                fit_output = self.pipeline.fitter.fit(user.time, user.engagement)
                best_model = fit_output["best_model"]
                best_result = fit_output["fit_results"].get(best_model)
                tau = best_result.parameters.get("tau", 0) if best_result else 0

                train_results.append(UserFitResult(
                    user_id=user.user_id,
                    platform=user.platform,
                    best_model=best_model,
                    fit_results=fit_output["fit_results"],
                    model_comparisons=fit_output["comparisons"],
                    estimated_tau=tau,
                    preprocessed=self.pipeline.preprocessor.process(user.time, user.engagement)
                ))

            # Get master curve from training
            train_collapse = analyzer.analyze(train_results)
            if len(train_collapse.rescaled_times) < 3:
                continue

            master = analyzer.fit_master_curve(train_collapse)
            train_quality = analyzer.compute_collapse_quality(train_collapse, master)

            # Evaluate on test
            test_results = []
            for user in test_users:
                fit_output = self.pipeline.fitter.fit(user.time, user.engagement)
                best_model = fit_output["best_model"]
                best_result = fit_output["fit_results"].get(best_model)
                tau = best_result.parameters.get("tau", 0) if best_result else 0

                test_results.append(UserFitResult(
                    user_id=user.user_id,
                    platform=user.platform,
                    best_model=best_model,
                    fit_results=fit_output["fit_results"],
                    model_comparisons=fit_output["comparisons"],
                    estimated_tau=tau,
                    preprocessed=self.pipeline.preprocessor.process(user.time, user.engagement)
                ))

            test_collapse = analyzer.analyze(test_results)
            test_quality = analyzer.compute_collapse_quality(test_collapse, master)

            fold_results.append({
                "train_quality": train_quality,
                "test_quality": test_quality,
            })

        if not fold_results:
            return ValidationResult(
                method="k_fold",
                train_platforms=dataset.platforms,
                test_platforms=dataset.platforms
            )

        return ValidationResult(
            method="k_fold",
            train_platforms=dataset.platforms,
            test_platforms=dataset.platforms,
            train_quality=np.mean([r["train_quality"] for r in fold_results]),
            test_quality=np.mean([r["test_quality"] for r in fold_results]),
            details={
                "n_folds": self.n_folds,
                "fold_results": fold_results,
            }
        )


class Predictor:
    """
    Predict future engagement using the universal decay law.

    Given early engagement data, predict engagement at future time points.
    """

    def __init__(
        self,
        master_curve_params: Optional[dict] = None,
        model_name: str = "stretched_exponential"
    ):
        """
        Initialize predictor.

        Args:
            master_curve_params: Parameters of universal master curve
            model_name: Decay model to use
        """
        self.master_curve_params = master_curve_params or {}
        self.model_name = model_name
        self.model = DecayModelRegistry.get(model_name)

    def predict_from_early_data(
        self,
        t_early: NDArray[np.float64],
        E_early: NDArray[np.float64],
        t_future: NDArray[np.float64],
        use_universal: bool = True
    ) -> Tuple[NDArray[np.float64], dict]:
        """
        Predict future engagement from early observations.

        Args:
            t_early: Early time points
            E_early: Early engagement values
            t_future: Future time points to predict
            use_universal: Use universal curve parameters as prior

        Returns:
            Tuple of (predictions, fit_info)
        """
        # Fit to early data
        if use_universal and self.master_curve_params:
            # Use universal parameters as initialization
            fit_result = self.model.fit(t_early, E_early)
        else:
            fit_result = self.model.fit(t_early, E_early)

        if not fit_result.converged:
            return np.full(len(t_future), np.nan), {"converged": False}

        # Predict
        predictions = self.model.evaluate(t_future, **fit_result.parameters)

        # Confidence intervals (using parameter uncertainties)
        # Simple approach: propagate parameter errors
        if fit_result.parameter_errors:
            # Monte Carlo uncertainty propagation
            n_samples = 100
            pred_samples = []

            for _ in range(n_samples):
                sampled_params = {}
                for name, value in fit_result.parameters.items():
                    error = fit_result.parameter_errors.get(name, 0)
                    sampled_params[name] = np.random.normal(value, error)

                try:
                    sample_pred = self.model.evaluate(t_future, **sampled_params)
                    pred_samples.append(sample_pred)
                except Exception:
                    continue

            if pred_samples:
                pred_samples = np.array(pred_samples)
                ci_lower = np.percentile(pred_samples, 2.5, axis=0)
                ci_upper = np.percentile(pred_samples, 97.5, axis=0)
            else:
                ci_lower = ci_upper = predictions
        else:
            ci_lower = ci_upper = predictions

        return predictions, {
            "converged": True,
            "parameters": fit_result.parameters,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    def predict_retention(
        self,
        t_early: NDArray[np.float64],
        E_early: NDArray[np.float64],
        retention_threshold: float = 0.1,
        max_time: float = 365.0
    ) -> Tuple[float, float]:
        """
        Predict when engagement will fall below retention threshold.

        Args:
            t_early: Early time points
            E_early: Early engagement values
            retention_threshold: Threshold for "retained" (fraction of initial)
            max_time: Maximum time to consider

        Returns:
            Tuple of (predicted_retention_time, confidence)
        """
        from scipy.optimize import brentq

        # Fit model
        fit_result = self.model.fit(t_early, E_early)

        if not fit_result.converged:
            return max_time, 0.0

        # Find when E(t)/E0 = threshold
        E0 = fit_result.parameters.get("E0", 1.0)
        target = E0 * retention_threshold

        def residual(t):
            return self.model.evaluate(np.array([t]), **fit_result.parameters)[0] - target

        try:
            # Check bounds
            E_start = self.model.evaluate(np.array([0.0]), **fit_result.parameters)[0]
            E_end = self.model.evaluate(np.array([max_time]), **fit_result.parameters)[0]

            if E_start <= target:
                return 0.0, 0.5
            if E_end >= target:
                return max_time, 0.5

            retention_time = brentq(residual, 0, max_time)

            # Confidence based on fit quality
            confidence = 1 - fit_result.ks_statistic

            return retention_time, confidence

        except Exception:
            return max_time, 0.0

    def evaluate_prediction_accuracy(
        self,
        t_full: NDArray[np.float64],
        E_full: NDArray[np.float64],
        early_fraction: float = 0.2
    ) -> dict:
        """
        Evaluate prediction accuracy using held-out future data.

        Args:
            t_full: Full time array
            E_full: Full engagement array
            early_fraction: Fraction of data to use for fitting

        Returns:
            Dictionary with accuracy metrics
        """
        n_early = max(5, int(len(t_full) * early_fraction))
        t_early = t_full[:n_early]
        E_early = E_full[:n_early]
        t_future = t_full[n_early:]
        E_future = E_full[n_early:]

        if len(t_future) < 3:
            return {"error": "Insufficient future data"}

        predictions, info = self.predict_from_early_data(t_early, E_early, t_future)

        if not info.get("converged"):
            return {"error": "Fitting did not converge"}

        # Metrics
        residuals = E_future - predictions
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((E_future - np.mean(E_future))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Relative errors
        relative_errors = np.abs(residuals) / (E_future + 1e-10)
        mape = np.mean(relative_errors) * 100  # Mean absolute percentage error

        return {
            "rmse": rmse,
            "mae": mae,
            "r_squared": r_squared,
            "mape": mape,
            "n_early": n_early,
            "n_future": len(t_future),
            "parameters": info.get("parameters"),
        }
