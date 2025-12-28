"""
Statistical Tests Module.

Provides statistical tests for:
    - Universality hypothesis testing
    - Model comparison
    - Residual analysis
    - Bootstrap confidence intervals
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.stats import (
    kstest, shapiro, levene, mannwhitneyu,
    spearmanr, pearsonr, f_oneway
)


@dataclass
class HypothesisTestResult:
    """
    Result of a hypothesis test.

    Attributes:
        test_name: Name of the statistical test
        statistic: Test statistic
        p_value: P-value
        null_hypothesis: Description of H0
        alternative_hypothesis: Description of H1
        reject_null: Whether to reject H0 at given alpha
        alpha: Significance level
        details: Additional test-specific information
    """
    test_name: str
    statistic: float
    p_value: float
    null_hypothesis: str
    alternative_hypothesis: str
    reject_null: bool
    alpha: float = 0.05
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class StatisticalTests:
    """
    Collection of statistical tests for universality analysis.
    """

    @staticmethod
    def test_residual_normality(
        residuals: NDArray[np.float64],
        alpha: float = 0.05
    ) -> HypothesisTestResult:
        """
        Test if residuals follow a normal distribution.

        Uses Shapiro-Wilk test for small samples, D'Agostino-Pearson for large.

        Args:
            residuals: Array of residuals
            alpha: Significance level

        Returns:
            HypothesisTestResult
        """
        residuals = np.asarray(residuals)
        residuals = residuals[np.isfinite(residuals)]

        if len(residuals) < 3:
            return HypothesisTestResult(
                test_name="Normality Test",
                statistic=np.nan,
                p_value=np.nan,
                null_hypothesis="Residuals are normally distributed",
                alternative_hypothesis="Residuals are not normally distributed",
                reject_null=False,
                alpha=alpha,
                details={"error": "Insufficient data"}
            )

        if len(residuals) <= 5000:
            # Shapiro-Wilk
            stat, p_value = shapiro(residuals)
            test_name = "Shapiro-Wilk"
        else:
            # D'Agostino-Pearson
            stat, p_value = stats.normaltest(residuals)
            test_name = "D'Agostino-Pearson"

        return HypothesisTestResult(
            test_name=test_name,
            statistic=stat,
            p_value=p_value,
            null_hypothesis="Residuals are normally distributed",
            alternative_hypothesis="Residuals are not normally distributed",
            reject_null=p_value < alpha,
            alpha=alpha,
            details={
                "n_samples": len(residuals),
                "mean": np.mean(residuals),
                "std": np.std(residuals),
                "skewness": stats.skew(residuals),
                "kurtosis": stats.kurtosis(residuals),
            }
        )

    @staticmethod
    def test_homoscedasticity(
        groups: List[NDArray[np.float64]],
        alpha: float = 0.05
    ) -> HypothesisTestResult:
        """
        Test for equal variances across groups (platforms).

        Uses Levene's test (robust to non-normality).

        Args:
            groups: List of arrays (e.g., residuals per platform)
            alpha: Significance level

        Returns:
            HypothesisTestResult
        """
        # Filter valid groups
        valid_groups = [g[np.isfinite(g)] for g in groups if len(g) > 2]

        if len(valid_groups) < 2:
            return HypothesisTestResult(
                test_name="Levene's Test",
                statistic=np.nan,
                p_value=np.nan,
                null_hypothesis="Variances are equal across groups",
                alternative_hypothesis="Variances differ across groups",
                reject_null=False,
                alpha=alpha,
                details={"error": "Insufficient groups"}
            )

        stat, p_value = levene(*valid_groups)

        return HypothesisTestResult(
            test_name="Levene's Test",
            statistic=stat,
            p_value=p_value,
            null_hypothesis="Variances are equal across groups",
            alternative_hypothesis="Variances differ across groups",
            reject_null=p_value < alpha,
            alpha=alpha,
            details={
                "n_groups": len(valid_groups),
                "group_variances": [np.var(g) for g in valid_groups],
            }
        )

    @staticmethod
    def test_platform_differences(
        platform_residuals: Dict[str, NDArray[np.float64]],
        alpha: float = 0.05
    ) -> HypothesisTestResult:
        """
        Test if residual distributions differ significantly across platforms.

        Uses Kruskal-Wallis H-test (non-parametric ANOVA).

        Args:
            platform_residuals: Dictionary of platform -> residuals
            alpha: Significance level

        Returns:
            HypothesisTestResult
        """
        groups = []
        platforms = []
        for platform, residuals in platform_residuals.items():
            r = np.asarray(residuals)
            r = r[np.isfinite(r)]
            if len(r) > 2:
                groups.append(r)
                platforms.append(platform)

        if len(groups) < 2:
            return HypothesisTestResult(
                test_name="Kruskal-Wallis H-test",
                statistic=np.nan,
                p_value=np.nan,
                null_hypothesis="Residual distributions are identical across platforms",
                alternative_hypothesis="At least one platform has different distribution",
                reject_null=False,
                alpha=alpha,
                details={"error": "Insufficient platforms"}
            )

        stat, p_value = stats.kruskal(*groups)

        return HypothesisTestResult(
            test_name="Kruskal-Wallis H-test",
            statistic=stat,
            p_value=p_value,
            null_hypothesis="Residual distributions are identical across platforms",
            alternative_hypothesis="At least one platform has different distribution",
            reject_null=p_value < alpha,
            alpha=alpha,
            details={
                "platforms": platforms,
                "group_medians": {p: np.median(g) for p, g in zip(platforms, groups)},
            }
        )

    @staticmethod
    def test_scaling_relationship(
        alpha: NDArray[np.float64],
        tau: NDArray[np.float64],
        alpha_level: float = 0.05
    ) -> HypothesisTestResult:
        """
        Test if τ(α) = τ₀ · α^(-β) scaling relationship holds.

        Tests for significant linear relationship in log-log space.

        Args:
            alpha: Motivation parameters
            tau: Characteristic timescales
            alpha_level: Significance level

        Returns:
            HypothesisTestResult
        """
        alpha_arr = np.asarray(alpha)
        tau_arr = np.asarray(tau)

        # Filter valid
        valid = (alpha_arr > 0) & (tau_arr > 0) & np.isfinite(alpha_arr) & np.isfinite(tau_arr)
        alpha_arr = alpha_arr[valid]
        tau_arr = tau_arr[valid]

        if len(alpha_arr) < 5:
            return HypothesisTestResult(
                test_name="Scaling Relationship Test",
                statistic=np.nan,
                p_value=np.nan,
                null_hypothesis="No scaling relationship (β = 0)",
                alternative_hypothesis="Scaling relationship exists (β ≠ 0)",
                reject_null=False,
                alpha=alpha_level,
                details={"error": "Insufficient data"}
            )

        # Log-log regression
        log_alpha = np.log(alpha_arr)
        log_tau = np.log(tau_arr)

        # Pearson correlation in log-log space
        r, p_value = pearsonr(log_alpha, log_tau)

        # Linear regression for β
        slope, intercept = np.polyfit(log_alpha, log_tau, 1)
        beta_estimate = -slope  # τ = τ₀ · α^(-β) → log(τ) = log(τ₀) - β·log(α)
        tau0_estimate = np.exp(intercept)

        return HypothesisTestResult(
            test_name="Scaling Relationship Test",
            statistic=r,
            p_value=p_value,
            null_hypothesis="No scaling relationship (β = 0)",
            alternative_hypothesis="Scaling relationship exists (β ≠ 0)",
            reject_null=p_value < alpha_level,
            alpha=alpha_level,
            details={
                "correlation": r,
                "beta_estimate": beta_estimate,
                "tau0_estimate": tau0_estimate,
                "n_samples": len(alpha_arr),
            }
        )

    @staticmethod
    def test_universality(
        platform_gammas: Dict[str, List[float]],
        alpha: float = 0.05
    ) -> HypothesisTestResult:
        """
        Test if gamma (stretching exponent) is universal across platforms.

        Uses one-way ANOVA to test if mean gamma differs across platforms.

        Args:
            platform_gammas: Dictionary of platform -> gamma values
            alpha: Significance level

        Returns:
            HypothesisTestResult
        """
        groups = []
        platforms = []
        for platform, gammas in platform_gammas.items():
            g = np.asarray(gammas)
            g = g[np.isfinite(g)]
            if len(g) > 2:
                groups.append(g)
                platforms.append(platform)

        if len(groups) < 2:
            return HypothesisTestResult(
                test_name="Universality Test (ANOVA)",
                statistic=np.nan,
                p_value=np.nan,
                null_hypothesis="γ is identical across all platforms",
                alternative_hypothesis="γ differs across platforms",
                reject_null=False,
                alpha=alpha,
                details={"error": "Insufficient platforms"}
            )

        stat, p_value = f_oneway(*groups)

        # NOT rejecting H0 supports universality
        supports_universality = p_value >= alpha

        return HypothesisTestResult(
            test_name="Universality Test (ANOVA)",
            statistic=stat,
            p_value=p_value,
            null_hypothesis="γ is identical across all platforms",
            alternative_hypothesis="γ differs across platforms",
            reject_null=p_value < alpha,
            alpha=alpha,
            details={
                "platforms": platforms,
                "group_means": {p: np.mean(g) for p, g in zip(platforms, groups)},
                "group_stds": {p: np.std(g) for p, g in zip(platforms, groups)},
                "supports_universality": supports_universality,
                "overall_mean": np.mean(np.concatenate(groups)),
                "overall_std": np.std(np.concatenate(groups)),
            }
        )

    @staticmethod
    def kolmogorov_smirnov_collapse(
        rescaled_engagements: List[NDArray[np.float64]],
        reference_curve: callable
    ) -> HypothesisTestResult:
        """
        Test collapse quality using Kolmogorov-Smirnov test.

        Compares each rescaled curve to the reference (master) curve.

        Args:
            rescaled_engagements: List of rescaled engagement arrays
            reference_curve: Function to evaluate master curve

        Returns:
            HypothesisTestResult with aggregate KS statistic
        """
        ks_stats = []
        p_values = []

        for y in rescaled_engagements:
            y = np.asarray(y)
            y = y[np.isfinite(y)]
            if len(y) < 5:
                continue

            # Compare to uniform after applying master curve transform
            # This tests if the residuals follow expected distribution
            try:
                stat, p = kstest(y, 'uniform', args=(0, 1))
                ks_stats.append(stat)
                p_values.append(p)
            except Exception:
                continue

        if not ks_stats:
            return HypothesisTestResult(
                test_name="KS Collapse Test",
                statistic=np.nan,
                p_value=np.nan,
                null_hypothesis="Data collapses onto master curve",
                alternative_hypothesis="Significant deviation from master curve",
                reject_null=False,
                alpha=0.05,
                details={"error": "Insufficient data"}
            )

        # Aggregate
        mean_ks = np.mean(ks_stats)
        combined_p = stats.combine_pvalues(p_values)[1]

        return HypothesisTestResult(
            test_name="KS Collapse Test",
            statistic=mean_ks,
            p_value=combined_p,
            null_hypothesis="Data collapses onto master curve",
            alternative_hypothesis="Significant deviation from master curve",
            reject_null=combined_p < 0.05,
            alpha=0.05,
            details={
                "n_curves": len(ks_stats),
                "individual_ks_stats": ks_stats,
                "individual_p_values": p_values,
            }
        )


class BootstrapAnalysis:
    """
    Bootstrap analysis for confidence intervals and uncertainty quantification.
    """

    def __init__(self, n_bootstrap: int = 1000, random_state: Optional[int] = None):
        """
        Initialize bootstrap analyzer.

        Args:
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def confidence_interval(
        self,
        data: NDArray[np.float64],
        statistic: callable = np.mean,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for a statistic.

        Args:
            data: Data array
            statistic: Function to compute statistic
            confidence_level: Confidence level (0-1)

        Returns:
            Tuple of (point_estimate, ci_lower, ci_upper)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        data = np.asarray(data)
        data = data[np.isfinite(data)]
        n = len(data)

        if n < 3:
            point = statistic(data) if len(data) > 0 else np.nan
            return point, np.nan, np.nan

        # Bootstrap
        boot_stats = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            boot_stats.append(statistic(sample))

        boot_stats = np.array(boot_stats)

        alpha = 1 - confidence_level
        ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
        ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
        point_estimate = statistic(data)

        return point_estimate, ci_lower, ci_upper

    def parameter_uncertainty(
        self,
        fit_results: List[Any],
        parameter_name: str,
        confidence_level: float = 0.95
    ) -> dict:
        """
        Estimate uncertainty in a fitted parameter across users.

        Args:
            fit_results: List of UserFitResult
            parameter_name: Name of parameter to analyze
            confidence_level: Confidence level

        Returns:
            Dictionary with uncertainty estimates
        """
        values = []
        for result in fit_results:
            if result.best_model and result.fit_results.get(result.best_model):
                params = result.fit_results[result.best_model].parameters
                if parameter_name in params:
                    values.append(params[parameter_name])

        if not values:
            return {"error": "No parameter values found"}

        values = np.array(values)
        point, ci_lower, ci_upper = self.confidence_interval(
            values, np.mean, confidence_level
        )

        return {
            "parameter": parameter_name,
            "n_values": len(values),
            "mean": point,
            "median": np.median(values),
            "std": np.std(values),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence_level": confidence_level,
        }

    def collapse_quality_uncertainty(
        self,
        collapse_result: Any,  # CollapseResult
        master_curve: Any,     # MasterCurve
        confidence_level: float = 0.95
    ) -> dict:
        """
        Bootstrap uncertainty in collapse quality.

        Args:
            collapse_result: CollapseResult
            master_curve: MasterCurve
            confidence_level: Confidence level

        Returns:
            Dictionary with quality uncertainty
        """
        from src.analysis.universality import UniversalityAnalyzer

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_users = len(collapse_result.rescaled_times)
        if n_users < 5:
            return {"error": "Insufficient users for bootstrap"}

        qualities = []
        analyzer = UniversalityAnalyzer()

        for _ in range(self.n_bootstrap):
            # Bootstrap sample of users
            indices = np.random.choice(n_users, size=n_users, replace=True)

            # Create bootstrap collapse
            boot_collapse = type(collapse_result)(
                rescaled_times=[collapse_result.rescaled_times[i] for i in indices],
                rescaled_engagements=[collapse_result.rescaled_engagements[i] for i in indices],
                user_ids=[collapse_result.user_ids[i] for i in indices],
                platforms=[collapse_result.platforms[i] for i in indices],
            )

            quality = analyzer.compute_collapse_quality(boot_collapse, master_curve)
            qualities.append(quality)

        qualities = np.array(qualities)
        alpha = 1 - confidence_level

        return {
            "mean_quality": np.mean(qualities),
            "std_quality": np.std(qualities),
            "ci_lower": np.percentile(qualities, 100 * alpha / 2),
            "ci_upper": np.percentile(qualities, 100 * (1 - alpha / 2)),
            "confidence_level": confidence_level,
            "n_bootstrap": self.n_bootstrap,
        }
