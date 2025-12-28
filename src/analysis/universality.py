"""
Universality Analysis Module - Scientifically Rigorous Version.

DESIGN PRINCIPLES (addressing Nature reviewer concerns):

1. NO CIRCULAR REASONING - τ is derived from data, not optimized for collapse
2. NO FORCED NORMALIZATION - use initial value, not max (which forces decay pattern)
3. HYPOTHESIS TESTING - test universality, don't assume it
4. TRAJECTORY CLASSIFICATION - identify behavioral patterns before analysis
5. TRANSPARENCY - report all methods and their potential biases

The Universal Decay Hypothesis:
    After proper rescaling, engagement curves from different platforms
    collapse onto a single universal function f(x).

IMPORTANT: This hypothesis must be TESTED, not assumed.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Dict
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from scipy.stats import kstest, wasserstein_distance, spearmanr
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import warnings

from src.models.base import DecayModel, DecayModelRegistry


class TrajectoryType(Enum):
    """
    Engagement trajectory classifications.

    The Universal Decay Law may only apply to DECAY trajectories.
    Other patterns must be identified and excluded from decay analysis.
    """
    DECAY = "decay"              # Decreasing engagement over time
    GROWTH = "growth"            # Increasing engagement over time
    STABLE = "stable"            # Relatively constant engagement
    ERRATIC = "erratic"          # High variance, no clear trend
    REVIVAL = "revival"          # Decay followed by resurgence
    UNKNOWN = "unknown"          # Cannot classify


@dataclass
class TrajectoryClassification:
    """Classification result for a single user's trajectory."""
    user_id: str
    trajectory_type: TrajectoryType
    confidence: float  # 0-1, how confident is the classification

    # Supporting statistics
    trend_slope: float  # Normalized slope
    trend_p_value: float  # Significance of trend
    coefficient_of_variation: float
    n_sign_changes: int  # Number of direction changes

    # Only for revival type
    revival_point: Optional[float] = None


@dataclass
class CollapseResult:
    """
    Result of universality collapse analysis.

    IMPORTANT: Includes full methodology transparency.
    """
    # Core data
    rescaled_times: List[NDArray[np.float64]]
    rescaled_engagements: List[NDArray[np.float64]]
    user_ids: List[str]
    platforms: List[str]

    # Quality metrics
    collapse_quality: float = 0.0
    residual_std: float = float('inf')

    # Hypothesis test results
    universality_p_value: float = 1.0  # p-value for universality hypothesis
    platform_homogeneity_p: float = 1.0  # Do platforms behave similarly?

    # Master curve
    master_curve_params: dict = field(default_factory=dict)
    master_curve_model: str = ""

    # Methodology transparency
    normalization_method: str = ""
    tau_estimation_method: str = ""
    n_users_excluded: int = 0
    exclusion_reasons: Dict[str, int] = field(default_factory=dict)

    # Trajectory breakdown
    trajectory_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class MasterCurve:
    """
    The universal master curve.

    Parameters are FITTED to data, not optimized for collapse.
    """
    model_name: str
    parameters: dict
    x_range: Tuple[float, float]
    interpolator: Optional[callable] = None

    # Goodness of fit
    r_squared: float = 0.0
    aic: float = float('inf')

    # Uncertainty
    parameter_uncertainties: dict = field(default_factory=dict)

    def evaluate(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate master curve at given rescaled times."""
        if self.interpolator is not None:
            return self.interpolator(x)

        model = DecayModelRegistry.get(self.model_name)
        return model.evaluate(x, **self.parameters)


class TrajectoryClassifier:
    """
    Classify user engagement trajectories BEFORE decay analysis.

    This addresses the Nature reviewer concern about forcing all users
    into a decay pattern when they may show growth, stability, or erratic behavior.
    """

    def __init__(
        self,
        min_observations: int = 10,
        trend_significance: float = 0.05,
        cv_threshold: float = 0.5,  # Coefficient of variation threshold for erratic
    ):
        self.min_observations = min_observations
        self.trend_significance = trend_significance
        self.cv_threshold = cv_threshold

    def classify(
        self,
        time: NDArray[np.float64],
        engagement: NDArray[np.float64],
        user_id: str = ""
    ) -> TrajectoryClassification:
        """
        Classify a user's engagement trajectory.

        Classifications:
        - DECAY: Significant negative trend
        - GROWTH: Significant positive trend
        - STABLE: No significant trend, low variance
        - ERRATIC: No significant trend, high variance
        - REVIVAL: Decay followed by resurgence
        """
        time = np.asarray(time, dtype=np.float64)
        engagement = np.asarray(engagement, dtype=np.float64)

        # Remove invalid
        valid = np.isfinite(time) & np.isfinite(engagement)
        time = time[valid]
        engagement = engagement[valid]

        if len(time) < self.min_observations:
            return TrajectoryClassification(
                user_id=user_id,
                trajectory_type=TrajectoryType.UNKNOWN,
                confidence=0.0,
                trend_slope=0.0,
                trend_p_value=1.0,
                coefficient_of_variation=0.0,
                n_sign_changes=0
            )

        # Sort by time
        sort_idx = np.argsort(time)
        time = time[sort_idx]
        engagement = engagement[sort_idx]

        # Normalize time to [0, 1] for comparable slopes
        t_norm = (time - time[0]) / (time[-1] - time[0] + 1e-10)

        # Compute trend using Spearman correlation (robust to outliers)
        rho, p_value = spearmanr(t_norm, engagement)
        if np.isnan(rho):
            rho = 0.0
            p_value = 1.0

        # Coefficient of variation
        mean_e = np.mean(engagement)
        std_e = np.std(engagement)
        cv = std_e / (mean_e + 1e-10) if mean_e > 0 else float('inf')

        # Count sign changes in derivative (for erratic detection)
        diffs = np.diff(engagement)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)

        # Check for revival pattern (local minimum followed by increase)
        revival_point = None
        if len(engagement) >= 5:
            # Find local minima
            half = len(engagement) // 2
            first_half_min = np.min(engagement[:half]) if half > 0 else engagement[0]
            second_half_max = np.max(engagement[half:])

            # Revival if second half max > first half min by significant margin
            if second_half_max > first_half_min * 1.5:
                revival_idx = half + np.argmax(engagement[half:])
                revival_point = time[revival_idx]

        # Classification logic
        trend_slope = rho  # Using correlation as normalized slope

        if revival_point is not None and p_value < self.trend_significance:
            trajectory_type = TrajectoryType.REVIVAL
            confidence = 1 - p_value

        elif p_value < self.trend_significance:
            # Significant trend detected
            if rho < -0.3:
                trajectory_type = TrajectoryType.DECAY
                confidence = abs(rho)
            elif rho > 0.3:
                trajectory_type = TrajectoryType.GROWTH
                confidence = abs(rho)
            else:
                # Weak but significant trend
                trajectory_type = TrajectoryType.STABLE
                confidence = 1 - abs(rho)

        elif cv > self.cv_threshold:
            # No significant trend but high variance
            trajectory_type = TrajectoryType.ERRATIC
            confidence = min(1.0, cv / (2 * self.cv_threshold))

        else:
            # No significant trend, low variance
            trajectory_type = TrajectoryType.STABLE
            confidence = 1 - cv / self.cv_threshold

        return TrajectoryClassification(
            user_id=user_id,
            trajectory_type=trajectory_type,
            confidence=confidence,
            trend_slope=trend_slope,
            trend_p_value=p_value,
            coefficient_of_variation=cv,
            n_sign_changes=sign_changes,
            revival_point=revival_point
        )

    def classify_batch(
        self,
        users_data: List[Dict[str, Any]]
    ) -> Tuple[List[TrajectoryClassification], Dict[str, int]]:
        """
        Classify multiple users and return distribution.

        Args:
            users_data: List of dicts with user_id, time, engagement

        Returns:
            Tuple of (classifications, type_counts)
        """
        classifications = []
        type_counts = {t.value: 0 for t in TrajectoryType}

        for user in users_data:
            classification = self.classify(
                user.get("time", np.array([])),
                user.get("engagement", np.array([])),
                user.get("user_id", "")
            )
            classifications.append(classification)
            type_counts[classification.trajectory_type.value] += 1

        return classifications, type_counts


class UniversalityAnalyzer:
    """
    Analyze universality of engagement decay - RIGOROUS VERSION.

    CRITICAL CHANGES from original:
    1. Uses INITIAL value normalization, not MAX (which forces decay)
    2. τ is FITTED from data, not optimized for collapse
    3. Trajectory classification BEFORE analysis
    4. Proper hypothesis testing
    5. Full transparency reporting

    Example:
        >>> analyzer = UniversalityAnalyzer()
        >>> classifications = analyzer.classify_trajectories(fit_results)
        >>> decay_users = [r for r, c in zip(fit_results, classifications)
        ...               if c.trajectory_type == TrajectoryType.DECAY]
        >>> collapse = analyzer.analyze(decay_users)
        >>> hypothesis_result = analyzer.test_universality_hypothesis(collapse)
    """

    def __init__(
        self,
        master_curve_model: str = "stretched_exponential",
        min_points: int = 10,
        x_max: float = 10.0,
        normalization: str = "initial"  # NOT "max"!
    ):
        """
        Initialize analyzer.

        Args:
            master_curve_model: Model to use for master curve
            min_points: Minimum points required per user
            x_max: Maximum rescaled time to consider
            normalization: "initial" (recommended) or "mean_initial"
                          NOTE: "max" is NOT supported as it creates artifacts
        """
        if normalization == "max":
            warnings.warn(
                "MAX normalization forces peak-then-decay pattern and is not supported. "
                "Using 'initial' normalization instead.",
                UserWarning
            )
            normalization = "initial"

        self.master_curve_model = master_curve_model
        self.min_points = min_points
        self.x_max = x_max
        self.normalization = normalization
        self.classifier = TrajectoryClassifier()

    def classify_trajectories(
        self,
        fit_results: List[Any]
    ) -> List[TrajectoryClassification]:
        """
        Classify all user trajectories before analysis.

        This MUST be called before analyze() to identify which users
        actually show decay behavior.
        """
        classifications = []

        for result in fit_results:
            if result.preprocessed is None:
                classifications.append(TrajectoryClassification(
                    user_id=result.user_id,
                    trajectory_type=TrajectoryType.UNKNOWN,
                    confidence=0.0,
                    trend_slope=0.0,
                    trend_p_value=1.0,
                    coefficient_of_variation=0.0,
                    n_sign_changes=0
                ))
                continue

            # Use RAW data for classification, not preprocessed
            # to avoid artifacts from preprocessing
            if hasattr(result.preprocessed, 'raw_time'):
                time = result.preprocessed.raw_time
                engagement = result.preprocessed.raw_engagement
            else:
                time = result.preprocessed.time
                engagement = result.preprocessed.engagement

            classification = self.classifier.classify(
                time, engagement, result.user_id
            )
            classifications.append(classification)

        return classifications

    def analyze(
        self,
        fit_results: List[Any],
        tau_source: str = "fitted",
        trajectory_filter: Optional[TrajectoryType] = TrajectoryType.DECAY
    ) -> CollapseResult:
        """
        Perform universality collapse analysis.

        Args:
            fit_results: List of UserFitResult from fitting pipeline
            tau_source: How to get τ - "fitted" uses model-estimated τ
            trajectory_filter: Only include users with this trajectory type
                              Set to None to include all (NOT recommended)

        Returns:
            CollapseResult with rescaled data and methodology transparency
        """
        rescaled_times = []
        rescaled_engagements = []
        user_ids = []
        platforms = []

        n_excluded = 0
        exclusion_reasons = {
            "no_preprocessed": 0,
            "insufficient_points": 0,
            "invalid_tau": 0,
            "wrong_trajectory": 0,
            "normalization_failed": 0
        }
        trajectory_counts = {t.value: 0 for t in TrajectoryType}

        # Classify trajectories first
        classifications = self.classify_trajectories(fit_results)

        for result, classification in zip(fit_results, classifications):
            trajectory_counts[classification.trajectory_type.value] += 1

            # Filter by trajectory type
            if trajectory_filter is not None:
                if classification.trajectory_type != trajectory_filter:
                    exclusion_reasons["wrong_trajectory"] += 1
                    n_excluded += 1
                    continue

            # Get preprocessed data
            if result.preprocessed is None:
                exclusion_reasons["no_preprocessed"] += 1
                n_excluded += 1
                continue

            t = result.preprocessed.time
            E = result.preprocessed.engagement

            if len(E) < self.min_points:
                exclusion_reasons["insufficient_points"] += 1
                n_excluded += 1
                continue

            # Get τ from fitting (NOT optimized)
            tau = result.estimated_tau
            if tau <= 0 or not np.isfinite(tau):
                exclusion_reasons["invalid_tau"] += 1
                n_excluded += 1
                continue

            # Rescale time
            x = t / tau

            # Rescale engagement using INITIAL value (not MAX!)
            E0 = self._get_initial_engagement(E)
            if E0 <= 0 or not np.isfinite(E0):
                exclusion_reasons["normalization_failed"] += 1
                n_excluded += 1
                continue

            y = E / E0

            # Filter to valid range
            valid = (x >= 0) & (x <= self.x_max) & np.isfinite(y)
            x = x[valid]
            y = y[valid]

            if len(x) < self.min_points:
                exclusion_reasons["insufficient_points"] += 1
                n_excluded += 1
                continue

            rescaled_times.append(x)
            rescaled_engagements.append(y)
            user_ids.append(result.user_id)
            platforms.append(result.platform)

        return CollapseResult(
            rescaled_times=rescaled_times,
            rescaled_engagements=rescaled_engagements,
            user_ids=user_ids,
            platforms=platforms,
            normalization_method=self.normalization,
            tau_estimation_method=tau_source,
            n_users_excluded=n_excluded,
            exclusion_reasons=exclusion_reasons,
            trajectory_distribution=trajectory_counts
        )

    def _get_initial_engagement(self, E: NDArray) -> float:
        """
        Get initial engagement value for normalization.

        Uses first few observations to be robust to noise.
        """
        if len(E) == 0:
            return 0.0

        if self.normalization == "initial":
            return E[0] if np.isfinite(E[0]) and E[0] > 0 else 0.0

        elif self.normalization == "mean_initial":
            n = min(3, len(E))
            initial = E[:n]
            valid = initial[np.isfinite(initial) & (initial > 0)]
            return np.mean(valid) if len(valid) > 0 else 0.0

        return E[0] if np.isfinite(E[0]) and E[0] > 0 else 0.0

    def fit_master_curve(
        self,
        collapse: CollapseResult,
        model_name: Optional[str] = None
    ) -> MasterCurve:
        """
        Fit a master curve to the collapsed data.

        The master curve is FIT to the data, not optimized for collapse.
        """
        model_name = model_name or self.master_curve_model
        model = DecayModelRegistry.get(model_name)

        # Combine all data
        all_x = np.concatenate(collapse.rescaled_times)
        all_y = np.concatenate(collapse.rescaled_engagements)

        # Sort by x
        sort_idx = np.argsort(all_x)
        all_x = all_x[sort_idx]
        all_y = all_y[sort_idx]

        # Fit model
        fit_result = model.fit(all_x, all_y)

        # Create interpolator from binned data
        n_bins = min(100, len(all_x) // 10)
        interpolator = None
        if n_bins > 5:
            x_bins = np.linspace(0, self.x_max, n_bins)
            y_means = []
            for i in range(len(x_bins) - 1):
                mask = (all_x >= x_bins[i]) & (all_x < x_bins[i + 1])
                if np.sum(mask) > 0:
                    y_means.append(np.mean(all_y[mask]))
                else:
                    y_means.append(np.nan)

            x_centers = (x_bins[:-1] + x_bins[1:]) / 2
            valid = np.isfinite(y_means)
            if np.sum(valid) > 3:
                interpolator = interp1d(
                    x_centers[valid],
                    np.array(y_means)[valid],
                    kind='cubic',
                    bounds_error=False,
                    fill_value='extrapolate'
                )

        # Store in collapse
        collapse.master_curve_params = fit_result.parameters
        collapse.master_curve_model = model_name

        x_range = (0.0, min(self.x_max, np.max(all_x)))

        return MasterCurve(
            model_name=model_name,
            parameters=fit_result.parameters,
            x_range=x_range,
            interpolator=interpolator,
            r_squared=fit_result.r_squared,
            aic=fit_result.aic,
            parameter_uncertainties=fit_result.parameter_errors or {}
        )

    def compute_collapse_quality(
        self,
        collapse: CollapseResult,
        master: MasterCurve
    ) -> float:
        """
        Compute collapse quality as a descriptive metric.

        NOTE: This is NOT used for optimization, only for reporting.
        """
        if not collapse.rescaled_times:
            return 0.0

        # Compute residuals
        all_residuals = []
        for x, y in zip(collapse.rescaled_times, collapse.rescaled_engagements):
            y_pred = master.evaluate(x)
            residuals = y - y_pred
            all_residuals.extend(residuals)

        all_residuals = np.array(all_residuals)
        residual_std = np.std(all_residuals)
        collapse.residual_std = residual_std

        # Quality based on residual standard deviation
        # Lower residual std = better collapse
        quality = np.exp(-residual_std * 3)
        collapse.collapse_quality = quality

        return quality

    def test_universality_hypothesis(
        self,
        collapse: CollapseResult,
        master: MasterCurve,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        PROPERLY test the universality hypothesis.

        Null Hypothesis H0: All platforms follow the same master curve
        Alternative H1: At least one platform deviates significantly

        This is HYPOTHESIS TESTING, not parameter optimization.
        """
        from scipy.stats import kruskal, levene, f_oneway

        results = {
            "supports_universality": False,
            "p_values": {},
            "test_statistics": {},
            "interpretation": ""
        }

        if len(collapse.rescaled_times) < 10:
            results["interpretation"] = "Insufficient data for hypothesis testing"
            return results

        # Group residuals by platform
        platform_residuals = {}
        for x, y, platform in zip(
            collapse.rescaled_times,
            collapse.rescaled_engagements,
            collapse.platforms
        ):
            y_pred = master.evaluate(x)
            residuals = y - y_pred

            if platform not in platform_residuals:
                platform_residuals[platform] = []
            platform_residuals[platform].extend(residuals.tolist())

        # Need at least 2 platforms
        valid_platforms = {k: np.array(v) for k, v in platform_residuals.items()
                          if len(v) >= 5}

        if len(valid_platforms) < 2:
            results["interpretation"] = "Need at least 2 platforms with sufficient data"
            return results

        # Test 1: Kruskal-Wallis - do residual distributions differ?
        groups = list(valid_platforms.values())
        try:
            kw_stat, kw_p = kruskal(*groups)
            results["p_values"]["kruskal_wallis"] = kw_p
            results["test_statistics"]["kruskal_wallis"] = kw_stat
        except Exception as e:
            results["p_values"]["kruskal_wallis"] = np.nan

        # Test 2: Levene's test - equal variances?
        try:
            lev_stat, lev_p = levene(*groups)
            results["p_values"]["levene"] = lev_p
            results["test_statistics"]["levene"] = lev_stat
        except Exception as e:
            results["p_values"]["levene"] = np.nan

        # Test 3: Are residual means close to zero for all platforms?
        platform_means = {k: np.mean(v) for k, v in valid_platforms.items()}
        platform_stds = {k: np.std(v) for k, v in valid_platforms.items()}
        results["platform_residual_means"] = platform_means
        results["platform_residual_stds"] = platform_stds

        # Universality supported if:
        # 1. Kruskal-Wallis fails to reject (p > alpha) - platforms are similar
        # 2. All platform mean residuals are close to zero

        kw_p = results["p_values"].get("kruskal_wallis", 0)
        max_mean_deviation = max(abs(m) for m in platform_means.values())

        if kw_p > alpha and max_mean_deviation < 0.1:
            results["supports_universality"] = True
            results["interpretation"] = (
                f"Universality hypothesis SUPPORTED (p={kw_p:.4f}). "
                "Platforms show consistent decay behavior."
            )
        else:
            results["supports_universality"] = False
            if kw_p <= alpha:
                results["interpretation"] = (
                    f"Universality hypothesis REJECTED (p={kw_p:.4f}). "
                    "Platforms show significantly different decay patterns."
                )
            else:
                results["interpretation"] = (
                    f"Weak support for universality (p={kw_p:.4f}), but "
                    f"some platforms show systematic bias (max deviation: {max_mean_deviation:.3f})."
                )

        collapse.universality_p_value = kw_p
        collapse.platform_homogeneity_p = results["p_values"].get("levene", 1.0)

        return results

    def identify_deviants(
        self,
        collapse: CollapseResult,
        master: MasterCurve,
        threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Identify users that deviate significantly from universal behavior.

        IMPORTANT: These users are scientifically interesting, not "bad data".
        """
        # Compute global residual statistics
        all_residuals = []
        for x, y in zip(collapse.rescaled_times, collapse.rescaled_engagements):
            y_pred = master.evaluate(x)
            all_residuals.extend((y - y_pred).tolist())

        global_std = np.std(all_residuals)
        global_mean = np.mean(all_residuals)

        # Identify deviants
        deviants = []
        all_users = []

        for i, (x, y, uid, platform) in enumerate(zip(
            collapse.rescaled_times,
            collapse.rescaled_engagements,
            collapse.user_ids,
            collapse.platforms
        )):
            y_pred = master.evaluate(x)
            user_residuals = y - y_pred
            user_rmse = np.sqrt(np.mean(user_residuals**2))
            user_mean = np.mean(user_residuals)

            # Deviation score: how many standard deviations from typical
            deviation_score = user_rmse / (global_std + 1e-10)

            user_info = {
                "user_id": uid,
                "platform": platform,
                "deviation_score": deviation_score,
                "rmse": user_rmse,
                "mean_residual": user_mean,
                "n_points": len(x),
                "is_deviant": deviation_score > threshold
            }

            all_users.append(user_info)

            if deviation_score > threshold:
                deviants.append(user_info)

        # Sort by deviation
        deviants.sort(key=lambda x: x["deviation_score"], reverse=True)

        return deviants

    def compute_platform_statistics(
        self,
        collapse: CollapseResult,
        master: MasterCurve
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-platform statistics for transparency reporting.
        """
        platform_data = {}

        for x, y, platform in zip(
            collapse.rescaled_times,
            collapse.rescaled_engagements,
            collapse.platforms
        ):
            if platform not in platform_data:
                platform_data[platform] = {
                    "residuals": [],
                    "n_users": 0,
                    "n_points": 0
                }

            y_pred = master.evaluate(x)
            residuals = y - y_pred

            platform_data[platform]["residuals"].extend(residuals.tolist())
            platform_data[platform]["n_users"] += 1
            platform_data[platform]["n_points"] += len(x)

        results = {}
        for platform, data in platform_data.items():
            residuals = np.array(data["residuals"])
            results[platform] = {
                "n_users": data["n_users"],
                "n_points": data["n_points"],
                "mean_residual": np.mean(residuals),
                "std_residual": np.std(residuals),
                "rmse": np.sqrt(np.mean(residuals**2)),
                "median_residual": np.median(residuals),
                "bias": np.mean(residuals),  # Systematic over/under prediction
            }

        return results

    def generate_transparency_report(
        self,
        collapse: CollapseResult,
        master: MasterCurve,
        hypothesis_result: Dict[str, Any]
    ) -> str:
        """
        Generate a full transparency report for publication.

        This report must be included in any publication to allow
        reproducibility and assessment of potential biases.
        """
        report = []
        report.append("=" * 70)
        report.append("UNIVERSALITY ANALYSIS - TRANSPARENCY REPORT")
        report.append("=" * 70)

        report.append("\n1. METHODOLOGY")
        report.append(f"   Normalization: {collapse.normalization_method}")
        report.append(f"   τ estimation: {collapse.tau_estimation_method}")
        report.append(f"   Master curve model: {collapse.master_curve_model}")
        report.append(f"   Minimum points per user: {self.min_points}")

        report.append("\n2. TRAJECTORY CLASSIFICATION")
        report.append("   Distribution of engagement patterns:")
        for ttype, count in collapse.trajectory_distribution.items():
            report.append(f"   - {ttype}: {count}")

        report.append("\n3. DATA EXCLUSIONS")
        report.append(f"   Total users excluded: {collapse.n_users_excluded}")
        for reason, count in collapse.exclusion_reasons.items():
            if count > 0:
                report.append(f"   - {reason}: {count}")

        report.append("\n4. ANALYSIS SAMPLE")
        report.append(f"   Users included: {len(collapse.user_ids)}")
        platform_counts = {}
        for p in collapse.platforms:
            platform_counts[p] = platform_counts.get(p, 0) + 1
        for p, c in platform_counts.items():
            report.append(f"   - {p}: {c}")

        report.append("\n5. MASTER CURVE FIT")
        report.append(f"   R²: {master.r_squared:.4f}")
        report.append(f"   AIC: {master.aic:.2f}")
        report.append("   Parameters:")
        for param, value in master.parameters.items():
            uncertainty = master.parameter_uncertainties.get(param, np.nan)
            report.append(f"   - {param}: {value:.4f} ± {uncertainty:.4f}")

        report.append("\n6. COLLAPSE QUALITY")
        report.append(f"   Quality score: {collapse.collapse_quality:.4f}")
        report.append(f"   Residual std: {collapse.residual_std:.4f}")

        report.append("\n7. HYPOTHESIS TEST RESULTS")
        report.append(f"   Universality supported: {hypothesis_result.get('supports_universality', 'N/A')}")
        for test, p in hypothesis_result.get('p_values', {}).items():
            report.append(f"   - {test}: p = {p:.4f}")
        report.append(f"   Interpretation: {hypothesis_result.get('interpretation', 'N/A')}")

        report.append("\n" + "=" * 70)

        return "\n".join(report)


# REMOVED: optimize_tau_scaling function
# This was the source of circular reasoning identified by the Nature reviewer.
# τ should be ESTIMATED from data, not OPTIMIZED for collapse quality.

def analyze_scaling_relationship(
    fit_results: List[Any],
    alpha_estimates: Dict[str, float]
) -> Dict[str, Any]:
    """
    Analyze the τ(α) scaling relationship using INDEPENDENT α estimates.

    IMPORTANT: α must be estimated INDEPENDENTLY, not derived from τ.
    This function only analyzes the relationship, it does not optimize it.

    Args:
        fit_results: List of UserFitResult with estimated_tau
        alpha_estimates: Dictionary of user_id -> independently estimated α

    Returns:
        Statistical analysis of the relationship
    """
    from scipy.stats import pearsonr, spearmanr

    tau_values = []
    alpha_values = []

    for result in fit_results:
        if result.user_id in alpha_estimates:
            tau = result.estimated_tau
            alpha = alpha_estimates[result.user_id]

            if tau > 0 and alpha > 0 and np.isfinite(tau) and np.isfinite(alpha):
                tau_values.append(tau)
                alpha_values.append(alpha)

    if len(tau_values) < 10:
        return {
            "error": "Insufficient data for scaling analysis",
            "n_valid": len(tau_values)
        }

    tau_arr = np.array(tau_values)
    alpha_arr = np.array(alpha_values)

    # Log-log analysis
    log_tau = np.log(tau_arr)
    log_alpha = np.log(alpha_arr)

    # Correlation tests
    pearson_r, pearson_p = pearsonr(log_alpha, log_tau)
    spearman_r, spearman_p = spearmanr(alpha_arr, tau_arr)

    # Linear fit in log-log space: log(τ) = log(τ₀) - β·log(α)
    slope, intercept = np.polyfit(log_alpha, log_tau, 1)
    beta_estimate = -slope
    tau0_estimate = np.exp(intercept)

    # R² for the fit
    predicted = intercept + slope * log_alpha
    ss_res = np.sum((log_tau - predicted) ** 2)
    ss_tot = np.sum((log_tau - np.mean(log_tau)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        "n_users": len(tau_values),
        "tau0_estimate": tau0_estimate,
        "beta_estimate": beta_estimate,
        "r_squared": r_squared,
        "pearson_correlation": pearson_r,
        "pearson_p_value": pearson_p,
        "spearman_correlation": spearman_r,
        "spearman_p_value": spearman_p,
        "relationship_detected": pearson_p < 0.05 and abs(pearson_r) > 0.3,
        "interpretation": (
            f"τ(α) = {tau0_estimate:.2f} · α^(-{beta_estimate:.3f}), R²={r_squared:.3f}. "
            f"{'Significant' if pearson_p < 0.05 else 'Not significant'} "
            f"(p={pearson_p:.4f})."
        )
    }
