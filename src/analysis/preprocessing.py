"""
Data Preprocessing Module - Scientifically Rigorous Version.

DESIGN PRINCIPLES (addressing Nature reviewer concerns):
1. NO survivorship bias - include ALL users, weight by observation count
2. NO artificial data shaping - no max normalization, no outlier clipping
3. FULL TRANSPARENCY - report all transformations and exclusions
4. PRESERVE RAW DATA - always keep original alongside any transformations

This module explicitly avoids practices that could create artifacts:
- Max normalization (forces peak-then-decay pattern)
- Outlier clipping (reduces natural variance)
- Aggressive filtering (survivorship bias)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Literal
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from scipy import stats
import warnings


class NormalizationMethod(Enum):
    """
    Available normalization methods.

    NONE: Recommended for unbiased analysis
    INITIAL: Divides by first observation E(0) - acceptable if E(0) is meaningful
    ZSCORE: Standard normalization - changes shape, use with caution

    MAX normalization is NOT included because it forces a peak-then-decay pattern,
    which creates the very artifact we're trying to detect.
    """
    NONE = "none"
    INITIAL = "initial"  # E(t) / E(0)
    MEAN_INITIAL = "mean_initial"  # E(t) / mean(E[0:N])
    ZSCORE = "zscore"  # (E - mean) / std


@dataclass
class DataQualityMetrics:
    """Metrics for assessing data quality without excluding data."""
    n_observations: int
    duration_days: float
    observation_density: float  # observations per day
    max_gap_days: float
    variance: float
    has_missing: bool
    monotonicity: float  # -1 to 1, negative = decreasing trend
    noise_level: float  # estimated noise as fraction of signal


@dataclass
class TransparencyReport:
    """
    Complete report of all data transformations applied.

    This MUST be included in any publication to allow reproducibility
    and assessment of potential biases.
    """
    # What was done
    normalization_method: str
    normalization_factor: float
    outlier_handling: str
    n_outliers_detected: int
    n_outliers_modified: int

    # Data characteristics
    n_original: int
    n_after_invalid_removal: int
    n_final: int

    # Warnings
    warnings: List[str] = field(default_factory=list)


@dataclass
class PreprocessedUser:
    """
    Preprocessed user data with full provenance tracking.

    IMPORTANT: Both raw and processed data are preserved for validation.
    """
    user_id: str
    platform: str

    # Processed data
    time: NDArray[np.float64]
    engagement: NDArray[np.float64]

    # Original raw data (ALWAYS preserved)
    raw_time: NDArray[np.float64]
    raw_engagement: NDArray[np.float64]

    # Metadata
    quality: DataQualityMetrics
    report: TransparencyReport

    # Weight for analysis (based on data quality, NOT filtering)
    analysis_weight: float = 1.0


@dataclass
class DatasetPreprocessingReport:
    """
    Dataset-level report for detecting survivorship bias.

    If inclusion_rate < 0.5, results may be subject to survivorship bias
    and should be interpreted with extreme caution.
    """
    total_users_in_source: int
    users_processed: int
    users_with_valid_data: int

    # Exclusion reasons (for transparency)
    excluded_empty: int = 0
    excluded_single_point: int = 0
    excluded_no_variance: int = 0

    # Distribution statistics (for ALL users, not just included)
    observation_count_distribution: Dict[str, float] = field(default_factory=dict)
    duration_distribution: Dict[str, float] = field(default_factory=dict)

    @property
    def inclusion_rate(self) -> float:
        if self.total_users_in_source == 0:
            return 0.0
        return self.users_with_valid_data / self.total_users_in_source

    @property
    def survivorship_bias_warning(self) -> Optional[str]:
        if self.inclusion_rate < 0.5:
            return (
                f"WARNING: Only {self.inclusion_rate:.1%} of users included. "
                "Results may be subject to survivorship bias."
            )
        return None


class RobustStatistics:
    """
    Robust statistical methods that work WITH outliers, not by removing them.
    """

    @staticmethod
    def huber_mean(x: NDArray, c: float = 1.345) -> float:
        """
        Huber M-estimator for robust mean estimation.

        Downweights outliers instead of removing them.
        """
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return np.nan

        mu = np.median(x)
        s = 1.4826 * np.median(np.abs(x - mu))

        if s < 1e-10:
            return mu

        for _ in range(50):  # Iterate to convergence
            r = (x - mu) / s
            weights = np.where(np.abs(r) <= c, 1.0, c / np.abs(r))
            mu_new = np.sum(weights * x) / np.sum(weights)
            if np.abs(mu_new - mu) < 1e-6:
                break
            mu = mu_new

        return mu

    @staticmethod
    def mad(x: NDArray) -> float:
        """Median Absolute Deviation - robust measure of spread."""
        x = x[np.isfinite(x)]
        return np.median(np.abs(x - np.median(x)))

    @staticmethod
    def robust_regression_weights(residuals: NDArray) -> NDArray:
        """
        Compute weights for weighted least squares that downweight outliers.

        Uses bisquare (Tukey) weights.
        """
        c = 4.685  # Tukey's constant
        mad = RobustStatistics.mad(residuals)

        if mad < 1e-10:
            return np.ones_like(residuals)

        u = residuals / (c * mad * 1.4826)
        weights = np.where(np.abs(u) < 1, (1 - u**2)**2, 0)
        return weights


class MinimalPreprocessor:
    """
    Minimal preprocessing that preserves data integrity.

    Philosophy: It's better to analyze noisy real data than clean artifacts.
    """

    def __init__(
        self,
        normalization: NormalizationMethod = NormalizationMethod.NONE,
        initial_window: int = 3,  # For MEAN_INITIAL
        detect_outliers: bool = True,  # Detect but don't remove/modify
        min_observations: int = 2,  # Absolute minimum
    ):
        self.normalization = normalization
        self.initial_window = initial_window
        self.detect_outliers = detect_outliers
        self.min_observations = min_observations

    def process(
        self,
        user_id: str,
        platform: str,
        time: NDArray,
        engagement: NDArray
    ) -> Optional[PreprocessedUser]:
        """
        Process user data with minimal transformation.

        Returns None ONLY if data is fundamentally unusable (empty, single point).
        """
        time = np.asarray(time, dtype=np.float64)
        engagement = np.asarray(engagement, dtype=np.float64)

        # Store raw data
        raw_time = time.copy()
        raw_engagement = engagement.copy()

        warnings_list = []

        # Remove only truly invalid values (NaN, Inf)
        valid_mask = np.isfinite(time) & np.isfinite(engagement)
        n_invalid = np.sum(~valid_mask)
        if n_invalid > 0:
            warnings_list.append(f"Removed {n_invalid} invalid values")

        time = time[valid_mask]
        engagement = engagement[valid_mask]

        # Sort by time
        sort_idx = np.argsort(time)
        time = time[sort_idx]
        engagement = engagement[sort_idx]

        n_original = len(raw_time)
        n_valid = len(time)

        # Minimum check - only truly unusable data
        if n_valid < self.min_observations:
            return None

        # Compute quality metrics (NOT for filtering, for weighting)
        quality = self._compute_quality(time, engagement)

        # Detect outliers (but don't remove them)
        n_outliers = 0
        if self.detect_outliers:
            outlier_mask = self._detect_outliers_mad(engagement)
            n_outliers = np.sum(outlier_mask)
            if n_outliers > 0:
                warnings_list.append(
                    f"Detected {n_outliers} potential outliers (preserved in data)"
                )

        # Apply normalization (if requested)
        norm_factor = 1.0
        if self.normalization == NormalizationMethod.INITIAL:
            if engagement[0] > 0:
                norm_factor = engagement[0]
                engagement = engagement / norm_factor
            else:
                warnings_list.append("Initial engagement is 0, normalization skipped")

        elif self.normalization == NormalizationMethod.MEAN_INITIAL:
            window = min(self.initial_window, len(engagement))
            initial_mean = np.mean(engagement[:window])
            if initial_mean > 0:
                norm_factor = initial_mean
                engagement = engagement / norm_factor

        elif self.normalization == NormalizationMethod.ZSCORE:
            mean_e = np.mean(engagement)
            std_e = np.std(engagement)
            if std_e > 1e-10:
                norm_factor = std_e
                engagement = (engagement - mean_e) / std_e
            else:
                warnings_list.append("Zero variance, z-score normalization skipped")

        # Create transparency report
        report = TransparencyReport(
            normalization_method=self.normalization.value,
            normalization_factor=norm_factor,
            outlier_handling="none (detected but preserved)",
            n_outliers_detected=n_outliers,
            n_outliers_modified=0,
            n_original=n_original,
            n_after_invalid_removal=n_valid,
            n_final=len(engagement),
            warnings=warnings_list
        )

        # Compute analysis weight (based on data quality)
        weight = self._compute_weight(quality)

        return PreprocessedUser(
            user_id=user_id,
            platform=platform,
            time=time,
            engagement=engagement,
            raw_time=raw_time,
            raw_engagement=raw_engagement,
            quality=quality,
            report=report,
            analysis_weight=weight
        )

    def _compute_quality(self, time: NDArray, engagement: NDArray) -> DataQualityMetrics:
        """Compute data quality metrics."""
        n = len(time)
        duration = time[-1] - time[0] if n > 1 else 0
        density = n / max(duration, 1)

        # Max gap
        if n > 1:
            gaps = np.diff(time)
            max_gap = float(np.max(gaps))
        else:
            max_gap = 0

        # Variance
        variance = float(np.var(engagement))

        # Monotonicity (Spearman correlation with time)
        if n > 2:
            monotonicity = float(stats.spearmanr(time, engagement)[0])
            if np.isnan(monotonicity):
                monotonicity = 0.0
        else:
            monotonicity = 0.0

        # Noise level (residual variance from linear fit)
        if n > 2:
            slope, intercept = np.polyfit(time, engagement, 1)
            fitted = slope * time + intercept
            residuals = engagement - fitted
            noise_level = np.std(residuals) / (np.std(engagement) + 1e-10)
        else:
            noise_level = 0.0

        return DataQualityMetrics(
            n_observations=n,
            duration_days=duration,
            observation_density=density,
            max_gap_days=max_gap,
            variance=variance,
            has_missing=False,  # Already filtered
            monotonicity=monotonicity,
            noise_level=noise_level
        )

    def _detect_outliers_mad(self, data: NDArray, threshold: float = 5.0) -> NDArray:
        """
        Detect outliers using Median Absolute Deviation.

        Uses a permissive threshold (5 MAD) to only flag extreme values.
        """
        median = np.median(data)
        mad = np.median(np.abs(data - median))

        if mad < 1e-10:
            return np.zeros(len(data), dtype=bool)

        modified_z = 0.6745 * np.abs(data - median) / mad
        return modified_z > threshold

    def _compute_weight(self, quality: DataQualityMetrics) -> float:
        """
        Compute analysis weight based on data quality.

        Higher weight = more reliable data, but NO exclusions.
        """
        weight = 1.0

        # More observations = higher weight
        weight *= min(1.0, quality.n_observations / 20)

        # Longer duration = higher weight
        weight *= min(1.0, quality.duration_days / 30)

        # Higher density = higher weight
        weight *= min(1.0, quality.observation_density / 0.3)

        return max(0.1, weight)  # Minimum weight, never exclude


class DatasetPreprocessor:
    """
    Process entire datasets while tracking potential biases.
    """

    def __init__(self, preprocessor: Optional[MinimalPreprocessor] = None):
        self.preprocessor = preprocessor or MinimalPreprocessor()
        self.report = DatasetPreprocessingReport(
            total_users_in_source=0,
            users_processed=0,
            users_with_valid_data=0
        )

    def process_dataset(
        self,
        users: List[Dict[str, Any]]
    ) -> Tuple[List[PreprocessedUser], DatasetPreprocessingReport]:
        """
        Process all users in a dataset.

        Args:
            users: List of dicts with user_id, platform, time, engagement

        Returns:
            Tuple of (processed_users, dataset_report)
        """
        self.report.total_users_in_source = len(users)
        processed = []

        all_observation_counts = []
        all_durations = []

        for user_data in users:
            self.report.users_processed += 1

            time = np.asarray(user_data.get("time", []))
            engagement = np.asarray(user_data.get("engagement", []))

            # Track statistics for ALL users (for bias detection)
            valid_mask = np.isfinite(time) & np.isfinite(engagement)
            n_valid = np.sum(valid_mask)
            all_observation_counts.append(n_valid)

            if n_valid > 1:
                valid_time = time[valid_mask]
                all_durations.append(valid_time[-1] - valid_time[0])

            # Check for trivially invalid data
            if len(time) == 0 or len(engagement) == 0:
                self.report.excluded_empty += 1
                continue

            if np.sum(valid_mask) < 2:
                self.report.excluded_single_point += 1
                continue

            if np.var(engagement[valid_mask]) < 1e-10:
                self.report.excluded_no_variance += 1
                continue

            # Process
            result = self.preprocessor.process(
                user_id=user_data.get("user_id", f"user_{self.report.users_processed}"),
                platform=user_data.get("platform", "unknown"),
                time=time,
                engagement=engagement
            )

            if result is not None:
                processed.append(result)
                self.report.users_with_valid_data += 1

        # Compute distribution statistics for ALL users
        if all_observation_counts:
            self.report.observation_count_distribution = {
                "min": float(np.min(all_observation_counts)),
                "median": float(np.median(all_observation_counts)),
                "mean": float(np.mean(all_observation_counts)),
                "max": float(np.max(all_observation_counts)),
                "p10": float(np.percentile(all_observation_counts, 10)),
                "p90": float(np.percentile(all_observation_counts, 90))
            }

        if all_durations:
            self.report.duration_distribution = {
                "min": float(np.min(all_durations)),
                "median": float(np.median(all_durations)),
                "mean": float(np.mean(all_durations)),
                "max": float(np.max(all_durations))
            }

        # Issue warning if significant exclusions
        if self.report.survivorship_bias_warning:
            warnings.warn(self.report.survivorship_bias_warning)

        return processed, self.report
