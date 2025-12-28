"""
Survivorship Bias Modeling Module.

CRITICAL FOR NATURE PUBLICATION:
The Universal Decay Law may only apply to a subset of users.
Ignoring users who "drop out" early creates survivorship bias.

This module implements a TWO-STAGE analysis:
1. Stage 1: Model who becomes engaged (selection model)
2. Stage 2: Apply decay law only to engaged users (outcome model)

This approach:
- Explicitly models the "zeroes" (users filtered out)
- Quantifies what fraction of users the law applies to
- Provides honest reporting of sample selection

THEORETICAL BASIS:
Heckman Selection Model (Nobel Prize 2000)
- Selection and outcome processes may be correlated
- Ignoring selection leads to biased estimates
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings


@dataclass
class UserEligibility:
    """
    Assessment of whether a user is eligible for decay analysis.

    Users may be ineligible due to:
    - Insufficient data points
    - Too short observation period
    - Never engaged (zero activity)
    - Non-decay trajectory (growth, stable, erratic)
    """
    user_id: str
    platform: str

    # Eligibility for decay analysis
    is_eligible: bool
    eligibility_probability: float  # From selection model

    # Reason for ineligibility (if applicable)
    exclusion_reason: Optional[str] = None

    # Features used in selection model
    initial_activity_level: float = 0.0
    observation_duration_days: float = 0.0
    n_observations: int = 0
    first_week_engagement: float = 0.0


@dataclass
class SelectionModelResult:
    """
    Results from the selection (Stage 1) model.

    This model predicts which users will become "engaged enough"
    to be included in the decay analysis.
    """
    # Model performance
    auc_roc: float
    accuracy: float
    precision: float
    recall: float

    # Model coefficients (for interpretation)
    feature_names: List[str]
    coefficients: List[float]
    intercept: float

    # Selection statistics
    n_total_users: int
    n_selected: int
    selection_rate: float

    # Cross-validation
    cv_auc_mean: float
    cv_auc_std: float

    # Interpretation
    key_predictors: Dict[str, str] = field(default_factory=dict)


@dataclass
class TwoStageResult:
    """
    Complete two-stage analysis result.

    Stage 1: Who becomes engaged (selection)
    Stage 2: How does engagement decay (outcome)
    """
    # Stage 1 results
    selection_model: SelectionModelResult

    # Stage 2 results (on selected sample)
    n_decay_users: int
    mean_tau: float
    std_tau: float
    mean_gamma: float

    # Comparison: selected vs. all users
    selection_bias_estimate: float  # How different is selected sample?

    # Generalizability
    law_applicability_rate: float  # What fraction of users does the law apply to?

    # Warnings and notes
    warnings: List[str] = field(default_factory=list)


class SelectionFeatureExtractor:
    """
    Extract features that predict user eligibility for decay analysis.

    These features must be available BEFORE we know if a user
    shows decay behavior.
    """

    @staticmethod
    def extract_features(
        user_data: Dict[str, Any],
        early_window_days: float = 7.0
    ) -> Dict[str, float]:
        """
        Extract features from early user data.

        Args:
            user_data: User activity data
            early_window_days: Window for "early" features

        Returns:
            Dictionary of feature name -> value
        """
        features = {}

        # Get activity data
        activities = user_data.get("activities", [])
        timestamps = []
        engagement_values = []

        for a in activities:
            ts = a.get("timestamp") or a.get("created_utc")
            eng = a.get("engagement") or a.get("value") or 1.0

            if ts is not None:
                if isinstance(ts, (int, float)):
                    timestamps.append(ts)
                else:
                    try:
                        from datetime import datetime
                        timestamps.append(datetime.fromisoformat(str(ts)).timestamp())
                    except (ValueError, TypeError):
                        continue
                engagement_values.append(eng)

        if not timestamps:
            return {
                "n_observations": 0,
                "observation_duration": 0,
                "first_week_engagement": 0,
                "initial_activity_rate": 0,
                "engagement_variance": 0,
            }

        timestamps = np.array(timestamps)
        engagement_values = np.array(engagement_values)

        # Sort by time
        sort_idx = np.argsort(timestamps)
        timestamps = timestamps[sort_idx]
        engagement_values = engagement_values[sort_idx]

        # Basic features
        features["n_observations"] = len(timestamps)
        features["observation_duration"] = (timestamps[-1] - timestamps[0]) / (24 * 3600)

        # Early window features
        early_cutoff = timestamps[0] + early_window_days * 24 * 3600
        early_mask = timestamps <= early_cutoff

        if np.sum(early_mask) > 0:
            early_engagement = engagement_values[early_mask]
            features["first_week_engagement"] = np.mean(early_engagement)
            features["first_week_max"] = np.max(early_engagement)
            features["first_week_count"] = np.sum(early_mask)
        else:
            features["first_week_engagement"] = 0
            features["first_week_max"] = 0
            features["first_week_count"] = 0

        # Activity rate (observations per day)
        if features["observation_duration"] > 0:
            features["initial_activity_rate"] = features["first_week_count"] / min(
                early_window_days, features["observation_duration"]
            )
        else:
            features["initial_activity_rate"] = 0

        # Engagement variance (normalized)
        if len(engagement_values) > 1:
            features["engagement_variance"] = np.std(engagement_values) / (
                np.mean(engagement_values) + 1e-10
            )
        else:
            features["engagement_variance"] = 0

        # Time between activities (regularity)
        if len(timestamps) > 1:
            intervals = np.diff(timestamps) / 3600  # Hours
            features["mean_interval_hours"] = np.mean(intervals)
            features["interval_variance"] = np.std(intervals) / (np.mean(intervals) + 1e-10)
        else:
            features["mean_interval_hours"] = 0
            features["interval_variance"] = 0

        return features


class SurvivorshipBiasAnalyzer:
    """
    Analyze and correct for survivorship bias in decay analysis.

    The key insight: Not all users show decay. Some:
    - Never engage enough to measure decay
    - Show growth or stable patterns
    - Have insufficient data

    Ignoring these users biases results toward "decay is universal."
    """

    def __init__(
        self,
        min_observations_for_decay: int = 10,
        min_duration_days: float = 7.0,
        selection_features: Optional[List[str]] = None
    ):
        """
        Initialize analyzer.

        Args:
            min_observations_for_decay: Minimum points to fit decay model
            min_duration_days: Minimum observation period
            selection_features: Features to use in selection model
        """
        self.min_observations = min_observations_for_decay
        self.min_duration = min_duration_days
        self.selection_features = selection_features or [
            "n_observations",
            "observation_duration",
            "first_week_engagement",
            "initial_activity_rate",
            "engagement_variance",
        ]

        self.feature_extractor = SelectionFeatureExtractor()
        self.selection_model = None
        self.scaler = StandardScaler()

    def assess_eligibility(
        self,
        users_data: List[Dict[str, Any]]
    ) -> Tuple[List[UserEligibility], Dict[str, int]]:
        """
        Assess which users are eligible for decay analysis.

        This is DESCRIPTIVE - it categorizes users by exclusion reason.

        Returns:
            Tuple of (eligibility list, exclusion counts)
        """
        eligibilities = []
        exclusion_counts = {
            "eligible": 0,
            "insufficient_observations": 0,
            "too_short_duration": 0,
            "no_activity": 0,
            "no_variance": 0,
        }

        for user_data in users_data:
            user_id = user_data.get("user_id", "unknown")
            platform = user_data.get("platform", "unknown")

            features = self.feature_extractor.extract_features(user_data)

            # Check eligibility criteria
            is_eligible = True
            exclusion_reason = None

            if features["n_observations"] == 0:
                is_eligible = False
                exclusion_reason = "no_activity"
                exclusion_counts["no_activity"] += 1

            elif features["n_observations"] < self.min_observations:
                is_eligible = False
                exclusion_reason = "insufficient_observations"
                exclusion_counts["insufficient_observations"] += 1

            elif features["observation_duration"] < self.min_duration:
                is_eligible = False
                exclusion_reason = "too_short_duration"
                exclusion_counts["too_short_duration"] += 1

            elif features.get("engagement_variance", 0) < 0.01:
                is_eligible = False
                exclusion_reason = "no_variance"
                exclusion_counts["no_variance"] += 1

            else:
                exclusion_counts["eligible"] += 1

            eligibilities.append(UserEligibility(
                user_id=user_id,
                platform=platform,
                is_eligible=is_eligible,
                eligibility_probability=1.0 if is_eligible else 0.0,
                exclusion_reason=exclusion_reason,
                initial_activity_level=features.get("first_week_engagement", 0),
                observation_duration_days=features.get("observation_duration", 0),
                n_observations=int(features.get("n_observations", 0)),
                first_week_engagement=features.get("first_week_engagement", 0)
            ))

        return eligibilities, exclusion_counts

    def fit_selection_model(
        self,
        users_data: List[Dict[str, Any]],
        is_selected: List[bool]
    ) -> SelectionModelResult:
        """
        Fit a model predicting which users will be selected for analysis.

        Stage 1 of the two-stage approach.

        Args:
            users_data: All user data
            is_selected: Boolean indicating if user was selected

        Returns:
            SelectionModelResult with model details
        """
        # Extract features
        feature_matrix = []
        valid_labels = []

        for user_data, selected in zip(users_data, is_selected):
            features = self.feature_extractor.extract_features(user_data)

            feature_vec = [features.get(f, 0) for f in self.selection_features]
            feature_matrix.append(feature_vec)
            valid_labels.append(1 if selected else 0)

        X = np.array(feature_matrix)
        y = np.array(valid_labels)

        # Handle case where all or none selected
        if y.sum() == 0 or y.sum() == len(y):
            return SelectionModelResult(
                auc_roc=0.5,
                accuracy=y.mean() if y.sum() == len(y) else 1 - y.mean(),
                precision=0.0,
                recall=0.0,
                feature_names=self.selection_features,
                coefficients=[0.0] * len(self.selection_features),
                intercept=0.0,
                n_total_users=len(y),
                n_selected=int(y.sum()),
                selection_rate=y.mean(),
                cv_auc_mean=0.5,
                cv_auc_std=0.0
            )

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit logistic regression
        self.selection_model = LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            random_state=42
        )
        self.selection_model.fit(X_scaled, y)

        # Predictions
        y_pred = self.selection_model.predict(X_scaled)
        y_prob = self.selection_model.predict_proba(X_scaled)[:, 1]

        # Metrics
        from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

        try:
            auc = roc_auc_score(y, y_prob)
        except ValueError:
            auc = 0.5

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)

        # Cross-validation
        try:
            cv_scores = cross_val_score(
                LogisticRegression(penalty='l2', C=1.0, max_iter=1000),
                X_scaled, y,
                cv=min(5, max(2, int(y.sum() / 2))),
                scoring='roc_auc'
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception:
            cv_mean = auc
            cv_std = 0.0

        # Interpret coefficients
        coefficients = self.selection_model.coef_[0].tolist()
        key_predictors = {}

        for i, (name, coef) in enumerate(zip(self.selection_features, coefficients)):
            if abs(coef) > 0.5:
                direction = "more likely" if coef > 0 else "less likely"
                key_predictors[name] = f"Higher values make selection {direction}"

        return SelectionModelResult(
            auc_roc=auc,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            feature_names=self.selection_features,
            coefficients=coefficients,
            intercept=self.selection_model.intercept_[0],
            n_total_users=len(y),
            n_selected=int(y.sum()),
            selection_rate=y.mean(),
            cv_auc_mean=cv_mean,
            cv_auc_std=cv_std,
            key_predictors=key_predictors
        )

    def predict_selection_probability(
        self,
        user_data: Dict[str, Any]
    ) -> float:
        """
        Predict probability that a user would be selected for analysis.

        Requires fit_selection_model to be called first.
        """
        if self.selection_model is None:
            return 0.5

        features = self.feature_extractor.extract_features(user_data)
        feature_vec = np.array([[features.get(f, 0) for f in self.selection_features]])
        feature_scaled = self.scaler.transform(feature_vec)

        return self.selection_model.predict_proba(feature_scaled)[0, 1]

    def compute_selection_bias(
        self,
        all_users: List[Dict[str, Any]],
        selected_users: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Quantify how different selected users are from all users.

        Returns metrics showing potential selection bias.
        """
        all_features = []
        selected_features = []

        for user in all_users:
            all_features.append(self.feature_extractor.extract_features(user))

        for user in selected_users:
            selected_features.append(self.feature_extractor.extract_features(user))

        bias_metrics = {}

        for feature_name in self.selection_features:
            all_values = [f.get(feature_name, 0) for f in all_features]
            selected_values = [f.get(feature_name, 0) for f in selected_features]

            all_values = np.array([v for v in all_values if np.isfinite(v)])
            selected_values = np.array([v for v in selected_values if np.isfinite(v)])

            if len(all_values) > 5 and len(selected_values) > 5:
                # Cohen's d effect size
                pooled_std = np.sqrt(
                    (np.var(all_values) + np.var(selected_values)) / 2
                )
                if pooled_std > 0:
                    cohens_d = (np.mean(selected_values) - np.mean(all_values)) / pooled_std
                else:
                    cohens_d = 0

                # Mann-Whitney U test
                try:
                    stat, p_value = stats.mannwhitneyu(
                        all_values, selected_values, alternative='two-sided'
                    )
                except ValueError:
                    p_value = 1.0

                bias_metrics[feature_name] = {
                    "effect_size": cohens_d,
                    "p_value": p_value,
                    "all_mean": np.mean(all_values),
                    "selected_mean": np.mean(selected_values),
                    "significant_bias": abs(cohens_d) > 0.5 and p_value < 0.05
                }

        return bias_metrics

    def run_two_stage_analysis(
        self,
        all_users: List[Dict[str, Any]],
        decay_results: List[Dict[str, Any]]
    ) -> TwoStageResult:
        """
        Run complete two-stage analysis.

        Args:
            all_users: All users in the dataset
            decay_results: Results from decay analysis (only for selected users)

        Returns:
            TwoStageResult with complete analysis
        """
        # Map decay results to user IDs
        decay_user_ids = {r.get("user_id") for r in decay_results}

        # Create selection labels
        is_selected = [
            user.get("user_id") in decay_user_ids
            for user in all_users
        ]

        # Stage 1: Fit selection model
        selection_result = self.fit_selection_model(all_users, is_selected)

        # Stage 2: Summarize decay results
        tau_values = [r.get("tau", 0) for r in decay_results if r.get("tau", 0) > 0]
        gamma_values = [r.get("gamma", 0) for r in decay_results if r.get("gamma")]

        mean_tau = np.mean(tau_values) if tau_values else 0
        std_tau = np.std(tau_values) if tau_values else 0
        mean_gamma = np.mean(gamma_values) if gamma_values else 0

        # Compute selection bias
        selected_users = [u for u, s in zip(all_users, is_selected) if s]
        bias_metrics = self.compute_selection_bias(all_users, selected_users)

        # Overall selection bias estimate (max effect size)
        max_bias = max(
            abs(m.get("effect_size", 0))
            for m in bias_metrics.values()
        ) if bias_metrics else 0

        # Warnings
        warnings_list = []

        if selection_result.selection_rate < 0.3:
            warnings_list.append(
                f"Only {selection_result.selection_rate:.1%} of users selected. "
                "Results may be subject to severe survivorship bias."
            )

        if max_bias > 0.8:
            warnings_list.append(
                "Large selection bias detected. Selected users are very "
                "different from the general population."
            )

        if selection_result.auc_roc > 0.8:
            warnings_list.append(
                f"Selection is highly predictable (AUC={selection_result.auc_roc:.2f}). "
                "This suggests systematic differences between selected and excluded users."
            )

        return TwoStageResult(
            selection_model=selection_result,
            n_decay_users=len(decay_results),
            mean_tau=mean_tau,
            std_tau=std_tau,
            mean_gamma=mean_gamma,
            selection_bias_estimate=max_bias,
            law_applicability_rate=selection_result.selection_rate,
            warnings=warnings_list
        )


def generate_survivorship_report(result: TwoStageResult) -> str:
    """
    Generate a transparency report on survivorship bias.

    This MUST be included in any publication.
    """
    report = []
    report.append("=" * 70)
    report.append("SURVIVORSHIP BIAS ANALYSIS REPORT")
    report.append("=" * 70)

    report.append("\n1. SAMPLE SELECTION")
    report.append(f"   Total users in dataset: {result.selection_model.n_total_users}")
    report.append(f"   Users included in decay analysis: {result.selection_model.n_selected}")
    report.append(f"   Selection rate: {result.selection_model.selection_rate:.1%}")

    report.append("\n2. SELECTION MODEL PERFORMANCE")
    report.append(f"   AUC-ROC: {result.selection_model.auc_roc:.3f}")
    report.append(f"   Cross-validated AUC: {result.selection_model.cv_auc_mean:.3f} ± {result.selection_model.cv_auc_std:.3f}")
    report.append(f"   Accuracy: {result.selection_model.accuracy:.3f}")

    report.append("\n3. KEY SELECTION PREDICTORS")
    for feature, interpretation in result.selection_model.key_predictors.items():
        report.append(f"   - {feature}: {interpretation}")

    report.append("\n4. SELECTION BIAS ASSESSMENT")
    report.append(f"   Maximum effect size: {result.selection_bias_estimate:.3f}")
    if result.selection_bias_estimate < 0.2:
        report.append("   Interpretation: Minimal selection bias")
    elif result.selection_bias_estimate < 0.5:
        report.append("   Interpretation: Small to moderate selection bias")
    elif result.selection_bias_estimate < 0.8:
        report.append("   Interpretation: Moderate to large selection bias")
    else:
        report.append("   Interpretation: LARGE selection bias - results may not generalize")

    report.append("\n5. DECAY ANALYSIS RESULTS (ON SELECTED SAMPLE)")
    report.append(f"   N users: {result.n_decay_users}")
    report.append(f"   Mean τ: {result.mean_tau:.2f} ± {result.std_tau:.2f}")
    report.append(f"   Mean γ: {result.mean_gamma:.3f}")

    report.append("\n6. GENERALIZABILITY")
    report.append(f"   The Universal Decay Law applies to approximately "
                  f"{result.law_applicability_rate:.1%} of users.")
    report.append("   This represents users who:")
    report.append("   - Had sufficient observation data")
    report.append("   - Showed clear decay patterns")
    report.append("   - Met minimum engagement thresholds")

    if result.warnings:
        report.append("\n7. WARNINGS")
        for warning in result.warnings:
            report.append(f"   ⚠ {warning}")

    report.append("\n" + "=" * 70)

    return "\n".join(report)
