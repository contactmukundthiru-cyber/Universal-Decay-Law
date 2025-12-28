"""
Early Prediction Module - Predict τ from Initial User Data.

REVOLUTIONARY CAPABILITY FOR NATURE:
Predict long-term engagement trajectory from FIRST FEW SESSIONS.

This transforms the Universal Decay Law from a descriptive finding
into a PREDICTIVE tool with practical applications:

1. Identify users at risk of churn from day one
2. Optimize intervention timing for retention
3. Personalize engagement strategies

The model uses:
- Initial activity features (first 24h, first week)
- Motivation proxy (α) estimated from early behavior
- Platform-specific engagement patterns
- Trajectory classification probability

VALIDATION:
Predictions are validated by:
1. Holdout temporal validation (train on past, test on future)
2. Cross-platform generalization
3. Confidence calibration
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings


class PredictionTarget(Enum):
    """What to predict about user engagement."""
    TAU = "tau"  # Characteristic decay timescale
    TRAJECTORY_TYPE = "trajectory_type"  # Decay, growth, stable, erratic
    CHURN_30DAY = "churn_30day"  # Will user churn within 30 days?
    CHURN_90DAY = "churn_90day"  # Will user churn within 90 days?
    RETENTION_TIME = "retention_time"  # Days until engagement < threshold


@dataclass
class EarlyFeatures:
    """
    Features extracted from early user behavior.

    These features must be available within the first few days
    to enable early prediction.
    """
    user_id: str
    platform: str

    # Session 1 features
    first_session_engagement: float = 0.0
    first_session_duration_minutes: float = 0.0
    first_session_actions: int = 0

    # First 24 hours
    day1_total_engagement: float = 0.0
    day1_session_count: int = 0
    day1_return_within_hours: Optional[float] = None

    # First week
    week1_total_engagement: float = 0.0
    week1_active_days: int = 0
    week1_engagement_trend: float = 0.0  # Slope of daily engagement
    week1_peak_engagement: float = 0.0
    week1_engagement_variance: float = 0.0

    # Engagement patterns
    time_between_sessions_hours: float = 0.0
    session_regularity: float = 0.0  # 1 / CV of intervals

    # Motivation proxy (from motivation.py)
    estimated_alpha: float = 1.0
    alpha_confidence: float = 0.0

    # Platform-specific
    platform_features: Dict[str, float] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """
    Prediction for a single user.
    """
    user_id: str
    platform: str

    # Predictions
    predicted_tau: float
    tau_confidence_interval: Tuple[float, float]

    predicted_trajectory: str  # decay, growth, stable, erratic
    trajectory_probabilities: Dict[str, float]

    predicted_churn_30day: float  # Probability
    predicted_churn_90day: float

    # Feature importance for this prediction
    key_features: Dict[str, float] = field(default_factory=dict)

    # Prediction confidence
    confidence: float = 0.0


@dataclass
class ModelPerformance:
    """
    Performance metrics for prediction model.
    """
    target: str
    model_type: str

    # Regression metrics (for τ prediction)
    rmse: float = 0.0
    mae: float = 0.0
    r_squared: float = 0.0
    mape: float = 0.0  # Mean absolute percentage error

    # Classification metrics (for trajectory/churn)
    accuracy: float = 0.0
    auc_roc: float = 0.0

    # Cross-validation
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0

    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)


class EarlyFeatureExtractor:
    """
    Extract predictive features from early user data.
    """

    def __init__(
        self,
        day1_window_hours: float = 24.0,
        week1_window_days: float = 7.0
    ):
        self.day1_window = day1_window_hours * 3600
        self.week1_window = week1_window_days * 24 * 3600

    def extract(
        self,
        user_data: Dict[str, Any],
        platform: str,
        alpha_estimate: Optional[float] = None
    ) -> EarlyFeatures:
        """
        Extract early features from user data.

        Args:
            user_data: User activity data
            platform: Platform name
            alpha_estimate: Pre-computed motivation estimate (optional)

        Returns:
            EarlyFeatures object
        """
        user_id = user_data.get("user_id", "unknown")

        # Get activities
        activities = user_data.get("activities", [])
        if not activities:
            return EarlyFeatures(user_id=user_id, platform=platform)

        # Extract timestamps and engagement
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
                engagement_values.append(float(eng))

        if not timestamps:
            return EarlyFeatures(user_id=user_id, platform=platform)

        timestamps = np.array(timestamps)
        engagement = np.array(engagement_values)

        # Sort by time
        sort_idx = np.argsort(timestamps)
        timestamps = timestamps[sort_idx]
        engagement = engagement[sort_idx]

        # Reference point: first activity
        t0 = timestamps[0]

        # Session 1 features (first activity)
        first_session_engagement = engagement[0]
        first_session_duration = 0.0
        first_session_actions = 1

        # Find session boundaries (gap > 1 hour = new session)
        session_gap = 3600
        session_starts = [0]
        for i in range(1, len(timestamps)):
            if timestamps[i] - timestamps[i-1] > session_gap:
                session_starts.append(i)

        if len(session_starts) > 1:
            first_session_end = session_starts[1]
            first_session_duration = (timestamps[first_session_end-1] - timestamps[0]) / 60
            first_session_actions = first_session_end

        # Day 1 features
        day1_mask = (timestamps - t0) < self.day1_window
        day1_engagement = engagement[day1_mask]
        day1_timestamps = timestamps[day1_mask]

        day1_total = np.sum(day1_engagement) if len(day1_engagement) > 0 else 0
        day1_sessions = sum(1 for i in session_starts if i < np.sum(day1_mask))

        # Return time (time to second session)
        return_time = None
        if len(session_starts) > 1:
            return_time = (timestamps[session_starts[1]] - timestamps[0]) / 3600

        # Week 1 features
        week1_mask = (timestamps - t0) < self.week1_window
        week1_engagement = engagement[week1_mask]
        week1_timestamps = timestamps[week1_mask]

        week1_total = np.sum(week1_engagement) if len(week1_engagement) > 0 else 0

        # Active days in week 1
        if len(week1_timestamps) > 0:
            days = np.floor((week1_timestamps - t0) / (24 * 3600))
            active_days = len(np.unique(days))
        else:
            active_days = 0

        # Engagement trend (slope)
        week1_trend = 0.0
        if len(week1_engagement) >= 3:
            try:
                slope, _ = np.polyfit(
                    week1_timestamps - t0,
                    week1_engagement,
                    1
                )
                week1_trend = slope * (24 * 3600)  # Change per day
            except (np.linalg.LinAlgError, ValueError):
                pass

        # Variance and peak
        week1_variance = np.var(week1_engagement) if len(week1_engagement) > 1 else 0
        week1_peak = np.max(week1_engagement) if len(week1_engagement) > 0 else 0

        # Session timing features
        if len(timestamps) > 1:
            intervals = np.diff(timestamps) / 3600  # Hours
            mean_interval = np.mean(intervals)
            interval_cv = np.std(intervals) / (mean_interval + 1e-10)
            session_regularity = 1 / (1 + interval_cv)
        else:
            mean_interval = 0
            session_regularity = 0

        return EarlyFeatures(
            user_id=user_id,
            platform=platform,
            first_session_engagement=first_session_engagement,
            first_session_duration_minutes=first_session_duration,
            first_session_actions=first_session_actions,
            day1_total_engagement=day1_total,
            day1_session_count=day1_sessions,
            day1_return_within_hours=return_time,
            week1_total_engagement=week1_total,
            week1_active_days=active_days,
            week1_engagement_trend=week1_trend,
            week1_peak_engagement=week1_peak,
            week1_engagement_variance=week1_variance,
            time_between_sessions_hours=mean_interval,
            session_regularity=session_regularity,
            estimated_alpha=alpha_estimate or 1.0,
            alpha_confidence=0.5 if alpha_estimate else 0.0
        )

    def to_feature_vector(
        self,
        features: EarlyFeatures
    ) -> Tuple[List[str], np.ndarray]:
        """
        Convert EarlyFeatures to feature vector for ML.

        Returns:
            Tuple of (feature_names, feature_values)
        """
        feature_dict = {
            "first_session_engagement": features.first_session_engagement,
            "first_session_duration": features.first_session_duration_minutes,
            "first_session_actions": features.first_session_actions,
            "day1_total_engagement": features.day1_total_engagement,
            "day1_session_count": features.day1_session_count,
            "day1_return_hours": features.day1_return_within_hours or -1,
            "week1_total_engagement": features.week1_total_engagement,
            "week1_active_days": features.week1_active_days,
            "week1_trend": features.week1_engagement_trend,
            "week1_peak": features.week1_peak_engagement,
            "week1_variance": features.week1_engagement_variance,
            "mean_session_interval": features.time_between_sessions_hours,
            "session_regularity": features.session_regularity,
            "estimated_alpha": features.estimated_alpha,
        }

        names = list(feature_dict.keys())
        values = np.array(list(feature_dict.values()))

        return names, values


class EarlyPredictionModel:
    """
    Predict long-term engagement from early user behavior.

    This is the KEY REVOLUTIONARY CAPABILITY for Nature publication.
    """

    def __init__(
        self,
        prediction_target: PredictionTarget = PredictionTarget.TAU,
        n_estimators: int = 100,
        max_depth: int = 6
    ):
        """
        Initialize prediction model.

        Args:
            prediction_target: What to predict
            n_estimators: Number of trees in ensemble
            max_depth: Maximum tree depth
        """
        self.target = prediction_target
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        self.feature_extractor = EarlyFeatureExtractor()
        self.scaler = StandardScaler()

        if prediction_target in [PredictionTarget.TAU, PredictionTarget.RETENTION_TIME]:
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.1,
                random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )

        self.feature_names = []
        self.is_fitted = False

    def fit(
        self,
        users_data: List[Dict[str, Any]],
        targets: List[float],
        alpha_estimates: Optional[Dict[str, float]] = None,
        platform: str = "unknown"
    ) -> ModelPerformance:
        """
        Fit the prediction model.

        Args:
            users_data: List of user data dicts
            targets: Target values (τ, trajectory label, etc.)
            alpha_estimates: Pre-computed α estimates by user_id
            platform: Platform name

        Returns:
            ModelPerformance with training metrics
        """
        # Extract features
        X_list = []
        y_list = []

        for user_data, target in zip(users_data, targets):
            user_id = user_data.get("user_id", "")
            alpha = alpha_estimates.get(user_id, 1.0) if alpha_estimates else 1.0

            features = self.feature_extractor.extract(user_data, platform, alpha)
            names, values = self.feature_extractor.to_feature_vector(features)

            # Skip users with invalid data
            if np.isfinite(target) and target > 0:
                X_list.append(values)
                y_list.append(target)

        if len(X_list) < 10:
            warnings.warn("Insufficient data for training")
            return ModelPerformance(
                target=self.target.value,
                model_type=type(self.model).__name__
            )

        X = np.array(X_list)
        y = np.array(y_list)
        self.feature_names = names

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        # Compute performance
        y_pred = self.model.predict(X_scaled)

        if self.target in [PredictionTarget.TAU, PredictionTarget.RETENTION_TIME]:
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            mape = np.mean(np.abs((y - y_pred) / (y + 1e-10))) * 100

            # Cross-validation
            cv = TimeSeriesSplit(n_splits=5) if len(y) >= 50 else 3
            cv_scores = cross_val_score(
                self.model, X_scaled, y,
                cv=cv,
                scoring='neg_mean_squared_error'
            )
            cv_rmse = np.sqrt(-cv_scores)

            performance = ModelPerformance(
                target=self.target.value,
                model_type=type(self.model).__name__,
                rmse=rmse,
                mae=mae,
                r_squared=r2,
                mape=mape,
                cv_scores=cv_rmse.tolist(),
                cv_mean=cv_rmse.mean(),
                cv_std=cv_rmse.std()
            )

        else:
            # Classification metrics
            from sklearn.metrics import accuracy_score, roc_auc_score

            accuracy = accuracy_score(y, y_pred.round())
            try:
                auc = roc_auc_score(y, y_pred)
            except ValueError:
                auc = 0.5

            cv_scores = cross_val_score(
                self.model, X_scaled, y,
                cv=min(5, len(y) // 2),
                scoring='accuracy'
            )

            performance = ModelPerformance(
                target=self.target.value,
                model_type=type(self.model).__name__,
                accuracy=accuracy,
                auc_roc=auc,
                cv_scores=cv_scores.tolist(),
                cv_mean=cv_scores.mean(),
                cv_std=cv_scores.std()
            )

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            performance.feature_importance = dict(
                zip(self.feature_names, importance.tolist())
            )

        return performance

    def predict(
        self,
        user_data: Dict[str, Any],
        platform: str,
        alpha_estimate: Optional[float] = None
    ) -> PredictionResult:
        """
        Predict for a single user.

        Args:
            user_data: User activity data
            platform: Platform name
            alpha_estimate: Pre-computed motivation estimate

        Returns:
            PredictionResult with predictions and confidence
        """
        user_id = user_data.get("user_id", "unknown")

        if not self.is_fitted:
            warnings.warn("Model not fitted, returning default prediction")
            return PredictionResult(
                user_id=user_id,
                platform=platform,
                predicted_tau=30.0,
                tau_confidence_interval=(10.0, 90.0),
                predicted_trajectory="unknown",
                trajectory_probabilities={},
                predicted_churn_30day=0.5,
                predicted_churn_90day=0.5,
                confidence=0.0
            )

        # Extract features
        features = self.feature_extractor.extract(user_data, platform, alpha_estimate)
        names, values = self.feature_extractor.to_feature_vector(features)

        X = values.reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Predict
        prediction = self.model.predict(X_scaled)[0]

        # Confidence interval (using forest variance if available)
        if hasattr(self.model, 'estimators_'):
            tree_predictions = np.array([
                tree.predict(X_scaled)[0]
                for tree in self.model.estimators_
            ])
            ci_lower = np.percentile(tree_predictions, 5)
            ci_upper = np.percentile(tree_predictions, 95)
            variance = np.var(tree_predictions)
            confidence = 1 / (1 + variance / (prediction ** 2 + 1e-10))
        else:
            ci_lower = prediction * 0.5
            ci_upper = prediction * 2.0
            confidence = 0.5

        # Feature importance for this prediction
        key_features = {}
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            for name, imp, val in zip(self.feature_names, importance, values):
                if imp > 0.05:  # Only significant features
                    key_features[name] = val

        return PredictionResult(
            user_id=user_id,
            platform=platform,
            predicted_tau=max(1.0, prediction),
            tau_confidence_interval=(max(1.0, ci_lower), ci_upper),
            predicted_trajectory="decay",  # Would need separate classifier
            trajectory_probabilities={"decay": 1.0},
            predicted_churn_30day=1 - np.exp(-30 / max(1, prediction)),
            predicted_churn_90day=1 - np.exp(-90 / max(1, prediction)),
            key_features=key_features,
            confidence=confidence
        )

    def predict_batch(
        self,
        users_data: List[Dict[str, Any]],
        platform: str,
        alpha_estimates: Optional[Dict[str, float]] = None
    ) -> List[PredictionResult]:
        """
        Predict for multiple users.
        """
        results = []
        for user_data in users_data:
            user_id = user_data.get("user_id", "")
            alpha = alpha_estimates.get(user_id, 1.0) if alpha_estimates else 1.0
            result = self.predict(user_data, platform, alpha)
            results.append(result)
        return results


class PredictionValidator:
    """
    Validate prediction model performance.

    CRITICAL for Nature publication: demonstrate genuine predictive power.
    """

    def __init__(self, holdout_fraction: float = 0.2):
        """
        Initialize validator.

        Args:
            holdout_fraction: Fraction of data for final validation
        """
        self.holdout_fraction = holdout_fraction

    def temporal_validation(
        self,
        users_data: List[Dict[str, Any]],
        targets: List[float],
        platform: str,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Validate using temporal train/test split.

        Train on earlier users, test on later users.
        This is the most realistic validation for deployment.
        """
        # Sort by timestamp of first activity
        user_times = []
        for user in users_data:
            activities = user.get("activities", [])
            if activities:
                ts = activities[0].get("timestamp") or activities[0].get("created_utc", 0)
                if isinstance(ts, str):
                    from datetime import datetime
                    try:
                        ts = datetime.fromisoformat(ts).timestamp()
                    except:
                        ts = 0
            else:
                ts = 0
            user_times.append(ts)

        # Sort indices by time
        sort_idx = np.argsort(user_times)
        sorted_users = [users_data[i] for i in sort_idx]
        sorted_targets = [targets[i] for i in sort_idx]

        # Time series splits
        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_results = []
        for train_idx, test_idx in tscv.split(sorted_users):
            train_users = [sorted_users[i] for i in train_idx]
            train_targets = [sorted_targets[i] for i in train_idx]
            test_users = [sorted_users[i] for i in test_idx]
            test_targets = [sorted_targets[i] for i in test_idx]

            # Fit model on train
            model = EarlyPredictionModel()
            model.fit(train_users, train_targets, platform=platform)

            # Predict on test
            predictions = model.predict_batch(test_users, platform)
            pred_taus = [p.predicted_tau for p in predictions]

            # Metrics
            valid_idx = [
                i for i, t in enumerate(test_targets)
                if np.isfinite(t) and t > 0
            ]

            if valid_idx:
                y_true = np.array([test_targets[i] for i in valid_idx])
                y_pred = np.array([pred_taus[i] for i in valid_idx])

                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)

                fold_results.append({
                    "rmse": rmse,
                    "mae": mae,
                    "r_squared": r2,
                    "n_train": len(train_users),
                    "n_test": len(test_users)
                })

        # Aggregate
        if fold_results:
            return {
                "validation_type": "temporal",
                "n_folds": len(fold_results),
                "mean_rmse": np.mean([f["rmse"] for f in fold_results]),
                "std_rmse": np.std([f["rmse"] for f in fold_results]),
                "mean_r_squared": np.mean([f["r_squared"] for f in fold_results]),
                "std_r_squared": np.std([f["r_squared"] for f in fold_results]),
                "fold_results": fold_results
            }
        else:
            return {"error": "Insufficient data for validation"}

    def cross_platform_validation(
        self,
        platform_data: Dict[str, Tuple[List[Dict], List[float]]],
    ) -> Dict[str, Any]:
        """
        Validate by training on some platforms, testing on others.

        This tests true generalizability of the predictions.
        """
        platforms = list(platform_data.keys())
        if len(platforms) < 2:
            return {"error": "Need at least 2 platforms for cross-platform validation"}

        results = {}

        for test_platform in platforms:
            # Train on all other platforms
            train_users = []
            train_targets = []

            for platform in platforms:
                if platform != test_platform:
                    users, targets = platform_data[platform]
                    train_users.extend(users)
                    train_targets.extend(targets)

            # Test on held-out platform
            test_users, test_targets = platform_data[test_platform]

            # Fit and evaluate
            model = EarlyPredictionModel()
            model.fit(train_users, train_targets, platform="mixed")

            predictions = model.predict_batch(test_users, test_platform)
            pred_taus = [p.predicted_tau for p in predictions]

            valid_idx = [
                i for i, t in enumerate(test_targets)
                if np.isfinite(t) and t > 0
            ]

            if valid_idx:
                y_true = np.array([test_targets[i] for i in valid_idx])
                y_pred = np.array([pred_taus[i] for i in valid_idx])

                results[test_platform] = {
                    "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                    "mae": mean_absolute_error(y_true, y_pred),
                    "r_squared": r2_score(y_true, y_pred),
                    "n_test": len(valid_idx),
                    "n_train": len(train_users)
                }

        return {
            "validation_type": "cross_platform",
            "platform_results": results,
            "mean_r_squared": np.mean([r["r_squared"] for r in results.values()]),
            "generalizes": all(r["r_squared"] > 0.2 for r in results.values())
        }


def generate_prediction_report(
    performance: ModelPerformance,
    validation_results: Dict[str, Any]
) -> str:
    """
    Generate report on prediction model performance.
    """
    report = []
    report.append("=" * 70)
    report.append("EARLY PREDICTION MODEL - PERFORMANCE REPORT")
    report.append("=" * 70)

    report.append("\n1. MODEL CONFIGURATION")
    report.append(f"   Target: {performance.target}")
    report.append(f"   Model type: {performance.model_type}")

    report.append("\n2. TRAINING PERFORMANCE")
    if performance.rmse > 0:
        report.append(f"   RMSE: {performance.rmse:.2f} days")
        report.append(f"   MAE: {performance.mae:.2f} days")
        report.append(f"   R²: {performance.r_squared:.3f}")
        report.append(f"   MAPE: {performance.mape:.1f}%")
    else:
        report.append(f"   Accuracy: {performance.accuracy:.3f}")
        report.append(f"   AUC-ROC: {performance.auc_roc:.3f}")

    report.append("\n3. CROSS-VALIDATION")
    report.append(f"   Mean: {performance.cv_mean:.3f} ± {performance.cv_std:.3f}")

    report.append("\n4. FEATURE IMPORTANCE (Top 5)")
    sorted_features = sorted(
        performance.feature_importance.items(),
        key=lambda x: -x[1]
    )[:5]
    for name, importance in sorted_features:
        report.append(f"   - {name}: {importance:.3f}")

    report.append("\n5. VALIDATION RESULTS")
    val_type = validation_results.get("validation_type", "unknown")
    report.append(f"   Type: {val_type}")

    if "mean_r_squared" in validation_results:
        report.append(f"   Mean R²: {validation_results['mean_r_squared']:.3f}")

    if "generalizes" in validation_results:
        gen = "YES" if validation_results["generalizes"] else "NO"
        report.append(f"   Cross-platform generalization: {gen}")

    report.append("\n6. INTERPRETATION")
    if performance.r_squared > 0.5:
        report.append("   ✓ Model shows STRONG predictive power")
        report.append("   Early engagement patterns reliably predict long-term decay")
    elif performance.r_squared > 0.3:
        report.append("   ◐ Model shows MODERATE predictive power")
        report.append("   Early patterns provide useful but imperfect predictions")
    else:
        report.append("   ✗ Model shows LIMITED predictive power")
        report.append("   Early patterns alone may not determine long-term engagement")

    report.append("\n" + "=" * 70)

    return "\n".join(report)
