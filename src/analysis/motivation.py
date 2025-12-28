"""
Motivation Parameter (α) Estimation - Independent Measurement.

CRITICAL REQUIREMENT FOR NATURE PUBLICATION:
α must be measured INDEPENDENTLY of τ to avoid circular reasoning.

This module implements three methods for independent α estimation:

1. PROXY VARIABLES: Use observable user behavior patterns that correlate
   with intrinsic/extrinsic motivation (e.g., contribution patterns,
   activity types, engagement contexts).

2. NATURAL EXPERIMENTS: Identify platform changes (gamification features,
   reward systems) and measure behavioral changes before/after.

3. EXPERIMENTAL FRAMEWORK: Design for future controlled experiments
   with pre-registered hypotheses.

The α parameter represents the degree of intrinsic motivation:
- High α (>1): Intrinsically motivated (curiosity, mastery, purpose)
- Low α (<1): Extrinsically motivated (rewards, social recognition, status)

THEORETICAL BASIS:
Self-Determination Theory (Deci & Ryan) distinguishes intrinsic motivation
(doing something for its inherent satisfaction) from extrinsic motivation
(doing something for external rewards or to avoid punishment).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from datetime import datetime, timedelta
import warnings


class MotivationProxy(Enum):
    """
    Observable proxy variables for motivation type.

    Each proxy has theoretical grounding in Self-Determination Theory.
    """
    # Content creation vs. consumption patterns
    CREATION_RATIO = "creation_ratio"

    # Engagement with niche vs. mainstream content
    NICHE_ENGAGEMENT = "niche_engagement"

    # Response to explicit rewards
    REWARD_SENSITIVITY = "reward_sensitivity"

    # Consistency over time (intrinsic = more consistent)
    TEMPORAL_CONSISTENCY = "temporal_consistency"

    # Engagement depth (comments vs. likes)
    ENGAGEMENT_DEPTH = "engagement_depth"

    # Social vs. content focus
    SOCIAL_FOCUS = "social_focus"

    # Time of engagement (late night = often more intrinsic)
    ENGAGEMENT_TIMING = "engagement_timing"


@dataclass
class MotivationEstimate:
    """
    Independent motivation (α) estimate for a user.

    Includes uncertainty and provenance for transparency.
    """
    user_id: str
    platform: str

    # Primary estimate
    alpha: float
    alpha_uncertainty: float

    # Method used
    estimation_method: str
    proxies_used: List[str] = field(default_factory=list)

    # Individual proxy scores (for transparency)
    proxy_scores: Dict[str, float] = field(default_factory=dict)

    # Confidence in estimate
    confidence: float = 0.0  # 0-1

    # Data quality
    n_observations: int = 0
    observation_period_days: float = 0.0


@dataclass
class NaturalExperimentResult:
    """
    Results from analyzing a natural experiment.

    Natural experiments occur when platforms introduce changes that
    affect motivation (e.g., new gamification features).
    """
    experiment_name: str
    platform: str
    intervention_date: datetime

    # Sample sizes
    n_users_before: int
    n_users_after: int
    n_users_both: int  # Users observed in both periods

    # Effect on engagement decay
    mean_tau_before: float
    mean_tau_after: float
    tau_change: float  # After - Before
    tau_change_p_value: float

    # Effect on other metrics
    engagement_change: float
    retention_change: float

    # Interpretation
    interpretation: str = ""


class PlatformProxyCalculator:
    """
    Calculate motivation proxies from platform-specific user data.

    Each platform has different observable behaviors that can serve
    as proxies for motivation type.
    """

    @staticmethod
    def calculate_reddit_proxies(user_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate motivation proxies for Reddit users.

        Intrinsic motivation indicators (high α):
        - Posts in small, niche subreddits
        - Long-form content creation
        - Discussion participation
        - Consistent posting regardless of karma

        Extrinsic motivation indicators (low α):
        - Posts in default/large subreddits
        - Karma farming behavior
        - Timing posts for maximum visibility
        - Deleting low-karma posts
        """
        proxies = {}

        posts = user_data.get("posts", [])
        comments = user_data.get("comments", [])

        if not posts and not comments:
            return {}

        # 1. Creation ratio: posts / (posts + comments)
        n_posts = len(posts)
        n_comments = len(comments)
        total = n_posts + n_comments
        if total > 0:
            proxies["creation_ratio"] = n_posts / total
        else:
            proxies["creation_ratio"] = 0.5

        # 2. Niche engagement: average log(subreddit_subscribers)
        # Lower = more niche = more intrinsic
        subreddit_sizes = []
        for post in posts:
            size = post.get("subreddit_subscribers", 100000)
            subreddit_sizes.append(size)

        if subreddit_sizes:
            avg_log_size = np.mean(np.log10(np.array(subreddit_sizes) + 1))
            # Normalize: 4 = 10k subscribers, 6 = 1M subscribers
            # Invert so higher = more intrinsic
            proxies["niche_engagement"] = max(0, (7 - avg_log_size) / 4)
        else:
            proxies["niche_engagement"] = 0.5

        # 3. Content depth: average post/comment length
        all_content = posts + comments
        if all_content:
            avg_length = np.mean([len(c.get("body", "")) for c in all_content])
            # Normalize: 100 chars = 0.2, 1000 chars = 0.8
            proxies["engagement_depth"] = min(1.0, avg_length / 1200)
        else:
            proxies["engagement_depth"] = 0.5

        # 4. Karma sensitivity: correlation between post timing and karma
        # High correlation = extrinsically motivated
        if len(posts) >= 10:
            karma_scores = [p.get("score", 0) for p in posts]
            # Check if user deletes low-karma posts (indicator of karma sensitivity)
            if np.min(karma_scores) > 0:
                # All positive karma = might be deleting low-karma posts
                proxies["reward_sensitivity"] = 0.3  # Higher = more extrinsic
            else:
                proxies["reward_sensitivity"] = 0.7  # Keeps negative karma = intrinsic
        else:
            proxies["reward_sensitivity"] = 0.5

        # 5. Temporal consistency: coefficient of variation of posting frequency
        if len(posts) >= 5:
            timestamps = [p.get("created_utc", 0) for p in posts if p.get("created_utc")]
            if len(timestamps) >= 5:
                timestamps = np.sort(timestamps)
                intervals = np.diff(timestamps)
                if np.mean(intervals) > 0:
                    cv = np.std(intervals) / np.mean(intervals)
                    # Lower CV = more consistent = more intrinsic
                    proxies["temporal_consistency"] = 1 / (1 + cv)
                else:
                    proxies["temporal_consistency"] = 0.5
            else:
                proxies["temporal_consistency"] = 0.5
        else:
            proxies["temporal_consistency"] = 0.5

        return proxies

    @staticmethod
    def calculate_github_proxies(user_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate motivation proxies for GitHub users.

        Intrinsic motivation indicators (high α):
        - Personal projects (not forks, not company repos)
        - Documentation contributions
        - Long-term project maintenance
        - Contributions to non-commercial projects

        Extrinsic motivation indicators (low α):
        - Contribution graphs (green squares)
        - Resume-driven development
        - Hacktoberfest spam
        - Only contributing to popular repositories
        """
        proxies = {}

        repos = user_data.get("repositories", [])
        contributions = user_data.get("contributions", [])

        if not repos and not contributions:
            return {}

        # 1. Personal project ratio
        if repos:
            personal = sum(1 for r in repos if not r.get("fork", False))
            proxies["creation_ratio"] = personal / len(repos)
        else:
            proxies["creation_ratio"] = 0.5

        # 2. Niche engagement: average log(repo_stars)
        star_counts = [r.get("stargazers_count", 0) for r in repos]
        if star_counts:
            avg_log_stars = np.mean(np.log10(np.array(star_counts) + 1))
            # Invert: fewer stars = more niche = more intrinsic
            proxies["niche_engagement"] = max(0, (3 - avg_log_stars) / 3)
        else:
            proxies["niche_engagement"] = 0.5

        # 3. Contribution depth: LOC changed / commits
        if contributions:
            total_changes = sum(c.get("additions", 0) + c.get("deletions", 0)
                              for c in contributions)
            n_commits = len(contributions)
            if n_commits > 0:
                avg_changes = total_changes / n_commits
                # Normalize: 10 lines = 0.2, 100 lines = 0.8
                proxies["engagement_depth"] = min(1.0, avg_changes / 120)
            else:
                proxies["engagement_depth"] = 0.5
        else:
            proxies["engagement_depth"] = 0.5

        # 4. Reward sensitivity: contribution streak sensitivity
        # If user only contributes to maintain streak = extrinsic
        if contributions and len(contributions) >= 30:
            dates = [c.get("date") for c in contributions if c.get("date")]
            if len(dates) >= 30:
                # Check for minimum viable contributions (1 commit days)
                commit_counts = {}
                for d in dates:
                    commit_counts[d] = commit_counts.get(d, 0) + 1

                min_commits = sum(1 for v in commit_counts.values() if v == 1)
                if len(commit_counts) > 0:
                    min_commit_ratio = min_commits / len(commit_counts)
                    # High ratio of 1-commit days = streak maintenance = extrinsic
                    proxies["reward_sensitivity"] = min_commit_ratio
                else:
                    proxies["reward_sensitivity"] = 0.5
            else:
                proxies["reward_sensitivity"] = 0.5
        else:
            proxies["reward_sensitivity"] = 0.5

        # 5. Temporal consistency
        if contributions and len(contributions) >= 10:
            dates = [c.get("date") for c in contributions if c.get("date")]
            if len(dates) >= 10:
                # Convert to timestamps
                try:
                    timestamps = [datetime.fromisoformat(d).timestamp() for d in dates]
                    timestamps = np.sort(timestamps)
                    intervals = np.diff(timestamps)
                    if len(intervals) > 0 and np.mean(intervals) > 0:
                        cv = np.std(intervals) / np.mean(intervals)
                        proxies["temporal_consistency"] = 1 / (1 + cv)
                    else:
                        proxies["temporal_consistency"] = 0.5
                except (ValueError, TypeError):
                    proxies["temporal_consistency"] = 0.5
            else:
                proxies["temporal_consistency"] = 0.5
        else:
            proxies["temporal_consistency"] = 0.5

        return proxies

    @staticmethod
    def calculate_strava_proxies(user_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate motivation proxies for Strava users.

        Intrinsic motivation indicators (high α):
        - Solo activities without social sharing
        - Varied activity types
        - Training in off-peak times
        - Focus on personal improvement over records

        Extrinsic motivation indicators (low α):
        - Segment hunting (KOM/QOM chasing)
        - Social features heavy use
        - Training for likes/kudos
        - Challenge completion focus
        """
        proxies = {}

        activities = user_data.get("activities", [])
        segments = user_data.get("segment_efforts", [])
        kudos = user_data.get("kudos_received", 0)

        if not activities:
            return {}

        # 1. Solo ratio: activities without social engagement
        if activities:
            solo = sum(1 for a in activities
                      if a.get("kudos_count", 0) == 0 and not a.get("has_photos"))
            proxies["social_focus"] = 1 - (solo / len(activities))
        else:
            proxies["social_focus"] = 0.5

        # 2. Activity variety
        if activities:
            types = set(a.get("type", "Run") for a in activities)
            # More variety = more intrinsic
            proxies["niche_engagement"] = min(1.0, len(types) / 5)
        else:
            proxies["niche_engagement"] = 0.5

        # 3. Segment effort ratio (KOM hunting indicator)
        if activities and segments:
            segment_ratio = len(segments) / len(activities)
            # Higher ratio = more segment focused = more extrinsic
            proxies["reward_sensitivity"] = min(1.0, segment_ratio * 2)
        else:
            proxies["reward_sensitivity"] = 0.5

        # 4. Training consistency
        if len(activities) >= 10:
            timestamps = [a.get("start_date_local") for a in activities
                         if a.get("start_date_local")]
            if len(timestamps) >= 10:
                try:
                    ts = [datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
                          for t in timestamps]
                    ts = np.sort(ts)
                    intervals = np.diff(ts)
                    if len(intervals) > 0 and np.mean(intervals) > 0:
                        cv = np.std(intervals) / np.mean(intervals)
                        proxies["temporal_consistency"] = 1 / (1 + cv)
                    else:
                        proxies["temporal_consistency"] = 0.5
                except (ValueError, TypeError):
                    proxies["temporal_consistency"] = 0.5
            else:
                proxies["temporal_consistency"] = 0.5
        else:
            proxies["temporal_consistency"] = 0.5

        # 5. Effort depth: average moving time / elapsed time
        if activities:
            effort_ratios = []
            for a in activities:
                moving = a.get("moving_time", 0)
                elapsed = a.get("elapsed_time", 1)
                if elapsed > 0:
                    effort_ratios.append(moving / elapsed)

            if effort_ratios:
                proxies["engagement_depth"] = np.mean(effort_ratios)
            else:
                proxies["engagement_depth"] = 0.5
        else:
            proxies["engagement_depth"] = 0.5

        return proxies


class MotivationEstimator:
    """
    Estimate motivation parameter (α) independently of τ.

    This is CRITICAL for avoiding circular reasoning in the scaling law.
    """

    def __init__(
        self,
        proxy_weights: Optional[Dict[str, float]] = None,
        min_observations: int = 10,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize estimator.

        Args:
            proxy_weights: Weights for each proxy (default: equal)
            min_observations: Minimum observations for reliable estimate
            confidence_threshold: Minimum confidence to report estimate
        """
        # Default equal weights
        self.proxy_weights = proxy_weights or {
            "creation_ratio": 1.0,
            "niche_engagement": 1.0,
            "reward_sensitivity": 1.0,  # Inverted in calculation
            "temporal_consistency": 1.0,
            "engagement_depth": 1.0,
            "social_focus": 1.0,  # Inverted in calculation
            "engagement_timing": 1.0,
        }

        self.min_observations = min_observations
        self.confidence_threshold = confidence_threshold

        self.proxy_calculator = PlatformProxyCalculator()

    def estimate(
        self,
        user_data: Dict[str, Any],
        platform: str
    ) -> MotivationEstimate:
        """
        Estimate α for a single user.

        Args:
            user_data: User activity data
            platform: Platform name

        Returns:
            MotivationEstimate with α and uncertainty
        """
        user_id = user_data.get("user_id", "unknown")

        # Get platform-specific proxies
        if platform == "reddit":
            proxies = self.proxy_calculator.calculate_reddit_proxies(user_data)
        elif platform == "github":
            proxies = self.proxy_calculator.calculate_github_proxies(user_data)
        elif platform == "strava":
            proxies = self.proxy_calculator.calculate_strava_proxies(user_data)
        else:
            # Generic proxies
            proxies = self._calculate_generic_proxies(user_data)

        if not proxies:
            return MotivationEstimate(
                user_id=user_id,
                platform=platform,
                alpha=1.0,  # Default neutral
                alpha_uncertainty=1.0,
                estimation_method="none",
                confidence=0.0
            )

        # Invert extrinsic indicators
        if "reward_sensitivity" in proxies:
            proxies["reward_sensitivity"] = 1 - proxies["reward_sensitivity"]
        if "social_focus" in proxies:
            proxies["social_focus"] = 1 - proxies["social_focus"]

        # Weighted average of proxies
        weighted_sum = 0.0
        weight_sum = 0.0
        proxy_values = []

        for proxy_name, proxy_value in proxies.items():
            weight = self.proxy_weights.get(proxy_name, 1.0)
            weighted_sum += weight * proxy_value
            weight_sum += weight
            proxy_values.append(proxy_value)

        if weight_sum > 0:
            # Raw score in [0, 1]
            raw_alpha = weighted_sum / weight_sum

            # Transform to α scale centered at 1.0
            # α = 0.5 means highly extrinsic, α = 2.0 means highly intrinsic
            alpha = 0.5 + 1.5 * raw_alpha

            # Uncertainty from proxy variance
            if len(proxy_values) > 1:
                uncertainty = np.std(proxy_values) * 1.5
            else:
                uncertainty = 0.5

            # Confidence based on number of proxies
            confidence = min(1.0, len(proxies) / 5)
        else:
            alpha = 1.0
            uncertainty = 1.0
            confidence = 0.0

        # Count observations
        n_obs = self._count_observations(user_data)
        obs_days = self._calculate_observation_period(user_data)

        # Adjust confidence by observation count
        if n_obs < self.min_observations:
            confidence *= n_obs / self.min_observations

        return MotivationEstimate(
            user_id=user_id,
            platform=platform,
            alpha=alpha,
            alpha_uncertainty=uncertainty,
            estimation_method="proxy_variables",
            proxies_used=list(proxies.keys()),
            proxy_scores=proxies,
            confidence=confidence,
            n_observations=n_obs,
            observation_period_days=obs_days
        )

    def estimate_batch(
        self,
        users_data: List[Dict[str, Any]],
        platform: str
    ) -> Tuple[Dict[str, float], List[MotivationEstimate]]:
        """
        Estimate α for multiple users.

        Returns:
            Tuple of (user_id -> alpha dict, full estimates list)
        """
        alpha_dict = {}
        estimates = []

        for user_data in users_data:
            estimate = self.estimate(user_data, platform)
            estimates.append(estimate)

            if estimate.confidence >= self.confidence_threshold:
                alpha_dict[estimate.user_id] = estimate.alpha

        return alpha_dict, estimates

    def _calculate_generic_proxies(
        self,
        user_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate generic proxies for unknown platforms."""
        proxies = {}

        activities = user_data.get("activities", [])
        if not activities:
            return proxies

        # Temporal consistency
        if len(activities) >= 5:
            timestamps = [a.get("timestamp") for a in activities if a.get("timestamp")]
            if len(timestamps) >= 5:
                try:
                    if isinstance(timestamps[0], (int, float)):
                        ts = np.array(timestamps)
                    else:
                        ts = np.array([datetime.fromisoformat(t).timestamp()
                                      for t in timestamps])
                    ts = np.sort(ts)
                    intervals = np.diff(ts)
                    if np.mean(intervals) > 0:
                        cv = np.std(intervals) / np.mean(intervals)
                        proxies["temporal_consistency"] = 1 / (1 + cv)
                except (ValueError, TypeError):
                    pass

        return proxies

    def _count_observations(self, user_data: Dict[str, Any]) -> int:
        """Count total observations for a user."""
        count = 0
        for key in ["posts", "comments", "activities", "contributions", "repositories"]:
            if key in user_data:
                count += len(user_data[key])
        return count

    def _calculate_observation_period(self, user_data: Dict[str, Any]) -> float:
        """Calculate observation period in days."""
        timestamps = []

        for key in ["posts", "comments", "activities", "contributions"]:
            if key in user_data:
                for item in user_data[key]:
                    ts = item.get("timestamp") or item.get("created_utc") or item.get("date")
                    if ts:
                        timestamps.append(ts)

        if len(timestamps) < 2:
            return 0.0

        try:
            if isinstance(timestamps[0], (int, float)):
                ts = np.array(timestamps)
            else:
                ts = np.array([datetime.fromisoformat(str(t)).timestamp()
                              for t in timestamps])

            return (np.max(ts) - np.min(ts)) / (24 * 3600)
        except (ValueError, TypeError):
            return 0.0


class NaturalExperimentAnalyzer:
    """
    Analyze natural experiments for causal motivation effects.

    Natural experiments occur when platforms change features that
    affect user motivation (e.g., introducing badges, leaderboards).
    """

    # Known platform changes that affected motivation
    KNOWN_EXPERIMENTS = {
        "strava_local_legend": {
            "platform": "strava",
            "date": datetime(2020, 6, 1),
            "description": "Introduction of Local Legend awards",
            "expected_effect": "Increase extrinsic motivation",
        },
        "github_contribution_graph": {
            "platform": "github",
            "date": datetime(2013, 1, 1),
            "description": "Introduction of contribution graph",
            "expected_effect": "Increase streak-based behavior",
        },
        "reddit_karma_display": {
            "platform": "reddit",
            "date": datetime(2008, 1, 1),  # Approximate
            "description": "Visible karma scores",
            "expected_effect": "Increase karma-seeking behavior",
        },
    }

    def __init__(
        self,
        min_users_before: int = 30,
        min_users_after: int = 30,
        window_days: int = 90
    ):
        self.min_users_before = min_users_before
        self.min_users_after = min_users_after
        self.window_days = window_days

    def analyze_experiment(
        self,
        experiment_name: str,
        users_before: List[Dict[str, Any]],
        users_after: List[Dict[str, Any]],
        tau_before: List[float],
        tau_after: List[float]
    ) -> NaturalExperimentResult:
        """
        Analyze a natural experiment's effect on engagement decay.

        Args:
            experiment_name: Name of the experiment
            users_before: User data before intervention
            users_after: User data after intervention
            tau_before: Estimated τ values before
            tau_after: Estimated τ values after

        Returns:
            NaturalExperimentResult with statistical analysis
        """
        exp_info = self.KNOWN_EXPERIMENTS.get(experiment_name, {})

        # Filter valid τ values
        tau_before = np.array([t for t in tau_before if t > 0 and np.isfinite(t)])
        tau_after = np.array([t for t in tau_after if t > 0 and np.isfinite(t)])

        if len(tau_before) < self.min_users_before:
            return NaturalExperimentResult(
                experiment_name=experiment_name,
                platform=exp_info.get("platform", "unknown"),
                intervention_date=exp_info.get("date", datetime.now()),
                n_users_before=len(tau_before),
                n_users_after=len(tau_after),
                n_users_both=0,
                mean_tau_before=0,
                mean_tau_after=0,
                tau_change=0,
                tau_change_p_value=1.0,
                engagement_change=0,
                retention_change=0,
                interpretation="Insufficient data before intervention"
            )

        if len(tau_after) < self.min_users_after:
            return NaturalExperimentResult(
                experiment_name=experiment_name,
                platform=exp_info.get("platform", "unknown"),
                intervention_date=exp_info.get("date", datetime.now()),
                n_users_before=len(tau_before),
                n_users_after=len(tau_after),
                n_users_both=0,
                mean_tau_before=np.mean(tau_before),
                mean_tau_after=0,
                tau_change=0,
                tau_change_p_value=1.0,
                engagement_change=0,
                retention_change=0,
                interpretation="Insufficient data after intervention"
            )

        # Statistical test: Mann-Whitney U (non-parametric)
        stat, p_value = stats.mannwhitneyu(tau_before, tau_after, alternative='two-sided')

        mean_before = np.mean(tau_before)
        mean_after = np.mean(tau_after)
        tau_change = mean_after - mean_before

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(tau_before) + np.var(tau_after)) / 2)
        if pooled_std > 0:
            cohens_d = tau_change / pooled_std
        else:
            cohens_d = 0

        # Interpretation
        if p_value < 0.05:
            if tau_change < 0:
                interpretation = (
                    f"Significant DECREASE in τ after intervention (p={p_value:.4f}, "
                    f"d={cohens_d:.2f}). This suggests the intervention INCREASED "
                    "extrinsic motivation, leading to faster decay."
                )
            else:
                interpretation = (
                    f"Significant INCREASE in τ after intervention (p={p_value:.4f}, "
                    f"d={cohens_d:.2f}). This suggests the intervention DECREASED "
                    "extrinsic motivation or had other positive effects."
                )
        else:
            interpretation = (
                f"No significant change in τ after intervention (p={p_value:.4f}). "
                "The intervention may not have significantly affected motivation."
            )

        return NaturalExperimentResult(
            experiment_name=experiment_name,
            platform=exp_info.get("platform", "unknown"),
            intervention_date=exp_info.get("date", datetime.now()),
            n_users_before=len(tau_before),
            n_users_after=len(tau_after),
            n_users_both=0,  # Would need user matching
            mean_tau_before=mean_before,
            mean_tau_after=mean_after,
            tau_change=tau_change,
            tau_change_p_value=p_value,
            engagement_change=0,  # Would need engagement data
            retention_change=0,
            interpretation=interpretation
        )

    def detect_structural_breaks(
        self,
        timestamps: NDArray[np.float64],
        tau_values: NDArray[np.float64],
        min_segment: int = 30
    ) -> List[Tuple[datetime, float]]:
        """
        Detect potential structural breaks in τ distribution over time.

        These could indicate platform changes that affected motivation.

        Returns:
            List of (date, significance) for potential breaks
        """
        if len(timestamps) < 2 * min_segment:
            return []

        # Sort by time
        sort_idx = np.argsort(timestamps)
        timestamps = timestamps[sort_idx]
        tau_values = tau_values[sort_idx]

        breaks = []

        # Sliding window approach
        for i in range(min_segment, len(timestamps) - min_segment):
            before = tau_values[:i]
            after = tau_values[i:]

            # Test for difference
            stat, p_value = stats.mannwhitneyu(before, after, alternative='two-sided')

            if p_value < 0.01:  # Strict threshold
                break_time = datetime.fromtimestamp(timestamps[i])
                breaks.append((break_time, 1 - p_value))

        # Remove redundant breaks (within 30 days)
        if breaks:
            filtered = [breaks[0]]
            for break_time, significance in breaks[1:]:
                if (break_time - filtered[-1][0]).days > 30:
                    filtered.append((break_time, significance))
            return filtered

        return breaks


def generate_alpha_validation_report(
    estimates: List[MotivationEstimate],
    tau_values: Dict[str, float]
) -> str:
    """
    Generate a validation report showing α is independent of τ.

    This is CRITICAL for demonstrating no circular reasoning.
    """
    report = []
    report.append("=" * 70)
    report.append("MOTIVATION PARAMETER (α) - INDEPENDENCE VALIDATION REPORT")
    report.append("=" * 70)

    # Extract matched pairs
    alpha_values = []
    matched_tau = []

    for est in estimates:
        if est.user_id in tau_values and est.confidence > 0.3:
            alpha_values.append(est.alpha)
            matched_tau.append(tau_values[est.user_id])

    report.append(f"\n1. SAMPLE SIZE")
    report.append(f"   Total estimates: {len(estimates)}")
    report.append(f"   With matched τ: {len(alpha_values)}")
    report.append(f"   With confidence > 0.3: {sum(1 for e in estimates if e.confidence > 0.3)}")

    if len(alpha_values) >= 10:
        alpha_arr = np.array(alpha_values)
        tau_arr = np.array(matched_tau)

        # Test correlation between α estimation method and τ
        pearson_r, pearson_p = stats.pearsonr(alpha_arr, tau_arr)
        spearman_r, spearman_p = stats.spearmanr(alpha_arr, tau_arr)

        report.append(f"\n2. INDEPENDENCE TEST")
        report.append(f"   Pearson correlation (α, τ): r = {pearson_r:.3f}, p = {pearson_p:.4f}")
        report.append(f"   Spearman correlation (α, τ): ρ = {spearman_r:.3f}, p = {spearman_p:.4f}")

        if abs(pearson_r) < 0.3 and pearson_p > 0.05:
            report.append("\n   ✓ VALIDATION PASSED: α estimation is independent of τ")
            report.append("     The correlation is not significant, confirming no circular reasoning.")
        else:
            report.append("\n   ⚠ WARNING: Potential correlation detected between α and τ")
            report.append("     This could indicate methodological issues.")
            report.append("     Review proxy calculations for potential bias.")

        report.append(f"\n3. α DISTRIBUTION")
        report.append(f"   Mean α: {np.mean(alpha_arr):.3f}")
        report.append(f"   Std α: {np.std(alpha_arr):.3f}")
        report.append(f"   Range: [{np.min(alpha_arr):.3f}, {np.max(alpha_arr):.3f}]")

    else:
        report.append("\n2. INDEPENDENCE TEST")
        report.append("   Insufficient data for correlation analysis")

    # Proxy usage summary
    report.append(f"\n4. PROXY VARIABLES USED")
    proxy_counts = {}
    for est in estimates:
        for proxy in est.proxies_used:
            proxy_counts[proxy] = proxy_counts.get(proxy, 0) + 1

    for proxy, count in sorted(proxy_counts.items(), key=lambda x: -x[1]):
        report.append(f"   - {proxy}: {count} users")

    report.append("\n" + "=" * 70)

    return "\n".join(report)
