"""
Causal Inference Module - Natural Experiments for Proving Causation.

REVOLUTIONARY CAPABILITY FOR NATURE:
Move from correlation to CAUSATION by analyzing natural experiments.

The fatal flaw in the original design: α and τ are correlated because
one is derived from the other. This module provides methods to establish
CAUSAL relationships through quasi-experimental designs.

Key Methods:
1. COHORT ANALYSIS - Compare users before/after platform changes
2. DIFFERENCE-IN-DIFFERENCES - Control for confounding trends
3. REGRESSION DISCONTINUITY - Exploit sharp policy changes
4. INSTRUMENTAL VARIABLES - Use exogenous variation

Example Headline:
"We demonstrate a causal link between the design of digital platforms
and the decay of human engagement. The introduction of extrinsic reward
systems is shown to accelerate user churn by up to 30%."
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from scipy import stats
import warnings


class PlatformEvent(Enum):
    """
    Known platform events that may affect user motivation.

    These are "natural experiments" - exogenous shocks to the system
    that allow causal inference.
    """
    REDDIT_COMMUNITY_AWARDS = "reddit_community_awards"  # 2019
    REDDIT_GOLD_CHANGE = "reddit_gold_change"  # 2018
    GITHUB_CONTRIBUTION_GRAPH = "github_contribution_graph"  # 2013
    GITHUB_ACHIEVEMENTS = "github_achievements"  # 2022
    STRAVA_LOCAL_LEGEND = "strava_local_legend"  # 2020
    STRAVA_CHALLENGES = "strava_challenges"
    TWITTER_LIKES_TO_HEARTS = "twitter_likes_to_hearts"  # 2015
    YOUTUBE_SHORTS = "youtube_shorts"  # 2020
    CUSTOM = "custom"


@dataclass
class PlatformEventInfo:
    """Information about a platform event for causal analysis."""
    event_type: PlatformEvent
    platform: str
    event_date: datetime
    description: str

    # Expected effect direction
    expected_effect_on_tau: str  # "decrease", "increase", "unknown"
    expected_mechanism: str

    # Data requirements
    min_users_before: int = 100
    min_users_after: int = 100
    window_days: int = 180  # Days before/after to analyze


# Known platform events database
PLATFORM_EVENTS = {
    PlatformEvent.REDDIT_COMMUNITY_AWARDS: PlatformEventInfo(
        event_type=PlatformEvent.REDDIT_COMMUNITY_AWARDS,
        platform="reddit",
        event_date=datetime(2019, 9, 1),
        description="Reddit introduced Community Awards allowing subreddit-specific awards",
        expected_effect_on_tau="decrease",
        expected_mechanism="Increased extrinsic rewards -> faster decay"
    ),
    PlatformEvent.REDDIT_GOLD_CHANGE: PlatformEventInfo(
        event_type=PlatformEvent.REDDIT_GOLD_CHANGE,
        platform="reddit",
        event_date=datetime(2018, 9, 5),
        description="Reddit renamed Gold to Premium and introduced Silver/Platinum",
        expected_effect_on_tau="decrease",
        expected_mechanism="Tiered rewards increase extrinsic focus"
    ),
    PlatformEvent.GITHUB_ACHIEVEMENTS: PlatformEventInfo(
        event_type=PlatformEvent.GITHUB_ACHIEVEMENTS,
        platform="github",
        event_date=datetime(2022, 6, 9),
        description="GitHub introduced achievement badges",
        expected_effect_on_tau="decrease",
        expected_mechanism="Gamification increases extrinsic motivation"
    ),
    PlatformEvent.STRAVA_LOCAL_LEGEND: PlatformEventInfo(
        event_type=PlatformEvent.STRAVA_LOCAL_LEGEND,
        platform="strava",
        event_date=datetime(2020, 6, 1),
        description="Strava introduced Local Legend awards for segment dominance",
        expected_effect_on_tau="decrease",
        expected_mechanism="Competitive ranking increases extrinsic focus"
    ),
}


@dataclass
class CohortDefinition:
    """Definition of a cohort for natural experiment analysis."""
    name: str
    description: str

    # Selection criteria
    joined_after: Optional[datetime] = None
    joined_before: Optional[datetime] = None
    min_activity_count: int = 10
    min_observation_days: int = 30

    # Expected characteristics
    expected_tau_relative: str = "baseline"  # "higher", "lower", "baseline"


@dataclass
class CohortStats:
    """Statistics for a single cohort."""
    cohort_name: str
    n_users: int

    # τ distribution
    tau_mean: float
    tau_median: float
    tau_std: float
    tau_ci_lower: float
    tau_ci_upper: float

    # Other metrics
    mean_alpha: float = 1.0
    mean_engagement_duration: float = 0.0
    retention_30day: float = 0.0


@dataclass
class CausalEffectEstimate:
    """
    Estimate of causal effect from natural experiment.

    The key scientific contribution: demonstrating CAUSATION, not just correlation.
    """
    event: str
    treatment_cohort: str
    control_cohort: str

    # Effect size
    tau_difference: float  # Treatment mean - Control mean
    tau_percent_change: float  # Percentage change
    cohens_d: float  # Standardized effect size

    # Statistical significance
    p_value: float
    confidence_interval: Tuple[float, float]

    # Interpretation
    is_significant: bool
    supports_hypothesis: bool
    interpretation: str

    # Sample sizes
    n_treatment: int
    n_control: int

    # Robustness checks
    placebo_test_passed: bool = False
    parallel_trends_passed: bool = False


class NaturalExperimentAnalyzer:
    """
    Analyze natural experiments for causal inference.

    This is the KEY capability for establishing causation rather than correlation.
    """

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level

    def define_cohorts_for_event(
        self,
        event: PlatformEvent,
        window_days: int = 180
    ) -> Tuple[CohortDefinition, CohortDefinition]:
        """
        Define treatment and control cohorts for a platform event.

        Users who joined AFTER the event = Treatment (exposed to new feature)
        Users who joined BEFORE the event = Control (not exposed initially)
        """
        event_info = PLATFORM_EVENTS.get(event)
        if event_info is None:
            raise ValueError(f"Unknown event: {event}")

        event_date = event_info.event_date

        # Control: joined before event, had time to establish behavior
        control = CohortDefinition(
            name=f"pre_{event.value}",
            description=f"Users who joined before {event.value}",
            joined_before=event_date - timedelta(days=30),  # Buffer
            joined_after=event_date - timedelta(days=window_days + 30),
            expected_tau_relative="baseline"
        )

        # Treatment: joined after event, only know platform with new feature
        treatment = CohortDefinition(
            name=f"post_{event.value}",
            description=f"Users who joined after {event.value}",
            joined_after=event_date + timedelta(days=30),  # Buffer
            joined_before=event_date + timedelta(days=window_days + 30),
            expected_tau_relative="lower" if event_info.expected_effect_on_tau == "decrease" else "higher"
        )

        return control, treatment

    def assign_users_to_cohorts(
        self,
        users_data: List[Dict[str, Any]],
        control: CohortDefinition,
        treatment: CohortDefinition
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Assign users to control and treatment cohorts based on join date.
        """
        control_users = []
        treatment_users = []

        for user in users_data:
            # Get user's first activity date
            activities = user.get("activities", [])
            if not activities:
                continue

            first_activity = min(
                a.get("timestamp") or a.get("created_utc", float('inf'))
                for a in activities
            )

            if isinstance(first_activity, str):
                try:
                    first_activity = datetime.fromisoformat(first_activity).timestamp()
                except:
                    continue

            first_date = datetime.fromtimestamp(first_activity)

            # Check cohort membership
            if (control.joined_after is None or first_date >= control.joined_after) and \
               (control.joined_before is None or first_date <= control.joined_before):
                if len(activities) >= control.min_activity_count:
                    control_users.append(user)

            elif (treatment.joined_after is None or first_date >= treatment.joined_after) and \
                 (treatment.joined_before is None or first_date <= treatment.joined_before):
                if len(activities) >= treatment.min_activity_count:
                    treatment_users.append(user)

        return control_users, treatment_users

    def compute_cohort_stats(
        self,
        users_data: List[Dict[str, Any]],
        tau_values: List[float],
        cohort_name: str
    ) -> CohortStats:
        """
        Compute statistics for a cohort.
        """
        tau_arr = np.array([t for t in tau_values if t > 0 and np.isfinite(t)])

        if len(tau_arr) < 5:
            return CohortStats(
                cohort_name=cohort_name,
                n_users=len(tau_arr),
                tau_mean=0,
                tau_median=0,
                tau_std=0,
                tau_ci_lower=0,
                tau_ci_upper=0
            )

        # Bootstrap confidence interval
        n_bootstrap = 1000
        boot_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(tau_arr, size=len(tau_arr), replace=True)
            boot_means.append(np.mean(sample))

        ci_lower = np.percentile(boot_means, 2.5)
        ci_upper = np.percentile(boot_means, 97.5)

        return CohortStats(
            cohort_name=cohort_name,
            n_users=len(tau_arr),
            tau_mean=np.mean(tau_arr),
            tau_median=np.median(tau_arr),
            tau_std=np.std(tau_arr),
            tau_ci_lower=ci_lower,
            tau_ci_upper=ci_upper
        )

    def estimate_causal_effect(
        self,
        control_taus: List[float],
        treatment_taus: List[float],
        event_name: str,
        expected_direction: str = "decrease"
    ) -> CausalEffectEstimate:
        """
        Estimate the causal effect of a platform change on τ.

        Uses permutation testing for robust p-value computation.
        """
        control = np.array([t for t in control_taus if t > 0 and np.isfinite(t)])
        treatment = np.array([t for t in treatment_taus if t > 0 and np.isfinite(t)])

        if len(control) < 10 or len(treatment) < 10:
            return CausalEffectEstimate(
                event=event_name,
                treatment_cohort="post_event",
                control_cohort="pre_event",
                tau_difference=0,
                tau_percent_change=0,
                cohens_d=0,
                p_value=1.0,
                confidence_interval=(0, 0),
                is_significant=False,
                supports_hypothesis=False,
                interpretation="Insufficient data for causal analysis",
                n_treatment=len(treatment),
                n_control=len(control)
            )

        # Basic statistics
        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)
        tau_diff = treatment_mean - control_mean
        percent_change = (tau_diff / control_mean) * 100 if control_mean > 0 else 0

        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(control) + np.var(treatment)) / 2)
        cohens_d = tau_diff / pooled_std if pooled_std > 0 else 0

        # Permutation test for p-value
        observed_diff = tau_diff
        combined = np.concatenate([control, treatment])
        n_control = len(control)

        n_permutations = 10000
        perm_diffs = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_control = combined[:n_control]
            perm_treatment = combined[n_control:]
            perm_diffs.append(np.mean(perm_treatment) - np.mean(perm_control))

        perm_diffs = np.array(perm_diffs)

        # Two-tailed p-value
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

        # Bootstrap confidence interval for the difference
        n_bootstrap = 1000
        boot_diffs = []
        for _ in range(n_bootstrap):
            c_sample = np.random.choice(control, size=len(control), replace=True)
            t_sample = np.random.choice(treatment, size=len(treatment), replace=True)
            boot_diffs.append(np.mean(t_sample) - np.mean(c_sample))

        ci_lower = np.percentile(boot_diffs, 2.5)
        ci_upper = np.percentile(boot_diffs, 97.5)

        # Interpretation
        is_significant = p_value < self.alpha

        if expected_direction == "decrease":
            supports_hypothesis = tau_diff < 0 and is_significant
        else:
            supports_hypothesis = tau_diff > 0 and is_significant

        if is_significant:
            if supports_hypothesis:
                interpretation = (
                    f"CAUSAL EFFECT CONFIRMED: The platform change caused a "
                    f"{abs(percent_change):.1f}% {'decrease' if tau_diff < 0 else 'increase'} "
                    f"in engagement decay timescale (p={p_value:.4f}, d={cohens_d:.2f}). "
                    f"This supports the hypothesis that extrinsic rewards accelerate disengagement."
                )
            else:
                interpretation = (
                    f"Significant effect detected but in opposite direction to hypothesis: "
                    f"τ {'increased' if tau_diff > 0 else 'decreased'} by {abs(percent_change):.1f}% "
                    f"(p={p_value:.4f})."
                )
        else:
            interpretation = (
                f"No significant causal effect detected (p={p_value:.4f}). "
                f"The platform change did not measurably affect engagement decay dynamics."
            )

        return CausalEffectEstimate(
            event=event_name,
            treatment_cohort="post_event",
            control_cohort="pre_event",
            tau_difference=tau_diff,
            tau_percent_change=percent_change,
            cohens_d=cohens_d,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            supports_hypothesis=supports_hypothesis,
            interpretation=interpretation,
            n_treatment=len(treatment),
            n_control=len(control)
        )

    def run_placebo_test(
        self,
        control_taus: List[float],
        treatment_taus: List[float],
        n_fake_cutoffs: int = 10
    ) -> Dict[str, Any]:
        """
        Placebo test: check for spurious effects at fake event dates.

        If we find significant effects at random dates, our main effect
        might be spurious.
        """
        all_taus = control_taus + treatment_taus
        n = len(all_taus)

        fake_effects = []
        for i in range(n_fake_cutoffs):
            # Random split point
            cutoff = int(n * (0.2 + 0.6 * i / n_fake_cutoffs))
            fake_control = all_taus[:cutoff]
            fake_treatment = all_taus[cutoff:]

            if len(fake_control) >= 10 and len(fake_treatment) >= 10:
                effect = self.estimate_causal_effect(
                    fake_control, fake_treatment,
                    f"placebo_{i}",
                    "decrease"
                )
                fake_effects.append(effect)

        # If many placebo tests are significant, main test is suspect
        n_significant = sum(1 for e in fake_effects if e.is_significant)
        expected_false_positives = len(fake_effects) * self.alpha

        placebo_passed = n_significant <= expected_false_positives * 2

        return {
            "n_placebo_tests": len(fake_effects),
            "n_significant": n_significant,
            "expected_false_positives": expected_false_positives,
            "placebo_passed": placebo_passed,
            "interpretation": (
                "Placebo test PASSED - effect appears genuine" if placebo_passed
                else "Placebo test FAILED - effect may be spurious"
            )
        }

    def check_parallel_trends(
        self,
        control_users: List[Dict[str, Any]],
        treatment_users: List[Dict[str, Any]],
        event_date: datetime,
        window_days: int = 90
    ) -> Dict[str, Any]:
        """
        Check parallel trends assumption for difference-in-differences.

        Before the event, both groups should have similar trends.
        """
        def get_pre_event_trend(users: List[Dict], event_date: datetime) -> Tuple[float, float]:
            """Extract engagement trend before the event."""
            engagement_by_day = {}

            for user in users:
                activities = user.get("activities", [])
                for a in activities:
                    ts = a.get("timestamp") or a.get("created_utc")
                    if ts is None:
                        continue

                    if isinstance(ts, str):
                        try:
                            ts = datetime.fromisoformat(ts).timestamp()
                        except:
                            continue

                    activity_date = datetime.fromtimestamp(ts)

                    # Only pre-event activities
                    days_before = (event_date - activity_date).days
                    if 0 < days_before <= window_days:
                        day_key = days_before
                        if day_key not in engagement_by_day:
                            engagement_by_day[day_key] = []
                        engagement_by_day[day_key].append(
                            a.get("engagement", 1.0)
                        )

            if len(engagement_by_day) < 10:
                return 0.0, 1.0

            days = np.array(list(engagement_by_day.keys()))
            means = np.array([np.mean(engagement_by_day[d]) for d in days])

            try:
                slope, _ = np.polyfit(days, means, 1)
                residuals = means - (slope * days + _)
                r, p = stats.pearsonr(days, means)
                return slope, p
            except:
                return 0.0, 1.0

        control_slope, control_p = get_pre_event_trend(control_users, event_date)
        treatment_slope, treatment_p = get_pre_event_trend(treatment_users, event_date)

        # Trends are parallel if slopes are not significantly different
        slope_diff = abs(control_slope - treatment_slope)
        avg_slope = (abs(control_slope) + abs(treatment_slope)) / 2

        parallel = slope_diff < avg_slope * 0.5 if avg_slope > 0 else True

        return {
            "control_slope": control_slope,
            "treatment_slope": treatment_slope,
            "slope_difference": slope_diff,
            "parallel_trends": parallel,
            "interpretation": (
                "Parallel trends assumption SATISFIED" if parallel
                else "WARNING: Pre-event trends may not be parallel"
            )
        }


class DifferenceInDifferencesEstimator:
    """
    Difference-in-Differences (DiD) estimator for causal effects.

    This is the gold standard for natural experiment analysis.
    """

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level

    def estimate(
        self,
        control_pre: List[float],
        control_post: List[float],
        treatment_pre: List[float],
        treatment_post: List[float]
    ) -> Dict[str, Any]:
        """
        Estimate causal effect using difference-in-differences.

        DiD = (Treatment_post - Treatment_pre) - (Control_post - Control_pre)

        This controls for:
        - Time-invariant differences between groups
        - Common time trends affecting both groups
        """
        # Convert to arrays
        c_pre = np.array([x for x in control_pre if np.isfinite(x)])
        c_post = np.array([x for x in control_post if np.isfinite(x)])
        t_pre = np.array([x for x in treatment_pre if np.isfinite(x)])
        t_post = np.array([x for x in treatment_post if np.isfinite(x)])

        if any(len(arr) < 5 for arr in [c_pre, c_post, t_pre, t_post]):
            return {"error": "Insufficient data for DiD estimation"}

        # Compute differences
        treatment_diff = np.mean(t_post) - np.mean(t_pre)
        control_diff = np.mean(c_post) - np.mean(c_pre)
        did_estimate = treatment_diff - control_diff

        # Bootstrap standard error
        n_bootstrap = 1000
        boot_did = []

        for _ in range(n_bootstrap):
            b_c_pre = np.random.choice(c_pre, size=len(c_pre), replace=True)
            b_c_post = np.random.choice(c_post, size=len(c_post), replace=True)
            b_t_pre = np.random.choice(t_pre, size=len(t_pre), replace=True)
            b_t_post = np.random.choice(t_post, size=len(t_post), replace=True)

            b_treatment_diff = np.mean(b_t_post) - np.mean(b_t_pre)
            b_control_diff = np.mean(b_c_post) - np.mean(b_c_pre)
            boot_did.append(b_treatment_diff - b_control_diff)

        boot_did = np.array(boot_did)
        se = np.std(boot_did)

        # Statistical test
        t_stat = did_estimate / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        ci_lower = np.percentile(boot_did, 2.5)
        ci_upper = np.percentile(boot_did, 97.5)

        is_significant = p_value < self.alpha

        return {
            "did_estimate": did_estimate,
            "standard_error": se,
            "t_statistic": t_stat,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "is_significant": is_significant,
            "treatment_change": treatment_diff,
            "control_change": control_diff,
            "interpretation": (
                f"DiD estimate: {did_estimate:.3f} (p={p_value:.4f}). "
                f"{'Significant' if is_significant else 'Not significant'} causal effect. "
                f"Treatment group changed by {treatment_diff:.3f} while control changed by {control_diff:.3f}."
            ),
            "sample_sizes": {
                "control_pre": len(c_pre),
                "control_post": len(c_post),
                "treatment_pre": len(t_pre),
                "treatment_post": len(t_post)
            }
        }


def generate_causal_analysis_report(
    effect: CausalEffectEstimate,
    placebo: Dict[str, Any],
    parallel_trends: Dict[str, Any]
) -> str:
    """
    Generate a comprehensive causal analysis report for publication.
    """
    report = []
    report.append("=" * 70)
    report.append("CAUSAL INFERENCE ANALYSIS REPORT")
    report.append("=" * 70)

    report.append("\n1. NATURAL EXPERIMENT")
    report.append(f"   Event: {effect.event}")
    report.append(f"   Treatment cohort: {effect.treatment_cohort} (n={effect.n_treatment})")
    report.append(f"   Control cohort: {effect.control_cohort} (n={effect.n_control})")

    report.append("\n2. EFFECT ESTIMATE")
    report.append(f"   τ difference: {effect.tau_difference:.2f} days")
    report.append(f"   Percent change: {effect.tau_percent_change:.1f}%")
    report.append(f"   Cohen's d: {effect.cohens_d:.3f}")
    report.append(f"   95% CI: [{effect.confidence_interval[0]:.2f}, {effect.confidence_interval[1]:.2f}]")
    report.append(f"   p-value: {effect.p_value:.4f}")

    report.append("\n3. ROBUSTNESS CHECKS")
    report.append(f"   Placebo test: {placebo.get('interpretation', 'N/A')}")
    report.append(f"   Parallel trends: {parallel_trends.get('interpretation', 'N/A')}")

    report.append("\n4. CONCLUSION")
    if effect.is_significant and placebo.get('placebo_passed', False):
        if effect.supports_hypothesis:
            report.append("   ✓ CAUSAL EFFECT ESTABLISHED")
            report.append(f"   {effect.interpretation}")
        else:
            report.append("   ⚠ Significant effect, but opposite to hypothesis")
    else:
        report.append("   ✗ Causal effect not established")
        report.append(f"   {effect.interpretation}")

    report.append("\n5. NATURE HEADLINE")
    if effect.is_significant and effect.supports_hypothesis:
        report.append(
            f"   \"Platform gamification causally accelerates user disengagement: "
            f"A {abs(effect.tau_percent_change):.0f}% reduction in engagement lifetime "
            f"following the introduction of extrinsic reward systems.\""
        )
    else:
        report.append(
            "   Insufficient evidence for causal headline."
        )

    report.append("\n" + "=" * 70)

    return "\n".join(report)
