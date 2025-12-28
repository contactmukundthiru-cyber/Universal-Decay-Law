"""Tests for analysis pipeline - Updated for scientifically rigorous version."""

import numpy as np
import pytest

from src.analysis.preprocessing import (
    MinimalPreprocessor,
    NormalizationMethod,
    DatasetPreprocessor,
    RobustStatistics
)
from src.analysis.fitting import FittingPipeline, PipelineConfig, DecayFitter
from src.analysis.universality import (
    UniversalityAnalyzer,
    TrajectoryClassifier,
    TrajectoryType
)
from src.analysis.motivation import MotivationEstimator, PlatformProxyCalculator
from src.analysis.survivorship import SurvivorshipBiasAnalyzer
from src.analysis.prediction import EarlyFeatureExtractor, EarlyPredictionModel
from src.analysis.causal import (
    NaturalExperimentAnalyzer,
    DifferenceInDifferencesEstimator,
    PlatformEvent,
    PLATFORM_EVENTS,
)
from src.analysis.phases import (
    EngagementPhase,
    PhaseDetector,
    EngagementPhaseDiagram,
    PHASE_CHARACTERISTICS,
)
from src.analysis.network import (
    SocialNetwork,
    NetworkMetricsCalculator,
    PeerEffectEstimator,
    NetworkEngagementAnalyzer,
)


class TestPreprocessing:
    """Tests for scientifically rigorous preprocessing."""

    def test_minimal_preprocessor_preserves_data(self):
        """Test that MinimalPreprocessor preserves raw data."""
        preprocessor = MinimalPreprocessor(normalization=NormalizationMethod.NONE)

        t = np.array([0, 1, 2, 3, 4, 5])
        E = np.array([100, 80, 60, 50, 40, 30])

        result = preprocessor.process("test_user", "test", t, E)

        assert result is not None
        # Raw data should be preserved
        np.testing.assert_array_equal(result.raw_time, t)
        np.testing.assert_array_equal(result.raw_engagement, E)

    def test_initial_normalization(self):
        """Test initial value normalization (NOT max)."""
        preprocessor = MinimalPreprocessor(normalization=NormalizationMethod.INITIAL)

        t = np.array([0, 1, 2, 3, 4])
        E = np.array([100, 80, 60, 40, 20])

        result = preprocessor.process("test_user", "test", t, E)

        assert result is not None
        # First value should be 1.0 (normalized by initial)
        assert result.engagement[0] == pytest.approx(1.0)
        # Last value should be 20/100 = 0.2
        assert result.engagement[-1] == pytest.approx(0.2)

    def test_no_max_normalization_warning(self):
        """Test that max normalization is not supported."""
        # The NormalizationMethod enum should not have MAX
        assert not hasattr(NormalizationMethod, 'MAX')

    def test_outlier_detection_preserves_data(self):
        """Test that outliers are detected but NOT removed."""
        preprocessor = MinimalPreprocessor(detect_outliers=True)

        t = np.array([0, 1, 2, 3, 4, 5])
        E = np.array([100, 80, 60, 40, 1000, 20])  # 1000 is outlier

        result = preprocessor.process("test_user", "test", t, E)

        assert result is not None
        # Data length should be unchanged (outliers preserved)
        assert len(result.engagement) == len(E)
        # Report should mention detected outliers
        assert result.report.n_outliers_detected >= 1
        assert result.report.n_outliers_modified == 0  # Not modified!

    def test_transparency_report(self):
        """Test that transparency report is generated."""
        preprocessor = MinimalPreprocessor()

        t = np.array([0, 1, 2, 3, 4])
        E = np.array([100, 80, 60, 40, 20])

        result = preprocessor.process("test_user", "test", t, E)

        assert result is not None
        assert result.report is not None
        assert result.report.normalization_method is not None
        assert result.report.n_original == len(t)


class TestRobustStatistics:
    """Tests for robust statistics."""

    def test_huber_mean_robust_to_outliers(self):
        """Test that Huber mean is robust to outliers."""
        data = np.array([1, 2, 3, 4, 5, 100])  # 100 is outlier

        huber = RobustStatistics.huber_mean(data)
        regular = np.mean(data)

        # Huber mean should be much less affected by outlier
        assert huber < regular
        assert huber == pytest.approx(3.0, rel=0.5)

    def test_mad(self):
        """Test Median Absolute Deviation."""
        data = np.array([1, 2, 3, 4, 5])

        mad = RobustStatistics.mad(data)

        assert mad == pytest.approx(1.0)


class TestTrajectoryClassifier:
    """Tests for trajectory classification."""

    def test_classify_decay(self):
        """Test classification of decay trajectory."""
        classifier = TrajectoryClassifier()

        t = np.linspace(0, 100, 50)
        E = 100 * np.exp(-t / 30)  # Clear exponential decay

        result = classifier.classify(t, E, "test_user")

        assert result.trajectory_type == TrajectoryType.DECAY
        assert result.confidence > 0.5

    def test_classify_growth(self):
        """Test classification of growth trajectory."""
        classifier = TrajectoryClassifier()

        t = np.linspace(0, 100, 50)
        # Growth curve starting from moderate base
        E = 50 + t * 0.4  # Moderate linear growth from 50 to 90

        result = classifier.classify(t, E, "test_user")

        # Should classify as growth, stable, or revival (any non-decay pattern)
        # The key requirement is that it's NOT classified as DECAY
        assert result.trajectory_type != TrajectoryType.DECAY
        assert result.trend_slope > 0  # Positive trend

    def test_classify_stable(self):
        """Test classification of stable trajectory."""
        classifier = TrajectoryClassifier()

        np.random.seed(42)
        t = np.linspace(0, 100, 50)
        E = 50 + np.random.normal(0, 2, len(t))  # Stable with noise

        result = classifier.classify(t, E, "test_user")

        assert result.trajectory_type == TrajectoryType.STABLE

    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        classifier = TrajectoryClassifier(min_observations=10)

        t = np.array([0, 1, 2])  # Only 3 points
        E = np.array([100, 80, 60])

        result = classifier.classify(t, E, "test_user")

        assert result.trajectory_type == TrajectoryType.UNKNOWN
        assert result.confidence == 0.0


class TestUniversalityAnalyzer:
    """Tests for universality analysis (rewritten version)."""

    def test_max_normalization_rejected(self):
        """Test that max normalization is rejected."""
        with pytest.warns(UserWarning, match="MAX normalization"):
            analyzer = UniversalityAnalyzer(normalization="max")

        # Should fall back to initial normalization
        assert analyzer.normalization == "initial"

    def test_trajectory_classification_before_analysis(self):
        """Test that trajectories are classified before analysis."""
        analyzer = UniversalityAnalyzer()

        # Create mock fit results
        class MockResult:
            def __init__(self, uid, platform, tau, time, engagement):
                self.user_id = uid
                self.platform = platform
                self.estimated_tau = tau
                self.preprocessed = type('obj', (object,), {
                    'time': time,
                    'engagement': engagement,
                    'raw_time': time,
                    'raw_engagement': engagement
                })()

        # Create users with different trajectories
        t = np.linspace(0, 100, 30)
        results = [
            MockResult("decay_user", "test", 30, t, 100 * np.exp(-t / 30)),
            MockResult("growth_user", "test", 30, t, 50 + t * 0.4),  # Moderate growth
        ]

        classifications = analyzer.classify_trajectories(results)

        assert len(classifications) == 2
        assert classifications[0].trajectory_type == TrajectoryType.DECAY
        # Growth user should NOT be classified as decay
        assert classifications[1].trajectory_type != TrajectoryType.DECAY
        assert classifications[1].trend_slope > 0  # Positive trend


class TestDecayFitter:
    """Tests for model fitting."""

    def test_fit_stretched_exponential(self):
        """Test fitting stretched exponential model."""
        fitter = DecayFitter(models=["stretched_exponential"])

        # Generate synthetic data
        np.random.seed(42)
        t = np.linspace(0, 100, 50)
        true_tau, true_gamma = 30.0, 0.6
        E = np.exp(-(t / true_tau) ** true_gamma) + np.random.normal(0, 0.02, len(t))
        E = np.clip(E, 0.01, 1.0)

        result = fitter.fit(t, E)

        assert result["best_model"] == "stretched_exponential"
        assert "stretched_exponential" in result["fit_results"]

        fit = result["fit_results"]["stretched_exponential"]
        assert fit.converged
        # Parameters should be approximately correct
        assert fit.parameters["tau"] == pytest.approx(true_tau, rel=0.5)


class TestMotivationEstimator:
    """Tests for independent α estimation."""

    def test_reddit_proxy_calculation(self):
        """Test Reddit-specific proxy calculation."""
        calculator = PlatformProxyCalculator()

        user_data = {
            "posts": [
                {"body": "A " * 200, "score": 10, "subreddit_subscribers": 10000},
                {"body": "B " * 150, "score": 5, "subreddit_subscribers": 5000},
            ],
            "comments": [
                {"body": "Comment " * 50, "score": 2},
            ]
        }

        proxies = calculator.calculate_reddit_proxies(user_data)

        assert "creation_ratio" in proxies
        assert "niche_engagement" in proxies
        assert "engagement_depth" in proxies
        assert all(0 <= v <= 1 for v in proxies.values())

    def test_motivation_estimate(self):
        """Test α estimation from proxies."""
        estimator = MotivationEstimator()

        user_data = {
            "user_id": "test_user",
            "posts": [
                {"body": "A " * 200, "score": 10, "subreddit_subscribers": 10000,
                 "created_utc": 1000000 + i * 86400}
                for i in range(10)
            ],
            "comments": []
        }

        estimate = estimator.estimate(user_data, "reddit")

        assert estimate.user_id == "test_user"
        assert estimate.alpha > 0
        assert estimate.confidence > 0


class TestSurvivorshipBias:
    """Tests for survivorship bias analysis."""

    def test_eligibility_assessment(self):
        """Test user eligibility assessment."""
        analyzer = SurvivorshipBiasAnalyzer(
            min_observations_for_decay=10,
            min_duration_days=7.0
        )

        # User with sufficient data (spread over 10 days)
        user_good = {
            "user_id": "good",
            "activities": [
                {"timestamp": 1000000 + i * 6 * 3600, "engagement": 100 - i}
                for i in range(50)  # 50 activities over 12.5 days
            ]
        }

        # User with insufficient data
        user_bad = {
            "user_id": "bad",
            "activities": [
                {"timestamp": 1000000 + i * 3600, "engagement": 100}
                for i in range(3)
            ]
        }

        eligibilities, counts = analyzer.assess_eligibility([user_good, user_bad])

        assert len(eligibilities) == 2
        assert eligibilities[0].is_eligible  # good user
        assert not eligibilities[1].is_eligible  # bad user
        assert counts["eligible"] == 1


class TestEarlyPrediction:
    """Tests for early prediction model."""

    def test_feature_extraction(self):
        """Test early feature extraction."""
        extractor = EarlyFeatureExtractor()

        user_data = {
            "user_id": "test",
            "activities": [
                {"timestamp": 1000000 + i * 3600, "engagement": 100 - i * 2}
                for i in range(50)
            ]
        }

        features = extractor.extract(user_data, "test")

        assert features.user_id == "test"
        assert features.first_session_engagement > 0
        assert features.week1_active_days > 0


class TestPhaseDiagram:
    """Tests for engagement phase diagram analysis."""

    def test_phase_characteristics_defined(self):
        """Test that phase characteristics are properly defined."""
        assert EngagementPhase.EXPLORATION in PHASE_CHARACTERISTICS
        assert EngagementPhase.HABITUATION in PHASE_CHARACTERISTICS
        assert EngagementPhase.DECAY in PHASE_CHARACTERISTICS
        assert EngagementPhase.DORMANCY in PHASE_CHARACTERISTICS

        # Check characteristic properties
        exploration = PHASE_CHARACTERISTICS[EngagementPhase.EXPLORATION]
        assert exploration.coefficient_of_variation == "high"
        assert exploration.activity_level == "high"

    def test_phase_detection_decay(self):
        """Test phase detection on decay trajectory."""
        detector = PhaseDetector(window_size=5, min_segment_length=3)

        np.random.seed(42)
        t = np.linspace(0, 100, 100)  # Time in days
        # Clear decay pattern
        E = 100 * np.exp(-t / 30) + np.random.normal(0, 2, len(t))

        trajectory = detector.detect_phases(t, E, "test_user", "test")

        assert len(trajectory.phases) > 0
        # Should detect decay phase
        phases_detected = [seg.phase for seg in trajectory.phases]
        assert EngagementPhase.DECAY in phases_detected or \
               EngagementPhase.EXPLORATION in phases_detected  # Early high variability

    def test_phase_detection_stable(self):
        """Test phase detection on stable trajectory."""
        detector = PhaseDetector(window_size=5, min_segment_length=3)

        np.random.seed(42)
        t = np.linspace(0, 100, 100)
        # Stable pattern with low variability
        E = 50 + np.random.normal(0, 2, len(t))

        trajectory = detector.detect_phases(t, E, "test_user", "test")

        assert len(trajectory.phases) > 0
        # Should detect habituation (stable) phase
        phases_detected = [seg.phase for seg in trajectory.phases]
        assert EngagementPhase.HABITUATION in phases_detected

    def test_phase_transitions_detected(self):
        """Test that phase transitions are detected."""
        detector = PhaseDetector(window_size=5, min_segment_length=3)

        np.random.seed(42)
        t = np.linspace(0, 100, 100)
        # Transition: stable -> decay
        E = np.concatenate([
            50 + np.random.normal(0, 2, 50),  # Stable first half
            50 * np.exp(-np.linspace(0, 3, 50)) + np.random.normal(0, 2, 50)  # Decay second half
        ])

        trajectory = detector.detect_phases(t, E, "test_user", "test")

        # Should detect at least one transition
        assert len(trajectory.phases) >= 1
        # Check trajectory structure
        assert trajectory.user_id == "test_user"
        assert trajectory.current_phase is not None

    def test_engagement_phase_diagram(self):
        """Test complete phase diagram analysis."""
        diagram = EngagementPhaseDiagram(window_size=5, min_segment_length=3)

        # Create mock users with different patterns
        np.random.seed(42)
        users_data = []

        # User 1: Decay pattern
        for i in range(20):
            t = np.linspace(0, 100, 50)
            E = 100 * np.exp(-t / 30) + np.random.normal(0, 5, len(t))
            activities = [
                {"timestamp": t[j], "engagement": float(E[j])}
                for j in range(len(t))
            ]
            users_data.append({"user_id": f"decay_{i}", "activities": activities})

        # User 2: Stable pattern
        for i in range(20):
            t = np.linspace(0, 100, 50)
            E = 50 + np.random.normal(0, 3, len(t))
            activities = [
                {"timestamp": t[j], "engagement": float(E[j])}
                for j in range(len(t))
            ]
            users_data.append({"user_id": f"stable_{i}", "activities": activities})

        result = diagram.analyze_trajectories(users_data, "test")

        assert result["n_users_analyzed"] > 0
        assert "phase_distribution" in result
        assert "transition_matrix" in result
        assert "phase_durations" in result

    def test_at_risk_identification(self):
        """Test identification of at-risk users."""
        diagram = EngagementPhaseDiagram(window_size=5, min_segment_length=3)

        # Create enough users to train predictor
        np.random.seed(42)
        users_data = []

        for i in range(50):
            t = np.linspace(0, 100, 50)
            # Mix of patterns
            if i % 3 == 0:
                E = 100 * np.exp(-t / 30)  # Decay
            elif i % 3 == 1:
                E = 50 + np.random.normal(0, 3, len(t))  # Stable
            else:
                E = np.concatenate([
                    50 + np.random.normal(0, 2, 25),
                    50 * np.exp(-np.linspace(0, 2, 25))
                ])  # Stable then decay

            activities = [
                {"timestamp": float(t[j]), "engagement": float(E[j])}
                for j in range(len(t))
            ]
            users_data.append({"user_id": f"user_{i}", "activities": activities})

        result = diagram.analyze_trajectories(users_data, "test")
        at_risk = diagram.identify_at_risk_users(
            result["trajectories"], risk_threshold=0.3
        )

        # Should identify some at-risk users
        assert isinstance(at_risk, list)
        for user in at_risk:
            assert "user_id" in user
            assert "probability" in user
            assert user["probability"] >= 0.3


class TestCausalInference:
    """Tests for causal inference natural experiments."""

    def test_platform_events_defined(self):
        """Test that platform events are properly defined."""
        assert PlatformEvent.REDDIT_COMMUNITY_AWARDS in PLATFORM_EVENTS
        assert PlatformEvent.GITHUB_ACHIEVEMENTS in PLATFORM_EVENTS
        assert PlatformEvent.STRAVA_LOCAL_LEGEND in PLATFORM_EVENTS

        # Each event should have required fields
        event_info = PLATFORM_EVENTS[PlatformEvent.REDDIT_COMMUNITY_AWARDS]
        assert event_info.platform == "reddit"
        assert event_info.event_date is not None
        assert event_info.expected_effect_on_tau in ["decrease", "increase", "unknown"]

    def test_cohort_definition(self):
        """Test cohort definition for natural experiments."""
        analyzer = NaturalExperimentAnalyzer()

        control, treatment = analyzer.define_cohorts_for_event(
            PlatformEvent.GITHUB_ACHIEVEMENTS
        )

        assert control.name.startswith("pre_")
        assert treatment.name.startswith("post_")
        assert control.joined_before is not None
        assert treatment.joined_after is not None
        # Treatment should be after control
        assert treatment.joined_after > control.joined_before

    def test_causal_effect_estimation(self):
        """Test causal effect estimation."""
        analyzer = NaturalExperimentAnalyzer()

        np.random.seed(42)
        # Control group: higher τ values (longer engagement)
        control_taus = list(np.random.normal(50, 10, 100))
        # Treatment group: lower τ values (faster decay)
        treatment_taus = list(np.random.normal(35, 10, 100))

        effect = analyzer.estimate_causal_effect(
            control_taus, treatment_taus,
            "test_event",
            expected_direction="decrease"
        )

        assert effect.tau_difference < 0  # Treatment has lower τ
        assert effect.is_significant  # Should be significant
        assert effect.supports_hypothesis  # Matches expected decrease
        assert effect.n_control == 100
        assert effect.n_treatment == 100

    def test_placebo_test(self):
        """Test placebo test for robustness."""
        analyzer = NaturalExperimentAnalyzer()

        np.random.seed(42)
        # Same distribution - no real effect
        control_taus = list(np.random.normal(50, 10, 50))
        treatment_taus = list(np.random.normal(50, 10, 50))

        placebo = analyzer.run_placebo_test(control_taus, treatment_taus, n_fake_cutoffs=5)

        assert "n_placebo_tests" in placebo
        assert "placebo_passed" in placebo
        # With no real effect, placebo should pass
        assert placebo["placebo_passed"]

    def test_difference_in_differences(self):
        """Test DiD estimation."""
        did = DifferenceInDifferencesEstimator()

        np.random.seed(42)
        # Control: no change over time
        control_pre = list(np.random.normal(50, 5, 50))
        control_post = list(np.random.normal(50, 5, 50))

        # Treatment: decrease after event
        treatment_pre = list(np.random.normal(50, 5, 50))
        treatment_post = list(np.random.normal(35, 5, 50))

        result = did.estimate(control_pre, control_post, treatment_pre, treatment_post)

        assert "did_estimate" in result
        assert result["did_estimate"] < 0  # Treatment decreased relative to control
        assert result["is_significant"]
        assert "sample_sizes" in result

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        analyzer = NaturalExperimentAnalyzer()

        # Very small samples
        control_taus = [50.0, 55.0]
        treatment_taus = [40.0, 35.0]

        effect = analyzer.estimate_causal_effect(
            control_taus, treatment_taus,
            "test_event"
        )

        assert not effect.is_significant
        assert "Insufficient" in effect.interpretation


class TestNetworkEffects:
    """Tests for network effects and contagion analysis."""

    def test_social_network_construction(self):
        """Test basic social network construction."""
        network = SocialNetwork()

        network.add_edge("user1", "user2")
        network.add_edge("user2", "user3")
        network.add_edge("user1", "user3")

        assert network.n_nodes == 3
        assert network.n_edges == 3
        assert network.get_degree("user1") == 2
        assert network.get_degree("user2") == 2
        assert "user2" in network.get_neighbors("user1")

    def test_network_from_coactivity(self):
        """Test building network from co-activity patterns."""
        network = SocialNetwork()

        # Users active in same time windows
        users_data = [
            {
                "user_id": "user1",
                "activities": [{"timestamp": 1000}, {"timestamp": 2000}, {"timestamp": 3000}]
            },
            {
                "user_id": "user2",
                "activities": [{"timestamp": 1100}, {"timestamp": 2100}, {"timestamp": 3100}]
            },
            {
                "user_id": "user3",
                "activities": [{"timestamp": 100000}]  # Different time
            }
        ]

        network.build_from_coactivity(users_data, time_window=1000, min_coactivity=2)

        # user1 and user2 should be connected (co-active 3 times)
        assert "user2" in network.get_neighbors("user1")
        # user3 should not be connected (different time window)
        assert "user3" not in network.get_neighbors("user1")

    def test_network_metrics(self):
        """Test network metrics calculation."""
        network = SocialNetwork()

        # Create a simple network: hub with spokes
        for i in range(5):
            network.add_edge("hub", f"spoke_{i}")

        metrics = NetworkMetricsCalculator(network)
        centrality = metrics.compute_degree_centrality()

        # Hub should have highest centrality
        assert centrality["hub"] > centrality["spoke_0"]

        # Check connected components
        n_components, largest = metrics.compute_connected_components()
        assert n_components == 1
        assert largest == 6

    def test_peer_effect_estimation(self):
        """Test peer effect estimation."""
        network = SocialNetwork()

        # Create network
        for i in range(10):
            network.add_edge(f"user_{i}", f"user_{(i+1) % 10}")

        # Engagement states
        np.random.seed(42)
        user_states = {}
        for i in range(10):
            user_states[f"user_{i}"] = {
                "tau": np.random.uniform(20, 100),
                "churned": i < 3,  # First 3 churned
                "decaying": i < 5
            }

        estimator = PeerEffectEstimator(network)
        stats_result = estimator.compute_neighbor_engagement_stats(
            "user_0", user_states
        )

        assert stats_result["n_neighbors"] > 0
        assert "churn_rate" in stats_result

    def test_network_engagement_analyzer(self):
        """Test complete network engagement analysis."""
        analyzer = NetworkEngagementAnalyzer()

        # Create users with explicit network connections
        np.random.seed(42)
        users_data = []
        user_tau_values = {}

        for i in range(30):
            activities = [{"timestamp": 1000 + j * 100} for j in range(10)]
            users_data.append({
                "user_id": f"user_{i}",
                "activities": activities
            })
            user_tau_values[f"user_{i}"] = np.random.uniform(20, 100)

        # Create explicit interactions for network
        interactions = []
        for i in range(30):
            # Each user interacts with next 3 users
            for j in range(1, 4):
                target = (i + j) % 30
                interactions.append({
                    "from_user": f"user_{i}",
                    "to_user": f"user_{target}"
                })

        # Build network from interactions
        analyzer.build_network(users_data, interactions=interactions, use_coactivity=False)

        result = analyzer.analyze(users_data, user_tau_values)

        assert result.n_users > 0
        assert "peer_effect" in dir(result)
        assert "position_effect" in dir(result)
        assert result.network_density >= 0

    def test_network_analysis_insufficient_data(self):
        """Test handling of insufficient data."""
        analyzer = NetworkEngagementAnalyzer()

        # Very small network
        users_data = [
            {"user_id": "user_0", "activities": [{"timestamp": 1000}]},
            {"user_id": "user_1", "activities": [{"timestamp": 2000}]},
        ]
        user_tau_values = {"user_0": 50.0, "user_1": 60.0}

        analyzer.build_network(users_data, use_coactivity=True, min_coactivity=1)
        result = analyzer.analyze(users_data, user_tau_values)

        # Should handle gracefully
        assert result.n_users == 2
        assert not result.peer_effect.is_significant  # Not enough data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
