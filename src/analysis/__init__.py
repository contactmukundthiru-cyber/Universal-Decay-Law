"""
Analysis Pipeline Module - Scientifically Rigorous Version.

This module provides the complete analysis pipeline for discovering
and validating the Universal Decay Law.

IMPORTANT: This version addresses Nature reviewer concerns:
1. NO CIRCULAR REASONING - α measured independently of τ
2. NO FORCED NORMALIZATION - no max normalization that creates artifacts
3. HYPOTHESIS TESTING - test universality, don't assume it
4. SURVIVORSHIP BIAS - explicitly model excluded users
5. PREDICTIVE VALIDATION - demonstrate genuine predictive power

Pipeline Components:

1. Preprocessing (preprocessing.py)
   - Minimal data transformation
   - Preserves raw data
   - Full transparency reporting
   - NO outlier clipping, NO max normalization

2. Model Fitting (fitting.py)
   - Per-user decay model fitting
   - Multiple model comparison (AIC/BIC)
   - Scientifically rigorous defaults

3. Motivation Estimation (motivation.py) - NEW
   - INDEPENDENT α measurement (not derived from τ)
   - Platform-specific proxy variables
   - Natural experiment analysis
   - Avoids circular reasoning

4. Universality Analysis (universality.py) - REWRITTEN
   - Trajectory classification BEFORE analysis
   - Initial value normalization (not max)
   - Hypothesis testing framework
   - τ fitted from data (not optimized for collapse)

5. Survivorship Bias Analysis (survivorship.py) - NEW
   - Two-stage selection modeling
   - Quantifies who the law applies to
   - Reports law applicability rate

6. Early Prediction (prediction.py) - NEW
   - Predict τ from first few sessions
   - Validated across time and platforms
   - Transforms law from descriptive to predictive

7. Validation (validation.py)
   - Cross-platform validation
   - Out-of-sample prediction
   - Bootstrap confidence intervals

8. Statistical Tests (statistics.py)
   - Universality hypothesis testing
   - Model comparison
   - Residual analysis
"""

# Preprocessing - Scientifically rigorous version
from src.analysis.preprocessing import (
    NormalizationMethod,
    MinimalPreprocessor,
    DatasetPreprocessor,
    PreprocessedUser,
    DataQualityMetrics,
    TransparencyReport,
    DatasetPreprocessingReport,
    RobustStatistics,
)

# Model Fitting
from src.analysis.fitting import (
    DecayFitter,
    BatchFitter,
    FittingPipeline,
    PipelineConfig,
    UserFitResult,
)

# Universality Analysis - Rewritten to avoid circular reasoning
from src.analysis.universality import (
    UniversalityAnalyzer,
    CollapseResult,
    MasterCurve,
    TrajectoryClassifier,
    TrajectoryClassification,
    TrajectoryType,
    analyze_scaling_relationship,
)

# Motivation - Independent α estimation
from src.analysis.motivation import (
    MotivationEstimator,
    MotivationEstimate,
    MotivationProxy,
    NaturalExperimentAnalyzer,
    NaturalExperimentResult,
    PlatformProxyCalculator,
    generate_alpha_validation_report,
)

# Survivorship Bias - Two-stage modeling
from src.analysis.survivorship import (
    SurvivorshipBiasAnalyzer,
    SelectionModelResult,
    TwoStageResult,
    UserEligibility,
    generate_survivorship_report,
)

# Early Prediction - Revolutionary capability
from src.analysis.prediction import (
    EarlyPredictionModel,
    EarlyFeatureExtractor,
    EarlyFeatures,
    PredictionResult,
    PredictionValidator,
    ModelPerformance,
    PredictionTarget,
    generate_prediction_report,
)

# Validation
from src.analysis.validation import (
    CrossValidator,
    Predictor,
    ValidationResult,
)

# Statistical Tests
from src.analysis.statistics import (
    StatisticalTests,
    BootstrapAnalysis,
    HypothesisTestResult,
)

# Causal Inference - Natural Experiments
from src.analysis.causal import (
    NaturalExperimentAnalyzer,
    DifferenceInDifferencesEstimator,
    CausalEffectEstimate,
    CohortDefinition,
    CohortStats,
    PlatformEvent,
    PlatformEventInfo,
    PLATFORM_EVENTS,
    generate_causal_analysis_report,
)

# Phase Diagram - Beyond Universal Law
from src.analysis.phases import (
    EngagementPhase,
    PhaseCharacteristics,
    PHASE_CHARACTERISTICS,
    PhaseSegment,
    PhaseTransition,
    PhaseTrajectory,
    PhaseDetector,
    TransitionPredictor,
    EngagementPhaseDiagram,
    generate_phase_diagram_report,
)

# Network Effects and Contagion
from src.analysis.network import (
    NetworkMetricType,
    UserNetworkProfile,
    ContagionEvent,
    NetworkEffect,
    NetworkAnalysisResult,
    SocialNetwork,
    NetworkMetricsCalculator,
    PeerEffectEstimator,
    ContagionAnalyzer,
    NetworkPositionAnalyzer,
    NetworkEngagementAnalyzer,
    generate_network_analysis_report,
)

__all__ = [
    # Preprocessing
    "NormalizationMethod",
    "MinimalPreprocessor",
    "DatasetPreprocessor",
    "PreprocessedUser",
    "DataQualityMetrics",
    "TransparencyReport",
    "DatasetPreprocessingReport",
    "RobustStatistics",

    # Fitting
    "DecayFitter",
    "BatchFitter",
    "FittingPipeline",
    "PipelineConfig",
    "UserFitResult",

    # Universality
    "UniversalityAnalyzer",
    "CollapseResult",
    "MasterCurve",
    "TrajectoryClassifier",
    "TrajectoryClassification",
    "TrajectoryType",
    "analyze_scaling_relationship",

    # Motivation (Independent α)
    "MotivationEstimator",
    "MotivationEstimate",
    "MotivationProxy",
    "NaturalExperimentAnalyzer",
    "NaturalExperimentResult",
    "PlatformProxyCalculator",
    "generate_alpha_validation_report",

    # Survivorship Bias
    "SurvivorshipBiasAnalyzer",
    "SelectionModelResult",
    "TwoStageResult",
    "UserEligibility",
    "generate_survivorship_report",

    # Early Prediction
    "EarlyPredictionModel",
    "EarlyFeatureExtractor",
    "EarlyFeatures",
    "PredictionResult",
    "PredictionValidator",
    "ModelPerformance",
    "PredictionTarget",
    "generate_prediction_report",

    # Validation
    "CrossValidator",
    "Predictor",
    "ValidationResult",

    # Statistics
    "StatisticalTests",
    "BootstrapAnalysis",
    "HypothesisTestResult",

    # Causal Inference
    "NaturalExperimentAnalyzer",
    "DifferenceInDifferencesEstimator",
    "CausalEffectEstimate",
    "CohortDefinition",
    "CohortStats",
    "PlatformEvent",
    "PlatformEventInfo",
    "PLATFORM_EVENTS",
    "generate_causal_analysis_report",

    # Phase Diagram
    "EngagementPhase",
    "PhaseCharacteristics",
    "PHASE_CHARACTERISTICS",
    "PhaseSegment",
    "PhaseTransition",
    "PhaseTrajectory",
    "PhaseDetector",
    "TransitionPredictor",
    "EngagementPhaseDiagram",
    "generate_phase_diagram_report",

    # Network Effects
    "NetworkMetricType",
    "UserNetworkProfile",
    "ContagionEvent",
    "NetworkEffect",
    "NetworkAnalysisResult",
    "SocialNetwork",
    "NetworkMetricsCalculator",
    "PeerEffectEstimator",
    "ContagionAnalyzer",
    "NetworkPositionAnalyzer",
    "NetworkEngagementAnalyzer",
    "generate_network_analysis_report",
]
