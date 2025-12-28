"""
Universal Decay Law of Human Engagement

A cross-platform empirical law governing the decline of human digital activity.

This package implements the theoretical framework, data collection, analysis pipeline,
and validation methodology for discovering universal patterns in human engagement decay
across diverse digital platforms and behaviors.

Core Components:
    - models: Mathematical decay models (stretched exponential, power-law, Weibull)
    - data: Platform-specific data connectors and loaders
    - analysis: Statistical fitting, normalization, and universality collapse
    - visualization: Publication-quality figure generation
    - api: FastAPI backend for the analysis dashboard

Mathematical Framework:
    E(t) = E₀ · f(t/τ(α))

    Where:
    - E(t): Engagement intensity at time t
    - E₀: Initial engagement level
    - f(·): Universal decay function
    - τ(α): Motivation-dependent timescale
    - α: Intrinsic/extrinsic motivation balance parameter

References:
    - Stretched Exponential: f(x) = exp(-x^γ)
    - Power-Law: f(x) = (1 + x)^(-γ)
    - Weibull: f(x) = exp(-(x/λ)^κ)

Author: Research Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__license__ = "MIT"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import DecayModel
    from src.analysis import UniversalityAnalyzer
