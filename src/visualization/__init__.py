"""
Visualization Module.

Publication-quality figure generation for the Universal Decay Law paper.

Figures:
    1. Raw decay curves across 12 platforms (diversity before normalization)
    2. Motivation-normalized curves collapsing to single law (master curve)
    3. Relationship between α and τ(α) (scaling law log-log plot)
    4. Predictive validation (prediction accuracy vs baseline)
    5. Deviant behaviors and why they deviate
    6. Mechanistic SDE model recovery

Style:
    - Nature/Nature Human Behaviour format
    - Clean, professional appearance
    - Colorblind-friendly palettes
    - High-resolution output (300+ DPI)
"""

from src.visualization.figures import (
    FigureGenerator,
    plot_raw_decay_curves,
    plot_master_curve_collapse,
    plot_scaling_relationship,
    plot_prediction_validation,
    plot_deviant_behaviors,
    plot_mechanistic_recovery,
)
from src.visualization.style import (
    set_publication_style,
    get_platform_colors,
    get_platform_markers,
)
from src.visualization.interactive import (
    create_interactive_dashboard,
    create_collapse_animation,
)

__all__ = [
    "FigureGenerator",
    "plot_raw_decay_curves",
    "plot_master_curve_collapse",
    "plot_scaling_relationship",
    "plot_prediction_validation",
    "plot_deviant_behaviors",
    "plot_mechanistic_recovery",
    "set_publication_style",
    "get_platform_colors",
    "get_platform_markers",
    "create_interactive_dashboard",
    "create_collapse_animation",
]
