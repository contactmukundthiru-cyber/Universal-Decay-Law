"""
Visualization Style Configuration.

Provides publication-quality styling for figures following
Nature/Science journal guidelines.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# Nature-style color palette (colorblind-friendly)
NATURE_COLORS = {
    # Primary
    "blue": "#3B7EA1",
    "red": "#C5203E",
    "green": "#228B22",
    "orange": "#E5811E",
    "purple": "#7B3294",
    "teal": "#008080",
    "gold": "#D4A017",
    "pink": "#E75480",
    # Secondary
    "light_blue": "#89CFF0",
    "light_red": "#F08080",
    "light_green": "#90EE90",
    "gray": "#808080",
    "dark_gray": "#404040",
}

# Platform-specific colors
PLATFORM_COLORS = {
    "reddit": "#FF4500",
    "github": "#333333",
    "wikipedia": "#636466",
    "strava": "#FC4C02",
    "lastfm": "#D51007",
    "twitter": "#1DA1F2",
    "duolingo": "#58CC02",
    "fitbit": "#00B0B9",
    "goodreads": "#553B08",
    "youtube": "#FF0000",
    "facebook": "#4267B2",
    "instagram": "#E1306C",
    # Generic
    "social": NATURE_COLORS["blue"],
    "learning": NATURE_COLORS["green"],
    "fitness": NATURE_COLORS["orange"],
    "collaboration": NATURE_COLORS["purple"],
    "consumption": NATURE_COLORS["teal"],
    "synthetic": NATURE_COLORS["gray"],
}

# Platform markers
PLATFORM_MARKERS = {
    "reddit": "o",
    "github": "s",
    "wikipedia": "^",
    "strava": "D",
    "lastfm": "v",
    "twitter": "<",
    "duolingo": ">",
    "fitbit": "p",
    "goodreads": "h",
    "youtube": "*",
    "facebook": "H",
    "instagram": "8",
    "synthetic": ".",
}


def get_platform_colors() -> Dict[str, str]:
    """Get platform color mapping."""
    return PLATFORM_COLORS.copy()


def get_platform_markers() -> Dict[str, str]:
    """Get platform marker mapping."""
    return PLATFORM_MARKERS.copy()


def get_color_for_platform(platform: str) -> str:
    """Get color for a platform, with fallback."""
    platform = platform.lower()
    if platform in PLATFORM_COLORS:
        return PLATFORM_COLORS[platform]

    # Categorize by keywords
    if any(s in platform for s in ["social", "reddit", "twitter", "facebook"]):
        return PLATFORM_COLORS["social"]
    elif any(s in platform for s in ["learn", "duo", "khan", "course"]):
        return PLATFORM_COLORS["learning"]
    elif any(s in platform for s in ["fit", "strava", "run", "health"]):
        return PLATFORM_COLORS["fitness"]
    elif any(s in platform for s in ["git", "wiki", "collab"]):
        return PLATFORM_COLORS["collaboration"]
    elif any(s in platform for s in ["music", "video", "read", "watch"]):
        return PLATFORM_COLORS["consumption"]

    return NATURE_COLORS["gray"]


def get_marker_for_platform(platform: str) -> str:
    """Get marker for a platform, with fallback."""
    platform = platform.lower()
    return PLATFORM_MARKERS.get(platform, "o")


def set_publication_style(
    style: str = "nature",
    font_size: int = 10,
    figure_width: float = 3.5,  # Single column width in inches
) -> None:
    """
    Set matplotlib style for publication-quality figures.

    Args:
        style: Style preset ("nature", "science", "default")
        font_size: Base font size in points
        figure_width: Figure width in inches
    """
    # Reset to defaults first
    plt.rcdefaults()

    # Common settings for all styles
    base_params = {
        # Figure
        "figure.figsize": (figure_width, figure_width * 0.75),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,

        # Axes
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelpad": 4,
        "axes.titlepad": 8,

        # Ticks
        "xtick.major.size": 3,
        "xtick.major.width": 0.8,
        "xtick.minor.size": 1.5,
        "xtick.minor.width": 0.6,
        "ytick.major.size": 3,
        "ytick.major.width": 0.8,
        "ytick.minor.size": 1.5,
        "ytick.minor.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Lines
        "lines.linewidth": 1.2,
        "lines.markersize": 4,

        # Legend
        "legend.frameon": False,
        "legend.borderaxespad": 0.5,

        # Grid
        "axes.grid": False,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,

        # Math text
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    }

    if style == "nature":
        # Nature-specific settings
        style_params = {
            "axes.prop_cycle": plt.cycler(color=[
                NATURE_COLORS["blue"],
                NATURE_COLORS["red"],
                NATURE_COLORS["green"],
                NATURE_COLORS["orange"],
                NATURE_COLORS["purple"],
                NATURE_COLORS["teal"],
                NATURE_COLORS["gold"],
                NATURE_COLORS["pink"],
            ]),
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    elif style == "science":
        style_params = {
            "axes.prop_cycle": plt.cycler(color=[
                "#0072B2", "#D55E00", "#009E73", "#CC79A7",
                "#F0E442", "#56B4E9", "#E69F00", "#000000"
            ]),
        }
    else:
        style_params = {}

    # Merge and apply
    params = {**base_params, **style_params}
    plt.rcParams.update(params)


def create_figure_panel(
    n_rows: int = 1,
    n_cols: int = 1,
    width_ratios: Optional[List[float]] = None,
    height_ratios: Optional[List[float]] = None,
    figure_width: float = 7.0,  # Double column width
    aspect_ratio: float = 0.75,
    **kwargs
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a multi-panel figure with consistent styling.

    Args:
        n_rows: Number of rows
        n_cols: Number of columns
        width_ratios: Relative widths of columns
        height_ratios: Relative heights of rows
        figure_width: Total figure width in inches
        aspect_ratio: Height/width ratio
        **kwargs: Additional gridspec arguments

    Returns:
        Tuple of (Figure, array of Axes)
    """
    fig_height = figure_width * aspect_ratio * n_rows / n_cols

    gridspec_kw = {}
    if width_ratios:
        gridspec_kw["width_ratios"] = width_ratios
    if height_ratios:
        gridspec_kw["height_ratios"] = height_ratios
    gridspec_kw.update(kwargs)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figure_width, fig_height),
        gridspec_kw=gridspec_kw if gridspec_kw else None,
        squeeze=False
    )

    return fig, axes


def add_panel_labels(
    axes: np.ndarray,
    labels: Optional[List[str]] = None,
    fontsize: int = 12,
    fontweight: str = "bold",
    x_offset: float = -0.1,
    y_offset: float = 1.05
) -> None:
    """
    Add panel labels (A, B, C, etc.) to figure.

    Args:
        axes: Array of axes
        labels: Custom labels (default: A, B, C, ...)
        fontsize: Label font size
        fontweight: Label font weight
        x_offset: Horizontal offset from axis origin
        y_offset: Vertical offset from axis origin
    """
    flat_axes = axes.flatten()

    if labels is None:
        labels = [chr(65 + i) for i in range(len(flat_axes))]  # A, B, C, ...

    for ax, label in zip(flat_axes, labels):
        ax.text(
            x_offset, y_offset, label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight=fontweight,
            va="bottom",
            ha="right"
        )


def colorbar(
    mappable,
    ax: plt.Axes,
    label: str = "",
    orientation: str = "vertical",
    shrink: float = 0.8,
    pad: float = 0.02,
    **kwargs
) -> plt.colorbar:
    """
    Add a well-formatted colorbar.

    Args:
        mappable: Mappable object (from imshow, scatter, etc.)
        ax: Axes to attach colorbar to
        label: Colorbar label
        orientation: "vertical" or "horizontal"
        shrink: Shrink factor
        pad: Padding from axes
        **kwargs: Additional colorbar arguments

    Returns:
        Colorbar object
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)

    if orientation == "vertical":
        cax = divider.append_axes("right", size="5%", pad=pad)
    else:
        cax = divider.append_axes("bottom", size="5%", pad=pad)

    cbar = plt.colorbar(mappable, cax=cax, orientation=orientation, **kwargs)
    cbar.set_label(label)

    return cbar


def save_figure(
    fig: plt.Figure,
    filename: str,
    formats: List[str] = ["pdf", "png", "svg"],
    dpi: int = 300
) -> List[str]:
    """
    Save figure in multiple formats.

    Args:
        fig: Figure to save
        filename: Base filename (without extension)
        formats: List of formats to save
        dpi: DPI for raster formats

    Returns:
        List of saved file paths
    """
    from pathlib import Path

    saved_files = []
    base_path = Path(filename)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        filepath = base_path.with_suffix(f".{fmt}")
        fig.savefig(
            filepath,
            format=fmt,
            dpi=dpi if fmt in ["png", "jpg", "tiff"] else None,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
            edgecolor="none"
        )
        saved_files.append(str(filepath))

    return saved_files
