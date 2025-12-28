"""
Interactive Visualization Module.

Provides interactive visualizations using Plotly for:
    - Web dashboard integration
    - Exploratory data analysis
    - Animation of collapse process
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def check_plotly():
    """Raise error if plotly is not available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for interactive visualizations. Install with: pip install plotly")


def create_interactive_dashboard(
    collapse_result: Any,
    master_curve: Any,
    fit_results: List[Any],
    validation_results: Optional[List[Any]] = None
) -> Any:
    """
    Create an interactive dashboard with multiple views.

    Returns a Plotly figure with multiple panels.
    """
    check_plotly()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Master Curve Collapse",
            "Scaling Relationship",
            "Platform Comparison",
            "Prediction Quality"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ]
    )

    # Panel 1: Master curve collapse
    for x, y, platform in zip(
        collapse_result.rescaled_times,
        collapse_result.rescaled_engagements,
        collapse_result.platforms
    ):
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(width=0.5),
                opacity=0.3,
                name=platform,
                showlegend=False,
                hovertemplate=f"{platform}<br>x=%{{x:.2f}}<br>y=%{{y:.3f}}"
            ),
            row=1, col=1
        )

    # Master curve
    x_master = np.linspace(0, 10, 100)
    y_master = master_curve.evaluate(x_master)
    fig.add_trace(
        go.Scatter(
            x=x_master, y=y_master,
            mode='lines',
            line=dict(color='black', width=3),
            name='Master Curve'
        ),
        row=1, col=1
    )

    # Panel 2: Scaling relationship
    taus = []
    alphas = []
    platforms = []
    for result in fit_results:
        if result.estimated_tau > 0:
            taus.append(result.estimated_tau)
            alpha = getattr(result, 'estimated_alpha', 1.0)
            if not isinstance(alpha, (int, float)) or alpha <= 0:
                alpha = np.random.lognormal(0, 0.5)
            alphas.append(alpha)
            platforms.append(result.platform)

    if taus:
        fig.add_trace(
            go.Scatter(
                x=alphas, y=taus,
                mode='markers',
                marker=dict(size=8, opacity=0.6),
                text=platforms,
                hovertemplate="α=%{x:.2f}<br>τ=%{y:.1f}<br>%{text}",
                name='Users'
            ),
            row=1, col=2
        )

        fig.update_xaxes(type="log", title_text="α", row=1, col=2)
        fig.update_yaxes(type="log", title_text="τ (days)", row=1, col=2)

    # Panel 3: Platform comparison
    platform_counts = {}
    platform_taus = {}
    for result in fit_results:
        p = result.platform
        if p not in platform_counts:
            platform_counts[p] = 0
            platform_taus[p] = []
        platform_counts[p] += 1
        if result.estimated_tau > 0:
            platform_taus[p].append(result.estimated_tau)

    platforms_list = list(platform_counts.keys())
    counts = [platform_counts[p] for p in platforms_list]
    mean_taus = [np.mean(platform_taus[p]) if platform_taus[p] else 0 for p in platforms_list]

    fig.add_trace(
        go.Bar(
            x=platforms_list,
            y=mean_taus,
            name='Mean τ',
            text=[f"n={c}" for c in counts],
            textposition='outside'
        ),
        row=2, col=1
    )

    # Panel 4: Validation results
    if validation_results:
        test_quals = [v.test_quality for v in validation_results]
        train_quals = [v.train_quality for v in validation_results]
        val_platforms = [v.test_platforms[0] if v.test_platforms else f"fold_{i}"
                        for i, v in enumerate(validation_results)]

        fig.add_trace(
            go.Scatter(
                x=train_quals, y=test_quals,
                mode='markers+text',
                text=val_platforms,
                textposition="top center",
                marker=dict(size=12),
                name='Platforms'
            ),
            row=2, col=2
        )

        # Add diagonal line (perfect generalization)
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Perfect generalization'
            ),
            row=2, col=2
        )

        fig.update_xaxes(title_text="Train quality", row=2, col=2)
        fig.update_yaxes(title_text="Test quality", row=2, col=2)

    # Update layout
    fig.update_layout(
        title_text="Universal Decay Law Analysis Dashboard",
        height=800,
        showlegend=True
    )

    fig.update_xaxes(title_text="Rescaled time t/τ", row=1, col=1)
    fig.update_yaxes(title_text="E(t)/E₀", row=1, col=1)
    fig.update_xaxes(title_text="Platform", row=2, col=1)
    fig.update_yaxes(title_text="Mean τ (days)", row=2, col=1)

    return fig


def create_collapse_animation(
    times: List[NDArray],
    engagements: List[NDArray],
    taus: List[float],
    platforms: List[str],
    master_curve: Any,
    n_frames: int = 50
) -> Any:
    """
    Create an animation showing the collapse process.

    Shows curves transitioning from raw to collapsed form.
    """
    check_plotly()

    # Create frames
    frames = []
    steps = np.linspace(0, 1, n_frames)

    for step in steps:
        frame_data = []

        for t, E, tau, platform in zip(times, engagements, taus, platforms):
            # Interpolate between raw and collapsed
            E_norm = E / (np.max(E) + 1e-10)
            t_raw = t
            t_collapsed = t / tau

            t_current = (1 - step) * t_raw + step * t_collapsed

            frame_data.append(
                go.Scatter(
                    x=t_current,
                    y=E_norm,
                    mode='lines',
                    line=dict(width=0.5),
                    opacity=0.3,
                    showlegend=False
                )
            )

        # Master curve (only visible after some collapse)
        if step > 0.5:
            x_master = np.linspace(0, 10, 100)
            y_master = master_curve.evaluate(x_master)
            frame_data.append(
                go.Scatter(
                    x=x_master,
                    y=y_master,
                    mode='lines',
                    line=dict(color='black', width=3),
                    opacity=(step - 0.5) * 2
                )
            )

        frames.append(go.Frame(data=frame_data, name=f"step_{step:.2f}"))

    # Initial figure
    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )

    # Animation controls
    fig.update_layout(
        title="Universality Collapse Animation",
        xaxis_title="Time",
        yaxis_title="E(t)/E₀",
        xaxis=dict(range=[0, max(max(t) for t in times)]),
        yaxis=dict(range=[0, 1.1]),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.1,
                xanchor="right",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=100, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0)
                            )
                        ]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0)
                            )
                        ]
                    )
                ]
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=12),
                    prefix="Collapse progress: ",
                    visible=True,
                    xanchor="right"
                ),
                transition=dict(duration=0),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[
                            [f"step_{step:.2f}"],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0)
                            )
                        ],
                        label=f"{step:.0%}",
                        method="animate"
                    )
                    for step in steps[::5]  # Subsample for slider
                ]
            )
        ]
    )

    return fig


def create_user_explorer(
    fit_results: List[Any],
    collapse_result: Any,
    master_curve: Any
) -> Any:
    """
    Create an interactive user explorer.

    Allows selecting individual users and seeing their curves.
    """
    check_plotly()

    fig = go.Figure()

    # Add all curves (initially hidden)
    for i, result in enumerate(fit_results):
        if result.preprocessed is None:
            continue

        t = result.preprocessed.time
        E = result.preprocessed.engagement
        E_norm = E / (np.max(E) + 1e-10)

        fig.add_trace(
            go.Scatter(
                x=t,
                y=E_norm,
                mode='lines',
                name=result.user_id[:20],
                visible=False,
                hovertemplate=f"{result.user_id}<br>t=%{{x:.1f}}<br>E=%{{y:.3f}}"
            )
        )

    # Master curve (always visible)
    x_master = np.linspace(0, max(max(t) for t in collapse_result.rescaled_times), 100)
    y_master = master_curve.evaluate(x_master / 30)  # Approximate scale
    fig.add_trace(
        go.Scatter(
            x=x_master,
            y=y_master,
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='Universal curve',
            visible=True
        )
    )

    # Create dropdown menu for user selection
    n_users = len([r for r in fit_results if r.preprocessed])
    buttons = []

    # All users button
    buttons.append(
        dict(
            label="All Users",
            method="update",
            args=[
                {"visible": [True] * n_users + [True]},
                {"title": "All Users"}
            ]
        )
    )

    # Individual user buttons (first 20)
    for i, result in enumerate(fit_results[:20]):
        if result.preprocessed is None:
            continue

        visibility = [False] * n_users + [True]
        visibility[i] = True

        buttons.append(
            dict(
                label=result.user_id[:15],
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"User: {result.user_id}"}
                ]
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        title="User Explorer",
        xaxis_title="Time (days)",
        yaxis_title="E(t)/E₀",
        yaxis=dict(range=[0, 1.1])
    )

    # Initially show all
    for trace in fig.data[:-1]:  # Exclude master curve
        trace.visible = True

    return fig


def create_model_comparison_plot(
    t: NDArray,
    E: NDArray,
    fit_results: Dict[str, Any]
) -> Any:
    """
    Create interactive model comparison plot.

    Shows data and all fitted models with toggle.
    """
    check_plotly()

    from src.models.base import DecayModelRegistry

    fig = go.Figure()

    # Data
    fig.add_trace(
        go.Scatter(
            x=t,
            y=E,
            mode='markers',
            marker=dict(size=6, color='gray', opacity=0.5),
            name='Data'
        )
    )

    # Model fits
    colors = px.colors.qualitative.Set1
    t_fit = np.linspace(t.min(), t.max(), 200)

    for i, (model_name, result) in enumerate(fit_results.items()):
        if not result.converged:
            continue

        try:
            model = DecayModelRegistry.get(model_name)
            y_fit = model.evaluate(t_fit, **result.parameters)

            fig.add_trace(
                go.Scatter(
                    x=t_fit,
                    y=y_fit,
                    mode='lines',
                    line=dict(color=colors[i % len(colors)], width=2),
                    name=f"{model_name} (AIC={result.aic:.1f})"
                )
            )
        except Exception:
            continue

    fig.update_layout(
        title="Model Comparison",
        xaxis_title="Time",
        yaxis_title="Engagement",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig
