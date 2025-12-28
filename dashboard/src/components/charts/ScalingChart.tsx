import Plot from 'react-plotly.js'
import type { ScalingData } from '@/types'

interface Props {
  data: ScalingData
  height?: number
}

export default function ScalingChart({ data, height = 320 }: Props) {
  const traces: Plotly.Data[] = []

  // Add data points
  if (data.points.length > 0) {
    traces.push({
      x: data.points.map((p) => p.alpha),
      y: data.points.map((p) => p.tau),
      type: 'scatter',
      mode: 'markers',
      marker: {
        size: 8,
        color: 'rgba(59, 130, 246, 0.6)',
        line: {
          width: 1,
          color: 'rgba(59, 130, 246, 0.9)',
        },
      },
      text: data.points.map((p) => p.user_id.slice(0, 15)),
      hovertemplate: 'α: %{x:.2f}<br>τ: %{y:.1f} days<br>%{text}<extra></extra>',
      name: 'Users',
    })
  }

  // Add fit line
  if (data.fit_line) {
    traces.push({
      x: data.fit_line.alpha,
      y: data.fit_line.tau,
      type: 'scatter',
      mode: 'lines',
      line: {
        width: 2.5,
        color: '#dc2626',
        dash: 'dash',
      },
      name: `τ = ${data.fit_line.tau0.toFixed(1)} · α^(-${data.fit_line.beta.toFixed(2)})`,
      hovertemplate: 'α: %{x:.2f}<br>τ: %{y:.1f} days<extra>Fit</extra>',
    })
  }

  const layout: Partial<Plotly.Layout> = {
    height,
    margin: { l: 60, r: 20, t: 30, b: 50 },
    xaxis: {
      title: 'Motivation parameter α',
      type: 'log',
      gridcolor: '#f1f5f9',
      zerolinecolor: '#e2e8f0',
    },
    yaxis: {
      title: 'Characteristic time τ (days)',
      type: 'log',
      gridcolor: '#f1f5f9',
      zerolinecolor: '#e2e8f0',
    },
    showlegend: true,
    legend: {
      x: 0.98,
      y: 0.98,
      xanchor: 'right',
      bgcolor: 'rgba(255,255,255,0.8)',
    },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    font: {
      family: 'Inter, system-ui, sans-serif',
      size: 12,
    },
    annotations: data.fit_line
      ? [
          {
            x: 0.02,
            y: 0.02,
            xref: 'paper',
            yref: 'paper',
            text: `R² = ${data.fit_line.r_squared.toFixed(3)}`,
            showarrow: false,
            font: { size: 11, color: '#64748b' },
          },
        ]
      : [],
  }

  return (
    <Plot
      data={traces}
      layout={layout}
      config={{
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false,
      }}
      style={{ width: '100%' }}
    />
  )
}
