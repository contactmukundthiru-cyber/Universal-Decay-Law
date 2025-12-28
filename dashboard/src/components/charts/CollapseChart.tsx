import Plot from 'react-plotly.js'
import type { CollapseData } from '@/types'

interface Props {
  data: CollapseData
  height?: number
}

export default function CollapseChart({ data, height = 320 }: Props) {
  const traces: Plotly.Data[] = []

  // Add individual curves
  data.curves.slice(0, 50).forEach((curve, i) => {
    traces.push({
      x: curve.rescaled_time,
      y: curve.rescaled_engagement,
      type: 'scatter',
      mode: 'lines',
      line: {
        width: 1,
        color: curve.is_deviant ? 'rgba(239, 68, 68, 0.5)' : 'rgba(59, 130, 246, 0.3)',
      },
      hovertemplate: `User: ${curve.user_id.slice(0, 15)}<br>t/τ: %{x:.2f}<br>E/E₀: %{y:.3f}<extra></extra>`,
      showlegend: false,
      name: `User ${i}`,
    })
  })

  // Add master curve
  if (data.master_curve) {
    traces.push({
      x: data.master_curve.x,
      y: data.master_curve.y,
      type: 'scatter',
      mode: 'lines',
      line: {
        width: 3,
        color: '#1e293b',
      },
      name: 'Master Curve',
      hovertemplate: 't/τ: %{x:.2f}<br>f(x): %{y:.3f}<extra>Master Curve</extra>',
    })
  }

  const layout: Partial<Plotly.Layout> = {
    height,
    margin: { l: 50, r: 20, t: 30, b: 50 },
    xaxis: {
      title: 'Rescaled time t/τ',
      range: [0, 6],
      gridcolor: '#f1f5f9',
      zerolinecolor: '#e2e8f0',
    },
    yaxis: {
      title: 'E(t)/E₀',
      range: [0, 1.05],
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
    annotations: data.collapse_quality
      ? [
          {
            x: 0.02,
            y: 0.02,
            xref: 'paper',
            yref: 'paper',
            text: `Quality: ${(data.collapse_quality * 100).toFixed(1)}%`,
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
