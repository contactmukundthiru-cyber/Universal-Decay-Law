import Plot from 'react-plotly.js'
import type { RawCurvesData } from '@/types'

interface Props {
  data: RawCurvesData
  height?: number
}

// Color palette for platforms/users
const colors = [
  '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
  '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
]

export default function RawCurvesChart({ data, height = 320 }: Props) {
  const traces: Plotly.Data[] = data.curves.slice(0, 30).map((curve, i) => ({
    x: curve.time,
    y: curve.engagement,
    type: 'scatter',
    mode: 'lines',
    line: {
      width: 1.5,
      color: colors[i % colors.length],
    },
    opacity: 0.7,
    hovertemplate: `User: ${curve.user_id.slice(0, 15)}<br>t: %{x:.1f} days<br>E/E₀: %{y:.3f}<extra></extra>`,
    showlegend: false,
    name: curve.user_id.slice(0, 10),
  }))

  const layout: Partial<Plotly.Layout> = {
    height,
    margin: { l: 50, r: 20, t: 30, b: 50 },
    xaxis: {
      title: 'Time (days)',
      gridcolor: '#f1f5f9',
      zerolinecolor: '#e2e8f0',
    },
    yaxis: {
      title: 'E(t)/E₀',
      range: [0, 1.05],
      gridcolor: '#f1f5f9',
      zerolinecolor: '#e2e8f0',
    },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    font: {
      family: 'Inter, system-ui, sans-serif',
      size: 12,
    },
    annotations: [
      {
        x: 0.98,
        y: 0.98,
        xref: 'paper',
        yref: 'paper',
        text: `n = ${data.n_total}`,
        showarrow: false,
        font: { size: 11, color: '#64748b' },
      },
    ],
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
