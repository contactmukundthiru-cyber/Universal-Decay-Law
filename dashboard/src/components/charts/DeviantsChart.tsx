import Plot from 'react-plotly.js'
import type { DeviantsData } from '@/types'

interface Props {
  data: DeviantsData
  height?: number
}

export default function DeviantsChart({ data, height = 320 }: Props) {
  const traces: Plotly.Data[] = []

  // Histogram of deviation scores
  if (data.deviation_distribution.values.length > 0) {
    traces.push({
      x: data.deviation_distribution.values,
      type: 'histogram',
      marker: {
        color: 'rgba(59, 130, 246, 0.6)',
        line: {
          color: 'rgba(59, 130, 246, 0.9)',
          width: 1,
        },
      },
      nbinsx: 30,
      name: 'Deviation Distribution',
      hovertemplate: 'Deviation: %{x:.2f}<br>Count: %{y}<extra></extra>',
    })
  }

  // Add threshold line
  const threshold = data.deviation_distribution.threshold

  const layout: Partial<Plotly.Layout> = {
    height,
    margin: { l: 50, r: 20, t: 30, b: 50 },
    xaxis: {
      title: 'Deviation Score (σ)',
      gridcolor: '#f1f5f9',
      zerolinecolor: '#e2e8f0',
    },
    yaxis: {
      title: 'Count',
      gridcolor: '#f1f5f9',
      zerolinecolor: '#e2e8f0',
    },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    font: {
      family: 'Inter, system-ui, sans-serif',
      size: 12,
    },
    shapes: [
      {
        type: 'line',
        x0: threshold,
        x1: threshold,
        y0: 0,
        y1: 1,
        yref: 'paper',
        line: {
          color: '#dc2626',
          width: 2,
          dash: 'dash',
        },
      },
    ],
    annotations: [
      {
        x: threshold,
        y: 0.95,
        yref: 'paper',
        text: `Threshold: ${threshold}σ`,
        showarrow: true,
        arrowhead: 0,
        ax: 40,
        ay: 0,
        font: { size: 11, color: '#dc2626' },
      },
      {
        x: 0.98,
        y: 0.98,
        xref: 'paper',
        yref: 'paper',
        text: `${data.n_deviants} deviants`,
        showarrow: false,
        font: { size: 11, color: '#64748b' },
      },
    ],
  }

  return (
    <div>
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

      {/* Top deviants list */}
      {data.deviants.length > 0 && (
        <div className="mt-4 border-t pt-4">
          <p className="text-sm font-medium text-gray-700 mb-2">Top Deviants</p>
          <div className="space-y-1 text-sm max-h-32 overflow-y-auto">
            {data.deviants.slice(0, 5).map((d, i) => (
              <div key={d.user_id} className="flex justify-between text-gray-600">
                <span className="truncate" style={{ maxWidth: '60%' }}>
                  {i + 1}. {d.user_id}
                </span>
                <span className="text-red-600 font-mono">
                  {d.deviation_score.toFixed(2)}σ
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
