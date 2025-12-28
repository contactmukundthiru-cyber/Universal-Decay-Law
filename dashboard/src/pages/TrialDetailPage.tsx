import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  ArrowLeft,
  Loader2,
  Download,
  TrendingDown,
  Users,
  Target,
  AlertTriangle,
} from 'lucide-react'

import { trialApi, visualizationApi } from '@/api/client'
import CollapseChart from '@/components/charts/CollapseChart'
import ScalingChart from '@/components/charts/ScalingChart'
import RawCurvesChart from '@/components/charts/RawCurvesChart'
import DeviantsChart from '@/components/charts/DeviantsChart'

function StatCard({
  icon: Icon,
  label,
  value,
  subtitle,
}: {
  icon: typeof TrendingDown
  label: string
  value: string | number
  subtitle?: string
}) {
  return (
    <div className="stat-card">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-primary-50 rounded-lg">
          <Icon className="w-5 h-5 text-primary-600" />
        </div>
        <div>
          <p className="text-sm text-gray-500">{label}</p>
          <p className="stat-value">{value}</p>
          {subtitle && <p className="text-xs text-gray-400 mt-0.5">{subtitle}</p>}
        </div>
      </div>
    </div>
  )
}

export default function TrialDetailPage() {
  const { id } = useParams<{ id: string }>()
  const trialId = parseInt(id!)

  const { data: trial, isLoading: trialLoading } = useQuery({
    queryKey: ['trial', trialId],
    queryFn: () => trialApi.get(trialId),
  })

  const { data: collapseData, isLoading: collapseLoading } = useQuery({
    queryKey: ['collapse', trialId],
    queryFn: () => visualizationApi.getCollapseData(trialId),
    enabled: trial?.status === 'completed',
  })

  const { data: scalingData, isLoading: scalingLoading } = useQuery({
    queryKey: ['scaling', trialId],
    queryFn: () => visualizationApi.getScalingData(trialId),
    enabled: trial?.status === 'completed',
  })

  const { data: rawCurvesData, isLoading: rawCurvesLoading } = useQuery({
    queryKey: ['raw-curves', trialId],
    queryFn: () => visualizationApi.getRawCurves(trialId),
    enabled: trial?.status === 'completed',
  })

  const { data: deviantsData, isLoading: deviantsLoading } = useQuery({
    queryKey: ['deviants', trialId],
    queryFn: () => visualizationApi.getDeviantsData(trialId),
    enabled: trial?.status === 'completed',
  })

  const handleExport = async (format: 'csv' | 'json') => {
    try {
      const blob = await visualizationApi.exportData(trialId, format)
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `trial_${trialId}.${format}`
      a.click()
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Export failed:', error)
    }
  }

  if (trialLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
      </div>
    )
  }

  if (!trial) {
    return (
      <div className="p-8">
        <div className="bg-red-50 text-red-700 p-4 rounded-lg">
          Trial not found
        </div>
      </div>
    )
  }

  const results = trial.results

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-4">
          <Link
            to="/trials"
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5 text-gray-500" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{trial.name}</h1>
            <p className="text-gray-500 mt-1">
              Trial #{trial.id} · Dataset #{trial.dataset_id}
            </p>
          </div>
        </div>

        <div className="flex gap-2">
          <button
            className="btn-secondary flex items-center gap-2"
            onClick={() => handleExport('csv')}
          >
            <Download className="w-4 h-4" />
            Export CSV
          </button>
          <button
            className="btn-secondary flex items-center gap-2"
            onClick={() => handleExport('json')}
          >
            <Download className="w-4 h-4" />
            Export JSON
          </button>
        </div>
      </div>

      {/* Stats Grid */}
      {results && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <StatCard
            icon={Target}
            label="Collapse Quality"
            value={`${(results.collapse_quality * 100).toFixed(1)}%`}
            subtitle="Universality measure"
          />
          <StatCard
            icon={TrendingDown}
            label="Mean τ"
            value={`${results.mean_tau.toFixed(1)}`}
            subtitle={`± ${results.std_tau.toFixed(1)} days`}
          />
          <StatCard
            icon={Users}
            label="Users Fitted"
            value={results.n_users_fitted.toLocaleString()}
            subtitle={`Best model: ${results.best_model}`}
          />
          <StatCard
            icon={AlertTriangle}
            label="Deviants"
            value={`${(results.deviant_fraction * 100).toFixed(1)}%`}
            subtitle={`${results.n_deviants} users`}
          />
        </div>
      )}

      {/* Scaling Parameters */}
      {results?.scaling_tau0 && (
        <div className="card mb-8">
          <h2 className="text-lg font-semibold mb-4">Scaling Relationship</h2>
          <div className="font-mono text-lg">
            τ(α) = {results.scaling_tau0.toFixed(2)} · α
            <sup>-{results.scaling_beta?.toFixed(3)}</sup>
          </div>
          <p className="text-sm text-gray-500 mt-2">
            R² = {results.scaling_r_squared?.toFixed(4)}
          </p>
        </div>
      )}

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Raw Curves */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Raw Decay Curves</h2>
          {rawCurvesLoading ? (
            <div className="h-80 flex items-center justify-center">
              <Loader2 className="w-6 h-6 animate-spin text-primary-600" />
            </div>
          ) : rawCurvesData ? (
            <RawCurvesChart data={rawCurvesData} />
          ) : null}
        </div>

        {/* Master Curve Collapse */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Master Curve Collapse</h2>
          {collapseLoading ? (
            <div className="h-80 flex items-center justify-center">
              <Loader2 className="w-6 h-6 animate-spin text-primary-600" />
            </div>
          ) : collapseData ? (
            <CollapseChart data={collapseData} />
          ) : null}
        </div>

        {/* Scaling Relationship */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">τ-α Scaling</h2>
          {scalingLoading ? (
            <div className="h-80 flex items-center justify-center">
              <Loader2 className="w-6 h-6 animate-spin text-primary-600" />
            </div>
          ) : scalingData ? (
            <ScalingChart data={scalingData} />
          ) : null}
        </div>

        {/* Deviants Analysis */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Deviant Behaviors</h2>
          {deviantsLoading ? (
            <div className="h-80 flex items-center justify-center">
              <Loader2 className="w-6 h-6 animate-spin text-primary-600" />
            </div>
          ) : deviantsData ? (
            <DeviantsChart data={deviantsData} />
          ) : null}
        </div>
      </div>

      {/* Model Comparison */}
      {results && (
        <div className="card mt-6">
          <h2 className="text-lg font-semibold mb-4">Model Comparison</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2 px-4">Model</th>
                  <th className="text-right py-2 px-4">Count</th>
                  <th className="text-right py-2 px-4">Percentage</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(results.model_counts || {}).map(([model, count]) => (
                  <tr key={model} className="border-b last:border-0">
                    <td className="py-2 px-4 capitalize">
                      {model.replace('_', ' ')}
                    </td>
                    <td className="text-right py-2 px-4">{count}</td>
                    <td className="text-right py-2 px-4">
                      {((count / results.n_users_fitted) * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
