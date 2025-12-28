import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Loader2, BarChart3 } from 'lucide-react'

import { trialApi, visualizationApi } from '@/api/client'
import type { Trial } from '@/types'
import CollapseChart from '@/components/charts/CollapseChart'
import ScalingChart from '@/components/charts/ScalingChart'

export default function AnalysisPage() {
  const [selectedTrialId, setSelectedTrialId] = useState<number | null>(null)

  const { data: trials, isLoading: trialsLoading } = useQuery({
    queryKey: ['trials'],
    queryFn: () => trialApi.list(),
  })

  const completedTrials = trials?.filter((t) => t.status === 'completed') || []

  const { data: collapseData, isLoading: collapseLoading } = useQuery({
    queryKey: ['collapse', selectedTrialId],
    queryFn: () => visualizationApi.getCollapseData(selectedTrialId!),
    enabled: !!selectedTrialId,
  })

  const { data: scalingData, isLoading: scalingLoading } = useQuery({
    queryKey: ['scaling', selectedTrialId],
    queryFn: () => visualizationApi.getScalingData(selectedTrialId!),
    enabled: !!selectedTrialId,
  })

  const selectedTrial = trials?.find((t) => t.id === selectedTrialId)

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Analysis</h1>
        <p className="text-gray-500 mt-1">
          Explore universality and scaling across trials
        </p>
      </div>

      {/* Trial Selector */}
      <div className="card mb-8">
        <label className="label">Select Trial to Analyze</label>
        {trialsLoading ? (
          <div className="flex items-center gap-2 py-2">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm text-gray-500">Loading trials...</span>
          </div>
        ) : completedTrials.length === 0 ? (
          <p className="text-sm text-gray-500 py-2">
            No completed trials available. Run a trial first.
          </p>
        ) : (
          <select
            className="select max-w-md"
            value={selectedTrialId || ''}
            onChange={(e) =>
              setSelectedTrialId(e.target.value ? parseInt(e.target.value) : null)
            }
          >
            <option value="">Select a trial...</option>
            {completedTrials.map((trial) => (
              <option key={trial.id} value={trial.id}>
                {trial.name} (Quality: {((trial.collapse_quality || 0) * 100).toFixed(1)}%)
              </option>
            ))}
          </select>
        )}
      </div>

      {/* No Selection State */}
      {!selectedTrialId && (
        <div className="text-center py-16">
          <BarChart3 className="w-16 h-16 text-gray-200 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-700">
            Select a trial to view analysis
          </h3>
          <p className="text-gray-500 mt-1">
            Choose a completed trial from the dropdown above
          </p>
        </div>
      )}

      {/* Analysis Results */}
      {selectedTrialId && selectedTrial && (
        <div className="space-y-6">
          {/* Summary Stats */}
          {selectedTrial.results && (
            <div className="card">
              <h2 className="text-lg font-semibold mb-4">Summary</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div>
                  <p className="text-sm text-gray-500">Collapse Quality</p>
                  <p className="text-2xl font-bold text-primary-600">
                    {(selectedTrial.results.collapse_quality * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Mean τ</p>
                  <p className="text-2xl font-bold">
                    {selectedTrial.results.mean_tau.toFixed(1)} days
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Universal γ</p>
                  <p className="text-2xl font-bold">
                    {selectedTrial.results.mean_gamma.toFixed(3)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Best Model</p>
                  <p className="text-2xl font-bold capitalize">
                    {selectedTrial.results.best_model.replace('_', ' ')}
                  </p>
                </div>
              </div>

              {/* Scaling Law */}
              {selectedTrial.results.scaling_tau0 && (
                <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                  <p className="text-sm text-gray-500 mb-2">Discovered Scaling Law</p>
                  <p className="text-xl font-mono">
                    τ(α) = {selectedTrial.results.scaling_tau0.toFixed(2)} · α
                    <sup>-{selectedTrial.results.scaling_beta?.toFixed(3)}</sup>
                  </p>
                  <p className="text-sm text-gray-500 mt-2">
                    R² = {selectedTrial.results.scaling_r_squared?.toFixed(4)} ·{' '}
                    This power-law relationship connects engagement timescales to
                    intrinsic motivation.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card">
              <h2 className="text-lg font-semibold mb-4">Master Curve Collapse</h2>
              <p className="text-sm text-gray-500 mb-4">
                All individual decay curves rescaled by their characteristic
                timescale τ collapse onto a universal master curve.
              </p>
              {collapseLoading ? (
                <div className="h-80 flex items-center justify-center">
                  <Loader2 className="w-6 h-6 animate-spin text-primary-600" />
                </div>
              ) : collapseData ? (
                <CollapseChart data={collapseData} height={350} />
              ) : null}
            </div>

            <div className="card">
              <h2 className="text-lg font-semibold mb-4">Scaling Relationship</h2>
              <p className="text-sm text-gray-500 mb-4">
                The characteristic timescale τ follows a power-law relationship
                with the motivation parameter α: τ ∝ α<sup>-β</sup>
              </p>
              {scalingLoading ? (
                <div className="h-80 flex items-center justify-center">
                  <Loader2 className="w-6 h-6 animate-spin text-primary-600" />
                </div>
              ) : scalingData ? (
                <ScalingChart data={scalingData} height={350} />
              ) : null}
            </div>
          </div>

          {/* Interpretation */}
          <div className="card">
            <h2 className="text-lg font-semibold mb-4">Physical Interpretation</h2>
            <div className="prose prose-sm max-w-none text-gray-600">
              <p>
                The universality demonstrated here reveals that human digital engagement
                follows a fundamental decay law that is independent of the specific
                platform or activity type. This suggests that:
              </p>
              <ul className="mt-4 space-y-2">
                <li>
                  <strong>Stretched exponential form:</strong> The decay function
                  f(x) = exp(-x<sup>γ</sup>) with γ ≈ 0.5-0.7 indicates heterogeneous
                  relaxation, analogous to disordered physical systems.
                </li>
                <li>
                  <strong>Motivation scaling:</strong> The power-law τ(α) = τ₀ · α<sup>-β</sup>
                  quantifies how intrinsic motivation extends engagement lifetime.
                </li>
                <li>
                  <strong>Universal γ:</strong> A common stretching exponent across
                  platforms suggests a shared cognitive mechanism underlying disengagement.
                </li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
