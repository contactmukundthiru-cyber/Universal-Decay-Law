import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  Plus,
  Play,
  Square,
  Trash2,
  Loader2,
  FlaskConical,
  ChevronRight,
  CheckCircle2,
  XCircle,
  Clock,
} from 'lucide-react'
import { format } from 'date-fns'
import clsx from 'clsx'

import { trialApi, datasetApi } from '@/api/client'
import type { Trial, TrialConfig, DecayModel } from '@/types'

const DECAY_MODELS: { value: DecayModel; label: string }[] = [
  { value: 'stretched_exponential', label: 'Stretched Exponential' },
  { value: 'power_law', label: 'Power Law' },
  { value: 'weibull', label: 'Weibull' },
  { value: 'double_exponential', label: 'Double Exponential' },
  { value: 'mechanistic', label: 'Mechanistic SDE' },
]

const statusConfig = {
  pending: { icon: Clock, color: 'text-gray-500', bg: 'bg-gray-100' },
  running: { icon: Loader2, color: 'text-blue-500', bg: 'bg-blue-100' },
  completed: { icon: CheckCircle2, color: 'text-green-500', bg: 'bg-green-100' },
  failed: { icon: XCircle, color: 'text-red-500', bg: 'bg-red-100' },
  cancelled: { icon: Square, color: 'text-orange-500', bg: 'bg-orange-100' },
}

function CreateTrialModal({
  isOpen,
  onClose,
}: {
  isOpen: boolean
  onClose: () => void
}) {
  const queryClient = useQueryClient()
  const [name, setName] = useState('')
  const [datasetId, setDatasetId] = useState<number | null>(null)
  const [selectedModels, setSelectedModels] = useState<DecayModel[]>([
    'stretched_exponential',
    'power_law',
  ])
  const [minDataPoints, setMinDataPoints] = useState(20)
  const [maxUsers, setMaxUsers] = useState<number | undefined>(undefined)
  const [crossValidate, setCrossValidate] = useState(true)
  const [computeScaling, setComputeScaling] = useState(true)
  const [deviationThreshold, setDeviationThreshold] = useState(2.0)

  const { data: datasets } = useQuery({
    queryKey: ['datasets'],
    queryFn: datasetApi.list,
  })

  const createMutation = useMutation({
    mutationFn: trialApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trials'] })
      onClose()
      resetForm()
    },
  })

  const resetForm = () => {
    setName('')
    setDatasetId(null)
    setSelectedModels(['stretched_exponential', 'power_law'])
    setMinDataPoints(20)
    setMaxUsers(undefined)
  }

  const handleModelToggle = (model: DecayModel) => {
    setSelectedModels((prev) =>
      prev.includes(model) ? prev.filter((m) => m !== model) : [...prev, model]
    )
  }

  const handleSubmit = () => {
    if (!name || !datasetId || selectedModels.length === 0) return

    const config: TrialConfig = {
      models: selectedModels,
      min_data_points: minDataPoints,
      max_users: maxUsers,
      fit_method: 'L-BFGS-B',
      cross_validate: crossValidate,
      compute_scaling: computeScaling,
      deviation_threshold: deviationThreshold,
    }

    createMutation.mutate({
      name,
      dataset_id: datasetId,
      config,
    })
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-lg p-6 max-h-[90vh] overflow-y-auto">
        <h2 className="text-xl font-semibold mb-4">Create Trial</h2>

        <div className="space-y-4">
          <div>
            <label className="label">Trial Name</label>
            <input
              type="text"
              className="input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Reddit Universality Analysis v1"
            />
          </div>

          <div>
            <label className="label">Dataset</label>
            <select
              className="select"
              value={datasetId || ''}
              onChange={(e) => setDatasetId(parseInt(e.target.value))}
            >
              <option value="">Select a dataset...</option>
              {datasets?.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.n_users} users)
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="label">Decay Models</label>
            <div className="space-y-2">
              {DECAY_MODELS.map((model) => (
                <label
                  key={model.value}
                  className="flex items-center gap-2 cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(model.value)}
                    onChange={() => handleModelToggle(model.value)}
                    className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  />
                  <span className="text-sm">{model.label}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label">Min Data Points</label>
              <input
                type="number"
                className="input"
                value={minDataPoints}
                onChange={(e) => setMinDataPoints(parseInt(e.target.value))}
                min={5}
              />
            </div>
            <div>
              <label className="label">Max Users (optional)</label>
              <input
                type="number"
                className="input"
                value={maxUsers || ''}
                onChange={(e) =>
                  setMaxUsers(e.target.value ? parseInt(e.target.value) : undefined)
                }
                placeholder="All"
              />
            </div>
          </div>

          <div>
            <label className="label">Deviation Threshold (σ)</label>
            <input
              type="number"
              className="input"
              value={deviationThreshold}
              onChange={(e) => setDeviationThreshold(parseFloat(e.target.value))}
              step={0.1}
              min={0}
            />
          </div>

          <div className="space-y-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={crossValidate}
                onChange={(e) => setCrossValidate(e.target.checked)}
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
              />
              <span className="text-sm">Cross-validate across platforms</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={computeScaling}
                onChange={(e) => setComputeScaling(e.target.checked)}
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
              />
              <span className="text-sm">Compute τ-α scaling relationship</span>
            </label>
          </div>
        </div>

        <div className="flex justify-end gap-3 mt-6">
          <button className="btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn-primary"
            onClick={handleSubmit}
            disabled={
              !name || !datasetId || selectedModels.length === 0 || createMutation.isPending
            }
          >
            {createMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              'Create'
            )}
          </button>
        </div>
      </div>
    </div>
  )
}

function TrialCard({ trial }: { trial: Trial }) {
  const queryClient = useQueryClient()
  const status = statusConfig[trial.status]
  const StatusIcon = status.icon

  const runMutation = useMutation({
    mutationFn: () => trialApi.run(trial.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trials'] })
    },
  })

  const cancelMutation = useMutation({
    mutationFn: () => trialApi.cancel(trial.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trials'] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: () => trialApi.delete(trial.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['trials'] })
    },
  })

  return (
    <div className="card hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div
            className={clsx(
              'w-10 h-10 rounded-lg flex items-center justify-center',
              status.bg
            )}
          >
            <StatusIcon
              className={clsx(
                'w-5 h-5',
                status.color,
                trial.status === 'running' && 'animate-spin'
              )}
            />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">{trial.name}</h3>
            <p className="text-sm text-gray-500 capitalize">
              {trial.status}
              {trial.status === 'running' && ` (${trial.n_users_processed} processed)`}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1">
          {trial.status === 'pending' && (
            <button
              className="p-2 text-gray-400 hover:text-green-500 rounded-lg hover:bg-green-50 transition-colors"
              onClick={() => runMutation.mutate()}
              disabled={runMutation.isPending}
            >
              <Play className="w-4 h-4" />
            </button>
          )}
          {trial.status === 'running' && (
            <button
              className="p-2 text-gray-400 hover:text-orange-500 rounded-lg hover:bg-orange-50 transition-colors"
              onClick={() => cancelMutation.mutate()}
              disabled={cancelMutation.isPending}
            >
              <Square className="w-4 h-4" />
            </button>
          )}
          <button
            className="p-2 text-gray-400 hover:text-red-500 rounded-lg hover:bg-red-50 transition-colors"
            onClick={() => deleteMutation.mutate()}
            disabled={deleteMutation.isPending}
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {trial.status === 'completed' && trial.results && (
        <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Collapse Quality:</span>
            <span className="font-medium ml-1">
              {(trial.results.collapse_quality * 100).toFixed(1)}%
            </span>
          </div>
          <div>
            <span className="text-gray-500">Mean τ:</span>
            <span className="font-medium ml-1">
              {trial.results.mean_tau.toFixed(1)} days
            </span>
          </div>
          <div>
            <span className="text-gray-500">Deviants:</span>
            <span className="font-medium ml-1">
              {(trial.results.deviant_fraction * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      )}

      {trial.status === 'failed' && trial.error_message && (
        <div className="mt-3 text-sm text-red-600 bg-red-50 p-2 rounded">
          {trial.error_message}
        </div>
      )}

      <div className="mt-4 flex items-center justify-between text-sm">
        <span className="text-gray-500">
          Created {format(new Date(trial.created_at), 'MMM d, yyyy HH:mm')}
        </span>

        {trial.status === 'completed' && (
          <Link
            to={`/trials/${trial.id}`}
            className="flex items-center gap-1 text-primary-600 hover:text-primary-700 font-medium"
          >
            View Results
            <ChevronRight className="w-4 h-4" />
          </Link>
        )}
      </div>
    </div>
  )
}

export default function TrialsPage() {
  const [showCreateModal, setShowCreateModal] = useState(false)

  const { data: trials, isLoading, error } = useQuery({
    queryKey: ['trials'],
    queryFn: () => trialApi.list(),
    refetchInterval: 5000, // Poll for updates
  })

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Trials</h1>
          <p className="text-gray-500 mt-1">
            Run universality analysis experiments
          </p>
        </div>

        <button
          className="btn-primary flex items-center gap-2"
          onClick={() => setShowCreateModal(true)}
        >
          <Plus className="w-4 h-4" />
          New Trial
        </button>
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
        </div>
      )}

      {error && (
        <div className="bg-red-50 text-red-700 p-4 rounded-lg">
          Failed to load trials: {(error as Error).message}
        </div>
      )}

      {trials && trials.length === 0 && (
        <div className="text-center py-12">
          <FlaskConical className="w-12 h-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900">No trials yet</h3>
          <p className="text-gray-500 mt-1">
            Create a trial to analyze engagement decay patterns
          </p>
        </div>
      )}

      {trials && trials.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {trials.map((trial) => (
            <TrialCard key={trial.id} trial={trial} />
          ))}
        </div>
      )}

      <CreateTrialModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
      />
    </div>
  )
}
