import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Plus, Trash2, Play, Loader2, Database, Settings } from 'lucide-react'
import { format } from 'date-fns'
import clsx from 'clsx'

import { datasetApi } from '@/api/client'
import type { Dataset, Platform } from '@/types'

const PLATFORMS: { value: Platform; label: string; requiresAuth: boolean }[] = [
  { value: 'reddit', label: 'Reddit', requiresAuth: true },
  { value: 'github', label: 'GitHub', requiresAuth: true },
  { value: 'wikipedia', label: 'Wikipedia', requiresAuth: false },
  { value: 'strava', label: 'Strava', requiresAuth: true },
  { value: 'lastfm', label: 'Last.fm', requiresAuth: true },
  { value: 'duolingo', label: 'Duolingo', requiresAuth: true },
  { value: 'khan_academy', label: 'Khan Academy', requiresAuth: true },
  { value: 'youtube', label: 'YouTube', requiresAuth: true },
  { value: 'twitter', label: 'Twitter', requiresAuth: true },
  { value: 'spotify', label: 'Spotify', requiresAuth: true },
  { value: 'goodreads', label: 'Goodreads', requiresAuth: true },
  { value: 'steam', label: 'Steam', requiresAuth: true },
]

function CreateDatasetModal({
  isOpen,
  onClose,
}: {
  isOpen: boolean
  onClose: () => void
}) {
  const queryClient = useQueryClient()
  const [name, setName] = useState('')
  const [platform, setPlatform] = useState<Platform>('reddit')
  const [description, setDescription] = useState('')
  const [targetUsers, setTargetUsers] = useState(100)

  const createMutation = useMutation({
    mutationFn: datasetApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      onClose()
      setName('')
      setDescription('')
    },
  })

  const selectedPlatform = PLATFORMS.find((p) => p.value === platform)

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-md p-6">
        <h2 className="text-xl font-semibold mb-4">Create Dataset</h2>

        <div className="space-y-4">
          <div>
            <label className="label">Name</label>
            <input
              type="text"
              className="input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Reddit Power Users 2024"
            />
          </div>

          <div>
            <label className="label">Platform</label>
            <select
              className="select"
              value={platform}
              onChange={(e) => setPlatform(e.target.value as Platform)}
            >
              {PLATFORMS.map((p) => (
                <option key={p.value} value={p.value}>
                  {p.label} {p.requiresAuth ? '(requires API key)' : '(no auth needed)'}
                </option>
              ))}
            </select>
            {selectedPlatform?.requiresAuth && (
              <p className="text-xs text-amber-600 mt-1">
                Configure API credentials in Settings before collecting data
              </p>
            )}
          </div>

          <div>
            <label className="label">Target Users</label>
            <input
              type="number"
              className="input"
              value={targetUsers}
              onChange={(e) => setTargetUsers(parseInt(e.target.value))}
              min={10}
              max={10000}
            />
            <p className="text-xs text-gray-500 mt-1">
              Number of users to collect engagement data from
            </p>
          </div>

          <div>
            <label className="label">Description</label>
            <textarea
              className="input"
              rows={3}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Optional description..."
            />
          </div>
        </div>

        <div className="flex justify-end gap-3 mt-6">
          <button className="btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn-primary"
            onClick={() =>
              createMutation.mutate({ name, platform, description })
            }
            disabled={!name || createMutation.isPending}
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

function CollectDataModal({
  isOpen,
  onClose,
  dataset,
}: {
  isOpen: boolean
  onClose: () => void
  dataset: Dataset | null
}) {
  const queryClient = useQueryClient()

  const collectMutation = useMutation({
    mutationFn: () => datasetApi.collect(dataset!.id, {}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      onClose()
    },
  })

  if (!isOpen || !dataset) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-md p-6">
        <h2 className="text-xl font-semibold mb-4">Collect Data</h2>

        <div className="space-y-4">
          <p className="text-gray-600">
            Start collecting real engagement data from <strong>{dataset.platform}</strong> for
            dataset "{dataset.name}".
          </p>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-sm text-blue-800">
              <strong>Note:</strong> Make sure you have configured the API credentials
              for {dataset.platform} in the Settings page before proceeding.
            </p>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm text-gray-600">
              <strong>Target:</strong> {dataset.target_users || 100} users<br />
              <strong>Platform:</strong> {dataset.platform}
            </p>
          </div>
        </div>

        <div className="flex justify-end gap-3 mt-6">
          <button className="btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn-primary flex items-center gap-2"
            onClick={() => collectMutation.mutate()}
            disabled={collectMutation.isPending}
          >
            {collectMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <>
                <Play className="w-4 h-4" />
                Start Collection
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}

function DatasetCard({
  dataset,
  onCollect,
}: {
  dataset: Dataset
  onCollect: (dataset: Dataset) => void
}) {
  const queryClient = useQueryClient()

  const deleteMutation = useMutation({
    mutationFn: () => datasetApi.delete(dataset.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
    },
  })

  const platformColors: Record<string, string> = {
    reddit: 'bg-platform-reddit',
    github: 'bg-platform-github',
    wikipedia: 'bg-platform-wikipedia',
    strava: 'bg-platform-strava',
    lastfm: 'bg-platform-lastfm',
    duolingo: 'bg-platform-duolingo',
    khan_academy: 'bg-platform-khan',
    youtube: 'bg-platform-youtube',
    twitter: 'bg-platform-twitter',
    spotify: 'bg-platform-spotify',
    goodreads: 'bg-platform-goodreads',
    steam: 'bg-platform-steam',
  }

  const statusColors: Record<string, string> = {
    pending: 'badge-neutral',
    collecting: 'badge-info',
    processing: 'badge-warning',
    ready: 'badge-success',
    failed: 'badge-error',
  }

  return (
    <div className="card hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div
            className={clsx(
              'w-10 h-10 rounded-lg flex items-center justify-center text-white',
              platformColors[dataset.platform] || 'bg-gray-500'
            )}
          >
            <Database className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">{dataset.name}</h3>
            <p className="text-sm text-gray-500 capitalize">
              {dataset.platform.replace('_', ' ')}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1">
          {dataset.n_users === 0 && (
            <button
              className="p-2 text-gray-400 hover:text-green-500 rounded-lg hover:bg-green-50 transition-colors"
              onClick={() => onCollect(dataset)}
              title="Collect data"
            >
              <Play className="w-4 h-4" />
            </button>
          )}
          <button
            className="p-2 text-gray-400 hover:text-red-500 rounded-lg hover:bg-red-50 transition-colors"
            onClick={() => deleteMutation.mutate()}
            disabled={deleteMutation.isPending}
            title="Delete dataset"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {dataset.description && (
        <p className="mt-3 text-sm text-gray-600">{dataset.description}</p>
      )}

      <div className="mt-4 flex items-center gap-4 text-sm">
        <div>
          <span className="text-gray-500">Users:</span>{' '}
          <span className="font-medium">{dataset.n_users.toLocaleString()}</span>
        </div>
        <div>
          <span className="text-gray-500">Created:</span>{' '}
          <span className="font-medium">
            {format(new Date(dataset.created_at), 'MMM d, yyyy')}
          </span>
        </div>
        <span className={clsx('badge', statusColors[dataset.status || 'pending'])}>
          {dataset.status || 'pending'}
        </span>
      </div>
    </div>
  )
}

export default function DatasetsPage() {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [collectDataset, setCollectDataset] = useState<Dataset | null>(null)

  const { data: datasets, isLoading, error } = useQuery({
    queryKey: ['datasets'],
    queryFn: datasetApi.list,
  })

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Datasets</h1>
          <p className="text-gray-500 mt-1">
            Manage engagement datasets collected from real platforms
          </p>
        </div>

        <div className="flex gap-3">
          <a
            href="/settings"
            className="btn-secondary flex items-center gap-2"
          >
            <Settings className="w-4 h-4" />
            API Settings
          </a>
          <button
            className="btn-primary flex items-center gap-2"
            onClick={() => setShowCreateModal(true)}
          >
            <Plus className="w-4 h-4" />
            New Dataset
          </button>
        </div>
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
        </div>
      )}

      {error && (
        <div className="bg-red-50 text-red-700 p-4 rounded-lg">
          Failed to load datasets: {(error as Error).message}
        </div>
      )}

      {datasets && datasets.length === 0 && (
        <div className="text-center py-12">
          <Database className="w-12 h-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900">No datasets yet</h3>
          <p className="text-gray-500 mt-1">
            Create a dataset and collect real engagement data from platforms
          </p>
          <p className="text-sm text-gray-400 mt-2">
            Configure API credentials in Settings first
          </p>
        </div>
      )}

      {datasets && datasets.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {datasets.map((dataset) => (
            <DatasetCard
              key={dataset.id}
              dataset={dataset}
              onCollect={setCollectDataset}
            />
          ))}
        </div>
      )}

      <CreateDatasetModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
      />
      <CollectDataModal
        isOpen={!!collectDataset}
        onClose={() => setCollectDataset(null)}
        dataset={collectDataset}
      />
    </div>
  )
}
