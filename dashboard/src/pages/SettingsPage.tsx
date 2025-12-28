import { useState } from 'react'
import { Save, RefreshCw } from 'lucide-react'

export default function SettingsPage() {
  const [apiUrl, setApiUrl] = useState('http://localhost:8000')
  const [redditClientId, setRedditClientId] = useState('')
  const [redditSecret, setRedditSecret] = useState('')
  const [githubToken, setGithubToken] = useState('')
  const [stravaClientId, setStravaClientId] = useState('')
  const [stravaSecret, setStravaSecret] = useState('')
  const [lastfmApiKey, setLastfmApiKey] = useState('')
  const [saved, setSaved] = useState(false)

  const handleSave = () => {
    // In a real app, these would be sent to the backend
    localStorage.setItem(
      'settings',
      JSON.stringify({
        apiUrl,
        redditClientId,
        redditSecret,
        githubToken,
        stravaClientId,
        stravaSecret,
        lastfmApiKey,
      })
    )
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-500 mt-1">
          Configure API connections and credentials
        </p>
      </div>

      <div className="max-w-2xl space-y-6">
        {/* API Configuration */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">API Configuration</h2>
          <div>
            <label className="label">Backend API URL</label>
            <input
              type="text"
              className="input"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="http://localhost:8000"
            />
          </div>
        </div>

        {/* Platform Credentials */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Platform Credentials</h2>
          <p className="text-sm text-gray-500 mb-4">
            Configure API credentials for data collection. These are stored locally
            and sent to the backend when collecting data.
          </p>

          <div className="space-y-4">
            {/* Reddit */}
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-medium text-gray-900 mb-3">Reddit</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Client ID</label>
                  <input
                    type="text"
                    className="input"
                    value={redditClientId}
                    onChange={(e) => setRedditClientId(e.target.value)}
                    placeholder="Your Reddit Client ID"
                  />
                </div>
                <div>
                  <label className="label">Client Secret</label>
                  <input
                    type="password"
                    className="input"
                    value={redditSecret}
                    onChange={(e) => setRedditSecret(e.target.value)}
                    placeholder="Your Reddit Secret"
                  />
                </div>
              </div>
            </div>

            {/* GitHub */}
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-medium text-gray-900 mb-3">GitHub</h3>
              <div>
                <label className="label">Personal Access Token</label>
                <input
                  type="password"
                  className="input"
                  value={githubToken}
                  onChange={(e) => setGithubToken(e.target.value)}
                  placeholder="ghp_..."
                />
              </div>
            </div>

            {/* Strava */}
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-medium text-gray-900 mb-3">Strava</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Client ID</label>
                  <input
                    type="text"
                    className="input"
                    value={stravaClientId}
                    onChange={(e) => setStravaClientId(e.target.value)}
                    placeholder="Your Strava Client ID"
                  />
                </div>
                <div>
                  <label className="label">Client Secret</label>
                  <input
                    type="password"
                    className="input"
                    value={stravaSecret}
                    onChange={(e) => setStravaSecret(e.target.value)}
                    placeholder="Your Strava Secret"
                  />
                </div>
              </div>
            </div>

            {/* Last.fm */}
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-medium text-gray-900 mb-3">Last.fm</h3>
              <div>
                <label className="label">API Key</label>
                <input
                  type="password"
                  className="input"
                  value={lastfmApiKey}
                  onChange={(e) => setLastfmApiKey(e.target.value)}
                  placeholder="Your Last.fm API Key"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Analysis Defaults */}
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Analysis Defaults</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium text-gray-900">Auto-run on create</p>
                <p className="text-sm text-gray-500">
                  Automatically start trials when created
                </p>
              </div>
              <input
                type="checkbox"
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium text-gray-900">Enable caching</p>
                <p className="text-sm text-gray-500">
                  Cache fit results for faster re-analysis
                </p>
              </div>
              <input
                type="checkbox"
                defaultChecked
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
              />
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4">
          <button
            className="btn-primary flex items-center gap-2"
            onClick={handleSave}
          >
            <Save className="w-4 h-4" />
            {saved ? 'Saved!' : 'Save Settings'}
          </button>
          <button
            className="btn-secondary flex items-center gap-2"
            onClick={() => window.location.reload()}
          >
            <RefreshCw className="w-4 h-4" />
            Reset to Defaults
          </button>
        </div>
      </div>
    </div>
  )
}
