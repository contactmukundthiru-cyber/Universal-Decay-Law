import { Routes, Route, NavLink } from 'react-router-dom'
import { Activity, Database, FlaskConical, BarChart3, Settings } from 'lucide-react'
import clsx from 'clsx'

import DatasetsPage from '@/pages/DatasetsPage'
import TrialsPage from '@/pages/TrialsPage'
import TrialDetailPage from '@/pages/TrialDetailPage'
import AnalysisPage from '@/pages/AnalysisPage'
import SettingsPage from '@/pages/SettingsPage'

const navItems = [
  { to: '/', icon: Database, label: 'Datasets' },
  { to: '/trials', icon: FlaskConical, label: 'Trials' },
  { to: '/analysis', icon: BarChart3, label: 'Analysis' },
  { to: '/settings', icon: Settings, label: 'Settings' },
]

function Sidebar() {
  return (
    <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
      <div className="p-6 border-b border-gray-100">
        <div className="flex items-center gap-3">
          <Activity className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="font-semibold text-gray-900">Universal Decay Law</h1>
            <p className="text-xs text-gray-500">Research Dashboard</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 p-4">
        <ul className="space-y-1">
          {navItems.map(({ to, icon: Icon, label }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-3 px-3 py-2 rounded-lg transition-colors',
                    isActive
                      ? 'bg-primary-50 text-primary-700'
                      : 'text-gray-600 hover:bg-gray-50'
                  )
                }
              >
                <Icon className="w-5 h-5" />
                <span className="font-medium">{label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      <div className="p-4 border-t border-gray-100">
        <div className="text-xs text-gray-400">
          <p>E(t) = E₀ · f(t/τ(α))</p>
          <p className="mt-1">τ(α) = τ₀ · α<sup>-β</sup></p>
        </div>
      </div>
    </aside>
  )
}

function App() {
  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />

      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<DatasetsPage />} />
          <Route path="/trials" element={<TrialsPage />} />
          <Route path="/trials/:id" element={<TrialDetailPage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
