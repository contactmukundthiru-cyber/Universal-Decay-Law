/**
 * TypeScript type definitions for the Universal Decay Law Dashboard.
 */

export type TrialStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export type Platform =
  | 'reddit'
  | 'github'
  | 'wikipedia'
  | 'strava'
  | 'lastfm'
  | 'duolingo'
  | 'khan_academy'
  | 'youtube'
  | 'twitter'
  | 'spotify'
  | 'goodreads'
  | 'steam'

export type DecayModel =
  | 'stretched_exponential'
  | 'power_law'
  | 'weibull'
  | 'double_exponential'
  | 'mechanistic'

export interface Dataset {
  id: number
  name: string
  platform: Platform
  description?: string
  status?: string
  target_users?: number
  n_users: number
  date_range_start?: string
  date_range_end?: string
  metadata?: Record<string, unknown>
  created_at: string
  updated_at?: string
}

export interface DatasetCreate {
  name: string
  platform: Platform
  description?: string
}

export interface Trial {
  id: number
  name: string
  dataset_id: number
  status: TrialStatus
  config: TrialConfig
  n_users_processed: number
  collapse_quality?: number
  results?: TrialResults
  error_message?: string
  started_at?: string
  completed_at?: string
  created_at: string
}

export interface TrialConfig {
  models: DecayModel[]
  min_data_points: number
  max_users?: number
  fit_method: string
  cross_validate: boolean
  compute_scaling: boolean
  deviation_threshold: number
}

export interface TrialCreate {
  name: string
  dataset_id: number
  config: TrialConfig
}

export interface TrialResults {
  n_users_fitted: number
  best_model: DecayModel
  model_counts: Record<DecayModel, number>
  mean_tau: number
  std_tau: number
  mean_gamma: number
  std_gamma: number
  collapse_quality: number
  scaling_tau0?: number
  scaling_beta?: number
  scaling_r_squared?: number
  n_deviants: number
  deviant_fraction: number
}

export interface UserFitResult {
  id: number
  trial_id: number
  user_id: string
  best_model: DecayModel
  estimated_tau: number
  estimated_alpha?: number
  deviation_score?: number
  is_deviant: boolean
  model_results: Record<DecayModel, ModelFitResult>
  rescaled_time?: number[]
  rescaled_engagement?: number[]
}

export interface ModelFitResult {
  converged: boolean
  parameters: Record<string, number>
  aic: number
  bic: number
  r_squared: number
  rmse: number
}

export interface MasterCurve {
  id: number
  trial_id: number
  model_name: DecayModel
  parameters: Record<string, number>
  collapse_quality: number
  n_curves: number
}

export interface ScalingRelationship {
  id: number
  trial_id: number
  tau0: number
  beta: number
  r_squared: number
  n_points: number
}

export interface RawCurvesData {
  curves: Array<{
    user_id: string
    time: number[]
    engagement: number[]
  }>
  n_total: number
}

export interface CollapseData {
  curves: Array<{
    user_id: string
    rescaled_time: number[]
    rescaled_engagement: number[]
    is_deviant: boolean
  }>
  master_curve?: {
    x: number[]
    y: number[]
    parameters: Record<string, number>
  }
  collapse_quality?: number
}

export interface ScalingData {
  points: Array<{
    user_id: string
    tau: number
    alpha: number
  }>
  fit_line?: {
    alpha: number[]
    tau: number[]
    tau0: number
    beta: number
    r_squared: number
  }
}

export interface DeviantsData {
  deviation_distribution: {
    values: number[]
    mean?: number
    std?: number
    threshold: number
  }
  deviants: Array<{
    user_id: string
    deviation_score: number
    tau: number
    rescaled_time?: number[]
    rescaled_engagement?: number[]
  }>
  n_deviants: number
}

export interface DashboardData {
  trial: {
    id: number
    name: string
    status: TrialStatus
    n_users: number
    collapse_quality?: number
  }
  master_curve: {
    model?: DecayModel
    parameters?: Record<string, number>
    quality?: number
  }
  scaling: {
    tau0?: number
    beta?: number
    r_squared?: number
  }
  summary_stats: Record<string, unknown>
  fit_results_sample: Array<{
    user_id: string
    tau: number
    alpha?: number
    deviation?: number
  }>
}

export interface ProgressUpdate {
  trial_id: number
  status: TrialStatus
  progress: number
  current_step: string
  n_processed: number
  n_total: number
  eta_seconds?: number
}

export interface PlotlyData {
  data: Array<{
    x: number[]
    y: number[]
    type: string
    mode?: string
    name?: string
    line?: Record<string, unknown>
    marker?: Record<string, unknown>
    opacity?: number
  }>
  layout: Record<string, unknown>
}
