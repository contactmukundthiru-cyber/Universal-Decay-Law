/**
 * API client for the Universal Decay Law backend.
 */

import axios, { AxiosError } from 'axios'
import type {
  Dataset,
  DatasetCreate,
  Trial,
  TrialCreate,
  RawCurvesData,
  CollapseData,
  ScalingData,
  DeviantsData,
  DashboardData,
} from '@/types'

const API_BASE = '/api'

const client = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Response interceptor for error handling
client.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    const message = (error.response?.data as { detail?: string })?.detail || error.message
    console.error('API Error:', message)
    return Promise.reject(new Error(message))
  }
)

// Dataset endpoints
export const datasetApi = {
  list: async (): Promise<Dataset[]> => {
    const { data } = await client.get('/datasets/')
    return data
  },

  get: async (id: number): Promise<Dataset> => {
    const { data } = await client.get(`/datasets/${id}`)
    return data
  },

  create: async (dataset: DatasetCreate): Promise<Dataset> => {
    const { data } = await client.post('/datasets/', dataset)
    return data
  },

  delete: async (id: number): Promise<void> => {
    await client.delete(`/datasets/${id}`)
  },

  collect: async (
    id: number,
    params: { limit?: number; query?: string }
  ): Promise<{ message: string; n_users: number }> => {
    const { data } = await client.post(`/datasets/${id}/collect`, params)
    return data
  },
}

// Trial endpoints
export const trialApi = {
  list: async (datasetId?: number): Promise<Trial[]> => {
    const params = datasetId ? { dataset_id: datasetId } : {}
    const { data } = await client.get('/trials/', { params })
    return data
  },

  get: async (id: number): Promise<Trial> => {
    const { data } = await client.get(`/trials/${id}`)
    return data
  },

  create: async (trial: TrialCreate): Promise<Trial> => {
    const { data } = await client.post('/trials/', trial)
    return data
  },

  run: async (id: number): Promise<{ message: string }> => {
    const { data } = await client.post(`/trials/${id}/run`)
    return data
  },

  cancel: async (id: number): Promise<{ message: string }> => {
    const { data } = await client.post(`/trials/${id}/cancel`)
    return data
  },

  delete: async (id: number): Promise<void> => {
    await client.delete(`/trials/${id}`)
  },

  getResults: async (id: number): Promise<Trial> => {
    const { data } = await client.get(`/trials/${id}/results`)
    return data
  },
}

// Visualization endpoints
export const visualizationApi = {
  getRawCurves: async (trialId: number, maxCurves = 50): Promise<RawCurvesData> => {
    const { data } = await client.get(`/visualization/raw-curves/${trialId}`, {
      params: { max_curves: maxCurves },
    })
    return data
  },

  getCollapseData: async (trialId: number, maxCurves = 100): Promise<CollapseData> => {
    const { data } = await client.get(`/visualization/collapse/${trialId}`, {
      params: { max_curves: maxCurves },
    })
    return data
  },

  getScalingData: async (trialId: number): Promise<ScalingData> => {
    const { data } = await client.get(`/visualization/scaling/${trialId}`)
    return data
  },

  getDeviantsData: async (trialId: number, threshold = 2.0): Promise<DeviantsData> => {
    const { data } = await client.get(`/visualization/deviants/${trialId}`, {
      params: { threshold },
    })
    return data
  },

  getDashboardData: async (trialId: number): Promise<DashboardData> => {
    const { data } = await client.get(`/visualization/dashboard/${trialId}`)
    return data
  },

  exportData: async (
    trialId: number,
    format: 'csv' | 'json'
  ): Promise<Blob> => {
    const { data } = await client.post(
      '/visualization/export',
      { trial_id: trialId, format },
      { responseType: 'blob' }
    )
    return data
  },
}

// Analysis endpoints
export const analysisApi = {
  fitUser: async (params: {
    trial_id: number
    user_id: string
    models?: string[]
  }): Promise<unknown> => {
    const { data } = await client.post('/analysis/fit', params)
    return data
  },

  compareModels: async (trialId: number): Promise<unknown> => {
    const { data } = await client.get(`/analysis/compare/${trialId}`)
    return data
  },

  predict: async (params: {
    trial_id: number
    user_id: string
    time_horizon: number
  }): Promise<unknown> => {
    const { data } = await client.post('/analysis/predict', params)
    return data
  },

  runStatisticalTests: async (trialId: number): Promise<unknown> => {
    const { data } = await client.get(`/analysis/stats/${trialId}`)
    return data
  },
}

export default client
