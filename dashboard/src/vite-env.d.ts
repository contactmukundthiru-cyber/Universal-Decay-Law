/// <reference types="vite/client" />

declare module 'react-plotly.js' {
  import Plotly from 'plotly.js'
  import * as React from 'react'

  interface PlotParams {
    data: Plotly.Data[]
    layout?: Partial<Plotly.Layout>
    config?: Partial<Plotly.Config>
    frames?: Plotly.Frame[]
    style?: React.CSSProperties
    className?: string
    useResizeHandler?: boolean
    onInitialized?: (
      figure: Readonly<{ data: Plotly.Data[]; layout: Partial<Plotly.Layout> }>,
      graphDiv: HTMLElement
    ) => void
    onUpdate?: (
      figure: Readonly<{ data: Plotly.Data[]; layout: Partial<Plotly.Layout> }>,
      graphDiv: HTMLElement
    ) => void
    onPurge?: (
      figure: Readonly<{ data: Plotly.Data[]; layout: Partial<Plotly.Layout> }>,
      graphDiv: HTMLElement
    ) => void
    onError?: (err: Error) => void
    divId?: string
  }

  const Plot: React.ComponentType<PlotParams>
  export default Plot
}
