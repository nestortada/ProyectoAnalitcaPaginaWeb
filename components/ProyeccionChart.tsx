"use client"

import dynamic from 'next/dynamic'
import React from 'react'
import type { PlotParams } from 'react-plotly.js'

// Cargamos react-plotly.js sólo en cliente para evitar errores de SSR
const Plot = dynamic<PlotParams>(() => import('react-plotly.js'), { ssr: false })

interface ChartProps {
  labels: (number | string)[]
  values: number[]
  municipio?: string
  cultivo?: string
}

export default function ProyeccionChart({ labels, values, municipio, cultivo }: ChartProps) {
  return (
    <div className="overflow-x-auto">
      <Plot
        data={[
          {
            x: labels,
            y: values,
            type: 'scatter',
            mode: 'lines+markers',
            marker: { color: '#047857' },
            name: 'Producción predicha',
          },
        ]}
        layout={{
          title: `${
            typeof municipio === 'string' && municipio ? `${municipio} — ` : ''
          }Proyección de producción${cultivo ? `: ${cultivo}` : ''}`,
          xaxis: { title: 'Año' },
          yaxis: { title: 'Producción (t)' },
          height: 400,
          autosize: true,
        }}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  )
}
