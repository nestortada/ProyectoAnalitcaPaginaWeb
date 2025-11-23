"use client"

import dynamic from 'next/dynamic'
import Link from 'next/link'
import { useEffect, useState } from 'react'
import { useStore } from '../../lib/state/useStore'
import Recomendaciones from '../../components/Recomendaciones'
import { loadDfSummary, loadClusterInfo, loadGbPredictions } from '../../lib/data/loadArtifacts'

const ProyeccionChart = dynamic(() => import('../../components/ProyeccionChart'), { ssr: false })

export default function ResultadoPage() {
  const resultData = useStore((s) => s.resultData)
  const formData = useStore((s) => s.formData)
  const [dfSummary, setDfSummary] = useState<any | null>(null)
  const [clusterInfo, setClusterInfo] = useState<any | null>(null)
  const [gbPreds, setGbPreds] = useState<Record<string, Record<string, Record<string, number | null>>>>({})

  useEffect(() => {
    async function init() {
      try {
        const [df, ci] = await Promise.all([loadDfSummary(), loadClusterInfo()])
        setDfSummary(df)
        setClusterInfo(ci)
        try {
          const gb = await loadGbPredictions()
          setGbPreds(gb)
        } catch (e) {
          // ignore missing gb preds
        }
      } catch (e) {
        console.error(e)
      }
    }
    init()
  }, [])

  if (!resultData || !formData) {
    return (
      <div>
        <p>No hay resultados para mostrar. Por favor completa el formulario de ingreso.</p>
        <Link href="/ingreso" className="text-primary-dark underline">
          Ir al formulario
        </Link>
      </div>
    )
  }

  const municipio = formData.municipio

  // compute available cultivos similar to the form behavior
  function computeAvailableCultivos(): string[] {
    if (!dfSummary) return []
    const muniEntry = municipio ? dfSummary.municipios[municipio] : undefined
    let list: string[] | undefined = muniEntry?.cultivos?.length ? muniEntry.cultivos : undefined
    if ((!list || list.length === 0) && clusterInfo && municipio) {
      const ciAny: any = clusterInfo as any
      const muniCrops = ciAny?.municipio_crops?.[municipio]
      if (Array.isArray(muniCrops) && muniCrops.length) {
        list = muniCrops.map((c: string) => (typeof c === 'string' ? c.trim() : String(c)))
      }
    }
    if (!list) list = dfSummary.cultivos
    // combine with cluster recommendations for the cluster
    try {
      const ciAny: any = clusterInfo as any
      if (ciAny && ciAny.municipio_cluster && municipio) {
        const clu = ciAny.municipio_cluster[municipio]
        if ((clu === 0 || clu === 1) && ciAny.cluster_crops && ciAny.cluster_crops[String(clu)]) {
          const clusterList: string[] = ciAny.cluster_crops[String(clu)].map((c: any) =>
            typeof c.cultivo === 'string' ? c.cultivo.trim() : String(c.cultivo)
          )
          const seen = new Set<string>(list.map((c: string) => (typeof c === 'string' ? c.trim() : String(c))))
          const combined = [...list]
          clusterList.forEach((c) => {
            if (!seen.has(c)) {
              seen.add(c)
              combined.push(c)
            }
          })
          list = combined
        }
      }
    } catch (e) {
      // ignore
    }
    return list
  }

  const availableCultivos = computeAvailableCultivos()

  // Determine details for each recommended crop (show up to 6 other crops)
  const recommended = resultData.recommended ?? []
  const otherCrops = recommended.map((r) => r.cultivo).filter((c) => c !== resultData.cultivo).slice(0, 6)

  return (
    <section className="space-y-6">
      <h1 className="text-2xl font-semibold">Resultado de la recomendación</h1>
      <div>
        <p className="mb-4">
          Municipio seleccionado: <strong>{municipio}</strong>
        </p>
        {resultData.cluster !== null && (
          <p className="mb-2">
            Se identificó el cluster <strong>{resultData.cluster}</strong> para tu municipio. Las recomendaciones se basan en la producción media de ese grupo.
          </p>
        )}
      </div>
      <div>
        <ProyeccionChart
          labels={resultData.yearsList}
          values={resultData.predictions}
          municipio={municipio}
          cultivo={resultData.cultivo}
        />
      </div>

      <div>
        <Recomendaciones items={resultData.recommended} />
      </div>

      <div>
        <h2 className="text-xl font-semibold mt-6">Otros cultivos posibles y detalles</h2>
        {!dfSummary ? (
          <p>Cargando información adicional…</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
            {otherCrops.map((cultivo) => {
              const ciclo = dfSummary.cultivo_ciclos?.[cultivo]
              const muniDefaults = dfSummary.municipios?.[municipio]?.defaults || {}
              // cluster mean production if present
              let meanProd: number | undefined = undefined
              try {
                const ciAny: any = clusterInfo as any
                const clu = ciAny?.municipio_cluster?.[municipio]
                const cc = ciAny?.cluster_crops?.[String(clu)] || []
                const found = cc.find((x: any) => (x.cultivo || '').trim() === cultivo)
                if (found) meanProd = found.mean_production
              } catch (e) {
                // ignore
              }
              const gbEntry = gbPreds?.[municipio]?.[cultivo]
              let gbYears: string[] = []
              let gbValues: number[] = []
              if (gbEntry) {
                gbYears = Object.keys(gbEntry).sort()
                gbValues = gbYears.map((y) => (gbEntry[y] as number) ?? 0)
              }
              return (
                <div key={cultivo} className="border rounded-md p-3">
                  <h3 className="font-semibold">{cultivo}</h3>
                  <p className="text-sm">Ciclo (días): <strong>{typeof ciclo === 'number' ? ciclo : 'N/A'}</strong></p>
                  <p className="text-sm">Temperatura promedio (municipio): <strong>{muniDefaults?.temperatura_avg ?? 'N/A'}</strong></p>
                  <p className="text-sm">Precipitación promedio (municipio): <strong>{muniDefaults?.precipitacion_avg ?? 'N/A'}</strong></p>
                  <p className="text-sm">pH promedio (municipio): <strong>{muniDefaults?.Ph_avg ?? 'N/A'}</strong></p>
                  {typeof meanProd === 'number' && (
                    <p className="text-sm">Producción media en cluster: <strong>{meanProd.toFixed(2)}</strong></p>
                  )}
                  <div className="mt-2">
                    {gbValues.length ? (
                      <ProyeccionChart labels={gbYears.map((y) => parseInt(y))} values={gbValues} municipio={municipio} cultivo={cultivo} />
                    ) : (
                      <p className="text-sm text-gray-600">No hay proyección previa para este cultivo en el municipio.</p>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      <div className="mt-6">
        <Link href="/ingreso" className="underline text-primary-dark">
          Volver a ingresar datos
        </Link>
      </div>
    </section>
  )
}