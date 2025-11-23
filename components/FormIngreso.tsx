"use client"

import { useEffect, useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'
import {
  loadNumericPreprocessor,
  loadCategoricalPreprocessor,
  loadLinearModel,
  loadClusterInfo,
  loadDfSummary,
  NumericPreprocessor,
  CategoricalPreprocessor,
  LinearRegressionModel,
  ClusterInfo,
  DfSummary,
} from '../lib/data/loadArtifacts'
import { applyPipeline } from '../lib/inference/pipeline'
import { predictLinear } from '../lib/inference/lr_predict'
import { getClusterForMunicipio, getRecommendedCrops } from '../lib/inference/cluster'
import { useStore } from '../lib/state/useStore'

const EXCLUDED_FIELDS = new Set(['Año', 'Periodo', 'Área cosechada'])

const AUTO_FIELDS = [
  'Ph_avg',
  'Capacidad de intercambio catiónico_avg',
  'Monóxido de Carbono_avg',
  'Porcentaje de acidez intercambiable_avg',
  'La saturación de aluminio del suelo_avg',
  'temperatura_avg',
  'precipitacion_avg',
]

const FIELD_LABELS: Record<string, string> = {
  'Área sembrada': 'Área sembrada (hectáreas)',
  'Ciclo del cultivo': 'Ciclo del cultivo (días)',
  Ph_avg: 'pH promedio del suelo',
  'Capacidad de intercambio catiónico_avg':
    'Capacidad de intercambio catiónico promedio (cmol(+)/kg)',
  'Monóxido de Carbono_avg': 'Monóxido de carbono promedio (ppm)',
  'Porcentaje de acidez intercambiable_avg': 'Acidez intercambiable promedio (%)',
  'La saturación de aluminio del suelo_avg': 'Saturación de aluminio en el suelo (%)',
  temperatura_avg: 'Temperatura promedio (°C)',
  precipitacion_avg: 'Precipitación promedio (mm)',
}

/**
 * Formulario principal de ingreso de datos.
 * Carga los artefactos JSON en el cliente y gestiona el estado local del formulario.
 */
export default function FormIngreso() {
  const router = useRouter()
  const setFormData = useStore((s) => s.setFormData)
  const setResultData = useStore((s) => s.setResultData)
  const [loading, setLoading] = useState(true)
  const [numPre, setNumPre] = useState<NumericPreprocessor | null>(null)
  const [catPre, setCatPre] = useState<CategoricalPreprocessor | null>(null)
  const [model, setModel] = useState<LinearRegressionModel | null>(null)
  const [clusterInfo, setClusterInfo] = useState<ClusterInfo | null>(null)
  const [municipios, setMunicipios] = useState<string[]>([])
  const [dfSummary, setDfSummary] = useState<DfSummary | null>(null)
  const [gbPreds, setGbPreds] = useState<Record<string, Record<string, Record<string, number | null>>>>({})

  // Form state
  const [municipio, setMunicipio] = useState('')
  const [intereses, setIntereses] = useState<string[]>([])
  const [numericValues, setNumericValues] = useState<Record<string, number | undefined>>({})
  const [years, setYears] = useState(3)
  const [growthRate, setGrowthRate] = useState(0) // porcentaje anual
  const [areaError, setAreaError] = useState<string | null>(null)

  // Cargar artefactos una vez
  useEffect(() => {
    async function init() {
      try {
        const [np, cp, lm, ci, summary] = await Promise.all([
          loadNumericPreprocessor(),
          loadCategoricalPreprocessor(),
          loadLinearModel(),
          loadClusterInfo(),
          loadDfSummary(),
        ])
        // intentar cargar predicciones del modelo GB si existen
        try {
          const gb = await (await fetch('/data/gb_predictions.json')).json()
          setGbPreds(gb)
        } catch (e) {
          // silencio: no hay predicciones GB
        }
        setNumPre(np)
        setCatPre(cp)
        setModel(lm)
        setClusterInfo(ci)
        setDfSummary(summary)
        // Municipios disponibles
        const muniList = Object.keys(ci.municipio_cluster).sort()
        setMunicipios(muniList)
        // Inicializar valores numéricos como vacíos
        const initial: Record<string, number | undefined> = {}
        np.features.forEach((f) => {
          initial[f] = undefined
        })
        setNumericValues(initial)
        setMunicipio(muniList[0] ?? '')
      } catch (err) {
        console.error(err)
      } finally {
        setLoading(false)
      }
    }
    init()
  }, [])

  function handleNumericChange(key: string, value: string) {
    setNumericValues((prev) => ({ ...prev, [key]: value === '' ? undefined : parseFloat(value) }))
  }

  function toggleInteres(cultivo: string) {
    setIntereses((prev) => {
      if (prev.includes(cultivo)) {
        return prev.filter((item) => item !== cultivo)
      }
      return [...prev, cultivo]
    })
  }

  const availableCultivos = useMemo(() => {
    if (!dfSummary) return []
    const muniEntry = municipio ? dfSummary.municipios[municipio] : undefined
    // Prefer the dfSummary per-municipio list; if missing, try clusterInfo's municipio_crops as fallback
    let list: string[] | undefined = muniEntry?.cultivos?.length ? muniEntry.cultivos : undefined
    if ((!list || list.length === 0) && clusterInfo && municipio) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const ciAny: any = clusterInfo as any
      const muniCrops = ciAny?.municipio_crops?.[municipio]
      if (Array.isArray(muniCrops) && muniCrops.length) {
        // Ensure items are strings and trimmed
        list = muniCrops.map((c: string) => (typeof c === 'string' ? c.trim() : String(c)))
      }
    }
    if (!list) list = dfSummary.cultivos
    // Si tenemos info de clusters, y el municipio pertenece a cluster 0 o 1,
    // restringimos la lista a los cultivos sugeridos para ese cluster.
    try {
      // access cluster info from state
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const ciAny: any = (clusterInfo as unknown) as any
      if (ciAny && ciAny.municipio_cluster && municipio) {
        const clu = ciAny.municipio_cluster[municipio]
        if ((clu === 0 || clu === 1) && ciAny.cluster_crops && ciAny.cluster_crops[String(clu)]) {
          // Instead of restricting (intersection), combine municipio list with cluster recommendations
          // preserving municipio items first and appending cluster suggestions that are missing.
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
      // si algo falla, devolvemos la lista original
    }
    return list
  }, [dfSummary, municipio])

  useEffect(() => {
    if (!availableCultivos.length) {
      setIntereses((prev) => (prev.length ? [] : prev))
      return
    }
    const allowed = new Set(availableCultivos)
    setIntereses((prev) => prev.filter((cultivo) => allowed.has(cultivo)))
  }, [availableCultivos])

  useEffect(() => {
    if (!dfSummary || !municipio) return
    const muniEntry = dfSummary.municipios[municipio]
    if (!muniEntry) return
    setNumericValues((prev) => {
      const next = { ...prev }
      AUTO_FIELDS.forEach((field) => {
        const raw = muniEntry.defaults[field]
        next[field] = typeof raw === 'number' && !Number.isNaN(raw) ? raw : undefined
      })
      return next
    })
  }, [dfSummary, municipio])

  useEffect(() => {
    if (!dfSummary) return
    const firstCultivo = intereses[0]
    const ciclo = firstCultivo ? dfSummary.cultivo_ciclos[firstCultivo] : undefined
    setNumericValues((prev) => {
      const nextValue = typeof ciclo === 'number' && !Number.isNaN(ciclo) ? ciclo : undefined
      if (prev['Ciclo del cultivo'] === nextValue) {
        return prev
      }
      return { ...prev, 'Ciclo del cultivo': nextValue }
    })
  }, [dfSummary, intereses])

  async function handleSubmit() {
    if (!numPre || !catPre || !model || !clusterInfo) return
    // Validación: 'Área sembrada' es obligatoria
    const areaSembrada = numericValues['Área sembrada']
    if (typeof areaSembrada !== 'number' || Number.isNaN(areaSembrada)) {
      setAreaError('El campo "Área sembrada" es obligatorio. Por favor ingresa un valor en hectáreas.')
      return
    }
    setAreaError(null)
    // Construir inputs categóricos
    const catInput: Record<string, string | undefined> = {}
    // Decidir qué cultivo usar para la predicción:
    // 1) si el usuario seleccionó uno (intereses[0]) lo usamos;
    // 2) si no, usamos la primera recomendación del cluster (si existe).
    let cultivoForPrediction: string | undefined = intereses[0]
    const cluster = getClusterForMunicipio(municipio, clusterInfo)
    const recommended = getRecommendedCrops(cluster, clusterInfo, municipio, intereses, 5)
    // Si no hay cultivo elegido por el usuario, seleccionamos uno de forma no determinista
    if (!cultivoForPrediction) {
      if (recommended.length > 0) {
        // muestreamos entre las recomendaciones con probabilidad proporcional a la produccion media
        const weights = recommended.map((r) => Math.max(r.mean_production ?? 1, 1))
        const sum = weights.reduce((a, b) => a + b, 0)
        let rnd = Math.random() * sum
        for (let idx = 0; idx < recommended.length; idx++) {
          rnd -= weights[idx]
          if (rnd <= 0) {
            cultivoForPrediction = recommended[idx].cultivo
            break
          }
        }
        cultivoForPrediction = cultivoForPrediction ?? recommended[0].cultivo
      } else if (dfSummary && dfSummary.municipios[municipio]?.cultivos?.length) {
        const list = dfSummary.municipios[municipio].cultivos
        cultivoForPrediction = list[Math.floor(Math.random() * list.length)]
      } else if (dfSummary && dfSummary.cultivos?.length) {
        const list = dfSummary.cultivos
        cultivoForPrediction = list[Math.floor(Math.random() * list.length)]
      }
    }
    catPre.features.forEach((f) => {
      if (f === 'Municipio') {
        catInput[f] = municipio
      } else if (f === 'Cultivo') {
        catInput[f] = cultivoForPrediction ?? ''
      } else if (f === 'Grupo cultivo') {
        catInput[f] = undefined
      } else if (f === 'Estado físico del cultivo') {
        catInput[f] = undefined
      }
    })
    // Guardar datos del formulario
    setFormData({
      municipio,
      intereses,
      numeric: numericValues as Record<string, number>,
      years,
      growthRate,
    })
    // Predecir para cada año
    const preds: number[] = []
    const yearsList: number[] = []
    // baseYear: si el usuario proporcionó 'Año' lo usamos; si no, tomamos el año actual
    const baseYear = typeof numericValues['Año'] === 'number' ? numericValues['Año']! : new Date().getFullYear()
    // preparar datos para dinámica GB si existen predicciones precomputadas
    const cultivoKey = cultivoForPrediction ?? ''
    const muniKey = municipio
    const gbEntryRoot = gbPreds?.[muniKey]?.[cultivoKey]
    let prevGbValue: number | null = null
    if (gbEntryRoot) {
      const availableYearsRoot = Object.keys(gbEntryRoot).sort()
      const baseYearStrRoot = String(baseYear)
      const baseValRawRoot = gbEntryRoot[baseYearStrRoot] ?? gbEntryRoot[availableYearsRoot[0]]
      prevGbValue = (baseValRawRoot as number) ?? 0
    }

    for (let i = 0; i < years; i++) {
      const factor = Math.pow(1 + growthRate / 100, i)
      const numericClone: Record<string, number | undefined> = { ...numericValues }
      if (numericClone['Área sembrada'] !== undefined) {
        numericClone['Área sembrada'] = numericClone['Área sembrada']! * factor
      }
      // Asegurar que 'Área cosechada' sea igual a 'Área sembrada' según requisito
      if (numericClone['Área sembrada'] !== undefined) {
        numericClone['Área cosechada'] = numericClone['Área sembrada']
      }
      // Ajustar año (Año) sumando i
      if (numericClone['Año'] !== undefined) {
        numericClone['Año'] = numericClone['Año']! + i
      }
      // Si no tenemos el 'Ciclo del cultivo' en los valores numéricos, intentar tomarlo del resumen (excel/json)
      if ((numericClone['Ciclo del cultivo'] === undefined || numericClone['Ciclo del cultivo'] === null) && dfSummary) {
        const cicloFromSummary = cultivoForPrediction ? dfSummary.cultivo_ciclos[cultivoForPrediction] : undefined
        if (typeof cicloFromSummary === 'number') {
          numericClone['Ciclo del cultivo'] = cicloFromSummary
        }
      }
      // intentamos usar predicción precomputada del GB (si existe) aplicando dinámica
      const yearForPred = baseYear + i
      let pred: number
      const gbEntry = gbEntryRoot
      if (gbEntry && prevGbValue !== null) {
        // deriva aleatoria en [-3%, +3%] más el growthRate del usuario
        const randomDrift = Math.random() * 0.06 - 0.03
        const drift = growthRate / 100 + randomDrift
        // ruido relativo ±2% del valor previo
        const noise = (Math.random() * 0.04 - 0.02) * Math.abs(prevGbValue)
        pred = prevGbValue * (1 + drift) + noise
        prevGbValue = pred
      } else {
        const vector = applyPipeline(numericClone, catInput, numPre, catPre)
        pred = predictLinear(vector, model)
      }
      preds.push(pred)
      yearsList.push(baseYear + i)
    }
    // Guardar resultado incluyendo el cultivo que se usó para la predicción
    setResultData({ predictions: preds, yearsList, recommended, cluster, cultivo: cultivoForPrediction })
    router.push('/resultado')
  }

  if (loading) {
    return <div>Cargando artefactos…</div>
  }
  if (!numPre || !catPre || !model || !clusterInfo) {
    return <div>No se pudieron cargar los artefactos necesarios.</div>
  }
  return (
    <form
      className="space-y-4"
      onSubmit={(e) => {
        e.preventDefault()
        handleSubmit()
      }}
    >
      {/* Selección de municipio */}
      <div>
        <label className="block text-sm font-medium mb-1" htmlFor="municipio">
          Municipio
        </label>
        <select
          id="municipio"
          className="w-full border rounded-md p-2"
          value={municipio}
          onChange={(e) => setMunicipio(e.target.value)}
        >
          {municipios.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </div>
      {/* Selección de cultivos de interés */}
      <div>
        <label className="block text-sm font-medium mb-1" htmlFor="interes">
          Cultivos de interés (opcional)
        </label>
        <div className="max-h-56 overflow-y-auto border rounded-md p-3 space-y-2">
          {availableCultivos.length === 0 ? (
            <p className="text-sm text-gray-500 dark:text-gray-400">
              No se encontraron cultivos asociados al municipio seleccionado.
            </p>
          ) : (
            availableCultivos.map((cultivo) => {
              const id = `cultivo-${cultivo}`
              return (
                <label key={cultivo} className="flex items-start gap-2 text-sm" htmlFor={id}>
                  <input
                    id={id}
                    type="checkbox"
                    className="mt-1"
                    checked={intereses.includes(cultivo)}
                    onChange={() => toggleInteres(cultivo)}
                  />
                  <span>{cultivo}</span>
                </label>
              )
            })
          )}
        </div>
        {intereses.length > 1 && (
          <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">
            Se utilizará el primer cultivo seleccionado para estimar el ciclo y ejecutar la proyección.
          </p>
        )}
      </div>
      {/* Variables numéricas */}
      <fieldset className="border rounded-md p-4">
        <legend className="text-sm font-semibold">Variables agronómicas y climáticas</legend>
        {numPre.features
          .filter((feature) => !EXCLUDED_FIELDS.has(feature))
          .map((feature) => {
            const isReadOnly = feature === 'Ciclo del cultivo'
            return (
              <div key={feature} className="mt-3">
                <label className="block text-xs font-medium" htmlFor={feature}>
                  {FIELD_LABELS[feature] ?? feature}
                </label>
                <input
                  id={feature}
                  type="number"
                  className="w-full border rounded-md p-2"
                  value={numericValues[feature] ?? ''}
                  onChange={
                    isReadOnly ? undefined : (e) => handleNumericChange(feature, e.target.value)
                  }
                  readOnly={isReadOnly}
                />
                  {feature === 'Área sembrada' && areaError && (
                    <p className="text-sm text-red-600 mt-1">{areaError}</p>
                  )}
              </div>
            )
          })}
      </fieldset>
      {/* Escenario de años y crecimiento */}
      <fieldset className="border rounded-md p-4">
        <legend className="text-sm font-semibold">Escenario de proyección</legend>
        <div className="mt-2">
          <label className="block text-xs font-medium" htmlFor="years">
            Número de años a proyectar: {years}
          </label>
          <input
            id="years"
            type="range"
            min={1}
            max={5}
            value={years}
            onChange={(e) => setYears(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
        <div className="mt-2">
          <label className="block text-xs font-medium" htmlFor="growth">
            Incremento anual del área sembrada (%): {growthRate}
          </label>
          <input
            id="growth"
            type="range"
            min={-20}
            max={50}
            step={1}
            value={growthRate}
            onChange={(e) => setGrowthRate(parseInt(e.target.value))}
            className="w-full"
          />
        </div>
      </fieldset>
      <button
        type="submit"
        className="mt-4 px-6 py-3 bg-primary text-white rounded-md hover:bg-primary-dark"
        disabled={typeof numericValues['Área sembrada'] !== 'number' || Number.isNaN(numericValues['Área sembrada'] as any)}
      >
        Calcular recomendación
      </button>
    </form>
  )
}