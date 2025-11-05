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

const EXCLUDED_FIELDS = new Set(['Año', 'Periodo', 'Área cosechada', 'Ciclo del cultivo'])

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

  // Form state
  const [municipio, setMunicipio] = useState('')
  const [intereses, setIntereses] = useState<string[]>([])
  const [numericValues, setNumericValues] = useState<Record<string, number | undefined>>({})
  const [years, setYears] = useState(3)
  const [growthRate, setGrowthRate] = useState(0) // porcentaje anual

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
    const list = muniEntry?.cultivos?.length ? muniEntry.cultivos : dfSummary.cultivos
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

  async function handleSubmit() {
    if (!numPre || !catPre || !model || !clusterInfo) return
    // Construir inputs categóricos
    const catInput: Record<string, string | undefined> = {}
    catPre.features.forEach((f) => {
      if (f === 'Municipio') {
        catInput[f] = municipio
      } else if (f === 'Cultivo') {
        // El cultivo seleccionado aquí no afecta al modelo de producción directamente; se usa 1er interés si existe
        catInput[f] = intereses[0] ?? ''
      } else if (f === 'Grupo cultivo') {
        // Derivar grupo cultivo del cultivo seleccionado si es posible; de lo contrario, undefined
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
    for (let i = 0; i < years; i++) {
      const factor = Math.pow(1 + growthRate / 100, i)
      const numericClone: Record<string, number | undefined> = { ...numericValues }
      if (numericClone['Área sembrada'] !== undefined) {
        numericClone['Área sembrada'] = numericClone['Área sembrada']! * factor
      }
      // Ajustar año (Año) sumando i
      if (numericClone['Año'] !== undefined) {
        numericClone['Año'] = numericClone['Año']! + i
      }
      const vector = applyPipeline(numericClone, catInput, numPre, catPre)
      const pred = predictLinear(vector, model)
      preds.push(pred)
      yearsList.push(i + 1)
    }
    // Cluster y recomendaciones
    const cluster = getClusterForMunicipio(municipio, clusterInfo)
    const recommended = getRecommendedCrops(cluster, clusterInfo, municipio, intereses, 5)
    setResultData({ predictions: preds, yearsList, recommended, cluster })
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
            Se utilizará el primer cultivo seleccionado para alimentar la proyección del modelo.
          </p>
        )}
      </div>
      {/* Variables numéricas */}
      <fieldset className="border rounded-md p-4">
        <legend className="text-sm font-semibold">Variables agronómicas y climáticas</legend>
        {numPre.features
          .filter((feature) => !EXCLUDED_FIELDS.has(feature))
          .map((feature) => {
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
                  onChange={(e) => handleNumericChange(feature, e.target.value)}
                />
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
      >
        Calcular recomendación
      </button>
    </form>
  )
}