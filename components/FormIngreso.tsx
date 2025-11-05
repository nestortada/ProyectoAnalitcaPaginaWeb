"use client"

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import {
  loadNumericPreprocessor,
  loadCategoricalPreprocessor,
  loadLinearModel,
  loadClusterInfo,
  NumericPreprocessor,
  CategoricalPreprocessor,
  LinearRegressionModel,
  ClusterInfo,
} from '../lib/data/loadArtifacts'
import { applyPipeline } from '../lib/inference/pipeline'
import { predictLinear } from '../lib/inference/lr_predict'
import { getClusterForMunicipio, getRecommendedCrops } from '../lib/inference/cluster'
import { useStore } from '../lib/state/useStore'

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
  const [cultivosList, setCultivosList] = useState<string[]>([])

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
        const [np, cp, lm, ci] = await Promise.all([
          loadNumericPreprocessor(),
          loadCategoricalPreprocessor(),
          loadLinearModel(),
          loadClusterInfo(),
        ])
        setNumPre(np)
        setCatPre(cp)
        setModel(lm)
        setClusterInfo(ci)
        // Municipios disponibles
        const muniList = Object.keys(ci.municipio_cluster).sort()
        setMunicipios(muniList)
        // Lista de cultivos a partir de las categorías del OneHotEncoder
        const cultivos = cp.categories['Cultivo'] || []
        setCultivosList(cultivos)
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

  function handleInteresChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const selected = Array.from(e.target.selectedOptions).map((o) => o.value)
    setIntereses(selected)
  }

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
        <select
          id="interes"
          multiple
          className="w-full border rounded-md p-2 h-32"
          value={intereses}
          onChange={handleInteresChange}
        >
          {cultivosList.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </div>
      {/* Variables numéricas */}
      <fieldset className="border rounded-md p-4">
        <legend className="text-sm font-semibold">Variables agronómicas y climáticas</legend>
        {numPre.features.map((feature) => (
          <div key={feature} className="mt-3">
            <label className="block text-xs font-medium" htmlFor={feature}>
              {feature}
            </label>
            <input
              id={feature}
              type="number"
              className="w-full border rounded-md p-2"
              value={numericValues[feature] ?? ''}
              onChange={(e) => handleNumericChange(feature, e.target.value)}
            />
          </div>
        ))}
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