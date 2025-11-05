import { NumericPreprocessor, CategoricalPreprocessor } from '../data/loadArtifacts'

/**
 * Aplica la imputación y el escalado a las variables numéricas según los parámetros de entrenamiento.
 * Los valores nulos o no numéricos se reemplazan por la mediana.
 * @param input Valores de entrada proporcionados por el usuario.
 * @param pre Configuración de imputación y escalado exportada desde Python.
 */
export function transformNumericas(
  input: Record<string, number | undefined>,
  pre: NumericPreprocessor
): number[] {
  const output: number[] = []
  for (let i = 0; i < pre.features.length; i++) {
    const feature = pre.features[i]
    const raw = input[feature]
    let value: number = typeof raw === 'number' && !Number.isNaN(raw) ? raw : pre.median[i]
    // Aplicar escalado: (x - mean) / scale
    value = (value - pre.mean[i]) / pre.scale[i]
    output.push(value)
  }
  return output
}

/**
 * Aplica imputación y one‑hot encoding a las variables categóricas.
 * Si el valor de entrada no existe en las categorías entrenadas, se ignora (handle_unknown='ignore').
 * @param input Valores de entrada.
 * @param pre Configuración de imputación y codificación.
 */
export function transformCategoricas(
  input: Record<string, string | undefined>,
  pre: CategoricalPreprocessor
): number[] {
  const encoded: number[] = []
  pre.features.forEach((feat, idx) => {
    const categories = pre.categories[feat] || []
    const raw = input[feat]
    // imputación: si no se proporciona valor, usar el más frecuente
    const value = raw ?? pre.most_frequent[idx]
    categories.forEach((cat) => {
      encoded.push(value === cat ? 1 : 0)
    })
  })
  return encoded
}

/**
 * Combina las transformaciones numéricas y categóricas en un único vector listo para la inferencia.
 * @param numericInput Objeto con valores numéricos.
 * @param categoricalInput Objeto con valores categóricos.
 * @param numPre Artefacto numérico.
 * @param catPre Artefacto categórico.
 */
export function applyPipeline(
  numericInput: Record<string, number | undefined>,
  categoricalInput: Record<string, string | undefined>,
  numPre: NumericPreprocessor,
  catPre: CategoricalPreprocessor
): number[] {
  const numVec = transformNumericas(numericInput, numPre)
  const catVec = transformCategoricas(categoricalInput, catPre)
  return [...numVec, ...catVec]
}