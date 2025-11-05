import { LinearRegressionModel } from '../data/loadArtifacts'

/**
 * Realiza la predicción de un modelo de regresión lineal dado un vector de características.
 * @param vector Vector de características resultante de applyPipeline().
 * @param model Objeto con coeficientes e intercepto.
 */
export function predictLinear(
  vector: number[],
  model: LinearRegressionModel
): number {
  let total = model.intercept
  const len = Math.min(vector.length, model.coef.length)
  for (let i = 0; i < len; i++) {
    total += vector[i] * model.coef[i]
  }
  return total
}