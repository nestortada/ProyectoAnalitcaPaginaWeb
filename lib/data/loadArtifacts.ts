// Utilidades para cargar artefactos JSON desde la carpeta public/data.
// Estas funciones se ejecutan en el lado del cliente. Si se usan en el servidor,
// asegúrese de proporcionar una URL absoluta o de usar fs/promises para leer los archivos.

export interface NumericPreprocessor {
  features: string[]
  median: number[]
  mean: number[]
  scale: number[]
}

export interface CategoricalPreprocessor {
  features: string[]
  most_frequent: string[]
  categories: Record<string, string[]>
}

export interface LinearRegressionModel {
  coef: number[]
  intercept: number
}

export interface ClusterInfo {
  municipio_cluster: Record<string, number>
  municipio_crops: Record<string, string[]>
  cluster_crops: Record<string, { cultivo: string; mean_production: number }[]>
}

export interface ClusterScaling {
  cluster_features: string[]
  mean: number[]
  scale: number[]
  centroids: Record<string, number[]>
}

export interface DfSummary {
  cultivos: string[]
  cultivo_ciclos: Record<string, number | null>
  municipios: Record<
    string,
    {
      cultivos: string[]
      defaults: Record<string, number | null>
    }
  >
}

async function fetchJSON<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`Error cargando ${url}: ${res.statusText}`)
  }
  return (await res.json()) as T
}

export async function loadNumericPreprocessor(): Promise<NumericPreprocessor> {
  return fetchJSON<NumericPreprocessor>('/data/numeric_preprocessor.json')
}

export async function loadCategoricalPreprocessor(): Promise<CategoricalPreprocessor> {
  return fetchJSON<CategoricalPreprocessor>('/data/categorical_preprocessor.json')
}

export async function loadLinearModel(): Promise<LinearRegressionModel> {
  return fetchJSON<LinearRegressionModel>('/data/linear_regression.json')
}

export async function loadClusterInfo(): Promise<ClusterInfo> {
  return fetchJSON<ClusterInfo>('/data/cluster_info.json')
}

export async function loadClusterScaling(): Promise<ClusterScaling> {
  return fetchJSON<ClusterScaling>('/data/cluster_scaling_centroids.json')
}

export async function loadDfSummary(): Promise<DfSummary> {
  return fetchJSON<DfSummary>('/data/df_final_summary.json')
}
