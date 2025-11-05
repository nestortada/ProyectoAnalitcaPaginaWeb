import { ClusterInfo } from '../data/loadArtifacts'

/** Devuelve la etiqueta de cluster para un municipio dado. */
export function getClusterForMunicipio(
  municipio: string,
  info: ClusterInfo
): number | null {
  return info.municipio_cluster[municipio] ?? null
}

/**
 * Obtiene las recomendaciones de cultivos para un cluster particular, filtrando los cultivos que ya se siembran
 * en el municipio y, opcionalmente, aquellos fuera de los intereses seleccionados.
 * @param cluster Etiqueta de cluster.
 * @param info Información de clusters y cultivos.
 * @param municipio Nombre del municipio para filtrar cultivos ya sembrados.
 * @param intereses Lista opcional de cultivos de interés seleccionados por el usuario.
 * @param top Número de recomendaciones a devolver.
 */
export function getRecommendedCrops(
  cluster: number | null,
  info: ClusterInfo,
  municipio: string,
  intereses?: string[],
  top = 5
) {
  if (cluster === null) return []
  const clusterKey = String(cluster)
  const candidates = info.cluster_crops[clusterKey] || []
  const cultivados = new Set(info.municipio_crops[municipio] || [])
  const filtered = candidates.filter((item) => {
    // filtrar los que ya se siembran
    if (cultivados.has(item.cultivo)) return false
    // si hay intereses, solo incluir los que están en intereses
    if (intereses && intereses.length > 0) {
      return intereses.includes(item.cultivo)
    }
    return true
  })
  return filtered.slice(0, top)
}