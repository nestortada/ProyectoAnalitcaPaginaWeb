"use client"

interface RecItem {
  cultivo: string
  mean_production: number
}

export default function Recomendaciones({ items }: { items: RecItem[] }) {
  if (!items || items.length === 0) {
    return <p>No se encontraron recomendaciones de cultivos para los criterios seleccionados.</p>
  }
  return (
    <div className="mt-4">
      <h2 className="text-xl font-semibold mb-2">Cultivos recomendados</h2>
      <ul className="space-y-1 list-disc list-inside">
        {items.map((item) => (
          <li key={item.cultivo}>
            <span className="font-medium">{item.cultivo}</span>
            {` — producción media en tu cluster: ${item.mean_production.toFixed(2)}`}
          </li>
        ))}
      </ul>
    </div>
  )
}