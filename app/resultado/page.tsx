"use client"

import dynamic from 'next/dynamic'
import Link from 'next/link'
import { useStore } from '../../lib/state/useStore'
import Recomendaciones from '../../components/Recomendaciones'

const ProyeccionChart = dynamic(() => import('../../components/ProyeccionChart'), { ssr: false })

export default function ResultadoPage() {
  const resultData = useStore((s) => s.resultData)
  const formData = useStore((s) => s.formData)
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
  return (
    <section className="space-y-6">
      <h1 className="text-2xl font-semibold">Resultado de la recomendación</h1>
      <div>
        <p className="mb-4">
          Municipio seleccionado: <strong>{formData.municipio}</strong>
        </p>
        {resultData.cluster !== null && (
          <p className="mb-2">
            Se identificó el cluster <strong>{resultData.cluster}</strong> para tu municipio. Las recomendaciones se basan en la producción media de ese grupo.
          </p>
        )}
      </div>
      <div>
        <ProyeccionChart labels={resultData.yearsList} values={resultData.predictions} />
      </div>
      <div>
        <Recomendaciones items={resultData.recommended} />
      </div>
      <div className="mt-6">
        <Link href="/ingreso" className="underline text-primary-dark">
          Volver a ingresar datos
        </Link>
      </div>
    </section>
  )
}