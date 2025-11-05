import dynamic from 'next/dynamic'

// Cargamos el formulario de manera dinámica para evitar problemas con SSR
const FormIngreso = dynamic(() => import('../../components/FormIngreso'), { ssr: false })

export default function IngresoPage() {
  return (
    <section className="space-y-6">
      <h1 className="text-2xl font-semibold">Ingreso de datos</h1>
      <p className="text-gray-600 dark:text-gray-400 max-w-prose">
        Completa el siguiente formulario con la información de tu parcela y tus cultivos de interés. Los datos
        serán usados únicamente para generar recomendaciones y proyecciones de producción; no se envían a ningún servidor.
      </p>
      <FormIngreso />
    </section>
  )
}