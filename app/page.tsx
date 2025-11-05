import Link from 'next/link'

export default function HomePage() {
  return (
    <section className="space-y-8">
      <h1 className="text-3xl font-bold text-primary-dark">Meta 2030: Duplicar la productividad agrícola</h1>
      <p className="max-w-prose text-lg">
        Contribuimos a la Agenda 2030 de las Naciones Unidas al proporcionar
        recomendaciones personalizadas de cultivos y proyecciones de producción para los próximos años.
        Introduce tu municipio, variables agronómicas y climáticas y descubre qué cultivos podrían
        incrementar tu productividad.
      </p>
      <div className="mt-6">
        <Link
          href="/ingreso"
          className="inline-flex items-center px-5 py-3 bg-primary text-white rounded-md shadow hover:bg-primary-dark transition-colors"
        >
          Empezar recomendación
        </Link>
      </div>
    </section>
  )
}