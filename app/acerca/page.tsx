export default function AcercaPage() {
  return (
    <section className="space-y-6">
      <h1 className="text-2xl font-semibold">Acerca de esta herramienta</h1>
      <p className="max-w-prose text-gray-700 dark:text-gray-300">
        Esta aplicación utiliza técnicas de aprendizaje automático para analizar datos históricos de producción
        agrícola y condiciones de clima y suelo en Colombia. A partir de tu municipio y variables agronómicas,
        se determina a qué grupo (cluster) perteneces y se calcula una proyección de producción para los próximos
        años utilizando un modelo de regresión lineal entrenado previamente. No utilizamos ninguna información
        personal ni enviamos tus datos a servidores: todo ocurre en tu navegador.
      </p>
      <h2 className="text-xl font-semibold">¿Qué es un cluster?</h2>
      <p className="max-w-prose">
        Un <em>cluster</em> agrupa municipios con características similares en cuanto a sus variables de suelo,
        clima y rendimiento. El algoritmo jerárquico utilizado calcula las distancias entre municipios en un espacio
        de características normalizado y los agrupa de forma que la variabilidad interna sea mínima. Así, las
        recomendaciones de cultivos se basan en lo que ha funcionado bien en municipios con condiciones
        parecidas a las tuyas.
      </p>
      <h2 className="text-xl font-semibold">¿Cómo leer la gráfica?</h2>
      <p className="max-w-prose">
        La gráfica muestra la producción estimada para los años futuros que seleccionaste. El eje horizontal
        representa los años (a partir del año base que ingresaste) y el eje vertical la producción (en toneladas).
        Puedes ajustar el incremento anual del área sembrada para ver cómo influyen distintas trayectorias en la
        producción predicha.
      </p>
      <h2 className="text-xl font-semibold">Glosario</h2>
      <ul className="list-disc list-inside space-y-1">
        <li>
          <strong>MAPE</strong>: Error absoluto porcentual medio, una medida de la precisión de un modelo de
          predicción; mientras más bajo, mejor.
        </li>
        <li>
          <strong>Cluster</strong>: Conjunto de municipios con comportamiento agronómico y climático similar.
        </li>
        <li>
          <strong>Escenario</strong>: Conjunto de supuestos sobre la evolución de tus variables (por ejemplo,
          cómo cambia el área sembrada cada año) para proyectar la producción.
        </li>
      </ul>
    </section>
  )
}