# Agro Recomendador

Esta aplicación web utiliza datos históricos de producción y condiciones ambientales para sugerir cultivos con alto rendimiento esperado y proyectar la producción bajo distintos escenarios de manejo. Está construida con **Next.js 14**, **TypeScript**, **React**, **Tailwind CSS** y **Plotly**. Todo el procesamiento se realiza del lado del cliente: los artefactos de los modelos se exportan a JSON y se consumen directamente en el navegador.

## Estructura del proyecto

```
agro_app/
├── app/               # Páginas de la App Router
│   ├── layout.tsx     # Diseño raíz e importación de estilos globales
│   ├── page.tsx       # Portada con el objetivo 2030
│   ├── ingreso/page.tsx  # Formulario de ingreso de datos
│   ├── resultado/page.tsx # Visualización de resultados y recomendaciones
│   └── acerca/page.tsx    # Información sobre la metodología y glosario
├── components/        # Componentes reutilizables (formularios, tarjetas, gráficos)
├── lib/
│   ├── data/          # Carga de artefactos y utilidades de datos
│   └── inference/     # Implementaciones de preprocesamiento y modelos en TS
├── public/data/       # Artefactos JSON y archivos de datos utilizados en el frontend
├── tailwind.config.ts # Configuración de Tailwind CSS
├── postcss.config.js  # Configuración de PostCSS
├── next.config.js     # Configuración de Next.js
├── vercel.json        # Configuración mínima para desplegar en Vercel
├── package.json       # Dependencias y scripts
├── tsconfig.json      # Configuración de TypeScript
└── README.md          # Este documento
```

## Instalación y ejecución local

1. Asegúrate de tener instalado [Node.js](https://nodejs.org/). Se recomiendan versiones 18 o superiores.
2. Instala las dependencias del proyecto:

   ```bash
   npm install
   ```

3. Ejecuta el servidor de desarrollo:

   ```bash
   npm run dev
   ```

   La aplicación estará disponible en `http://localhost:3000`.

4. Para generar una versión de producción usa:

   ```bash
   npm run build
   npm start
   ```

## Artefactos de modelos

Los modelos entrenados originalmente en Python se convirtieron a formato JSON para permitir su ejecución en el navegador. Estos archivos se encuentran en `public/data/`:

- **numeric_preprocessor.json**: Medianas, medias y desviaciones estándar de las variables numéricas utilizadas para escalado y imputación.
- **categorical_preprocessor.json**: Categorías posibles de las variables categóricas y sus valores más frecuentes (para imputar).
- **linear_regression.json**: Coeficientes e intercepto del modelo de regresión lineal que predice la producción a partir de las variables transformadas.
- **cluster_info.json**: Mapeo de municipios a sus etiquetas de cluster, lista de cultivos por municipio y ranking de cultivos por cluster según producción histórica.
- **cluster_scaling_centroids.json**: Parámetros del escalado y centroides de los clusters (para posibles extensiones).
- **df_final_con_clusters.xlsx**, **df_final_limpio.xlsx**: Archivos originales de datos para auditoría.

El código en `lib/inference/` implementa las mismas transformaciones que en Python (imputación, escalado, one‑hot encoding y regresión lineal) usando TypeScript. No hay comunicación con un servidor backend; toda la lógica se ejecuta en el navegador.

## Pruebas y validación

Se incluyen pruebas ligeras en la carpeta `lib/inference/__tests__/` (no mostrada aquí) que verifican que el preprocesamiento produce el mismo número de características que el modelo de entrenamiento y que la predicción para un registro de ejemplo es coherente. Para ejecutar las pruebas puedes usar `npm test` si configuras un entorno de pruebas como Jest.

## Licencia

Este proyecto se entrega como referencia educativa. Puedes adaptarlo y modificarlo según tus necesidades.