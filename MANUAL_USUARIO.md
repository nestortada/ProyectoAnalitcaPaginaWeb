# Manual de Usuario - Agro Recomendador

Bienvenido al manual de usuario de **Agro Recomendador**. Este documento le guiará a través de la instalación, ejecución y uso de la aplicación web diseñada para ayudar en la toma de decisiones agrícolas mediante análisis de datos y modelos predictivos.

## 1. ¿Qué es y para qué sirve?

**Agro Recomendador** es una herramienta web interactiva que utiliza datos históricos y modelos de aprendizaje automático para:
*   **Sugerir cultivos** con alto potencial de rendimiento para un municipio específico.
*   **Proyectar la producción** estimada de cultivos bajo diferentes escenarios climáticos y de manejo.
*   **Analizar condiciones** como temperatura, precipitación y características del suelo (pH) para optimizar la siembra.

El objetivo principal es apoyar a agricultores y planificadores en la selección de cultivos más rentables y sostenibles, alineándose con objetivos de seguridad alimentaria y producción responsable.

---

## 2. Requisitos del Sistema

Para ejecutar esta aplicación en su máquina local, necesitará lo siguiente:

### Software
*   **Node.js**: Entorno de ejecución para JavaScript. Se recomienda la versión **18 LTS** o superior.
    *   [Descargar Node.js aquí](https://nodejs.org/)
*   **Navegador Web**: Google Chrome, Mozilla Firefox, Microsoft Edge o Safari (versiones recientes).
*   **Git** (Opcional): Para clonar el repositorio si no descargó el archivo ZIP.

### Hardware
*   Cualquier computadora moderna (Windows, macOS o Linux) con acceso a internet para la instalación inicial de dependencias.
*   Mínimo 4GB de RAM recomendado.

---

## 3. Instalación en su Máquina

Siga estos pasos para instalar la aplicación en su computadora:

1.  **Obtener el código fuente**:
    *   Si tiene el archivo comprimido, descomprímalo en una carpeta de su elección.
    *   Si usa Git, clone el repositorio:
        ```bash
        git clone <URL_DEL_REPOSITORIO>
        ```

2.  **Abrir la terminal**:
    *   En Windows: Puede usar PowerShell o el Símbolo del sistema (CMD).
    *   En Mac/Linux: Use la Terminal.

3.  **Navegar a la carpeta del proyecto**:
    Use el comando `cd` para entrar a la carpeta donde están los archivos. Ejemplo:
    ```bash
    cd ruta/a/la/carpeta/agro_app
    ```

4.  **Instalar dependencias**:
    Ejecute el siguiente comando para descargar las librerías necesarias (esto puede tardar unos minutos):
    ```bash
    npm install
    ```

---

## 4. Cómo correr la página web

Una vez instalada, tiene dos opciones para ejecutar la aplicación:

### Opción A: Modo Desarrollo (Recomendado para pruebas)
Este modo le permite ver cambios en tiempo real y es ideal para probar la aplicación rápidamente.

1.  En la terminal, dentro de la carpeta del proyecto, ejecute:
    ```bash
    npm run dev
    ```
2.  Verá un mensaje indicando que el servidor está listo, generalmente en `http://localhost:3000`.
3.  Abra su navegador web y visite esa dirección.

### Opción B: Modo Producción (Para uso estable)
Este modo optimiza la aplicación para que funcione más rápido.

1.  Construya la aplicación:
    ```bash
    npm run build
    ```
2.  Inicie el servidor:
    ```bash
    npm start
    ```
3.  Abra su navegador en `http://localhost:3000`.

---

## 5. Guía de Uso

Una vez que la aplicación esté corriendo en su navegador, siga este flujo de trabajo:

### Paso 1: Pantalla de Inicio
Encontrará una introducción al proyecto y su alineación con los Objetivos de Desarrollo Sostenible (ODS). Haga clic en el botón para **Ingresar Datos** o **Comenzar**.

### Paso 2: Formulario de Ingreso
En esta sección deberá proporcionar la información base para el análisis:
*   **Municipio**: Seleccione el municipio de interés (la lista se carga automáticamente).
*   **Cultivo (Opcional)**: Puede seleccionar un cultivo específico si desea ver su proyección, o dejarlo en blanco para ver recomendaciones generales.
*   **Año**: El año para el cual desea la proyección.
*   **Semestre**: (Si aplica) Temporada de siembra.
*   **Datos Climáticos y de Suelo**: El sistema intentará cargar valores promedio históricos para el municipio. Puede ajustarlos si tiene datos más precisos de su parcela (Temperatura, Precipitación, pH, etc.).

Al finalizar, haga clic en el botón **"Analizar / Calcular"**.

### Paso 3: Resultados y Recomendaciones
La aplicación procesará los datos (todo ocurre en su navegador, no se envían datos a servidores externos) y le mostrará:

1.  **Gráfica de Proyección**: Una visualización de la producción estimada a lo largo del tiempo para el cultivo seleccionado.
2.  **Recomendaciones**: Una lista de otros cultivos que históricamente han tenido buen rendimiento en ese municipio o en municipios con condiciones similares (Clusters).
3.  **Detalles del Cultivo**: Información clave como el ciclo de vida del cultivo (días), y comparativas de las condiciones de su municipio vs. los requerimientos ideales.

### Paso 4: Nueva Consulta
Si desea realizar otro análisis, busque el botón o enlace **"Volver"** o **"Ingresar nuevos datos"** al final de la página de resultados.

---

## 6. Solución de Problemas Comunes

*   **Error "command not found: npm"**: Asegúrese de haber instalado Node.js correctamente. Cierre y vuelva a abrir su terminal.
*   **La página no carga en localhost:3000**: Verifique que no haya otro programa usando el puerto 3000. Si es así, Next.js usualmente intentará usar el 3001 (mire la salida en la terminal).
*   **Los gráficos no aparecen**: Asegúrese de tener una conexión a internet estable la primera vez para cargar las librerías de gráficos, o verifique que no tenga bloqueadores de scripts agresivos en su navegador.
