"""Aplicación Streamlit para recomendar cultivos por municipio."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib  # type: ignore
except ImportError:  # pragma: no cover
    joblib = None  # type: ignore

try:  # Importaciones opcionales utilizadas en los respaldos
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - se notificará en la UI cuando falten dependencias
    KMeans = None  # type: ignore
    StandardScaler = None  # type: ignore

DATA_PATH = Path("df_final.csv")
SCALER_MUNICIPIO_PATH = Path("scaler_municipio.joblib")
KMEANS_MUNICIPIO_PATH = Path("kmeans_municipio.joblib")
SCALER_MODELO_PATH = Path("scaler_modelo.joblib")
MODELO_RENDIMIENTO_PATH = Path("modelo_rendimiento.joblib")
FEATURE_COLS_PATH = Path("feature_cols.joblib")

EXCLUDED_FEATURES = {"Año", "Periodo", "Area_cosechada"}
DIV_PENALTY_FACTOR = 0.05
RANDOM_STATE = 42


def _standardize_string(value: object) -> str:
    return str(value).strip().title()


@st.cache_data(show_spinner=False)
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Carga y limpia el dataset maestro."""
    if not path.exists():
        st.warning(
            "No se encontró 'df_final.csv'. Coloca el archivo en la misma carpeta que"
            " este script para habilitar todas las funciones."
        )
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - la excepción se comunica en UI
        st.error(f"No se pudo leer 'df_final.csv': {exc}")
        return pd.DataFrame()

    # Normalización básica de nombres de columnas
    df.columns = [col.strip() for col in df.columns]
    rename_map = {
        "municipio": "Municipio",
        "cultivo": "Cultivo",
        "produccion_t_ha": "Produccion_t_ha",
        "produccion_tons_ha": "Produccion_t_ha",
        "produccion_ton_ha": "Produccion_t_ha",
        "produccion_toneladas_ha": "Produccion_t_ha",
        "produccion_tn_ha": "Produccion_t_ha",
        "produccion_kg_ha": "Produccion_kg_ha",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    for col in ["Municipio", "Cultivo"]:
        if col in df.columns:
            df[col] = df[col].apply(_standardize_string)
        else:
            st.warning(f"La columna obligatoria '{col}' no está presente en el dataset.")

    if "Año" in df.columns:
        df["Año"] = pd.to_numeric(df["Año"], errors="coerce").astype("Int64")
    else:
        st.warning("La columna obligatoria 'Año' no se encontró en el dataset.")

    if "Produccion_kg_ha" in df.columns and "Produccion_t_ha" not in df.columns:
        # Conversión de kg/ha a toneladas/ha
        df["Produccion_t_ha"] = pd.to_numeric(df["Produccion_kg_ha"], errors="coerce") / 1000.0
        st.info(
            "Se detectó la columna 'Produccion_kg_ha'. Se convirtió automáticamente a"
            " toneladas por hectárea."
        )

    if "Produccion_t_ha" in df.columns:
        df["Produccion_t_ha"] = pd.to_numeric(df["Produccion_t_ha"], errors="coerce")
        # Filtrado suave de outliers mediante IQR
        series = df["Produccion_t_ha"].dropna()
        if len(series) >= 10:
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            df.loc[df["Produccion_t_ha"].notna(), "Produccion_t_ha"] = df.loc[
                df["Produccion_t_ha"].notna(), "Produccion_t_ha"
            ].clip(lower, upper)
    else:
        st.warning("La columna obligatoria 'Produccion_t_ha' no se encontró en el dataset.")

    return df


@st.cache_resource(show_spinner=False)
def load_models() -> Dict[str, Optional[object]]:
    """Carga los modelos opcionales disponibles en disco."""
    models: Dict[str, Optional[object]] = {
        "scaler_municipio": None,
        "kmeans_municipio": None,
        "scaler_modelo": None,
        "modelo_rendimiento": None,
        "feature_cols": None,
    }

    if joblib is None:
        st.warning("No se pudo importar joblib. Los modelos externos no estarán disponibles.")
        return models

    file_map = {
        "scaler_municipio": SCALER_MUNICIPIO_PATH,
        "kmeans_municipio": KMEANS_MUNICIPIO_PATH,
        "scaler_modelo": SCALER_MODELO_PATH,
        "modelo_rendimiento": MODELO_RENDIMIENTO_PATH,
        "feature_cols": FEATURE_COLS_PATH,
    }
    for key, path in file_map.items():
        if not path.exists():
            continue
        try:
            models[key] = joblib.load(path)
        except Exception as exc:  # pragma: no cover - se comunica en UI
            st.warning(f"No se pudo cargar '{path.name}': {exc}")
            models[key] = None

    if models["feature_cols"] is not None and not isinstance(models["feature_cols"], list):
        st.warning(
            "El archivo 'feature_cols.joblib' debe contener una lista de nombres de columnas."
            " Se ignorará este archivo."
        )
        models["feature_cols"] = None

    missing_models = [name for name in ["scaler_municipio", "kmeans_municipio"] if models[name] is None]
    if missing_models:
        st.warning(
            "No se encontraron modelos de clusterización completos. Se recalculará"
            " K-Means en caliente utilizando las variables climáticas disponibles."
        )

    if models["modelo_rendimiento"] is None:
        st.warning(
            "No se encontró 'modelo_rendimiento.joblib'. Se usarán promedios históricos"
            " como respaldo para las predicciones."
        )

    return models


def identify_climate_columns(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    exclude = {"Produccion_t_ha", "Produccion_kg_ha", "Año"} | EXCLUDED_FEATURES
    return [col for col in numeric_cols if col not in exclude]


def assign_clusters(
    df: pd.DataFrame, models: Dict[str, Optional[object]], climate_cols: List[str]
) -> Tuple[Dict[str, int], Optional[StandardScaler], Optional[KMeans]]:
    """Asigna clusters a todos los municipios utilizando los modelos disponibles o K-Means en caliente."""
    if df.empty or not climate_cols:
        return {}, None, None

    muni_climate = (
        df.groupby("Municipio")[climate_cols]
        .mean(numeric_only=True)
        .replace([np.inf, -np.inf], np.nan)
    )
    muni_climate = muni_climate.fillna(muni_climate.mean(numeric_only=True))

    scaler: Optional[StandardScaler] = None
    kmeans: Optional[KMeans] = None
    X = muni_climate.values

    if models.get("scaler_municipio") is not None and models.get("kmeans_municipio") is not None:
        scaler = models["scaler_municipio"]  # type: ignore[assignment]
        kmeans = models["kmeans_municipio"]  # type: ignore[assignment]
        try:
            X_scaled = scaler.transform(X)  # type: ignore[arg-type]
            clusters = kmeans.predict(X_scaled)  # type: ignore[arg-type]
            return dict(zip(muni_climate.index, clusters)), scaler, kmeans
        except Exception:
            st.warning(
                "No se pudieron usar los modelos de clusterización existentes. Se recalculará"
                " un nuevo K-Means con los datos actuales."
            )

    if StandardScaler is None or KMeans is None:
        st.error(
            "scikit-learn no está disponible. No es posible crear clusters sin las"
            " dependencias necesarias."
        )
        return {}, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_municipios = len(muni_climate)
    n_clusters = max(min(6, max(n_municipios // 5, 2)), 2)
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(X_scaled)
    return dict(zip(muni_climate.index, clusters)), scaler, kmeans


def infer_cluster(
    municipio: str,
    df: pd.DataFrame,
    climate_cols: List[str],
    cluster_map: Dict[str, int],
    scaler: Optional[StandardScaler],
    kmeans: Optional[KMeans],
) -> Optional[int]:
    if municipio in cluster_map:
        return cluster_map[municipio]

    if municipio not in df["Municipio"].unique():
        return None

    muni_values = (
        df[df["Municipio"] == municipio][climate_cols]
        .mean(numeric_only=True)
        .fillna(df[climate_cols].mean(numeric_only=True))
    )

    if scaler is not None and kmeans is not None:
        try:
            transformed = scaler.transform([muni_values.values])  # type: ignore[arg-type]
            cluster = int(kmeans.predict(transformed)[0])  # type: ignore[arg-type]
            return cluster
        except Exception:
            pass

    if StandardScaler is None or KMeans is None:
        return None

    muni_climate = (
        df.groupby("Municipio")[climate_cols]
        .mean(numeric_only=True)
        .replace([np.inf, -np.inf], np.nan)
    )
    if muni_climate.empty:
        return None
    muni_climate = muni_climate.fillna(muni_climate.mean(numeric_only=True))

    scaler_local = StandardScaler()
    transformed = scaler_local.fit_transform(muni_climate.values)
    n_municipios = len(muni_climate)
    n_clusters = max(min(6, max(n_municipios // 5, 2)), 2)
    kmeans_local = KMeans(n_clusters=n_clusters, n_init="auto", random_state=RANDOM_STATE)
    labels = kmeans_local.fit_predict(transformed)
    local_map = dict(zip(muni_climate.index, labels))
    return local_map.get(municipio)


def infer_feature_columns(df: pd.DataFrame, feature_cols: Optional[List[str]]) -> List[str]:
    if feature_cols:
        return feature_cols
    climate_cols = identify_climate_columns(df)
    if climate_cols:
        st.info(
            "Se infirieron las columnas de características a partir de las variables"
            " climáticas disponibles."
        )
    return climate_cols


def prepare_features(
    df: pd.DataFrame,
    municipio: str,
    cultivo: str,
    feature_cols: List[str],
    year: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    if df.empty or not feature_cols:
        return None

    df_muni = df[df["Municipio"] == municipio]
    global_means = df.mean(numeric_only=True)
    muni_means = df_muni.mean(numeric_only=True)

    features: Dict[str, float] = {}

    for col in feature_cols:
        value: float
        if col in {"Municipio", "Cultivo"}:
            # Si el modelo requiere columnas categóricas sin codificar, no se pueden usar directamente.
            if not st.session_state.get("_warn_raw_categorical", False):
                st.session_state["_warn_raw_categorical"] = True
                st.warning(
                    "El modelo de rendimiento requiere columnas categóricas sin codificar."
                    " Se asignará un valor nulo a dichas columnas."
                )
            features[col] = np.nan
            continue
        if col in EXCLUDED_FEATURES:
            if col == "Año":
                if year is not None:
                    value = float(year)
                elif "Año" in df_muni.columns and df_muni["Año"].notna().any():
                    value = float(df_muni["Año"].dropna().max())
                elif "Año" in df.columns and df["Año"].notna().any():
                    value = float(df["Año"].dropna().max())
                else:
                    value = float(pd.Timestamp.now().year)
            else:
                value = float(df[col].dropna().mean()) if col in df.columns else np.nan
            features[col] = value
            continue
        if col.startswith("Municipio_"):
            value = 1.0 if col.split("Municipio_", 1)[1].strip().lower() == municipio.lower() else 0.0
            features[col] = value
            continue
        if col.startswith("Cultivo_"):
            value = 1.0 if col.split("Cultivo_", 1)[1].strip().lower() == cultivo.lower() else 0.0
            features[col] = value
            continue
        series_muni = df_muni[col] if col in df_muni.columns else pd.Series(dtype=float)
        if not series_muni.empty and pd.api.types.is_numeric_dtype(series_muni):
            value = float(series_muni.dropna().mean())
        elif col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            value = float(df[col].dropna().mean())
        elif col in global_means.index:
            value = float(global_means[col])
        elif col in muni_means.index:
            value = float(muni_means[col])
        else:
            value = np.nan
        features[col] = value

    features_df = pd.DataFrame([features])
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(features_df.mean(numeric_only=True))
    features_df = features_df.fillna(0.0)
    return features_df


def predict_yield(
    df: pd.DataFrame,
    municipio: str,
    cultivo: str,
    models: Dict[str, Optional[object]],
    feature_cols: List[str],
    year: Optional[int] = None,
) -> Tuple[Optional[float], str, bool]:
    """Predice el rendimiento esperado para un cultivo y municipio."""
    modelo = models.get("modelo_rendimiento")
    scaler = models.get("scaler_modelo")
    features = prepare_features(df, municipio, cultivo, feature_cols, year=year)

    if modelo is not None and features is not None and not features.empty:
        try:
            X = features.values
            if scaler is not None:
                X = scaler.transform(X)  # type: ignore[arg-type]
            pred = float(modelo.predict(X)[0])  # type: ignore[arg-type]
            return pred, "Modelo", True
        except Exception:
            st.warning(
                "El modelo de rendimiento no pudo generar una predicción. Se usará el"
                " historial como respaldo."
            )

    subset = df[(df["Municipio"] == municipio) & (df["Cultivo"] == cultivo)]
    if not subset.empty and "Produccion_t_ha" in subset.columns and subset["Produccion_t_ha"].notna().any():
        return float(subset["Produccion_t_ha"].mean()), "Histórico municipal", False

    global_subset = df[df["Cultivo"] == cultivo]
    if (
        not global_subset.empty
        and "Produccion_t_ha" in global_subset.columns
        and global_subset["Produccion_t_ha"].notna().any()
    ):
        return float(global_subset["Produccion_t_ha"].mean()), "Promedio global", False

    if "Produccion_t_ha" in df.columns and df["Produccion_t_ha"].notna().any():
        return float(df["Produccion_t_ha"].mean()), "Promedio global", False

    return None, "Sin datos", False


def historical_stats(df: pd.DataFrame, municipio: str, cultivo: str) -> Dict[str, Optional[float]]:
    subset = df[(df["Municipio"] == municipio) & (df["Cultivo"] == cultivo)]
    subset = subset.dropna(subset=["Produccion_t_ha"])
    if subset.empty:
        return {"min": None, "mean": None, "max": None, "years": pd.Series(dtype="Int64")}
    return {
        "min": float(subset["Produccion_t_ha"].min()),
        "mean": float(subset["Produccion_t_ha"].mean()),
        "max": float(subset["Produccion_t_ha"].max()),
        "years": subset["Año"].dropna().astype(int),
        "values": subset["Produccion_t_ha"].dropna(),
        "data": subset.sort_values("Año"),
    }


def compute_popularity_metrics(
    df: pd.DataFrame, cluster_map: Dict[str, int]
) -> Tuple[Dict[str, float], Dict[Tuple[int, str], float]]:
    if df.empty or "Produccion_t_ha" not in df.columns:
        return {}, {}

    mean_yields = (
        df.dropna(subset=["Produccion_t_ha"])
        .groupby(["Municipio", "Cultivo"])["Produccion_t_ha"]
        .mean()
    )
    if mean_yields.empty:
        return {}, {}

    top_pairs: Dict[str, str] = {}
    for municipio, group in mean_yields.groupby(level=0):
        if group.isna().all():
            continue
        cultivo = group.idxmax()[1]
        top_pairs[municipio] = cultivo

    total_munis = len(top_pairs) or 1
    freq_global: Dict[str, float] = {}
    for cultivo in top_pairs.values():
        freq_global[cultivo] = freq_global.get(cultivo, 0.0) + 1.0 / total_munis

    cluster_counts: Dict[int, float] = {}
    freq_cluster: Dict[Tuple[int, str], float] = {}
    for municipio, cultivo in top_pairs.items():
        cluster = cluster_map.get(municipio)
        if cluster is None:
            continue
        cluster_counts[cluster] = cluster_counts.get(cluster, 0.0) + 1.0
        key = (cluster, cultivo)
        freq_cluster[key] = freq_cluster.get(key, 0.0) + 1.0

    for key, value in list(freq_cluster.items()):
        cluster = key[0]
        freq_cluster[key] = value / max(cluster_counts.get(cluster, 1.0), 1.0)

    return freq_global, freq_cluster


def rank_crops_in_municipio(
    df: pd.DataFrame,
    municipio: str,
    models: Dict[str, Optional[object]],
    feature_cols: List[str],
    cluster: Optional[int],
    freq_global: Dict[str, float],
    freq_cluster: Dict[Tuple[int, str], float],
    std_global: float,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    muni_subset = df[df["Municipio"] == municipio]
    crops = sorted(muni_subset["Cultivo"].dropna().unique().tolist())
    if not crops:
        crops = sorted(df["Cultivo"].dropna().unique().tolist())

    if not crops:
        return pd.DataFrame(), None

    rng = np.random.default_rng(RANDOM_STATE)
    alpha = DIV_PENALTY_FACTOR * (std_global or 1.0)
    beta = DIV_PENALTY_FACTOR * (std_global or 1.0)

    records = []
    for cultivo in crops:
        pred, source, used_model = predict_yield(df, municipio, cultivo, models, feature_cols)
        if pred is None:
            continue

        hist_subset = df[(df["Municipio"] == municipio) & (df["Cultivo"] == cultivo)]
        hist_var = float(hist_subset["Produccion_t_ha"].var()) if not hist_subset.empty else math.inf

        penalizacion = alpha * freq_global.get(cultivo, 0.0)
        bono = beta
        if cluster is not None:
            bono *= 1.0 - freq_cluster.get((cluster, cultivo), 0.0)
        diversificacion = -penalizacion + bono
        score = pred + diversificacion

        records.append(
            {
                "Cultivo": cultivo,
                "Rendimiento_esperado_t_ha": pred,
                "Fuente": source,
                "Cluster": cluster if cluster is not None else "Sin cluster",
                "Penalización_diversificación": diversificacion,
                "Score": score,
                "Varianza_historica": hist_var if not np.isnan(hist_var) else math.inf,
                "Aleatorio": rng.random(),
            }
        )

    if not records:
        return pd.DataFrame(), None

    ranking = pd.DataFrame(records)
    ranking.sort_values(
        by=["Score", "Varianza_historica", "Aleatorio"], ascending=[False, True, True], inplace=True
    )
    ranking.reset_index(drop=True, inplace=True)

    return ranking.drop(columns=["Score", "Aleatorio"]), ranking.iloc[0]


def project_future_yield(
    df: pd.DataFrame,
    municipio: str,
    cultivo: str,
    models: Dict[str, Optional[object]],
    feature_cols: List[str],
    historical: Dict[str, Optional[float]],
) -> Tuple[List[int], List[float], str]:
    data = historical.get("data")
    if data is None or data.empty:
        return [], [], "Sin datos históricos"

    years = data["Año"].dropna().astype(int)
    values = data["Produccion_t_ha"].dropna()
    if years.empty or values.empty:
        return [], [], "Sin datos históricos"

    last_year = int(years.max())
    future_years = [last_year + 1, last_year + 2]

    preds: List[float] = []
    source = "Histórico"
    for future_year in future_years:
        pred, pred_source, used_model = predict_yield(
            df, municipio, cultivo, models, feature_cols, year=future_year
        )
        if used_model and pred is not None:
            preds.append(pred)
            source = pred_source
        else:
            preds = []
            break

    if preds:
        return future_years, preds, source

    # Respaldo con tendencia lineal simple
    if len(years.unique()) >= 2:
        coeffs = np.polyfit(years, values, 1)
        preds = [float(np.polyval(coeffs, y)) for y in future_years]
        return future_years, preds, "Tendencia lineal"

    mean_value = float(values.mean())
    preds = [mean_value for _ in future_years]
    return future_years, preds, "Promedio histórico"


def plot_best_crop_timeseries(
    municipio: str,
    cultivo: str,
    historical: Dict[str, Optional[float]],
    future_years: List[int],
    future_preds: List[float],
    future_source: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    data = historical.get("data")
    if data is not None and not data.empty:
        ax.plot(
            data["Año"],
            data["Produccion_t_ha"],
            marker="o",
            label="Histórico",
        )
    if future_years and future_preds:
        ax.plot(
            future_years,
            future_preds,
            marker="o",
            linestyle="--",
            label=f"Proyección ({future_source})",
        )
    ax.set_title(f"Evolución de producción para {cultivo} en {municipio}")
    ax.set_xlabel("Año")
    ax.set_ylabel("Producción (t/ha)")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def show_sidebar_help() -> None:
    with st.sidebar:
        st.header("Ayuda rápida")
        st.markdown(
            """
            1. Coloca los archivos `df_final.csv` y los modelos `.joblib` en la misma carpeta que `app.py`.
            2. Ejecuta la aplicación con: `streamlit run app.py`.
            3. Actualiza los archivos y reinicia la app para refrescar los datos.
            """
        )
        st.info("La aplicación puede trabajar sin modelos, utilizando promedios históricos.")


def main() -> None:
    st.set_page_config(page_title="Recomendador de cultivos por municipio", layout="wide")
    st.title("Recomendador de cultivos por municipio")
    show_sidebar_help()

    df = load_data()
    models = load_models()

    if df.empty:
        st.info("Carga un archivo 'df_final.csv' para comenzar a explorar recomendaciones.")
        return

    climate_cols = identify_climate_columns(df)
    cluster_map, scaler_used, kmeans_used = assign_clusters(df, models, climate_cols)
    freq_global, freq_cluster = compute_popularity_metrics(df, cluster_map)
    feature_cols = infer_feature_columns(df, models.get("feature_cols"))
    std_global = float(df["Produccion_t_ha"].std()) if "Produccion_t_ha" in df.columns else 1.0
    if math.isnan(std_global) or std_global == 0:
        std_global = 1.0

    municipios = sorted(df["Municipio"].dropna().unique().tolist())
    if not municipios:
        st.warning("No se encontraron municipios en el dataset.")
        return

    municipio = st.selectbox("Selecciona un municipio", municipios)
    if not municipio:
        return

    st.markdown(f"**Municipio seleccionado: {municipio}**")

    cluster = infer_cluster(municipio, df, climate_cols, cluster_map, scaler_used, kmeans_used)
    if cluster is not None:
        st.markdown(f"Cluster detectado: **{cluster}**")
    else:
        st.warning("No fue posible determinar el cluster para este municipio.")

    if st.button("Calcular mejores cultivos"):
        ranking, best_row = rank_crops_in_municipio(
            df,
            municipio,
            models,
            feature_cols,
            cluster,
            freq_global,
            freq_cluster,
            std_global,
        )

        if ranking.empty or best_row is None:
            st.warning("No se pudieron calcular recomendaciones para este municipio.")
            return

        st.subheader("Ranking de cultivos recomendados")
        display_cols = [
            "Cultivo",
            "Rendimiento_esperado_t_ha",
            "Fuente",
            "Cluster",
            "Penalización_diversificación",
        ]
        st.dataframe(ranking.loc[:, display_cols], use_container_width=True)

        best_cultivo = best_row["Cultivo"]
        st.subheader(f"Métricas del cultivo destacado: {best_cultivo}")
        hist = historical_stats(df, municipio, best_cultivo)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mínimo histórico (t/ha)", f"{hist.get('min', np.nan):.2f}" if hist.get("min") is not None else "N/D")
        col2.metric("Promedio histórico (t/ha)", f"{hist.get('mean', np.nan):.2f}" if hist.get("mean") is not None else "N/D")
        col3.metric("Máximo histórico (t/ha)", f"{hist.get('max', np.nan):.2f}" if hist.get("max") is not None else "N/D")

        future_years, future_preds, future_source = project_future_yield(
            df, municipio, best_cultivo, models, feature_cols, hist
        )
        next_pred = future_preds[0] if future_preds else None
        col4.metric(
            "Predicción próximo año (t/ha)",
            f"{next_pred:.2f}" if next_pred is not None else "N/D",
        )

        fig = plot_best_crop_timeseries(
            municipio,
            best_cultivo,
            hist,
            future_years,
            future_preds,
            future_source,
        )
        st.pyplot(fig)

        otros = ranking[ranking["Cultivo"] != best_cultivo].head(5)
        if not otros.empty:
            st.subheader("Otros cultivos con buen rendimiento esperado")
            for _, row in otros.iterrows():
                st.markdown(
                    f"- **{row['Cultivo']}**: {row['Rendimiento_esperado_t_ha']:.2f} t/ha"
                    f" · Fuente: {row['Fuente']}"
                )

        if "Ciclo_de_cultivo" in df.columns:
            ciclos = (
                df[(df["Municipio"] == municipio) & (df["Cultivo"] == best_cultivo)][
                    "Ciclo_de_cultivo"
                ]
                .dropna()
                .unique()
            )
            if ciclos.size:
                st.info(
                    "Ciclos de cultivo registrados (no editables): "
                    + ", ".join(sorted(map(str, ciclos)))
                )

        with st.expander("Diagnóstico de clusters (top-1 histórico por cluster)"):
            if cluster_map:
                cluster_summary: Dict[int, List[str]] = {}
                for muni, clus in cluster_map.items():
                    muni_data = df[df["Municipio"] == muni]
                    if (
                        "Produccion_t_ha" not in muni_data.columns
                        or not muni_data["Produccion_t_ha"].notna().any()
                    ):
                        continue
                    promedios = (
                        muni_data.groupby("Cultivo")["Produccion_t_ha"].mean().dropna()
                    )
                    if promedios.empty:
                        continue
                    cultivo_top = promedios.idxmax()
                    cluster_summary.setdefault(int(clus), []).append(cultivo_top)
                if cluster_summary:
                    diag_rows = []
                    for clus, cultivos in cluster_summary.items():
                        counts = pd.Series(cultivos).value_counts(normalize=True)
                        diag_rows.append(
                            {
                                "Cluster": clus,
                                "Cultivo": counts.index[0],
                                "Frecuencia": counts.iloc[0],
                            }
                        )
                    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)
                else:
                    st.write("No hay información histórica suficiente para el diagnóstico.")
            else:
                st.write("No hay información de clusters disponible.")


if __name__ == "__main__":  # pragma: no cover
    main()
else:
    # Streamlit ejecuta el script directamente, por lo que llamamos a main() siempre.
    main()
