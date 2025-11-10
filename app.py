import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib  # type: ignore
except ImportError:  # pragma: no cover
    joblib = None
import pickle


def load_data(path: Path = Path("df_final.csv")) -> pd.DataFrame:
    """Load the historical dataset if available and clean key columns."""
    if not path.exists():
        st.warning(
            "No se encontró 'df_final.csv'. La aplicación funcionará en modo demostrativo"
            " sin datos hasta que cargues el archivo requerido."
        )
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - feedback to UI
        st.error(f"No se pudo leer 'df_final.csv': {exc}")
        return pd.DataFrame()

    for col in ["Municipio", "Cultivo"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
        else:
            st.warning(f"La columna obligatoria '{col}' no se encontró en la base de datos.")
    if "Año" in df.columns:
        df["Año"] = pd.to_numeric(df["Año"], errors="coerce").astype("Int64")
    else:
        st.warning("La columna obligatoria 'Año' no se encontró en la base de datos.")
    if "Produccion_T_Ha" in df.columns:
        df.rename(columns={"Produccion_T_Ha": "Produccion_t_ha"}, inplace=True)
    if "Produccion_t_ha" in df.columns:
        df["Produccion_t_ha"] = pd.to_numeric(df["Produccion_t_ha"], errors="coerce")
    else:
        st.warning(
            "La columna obligatoria 'Produccion_t_ha' no se encontró en la base de datos."
        )

    return df


def load_model_and_features(
    model_path: Path = Path("model.pkl"), features_path: Path = Path("feature_cols.json")
) -> Tuple[Optional[object], Optional[List[str]]]:
    """Load the optional trained model and its feature list."""
    if not model_path.exists() or not features_path.exists():
        return None, None

    model: Optional[object] = None
    feature_cols: Optional[List[str]] = None

    try:
        if joblib is not None:
            model = joblib.load(model_path)
        else:  # pragma: no cover - fallback when joblib is absent
            with model_path.open("rb") as file:
                model = pickle.load(file)
    except Exception as exc:
        st.warning(f"No se pudo cargar 'model.pkl': {exc}")
        model = None

    try:
        with features_path.open("r", encoding="utf-8") as file:
            feature_cols = json.load(file)
            if not isinstance(feature_cols, list):
                raise ValueError("El archivo feature_cols.json debe contener una lista.")
    except Exception as exc:
        st.warning(f"No se pudo cargar 'feature_cols.json': {exc}")
        feature_cols = None

    if model is None or feature_cols is None:
        return None, None
    return model, feature_cols


def historical_yield(df: pd.DataFrame, municipio: str, cultivo: str) -> pd.DataFrame:
    """Return the subset of historical data for a given municipality and crop."""
    if df.empty:
        return pd.DataFrame()
    required_cols = {"Municipio", "Cultivo", "Año", "Produccion_t_ha"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()
    subset = df[(df["Municipio"] == municipio) & (df["Cultivo"] == cultivo)].copy()
    subset = subset.dropna(subset=["Año", "Produccion_t_ha"]).sort_values("Año")
    return subset


def _sanitize_label(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
        .replace(" ", "_")
    )


def prepare_features(
    df: pd.DataFrame,
    municipio: str,
    cultivo: str,
    feature_cols: Optional[List[str]],
    year: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Build the feature vector for model inference using municipal averages."""
    if feature_cols is None or df.empty:
        return None

    available_cols = set(df.columns)
    df_mun = df[df["Municipio"] == municipio] if "Municipio" in df.columns else pd.DataFrame()
    features: Dict[str, float] = {}
    global_means = df.mean(numeric_only=True)
    muni_means = df_mun.mean(numeric_only=True) if not df_mun.empty else pd.Series(dtype=float)
    used_proxies = False

    for col in feature_cols:
        value: Optional[float] = None
        if col == "Municipio":
            value = municipio
        elif col == "Cultivo":
            value = cultivo
        elif col == "Año":
            if year is not None:
                value = year
            else:
                if "Año" in df_mun.columns and not df_mun["Año"].dropna().empty:
                    value = float(df_mun["Año"].dropna().iloc[-1])
                elif "Año" in df.columns and not df["Año"].dropna().empty:
                    value = float(df["Año"].dropna().max())
                else:
                    value = datetime.datetime.now().year
        elif col.startswith("Municipio_"):
            value = 1.0 if _sanitize_label(municipio) == _sanitize_label(col.split("Municipio_", 1)[1]) else 0.0
        elif col.startswith("Cultivo_"):
            value = 1.0 if _sanitize_label(cultivo) == _sanitize_label(col.split("Cultivo_", 1)[1]) else 0.0
        elif col in available_cols:
            serie_mun = df_mun[col] if not df_mun.empty and col in df_mun.columns else pd.Series(dtype=float)
            if pd.api.types.is_numeric_dtype(df[col]):
                if not serie_mun.empty:
                    value = float(serie_mun.dropna().mean())
                if value is None or np.isnan(value):
                    value = float(df[col].dropna().mean()) if not df[col].dropna().empty else None
            else:
                if not serie_mun.empty:
                    mode = serie_mun.dropna().mode()
                    if not mode.empty:
                        value = mode.iloc[0]
                if value is None:
                    mode = df[col].dropna().mode()
                    if not mode.empty:
                        value = mode.iloc[0]
        else:
            st.warning(f"La característica '{col}' no está presente en los datos históricos.")

        if value is None:
            if col in muni_means.index and not np.isnan(muni_means[col]):
                value = float(muni_means[col])
                used_proxies = True
            elif col in global_means.index and not np.isnan(global_means[col]):
                value = float(global_means[col])
                used_proxies = True
            else:
                value = 0.0
                used_proxies = True
        features[col] = value

    feature_df = pd.DataFrame([features])
    feature_df.attrs["used_proxies"] = used_proxies
    return feature_df


def _historical_projection(hist_df: pd.DataFrame, target_year: int) -> Tuple[Optional[float], bool]:
    if hist_df.empty:
        return None, False
    years = hist_df["Año"].to_numpy(dtype=float)
    values = hist_df["Produccion_t_ha"].to_numpy(dtype=float)
    if len(np.unique(years)) > 1:
        try:
            slope, intercept = np.polyfit(years, values, 1)
            prediction = float(slope * target_year + intercept)
            return prediction, True
        except Exception:  # pragma: no cover - numerical edge cases
            pass
    return float(np.nanmean(values)), False


def predict_yield(
    model: Optional[object],
    feature_cols: Optional[List[str]],
    municipio: str,
    cultivo: str,
    df: pd.DataFrame,
    year: Optional[int] = None,
) -> Tuple[Optional[float], bool]:
    """Predict the expected yield using the trained model if available."""
    if model is None or feature_cols is None:
        return None, False

    features_df = prepare_features(df, municipio, cultivo, feature_cols, year=year)
    if features_df is None:
        return None, False

    used_proxies = bool(features_df.attrs.get("used_proxies"))
    try:
        prediction = model.predict(features_df)[0]
        return float(prediction), used_proxies
    except Exception as exc:
        st.warning(f"No se pudo generar una predicción para {cultivo}: {exc}")
        return None, used_proxies


def main() -> None:
    st.set_page_config(page_title="Recomendador de cultivos por municipio", layout="wide")
    st.sidebar.markdown(
        "**Instrucciones**\n\n"
        "Coloca `df_final.csv` (y opcionalmente `model.pkl` junto con `feature_cols.json`)\n"
        "en la misma carpeta que este archivo.\n\n"
        "Ejecuta: `streamlit run app.py`."
    )

    st.title("Recomendador de cultivos por municipio")

    df = load_data()
    model, feature_cols = load_model_and_features()

    if model is None or feature_cols is None:
        st.info(
            "No se encontró un modelo entrenado. Se utilizará el respaldo histórico para"
            " estimar los rendimientos."
        )

    if df.empty or not {"Municipio", "Cultivo"}.issubset(df.columns):
        st.stop()

    municipios = sorted(df["Municipio"].dropna().unique())
    if not municipios:
        st.info("No hay municipios disponibles para mostrar.")
        st.stop()

    municipio = st.selectbox("Municipio", municipios, index=0)

    df_municipio = df[df["Municipio"] == municipio]
    cultivos_municipio = sorted(df_municipio["Cultivo"].dropna().unique())
    if not cultivos_municipio:
        st.info("No hay cultivos registrados para este municipio.")
        st.stop()
    cultivo_principal = st.selectbox("Cultivo principal", cultivos_municipio, index=0)

    candidatos = [c for c in cultivos_municipio if c != cultivo_principal]
    otros_cultivos = st.multiselect(
        "Otros cultivos a evaluar",
        cultivos_municipio,
        default=candidatos,
    )

    if "auto_run_done" not in st.session_state:
        st.session_state["auto_run_done"] = False

    ejecutar = st.button("Evaluar")
    if not st.session_state["auto_run_done"]:
        ejecutar = True
        st.session_state["auto_run_done"] = True

    if not ejecutar:
        st.stop()

    current_year = datetime.datetime.now().year

    st.subheader("A. Evolución del cultivo principal")
    hist_df = historical_yield(df, municipio, cultivo_principal)
    if hist_df.empty:
        st.info(
            "No se encontraron registros históricos suficientes para el cultivo seleccionado."
        )
    else:
        min_prod = float(hist_df["Produccion_t_ha"].min())
        max_prod = float(hist_df["Produccion_t_ha"].max())
        mean_prod = float(hist_df["Produccion_t_ha"].mean())

        pred_model, used_proxy_model = predict_yield(
            model, feature_cols, municipio, cultivo_principal, df, year=current_year
        )
        pred_hist, used_regression = _historical_projection(hist_df, current_year)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mínimo histórico (t/ha)", f"{min_prod:.2f}")
        col2.metric("Máximo histórico (t/ha)", f"{max_prod:.2f}")
        col3.metric("Promedio histórico (t/ha)", f"{mean_prod:.2f}")
        if pred_model is not None:
            comentario = "con promedios municipales" if used_proxy_model else ""
            col4.metric(
                "Predicción próxima campaña (t/ha)",
                f"{pred_model:.2f}",
                comentario,
            )
        elif pred_hist is not None:
            comentario = "Regresión histórica" if used_regression else "Promedio histórico"
            col4.metric("Proyección próxima campaña (t/ha)", f"{pred_hist:.2f}", comentario)
        else:
            col4.metric("Proyección próxima campaña (t/ha)", "Sin datos")

        fig_hist, ax_hist = plt.subplots()
        ax_hist.plot(hist_df["Año"], hist_df["Produccion_t_ha"], marker="o", label="Producción histórica")

        future_years = []
        if model is not None and feature_cols is not None:
            last_year = int(hist_df["Año"].max()) if not hist_df["Año"].empty else current_year
            future_years = list(range(last_year + 1, current_year + 3))
            predicted_values = []
            for year in future_years:
                pred_value, _ = predict_yield(model, feature_cols, municipio, cultivo_principal, df, year=year)
                if pred_value is not None:
                    predicted_values.append((year, pred_value))
            if predicted_values:
                years_pred, values_pred = zip(*predicted_values)
                ax_hist.plot(years_pred, values_pred, marker="^", label="Predicción")

        ax_hist.set_title(f"Producción de {cultivo_principal} en {municipio}")
        ax_hist.set_xlabel("Año")
        ax_hist.set_ylabel("Producción (t/ha)")
        ax_hist.legend()
        st.pyplot(fig_hist)

        if future_years and model is not None:
            st.caption(
                "Las predicciones futuras utilizan el modelo cargado y, cuando no hay"
                " datos climáticos específicos por año, se emplean los promedios"
                " municipales como aproximación."
            )

    st.subheader("B. Sugerencia de otros cultivos")
    resultados: List[Dict[str, object]] = []
    global_mean_all = (
        float(df["Produccion_t_ha"].mean())
        if "Produccion_t_ha" in df.columns and not df["Produccion_t_ha"].dropna().empty
        else np.nan
    )

    if not otros_cultivos:
        st.info("Selecciona al menos un cultivo adicional para evaluar.")
    else:
        for cultivo in otros_cultivos:
            metodo = ""
            comentario = ""
            rendimiento = None

            pred, used_proxy = predict_yield(model, feature_cols, municipio, cultivo, df, year=current_year)
            if pred is not None:
                rendimiento = pred
                metodo = "Modelo"
                if used_proxy:
                    comentario = "Se usaron promedios municipales para completar variables."
            else:
                hist_df_cultivo = historical_yield(df, municipio, cultivo)
                rendimiento_hist, used_regression = _historical_projection(hist_df_cultivo, current_year)
                if rendimiento_hist is not None and not np.isnan(rendimiento_hist):
                    rendimiento = rendimiento_hist
                    metodo = "Histórico municipal"
                    if not used_regression:
                        comentario = "Promedio histórico municipal."
                    else:
                        comentario = "Regresión lineal histórica."
                else:
                    global_df = df[df["Cultivo"] == cultivo]
                    if not global_df.empty:
                        rendimiento = float(global_df["Produccion_t_ha"].mean())
                        metodo = "Promedio global"
                        comentario = "Sin histórico municipal, se usa promedio global."
                    else:
                        rendimiento = global_mean_all
                        metodo = "Promedio global"
                        comentario = (
                            "Sin datos del cultivo, se usa el promedio global de la base."
                        )

            if rendimiento is not None and not np.isnan(rendimiento):
                resultados.append(
                    {
                        "Cultivo": cultivo,
                        "Rendimiento_esperado_t_ha": float(rendimiento),
                        "Metodo_estimacion": metodo,
                        "Comentario": comentario,
                    }
                )

        if resultados:
            resultados_df = pd.DataFrame(resultados)
            resultados_df = resultados_df.sort_values(
                "Rendimiento_esperado_t_ha", ascending=False
            ).reset_index(drop=True)
            st.dataframe(resultados_df)

            fig_bar, ax_bar = plt.subplots()
            ax_bar.bar(resultados_df["Cultivo"], resultados_df["Rendimiento_esperado_t_ha"])
            ax_bar.set_ylabel("Rendimiento esperado (t/ha)")
            ax_bar.set_title(f"Rendimientos estimados en {municipio}")
            st.pyplot(fig_bar)
        else:
            st.info("No fue posible calcular rendimientos esperados para los cultivos seleccionados.")


if __name__ == "__main__":  # pragma: no cover
    main()
