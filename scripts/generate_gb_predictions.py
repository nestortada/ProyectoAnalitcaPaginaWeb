from pathlib import Path
import math
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path('.')
DATA_DIR = ROOT / 'public' / 'data'
MODEL_PATH = DATA_DIR / 'reg_Producción_GradientBoostingRegressor.joblib'
DF_PATH = DATA_DIR / 'df_final_limpio.xlsx'
OUT_PATH = DATA_DIR / 'gb_predictions.json'

RANDOM_STATE = 42


def _standardize_string(value):
    return str(value).strip().title()


def load_df():
    if not DF_PATH.exists():
        raise FileNotFoundError(DF_PATH)
    df = pd.read_excel(DF_PATH)
    df.columns = [c.strip() for c in df.columns]
    for col in ['Municipio', 'Cultivo']:
        if col in df.columns:
            df[col] = df[col].apply(_standardize_string)
    if 'Año' in df.columns:
        df['Año'] = pd.to_numeric(df['Año'], errors='coerce').astype('Int64')
    return df


def prepare_features_for_model(df, municipio, cultivo, feature_names, year=None):
    df_muni = df[df['Municipio'] == municipio]
    global_means = df.mean(numeric_only=True)
    muni_means = df_muni.mean(numeric_only=True)

    features = {}
    for col in feature_names:
        if col in {'Municipio', 'Cultivo'}:
            features[col] = np.nan
            continue
        if col == 'Año':
            if year is not None:
                value = float(year)
            elif 'Año' in df_muni.columns and df_muni['Año'].notna().any():
                value = float(df_muni['Año'].dropna().max())
            elif 'Año' in df.columns and df['Año'].notna().any():
                value = float(df['Año'].dropna().max())
            else:
                value = float(pd.Timestamp.now().year)
            features[col] = value
            continue
        if col == 'Área sembrada' or col == 'Área cosechada' or col == 'Ciclo del cultivo':
            # try to get municipal default if present
            if col in df_muni.columns and df_muni[col].notna().any():
                features[col] = float(df_muni[col].dropna().mean())
            elif col in df.columns and df[col].notna().any():
                features[col] = float(df[col].dropna().mean())
            elif col in global_means.index:
                features[col] = float(global_means[col])
            else:
                features[col] = 0.0
            continue
        # For other numeric columns
        if col in df_muni.columns and pd.api.types.is_numeric_dtype(df_muni[col]):
            val = df_muni[col].dropna().mean()
            features[col] = float(val) if not math.isnan(val) else 0.0
        elif col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            val = df[col].dropna().mean()
            features[col] = float(val) if not math.isnan(val) else 0.0
        elif col in global_means.index:
            features[col] = float(global_means[col])
        else:
            # fallback zero
            features[col] = 0.0
    # some pipelines expect exact column order and dtypes
    features_df = pd.DataFrame([features])
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(features_df.mean(numeric_only=True))
    features_df = features_df.fillna(0.0)
    return features_df


def main():
    print('Loading df...')
    df = load_df()
    print('Loading model...')
    model = joblib.load(MODEL_PATH)
    # model is a pipeline; find the final estimator and feature names
    feature_names = None
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
    else:
        # try to get from steps
        try:
            for step in model.steps:
                if hasattr(step[1], 'feature_names_in_'):
                    feature_names = list(step[1].feature_names_in_)
                    break
        except Exception:
            pass
    if feature_names is None:
        raise RuntimeError('No se pudieron determinar feature names del modelo')
    print('Feature names:', feature_names)

    municipios = sorted(df['Municipio'].dropna().unique().tolist())
    crops = sorted(df['Cultivo'].dropna().unique().tolist())

    # for each municipio, pick top crop by historical mean
    gb_preds = {}
    for municipio in municipios:
        muni_df = df[df['Municipio'] == municipio]
        if muni_df.empty:
            continue
        # use 'Rendimiento' or 'Producción' as the yield column if present
        if 'Rendimiento' in muni_df.columns:
            yield_col = 'Rendimiento'
        elif 'Producción' in muni_df.columns:
            yield_col = 'Producción'
        else:
            # fallback: try to find a numeric column that looks like production
            candidates = [c for c in muni_df.columns if 'produ' in c.lower() or 'rendim' in c.lower()]
            yield_col = candidates[0] if candidates else None
        if yield_col is None:
            continue
        mean_yields = (
            muni_df.dropna(subset=[yield_col])
            .groupby('Cultivo')[yield_col]
            .mean()
        )
        if mean_yields.empty:
            continue
        top_cultivo = mean_yields.idxmax()
        gb_preds.setdefault(municipio, {})
        gb_preds[municipio].setdefault(top_cultivo, {})
        # generate predictions for next 3 years
        last_year = int(muni_df['Año'].dropna().max()) if 'Año' in muni_df.columns and muni_df['Año'].notna().any() else pd.Timestamp.now().year
        for y in range(last_year + 1, last_year + 4):
            feat_df = prepare_features_for_model(df, municipio, top_cultivo, feature_names, year=y)
            try:
                pred = float(model.predict(feat_df.values)[0])
            except Exception:
                # pipeline.predict expects DataFrame maybe; try passing feat_df
                try:
                    pred = float(model.predict(feat_df)[0])
                except Exception as exc:
                    print('Prediction failed for', municipio, top_cultivo, y, exc)
                    pred = None
            gb_preds[municipio][top_cultivo][str(y)] = pred
    # write JSON
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(gb_preds, f, ensure_ascii=False, indent=2)
    print('Wrote', OUT_PATH)


if __name__ == '__main__':
    main()
