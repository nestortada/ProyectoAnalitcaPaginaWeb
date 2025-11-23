import joblib
path = 'public/data/reg_Producción_GradientBoostingRegressor.joblib'
print('Loading', path)
model = joblib.load(path)
print('Type:', type(model))
print('n_features_in_:', getattr(model, 'n_features_in_', None))
print('feature_names_in_ present?:', hasattr(model, 'feature_names_in_'))
if hasattr(model, 'feature_names_in_'):
    print('feature_names_in_ length:', len(model.feature_names_in_))
    print('feature_names_in_ sample:', list(model.feature_names_in_)[:50])
print('get_params keys sample:', list(model.get_params().keys())[:80])
print('has predict?:', hasattr(model, 'predict'))
