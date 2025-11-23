import pandas as pd
path = 'public/data/df_final_limpio.xlsx'
print('Reading', path)
df = pd.read_excel(path)
print('Columns:', list(df.columns))
print('\nSample head:')
print(df.head(5).to_string())
