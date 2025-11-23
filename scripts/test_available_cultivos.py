import json
from pathlib import Path

ci = json.loads(Path('public/data/cluster_info.json').read_text(encoding='utf-8'))
df = json.loads(Path('public/data/df_final_summary.json').read_text(encoding='utf-8'))

for municipio in ['Yopal', 'Manizales', 'Pereira']:
    muni_entry = df.get('municipios', {}).get(municipio)
    list_ = muni_entry.get('cultivos') if muni_entry and muni_entry.get('cultivos') else None
    if (not list_ or len(list_)==0) and ci and municipio in ci.get('municipio_crops', {}):
        muni_crops = ci['municipio_crops'][municipio]
        if isinstance(muni_crops, list) and len(muni_crops):
            list_ = [c.strip() for c in muni_crops]
    if not list_:
        list_ = df.get('cultivos', [])
    clu = ci.get('municipio_cluster', {}).get(municipio)
    combined = list_.copy()
    if clu in (0,1) and ci.get('cluster_crops') and str(clu) in ci['cluster_crops']:
        clusterList = [ (c.get('cultivo') or '').strip() for c in ci['cluster_crops'][str(clu)]]
        seen = set([ (c or '').strip() for c in list_])
        for c in clusterList:
            if c not in seen:
                combined.append(c)
    print('---')
    print('Municipio:', municipio)
    print('Cluster:', clu)
    print('Contains Caña?', 'Caña' in combined)
    print('Sample available (first 20):', combined[:20])
