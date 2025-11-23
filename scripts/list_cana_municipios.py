import json
from pathlib import Path
p = Path('public/data/cluster_info.json')
obj = json.loads(p.read_text(encoding='utf-8'))
municipios = []
for muni, crops in obj.get('municipio_crops', {}).items():
    if any(c.strip().lower() == 'caña' for c in crops):
        municipios.append(muni)
print('\n'.join(sorted(municipios)))
