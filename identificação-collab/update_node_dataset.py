import json
import re

nb_path = r'c:\Users\vicio\Documents\SYSTEM-IDENTIFICATION-AERO-main\identificação-collab\NODE_AEROPENDULO.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Change the training dataset
        if "seq_20260529_195254.csv" in source:
            source = source.replace("seq_20260529_195254.csv", "multi-sine-2_0731_17-38.csv")
            
        cell['source'] = source.splitlines(True)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
