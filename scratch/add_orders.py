import json
import re

def update_orders(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # The exact string to look for
            target = "ordens_teste = [(1,1),(1,2),(2,1),(2,2),(3,2),(2,3),(3,3),(4,3),(4,4),(8, 24), (24, 8), (30, 30), (30, 50), (40, 40), (50, 30), (50, 50), (60, 60)]"
            
            if target in source:
                new_orders = "ordens_teste = [(1,1), (1,2), (2,1), (2,2), (3,2), (2,3), (3,3), (4,3), (4,4), (5,5), (8,8), (10,10), (10,20), (15,15), (20,10), (20,20), (8, 24), (24, 8), (25,25), (30, 30), (35,35), (30, 50), (40, 40), (45,45), (50, 30), (50, 50), (55,55), (60, 60)]"
                source = source.replace(target, new_orders)
                cell['source'] = [line + '\n' if i < len(source.split('\n')) - 1 else line for i, line in enumerate(source.split('\n'))]

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

update_orders('identificação-collab/BLS_AEROPENDULO.ipynb')
