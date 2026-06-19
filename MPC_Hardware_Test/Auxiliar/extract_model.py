import json

notebook_path = r"c:\Users\mathe\OneDrive - Grupo Marista\PUCPR\15 Semestre\CONTROLE AVANÇADO\GITs\1-4 DRONE - GIT\SYSTEM-IDENTIFICATION-AERO\narx_mpc_14_drone_fixed.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'def f_system(' in source or 'f = Function' in source or 'CasADi' in source or 'SX.sym' in source:
            print(f"--- Cell {idx} ---")
            print(source)
