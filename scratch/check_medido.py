import os
import re

# Check scripts for 'Medido'
scripts = [
    'sysid/notebooks/identificação-collab/IDENT-HELON/narx.py',
    'sysid/notebooks/identificação-collab/IDENT-HELON/mpc_v4.py',
    'fluxograma-tcc/04_modelagem_dinamica/1_narx_polinomial/narx.py',
    'fluxograma-tcc/05_nmpc_casadi_e_dataset/scripts_mpc/mpc_v4.py',
    'fluxograma-tcc/modelagem-narx/narx/narx.py',
    'fluxograma-tcc/modelagem-narx/mpc-narx/mpc_v4.py'
]

for s in scripts:
    if os.path.exists(s):
        with open(s, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if 'Medido' in line:
                    print(f'{s}:{i+1}: {line.strip()}')

