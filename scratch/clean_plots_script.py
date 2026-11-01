import os
import re

files = [
    'sysid/notebooks/identificação-collab/IDENT-HELON/narx.py',
    'fluxograma-tcc/04_modelagem_dinamica/1_narx_polinomial/narx.py',
    'fluxograma-tcc/modelagem-narx/narx/narx.py'
]

for f in files:
    if not os.path.exists(f):
        continue
    with open(f, 'r', encoding='utf-8') as fp:
        c = fp.read()
    
    # 1. Single validation metric_text: use string concatenation with newline
    pattern1 = r'metric_text\s*=\s*\([^\)]*FIT = \{res\[\'fit\'\]:\.1f\}[^\)]*\)'
    replacement1 = 'metric_text = f"RMSE = {res[\'rmse\']:.3f}°\\\\nMAE = {res[\'mae\']:.3f}°\\\\nFIT = {res[\'fit\']:.1f}%"'
    c, n1 = re.subn(pattern1, replacement1, c)
    
    # 2. Consolidated validation txt
    pattern2 = r'txt\s*=\s*f"RMSE = \{res\[\'rmse\'\]:\.2f\}[^"\n]*[\s\S]*?FIT = \{res\[\'fit\'\]:\.1f\}%"'
    replacement2 = 'txt = f"RMSE = {res[\'rmse\']:.2f}°\\\\nFIT = {res[\'fit\']:.1f}%"'
    c, n2 = re.subn(pattern2, replacement2, c)
    
    with open(f, 'w', encoding='utf-8') as fp:
        fp.write(c)
    print(f'File {f}: replaced {n1} single blocks, {n2} consolidated blocks')

# Also fix mpc_v4 files
mpc_files = [
    'sysid/notebooks/identificação-collab/IDENT-HELON/mpc_v4.py',
    'fluxograma-tcc/05_nmpc_casadi_e_dataset/scripts_mpc/mpc_v4.py',
    'fluxograma-tcc/modelagem-narx/mpc-narx/mpc_v4.py'
]

for f in mpc_files:
    if not os.path.exists(f):
        continue
    with open(f, 'r', encoding='utf-8') as fp:
        c = fp.read()
    pattern = r'metric_text\s*=\s*\([^\)]*FIT = \{fit:\.1f\}[^\)]*\)'
    replacement = 'metric_text = f"RMSE = {rmse:.3f}°\\\\nMAE = {mae:.3f}°\\\\nFIT = {fit:.1f}%"'
    c, n = re.subn(pattern, replacement, c)
    with open(f, 'w', encoding='utf-8') as fp:
        fp.write(c)
    print(f'File {f}: replaced {n} mpc blocks')

print('Done fixing!')
