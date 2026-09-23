import os

with open(r'mpc_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

import_block = """import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
PLOTS_DIR = os.path.join(root_dir, "data", "experimentos", "sil", "graficos")
os.makedirs(PLOTS_DIR, exist_ok=True)
"""
content = content.replace("import os, sys\ncurrent_dir", import_block + "current_dir")

savefig_code = r'''plt.savefig(os.path.join(PLOTS_DIR, "".join([c if c.isalnum() else "_" for c in (plt.gca().get_title() or str(id(plt.gcf())))]) + ".png")); plt.show('''

content = content.replace('plt.show(', savefig_code)

with open(r'mpc_v2.py', 'w', encoding='utf-8') as f:
    f.write(content)
