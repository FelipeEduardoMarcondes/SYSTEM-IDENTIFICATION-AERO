import sys

fname = r'c:\Users\vicio\Documents\AEROPENDULO\identificação-collab\IDENT-HELON\mpc_v2.py'
with open(fname, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if i == 626: # Line 627 (0-indexed)
        skip = True
        new_lines.append("    'u_pct': u_sim,\n")
        new_lines.append("    'referencia': x2ref_val\n")
        new_lines.append("})\n")
        new_lines.append("csv_filename = 'simulacao_python.csv'\n")
        new_lines.append("df_export.to_csv(csv_filename, index=False)\n")
        new_lines.append("print(f'\\nSimulation data exported to {csv_filename}!')\n\n")

    if skip and line.startswith('def export_ann_to_c('):
        skip = False
        new_lines.append('# (Optional) We can also export ANN weights as before\n')
    
    if not skip:
        new_lines.append(line)

with open(fname, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
