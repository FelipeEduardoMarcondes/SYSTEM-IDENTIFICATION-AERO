import json

def dump_notebook(filename, output_txt):
    with open(filename, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    with open(output_txt, 'w', encoding='utf-8') as out:
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                out.write(f"# --- CELL {i} ---\n")
                out.write("".join(cell['source']))
                out.write("\n\n")

dump_notebook('identificação-collab/RLS_AEROPENDULO.ipynb', 'scratch/RLS_code.py')
dump_notebook('identificação-collab/BLS_AEROPENDULO.ipynb', 'scratch/BLS_code.py')
