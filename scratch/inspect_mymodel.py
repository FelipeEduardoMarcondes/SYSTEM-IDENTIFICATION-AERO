import json

with open('identificação-collab/ANN_NARX_AEROPENDULO.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'MyModel' in "".join(cell['source']):
            print(f"Cell {i}:")
            print("".join(cell['source']))
